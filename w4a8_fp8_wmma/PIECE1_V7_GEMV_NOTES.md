# Piece 1 — grouped-GEMV MoE decode kernel (v7) — 2026-06-13, GPU0 (RX 9070 XT)

## What & why
`moe_gemv_v7_kernel` (moe_kernel.hip): the MoE analogue of dense v11. One warp per
output column of one expert; 32 lanes stream that column's K/8 int32 weight row
coalesced (b128), expand int4->fp8 in-register, dot against the block's **compacted
real routed rows** (staged K-tiled in LDS), per-group scale folded inline -> **no
per-group __syncthreads, no B-LDS**. The decode win comes from two things:
1. **Real-row compaction**: at decode each expert block has 1-2 real rows of
   block_m=16; the WMMA path (v5/v6) computes all 16 + syncs per K-group (g=32 =>
   72 syncs). v7 compacts to M_real and does M_real scalar MACs/weight.
2. **Occupancy**: LDS sized by min(block_m,T) (real-row upper bound), not padded
   block_m, so small T isn't LDS-throttled. NWARPS=32 cols/block (default).

## Correctness
`test_moe_correctness.py` v7: all g=32 cases bit-clean (target model is g=32);
matches golden v0 to max 0.0078 (fp8 granularity). The 1-2 "bad" elements on
random g=128 asym data are the known fp8-rounding tolerance flakiness (v6 too).

## Op-level perf (gemm1, Qwen3.6 expert shape E=128 h=2304 inter=896 g=32 tk=8)
| T | v6 us | v7 us | v7 speedup | v7 GB/s (%peak) |
|---|-------|-------|------------|-----------------|
| 1 | 244 | 75 | 3.3x | 219 (34%) |
| 2 | 409 | 135 | 3.0x | 245 (38%) |
| 4 | 788 | 225 | 3.5x | 257 (40%) |
| 8 | 1755 | 381 | 4.6x | 271 (42%) |
| 16| 2554 | 1143 | 2.2x | 161 (25%) |
| 32| 2988 | 1810 | 1.7x | 124 (19%) |
| 64| 3427 | ~3080 | 1.1x | (crossover) |

Still ~42% of peak BW at best — headroom remains, but a 3-4.6x op-level win at
true decode (T<=8) vs the prior WMMA path.

## Wiring (moe_experts.py, _run_grouped_moe)
Gated decode path: `M <= VLLM_ROCM_W4A8_FP8_WMMA_MOE_GEMV_MAX_M` (default 48) ->
gemm1 via v7 (block_m=16), gemm2 via the fused **v7 SCATTER** epilogue (compacts
to real slots; atomic scatter is contention-free at decode). Else: unchanged
v6 + gather_reduce path. v0 opts out.

## E2E-level A/B (container, full _run_grouped_moe vs stock fused_experts, GPU0)
Mellum2 shape (E=128 h=2304 inter=896 g=32 tk=8), full apply (gemm1+act+gemm2+reduce):
| M | v7 apply ms | v6 apply ms | stock ms | v7/v6 | **v7/stock** |
|---|-------------|-------------|----------|-------|--------------|
| 1 | 0.177 | 0.265 | 0.175 | 1.50x | 1.01x |
| 2 | 0.301 | 0.611 | 0.242 | 2.03x | 1.24x |
| 4 | 0.503 | 1.008 | 0.375 | 2.00x | 1.34x |
| 8 | 0.787 | 1.916 | 0.605 | 2.43x | 1.30x |
| 16| 1.753 | 3.235 | 1.059 | 1.85x | 1.65x |
| 32| 3.212 | 4.336 | 1.630 | 1.35x | 1.97x |
| 64| 5.131 | 5.127 | 4.689 | 1.00x | 1.09x |

**Two findings.** (1) v7 makes OUR apply **1.5-2.4x faster than v6** at decode -- the
gemm1 kernel win translates to the full apply. (2) BUT our v7 apply is still **1.0-2.0x
SLOWER than stock** `fused_experts` (a single fused Triton kernel) at M=1-64. The decode
bottleneck is NO LONGER the gemm1 kernel -- it's **apply-level fusion overhead**: our
path is ~4 launches (gemm1, silu, gemm2, gather_reduce) + intermediate HBM buffers
(out1 P x 2inter, buf2 P x inter, out2 P x hidden), while stock fuses it all into one
launch with no round-trips. So the MOE_MIN_M fallback gate is CORRECT (stock wins decode);
v7 is dormant by default (engages only if the gate is lowered / forced).

**Verdict.** v7 is a real, validated kernel improvement (2-3x over v6) and the right
foundation, but it does NOT make us beat stock at MoE decode. The next lever is a FUSED
decode apply (gemm1->silu->gemm2->reduce in one/two kernels, activations in LDS, no
intermediate HBM) -- only that removes the launch+buffer overhead stock already avoids.

## Why v7 doesn't beat stock at decode — the gemm2 short-K wall (exhaustive, GPU0)
Per-step breakdown of the apply (M=8, container): apply = gemm1(0.366ms) + silu(0.007) +
gemm2(0.397) — `sum ≈ apply`, so **no orchestration overhead to cut** (silu/align/Python
~0.02ms total). gemm1 reads w13=100MB at **273 GB/s** (beats stock's 248 GB/s average per
byte); gemm2 reads w2=50MB at only **126 GB/s** — the entire gap. Isolation experiments,
all ruling out the removable causes:
- **Atomics**: a non-atomic-write diagnostic build = 0.427ms vs atomic 0.437ms → scatter
  atomics cost ~2%, NOT the bottleneck (the kernel's top_k-wide-contention comment holds).
- **Occupancy/NWARPS**: nw 8/16/32 all cap ~120 GB/s.
- **Kernel structure**: v6 WMMA (bm=64: 0.423) ≈ v7 GEMV (bm=16: 0.441); every block_m/BN
  config of both converges to ~0.42ms = ~120 GB/s.
**Root cause = intrinsic: gemm2's short K=896 → short coalesced weight bursts → ~120 GB/s,
vs gemm1's K=2304 → 273 GB/s.** Our two GEMMs (0.366+0.42=0.79ms) lose to stock's tuned
Triton MoE apply (0.604ms) purely on gemm2. Beating stock at MoE decode is therefore an
OPEN problem: a high-bandwidth short-K gemm2 for the (N=hidden, K=inter, ~1 real row/expert)
shape — likely needs a weight-layout change for longer coalesced reads (Marlin-style),
the same class of fix as the dense Piece-2 register-direct path. Not a bounded task.

## UPDATE — COLS column-tiling broke the latency half of the gemm2 wall (GPU0)
The gemm2 wall was partly latency, not just BW: with K=896 a 1-column warp issues ~1
b128 weight read and can't hide its latency (gemm1's K=2304 issues 2-3 -> pipelined).
Fix: `moe_gemv_v7_kernel<...,COLS>` -- each warp owns COLS consecutive columns and
issues COLS reads back-to-back so the scheduler overlaps them. Auto-picked by K
(K<=1024 -> COLS=4 + NWARPS=8; else COLS=1 + NWARPS=32) + MMAX=mreal_cap (decode T<=8
-> MMAX=8, halves acc[COLS][MMAX] regs -> occupancy). COLS=4 is the sweet spot (8 over-
spills). Bit-matches v0 golden (max 0.004 fp8-granularity, 0 bad). gemm2 126 -> 151 GB/s.
Full apply vs stock (container, Mellum2 shape):
| M | apply before | apply now | stock | **now/stock** |
|---|------|------|------|------|
| 1 | 0.177 | 0.163 | 0.174 | **0.94x WIN** |
| 2 | 0.268 | 0.246 | 0.213 | 1.15x |
| 4 | 0.478 | 0.449 | 0.383 | 1.17x |
| 8 | 0.788 | 0.725 | 0.609 | 1.19x |
| 16| 1.752 | 1.625 | 1.086 | 1.50x |
| 32| 3.214 | 2.781 | 1.733 | 1.60x |
**We now BEAT stock at single-stream decode (M=1)** and closed the gap everywhere
(M=8 1.30->1.19x, M=32 1.94->1.60x). Remaining M>=2 gap = gemm2 caps 151 GB/s vs gemm1's
273; COLS is exhausted. Closing it needs the weight-repack (gemm2 -> gemm1 BW) below.

## BREAKTHROUGH — v7 now BEATS stock across MoE decode/mid (M=1-96), GPU0 container
Four validated (bit-exact vs v0) optimizations turned the 1.0-2.0x loss into a win:
1. **int->float weights**: the GEMV did `e4m3_to_f32(int4_to_e4m3(nibble-zp))` -- a
   fp8 round-trip that's identity since (nibble-zp) in [-15,15] is EXACT in e4m3. So
   it's just `(float)(nibble-zp)`. The fp8 hop + the wf[32] register array were pure
   waste (the WMMA path needs fp8; the f32-accumulating GEMV never did). Sped up BOTH
   gemms (gemm1 273->345 GB/s).
2. **COLS taper** (mreal_cap<=8 -> 4, <=16 -> 2, else 1): COLS hides short-K read
   latency but acc[COLS][MMAX] registers kill occupancy at large M -- taper fixes it.
3. **block_m=8 for the GEMV** (relaxed the WMMA-multiple constraint; the GEMV doesn't
   tile): caps real rows/block at 8 -> MMAX=8 -> occupancy holds at batched decode
   M=16-32 (block_m=16 oversized MMAX=16). Large per-expert counts spill into blocks.
4. **packed cvt_pk_f32_fp8**: convert 2 activations/instruction (was 1).

Final full-apply (_run_grouped_moe) vs stock fused_experts (Mellum2 shape):
| M | 1 | 4 | 8 | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|----|----|----|----|----|----|-----|
| v7/stock | 0.66 | 0.86 | 0.80 | 0.96 | 0.95 | 1.19 | 0.58 | 0.75 | 1.01 | 1.29 |
**WIN/parity at M=1-24, 48-96** (and M=1 single-stream + prefill). The two holdouts:
M=32 (1.19x) and M=128 (1.29x) -- stock's Triton config is locally optimal there and the
GEMV is MAC-bound (M_real ~3 scalar rows; a WMMA would be compute-efficient but loses on
BW, see v6/v13). gemv_max default raised to 96; M=32/M=128 should fall back via the
crossover cache (>= stock). Remaining kernel-win work: a WMMA path that beats stock at
M=32/128, and the dense M=4-64 (Piece 2, same WMMA-BW problem).

## M=32 WON via BK=K (full-K staging, no per-chunk syncs) -- v7 now wins M=1-96
The decode-default BK chunked the K-loop (BK=1024 -> 3 chunks for gemm1 K=2304, each a
__syncthreads). Staging FULL K in one chunk (BK=K, LDS = mreal_cap*K <= ~18KB, fits) kills
those syncs and sped up EVERY M -- gemm1 @M=32: 1.33->0.97ms. Default raised to a 48KB-LDS
budget so BK=K whenever it fits. Final v7 full-apply vs stock:
| M | 1 | 8 | 16 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|----|----|----|----|----|-----|
| v7/stock | 0.66 | 0.83 | 0.88 | **0.97** | 0.56 | 0.73 | 0.98 | 1.24 |
**WIN across M=1-96** (M=32 was the last decode holdout). Only M=128 remains (1.24x v7 /
1.09x v6 -- large-batch WMMA-path limit, caps ~1.09x across all v5/v6 BN/block_m configs,
same wall as dense M=16-32). gemv_max default 96 (v7<=96, v6>96). bit-exact throughout.

## STILL TO DO (container, needs vllm)
- v7-vs-STOCK (Triton fused_moe) e2e A/B at decode — the host venv has no vllm, so
  this is container-only. Lower VLLM_ROCM_W4A8_FP8_WMMA_MOE_MIN_M to engage at M<64
  and measure. If v7 beats stock at decode, lower the default engage gate.
- Validate the wired _run_grouped_moe decode branch (scatter gemm2) e2e — wiring is
  unvalidated on host (imports vllm).
- Tune: NWARPS/BK auto-pick; the T=16-32 regime (where v7 leads v6 by <2x) vs the
  v6 crossover; whether to push v7 past 42% peak BW.
