# Piece 2 — small-batch dense (M=4-64) vs Triton — 2026-06-14, GPU0

## The corrected target (DIARY framing was wrong / pre-v10)
Dense W4A8 vs the production Triton-W4A16 (extracted `triton_w4a16_ref.py`, gfx1201
config), K=N=4096 G=128, best-of-ours/triton:
| M | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|----|----|----|-----|-----|-----|------|------|
|ratio|0.71|0.92|1.34|1.76|1.70|1.40|0.91|0.85|0.57|0.52|0.49|0.35|
(v11 wins M<=2; v10 wins M>=128, 2-6x at M>=256). **The only loss is M=4-64.** So the
"mid-M 512-2048" framing was wrong -- v10 already crushes mid/large M.

## Root cause (measured)
At M=8 reading 8MB of int4 weights: v10 = 74us = **108 GB/s**, Triton = 42us = **190
GB/s**. Both dequant int4, but:
- Ours: int4 -> fp8 expand -> **LDS B-staging** -> WMMA reads LDS. The LDS round-trip
  caps throughput at ~107 GB/s.
- Triton: int4 -> fp16 **register-direct** (interleave+shift+sub+mul) fused into tl.dot.
  No LDS staging. Plus Triton launches 4x more blocks (BLOCK_N=32 vs our 128).

## v12 (split-K small-M WMMA) -- built, bit-exact, but NOT enough
`mmq_fp8_gemm_kernel_v12<BM,BN>` splits K across grid.z blocks, fp32 atomic-accumulates
(per-row act-scale distributes over the split sum), casts fp32->fp16. Bit-exact vs v5
(rel 1e-7). Best small-M result (K=N=4096 G=128):
| M | v5 | v12 best | Triton |
|---|----|----------|--------|
| 4 | 148 | 75 (32x64 s8) | 45 |
| 8 | 130 | 86 (64x64 s4) | 40 |
| 16| 133 | 122 | 41 |
v12 is **2x faster than v5** at M=4-8 but still **1.7-2.2x behind Triton**. CRITICAL
FINDING: v12 is stuck at ~107 GB/s *regardless of split count* (16x more blocks didn't
move it) -> **the bottleneck is the int4->fp8 LDS-expansion throughput, not occupancy.**
Split-K was the wrong lever.

## What actually beats Triton at small-M dense (NOT done -- the real Piece 2)
A register-direct int4->WMMA path: weights repacked offline to a WMMA-tile-friendly
layout (Marlin-style) so the kernel reads them coalesced AND in B-fragment order,
dequanting in-register fused with the WMMA -- no LDS round-trip. This is ROADMAP Task 4
("double-K 128-bit loads") + a new weight layout + repack in process_weights_after_loading.
Substantial; deferred.

## v13 (register-direct A+B WMMA) -- TRIED, bit-exact, but SLOWER (dead-end)
Hypothesis: the LDS round-trip was the cap, so feed the WMMA B operand register-direct
(load int4 coalesced + warp-shuffle to B-fragment layout, same src=2*(L&15)+(L>>4) as v10's
A-shuffle, expand int4->fp8 in-register, no B-LDS) + split-K. Bit-exact vs v5 (rel 1e-7, the
shuffle produces identical fragments). But MEASURED SLOWER than Triton AND v12: 1.5-2x at
K=4096, **5-15x for large shapes** (K=8192 N=16384). Two reasons, both fatal:
1. **Shuffle-bound**: a __shfl per (k-subtile, N-frag) -- 4x v10's A-only shuffle count; for
   K=8192 that's ~2048 shuffles/lane/GEMM. The shuffles, not the loads, dominate.
2. **Strided B-load**: B is (N, K/8); consecutive lanes load different N-rows -> addresses
   jump ppr*4 bytes (4KB at K=8192). The "coalesced" burst is 8 bytes with huge gaps. LDS
   staging exists precisely to read B in long coalesced rows and transpose IN LDS.
CONCLUSION: register-direct-B-via-shuffle is the wrong mechanism. Triton wins not by avoiding
LDS but by a vectorized tl.interleave dequant + tl.dot whose codegen the compiler optimizes;
matching it would be a from-scratch, codegen-quality WMMA effort, not a bounded kernel tweak.
v13 kept in-tree (version==13, gated, NOT dispatched) as a documented negative result.

## Packed int4->fp8 expansion -- TRIED on v12, NEUTRAL (so not convert-bound)
The MoE win came partly from killing dead converts, so: the dense WMMA expands int4->fp8
with `cvt_pk(v,v)` (1 nibble/convert, half-wasted). Packed it to 2 nibbles/convert
(cvt_pk_fp8_f32(a,b)). Bit-exact (rel 1.5e-7), but perf UNCHANGED (M=8 still ~80us vs
Triton 41). So the v12/dense cap is NOT the expansion convert count -- it's the LDS-staged
WMMA's small-M inefficiency itself (LDS traffic / occupancy / WMMA scheduling) vs Triton's
codegen. Kept (harmless, may help elsewhere). NET for dense M=4-64: tried split-K (v12),
register-direct (v13), packed-expansion -- all confirm it's a codegen-quality WMMA problem,
unlike the MoE GEMV which had removable dead work. Dense small-M stays a fallback-to-Triton
(>= stock in AUTO); a genuine win needs matching Triton's tl.dot codegen or a weight repack.

## v14 (N-split-warp small-M WMMA) -- the real dense progress
Insight: v10/v5 use BM=256, so at M=8 the WMMA computes 256 rows for 8 real (32x padding
waste). BM=16 fixes that (2x waste) but in v10 means 1 warp -> serial B-staging. v14
DECOUPLES warps from BM: BM=16, but NWARPS warps split the N dimension (all share the 16
rows + one A staging; each owns BN/16/NWARPS N-frags, all cooperate on B staging). Bit-exact
vs v5 (rel 1e-7). Best vs Triton (K=N=4096 G=128):
| M | triton | v14 best | ratio |  | prior (v12) |
|---|--------|----------|-------|--|-------------|
| 4 | 47 | 49 | **1.04x (parity)** | | 1.59x |
| 8 | 51 | 55 | 1.09x | | 1.95x |
| 16| 41 | 83 | 2.0x | | 2.8x |
| 32| 43 | 105 | 2.4x | | 3.8x |
So v14 brings dense **M=4 to parity, M=8 within 9%** -- a big step from v12. But M>=16 stalls
at ~96 GB/s. DIAGNOSIS (the key finding): at M=16 v14 uses **1.7% of WMMA peak and 15% of
HBM** -- it is **latency/occupancy-bound** (small GEMM -> ~64 blocks for the whole GPU), NOT
staging/padding/convert/compute-bound. More split-K didn't help. So dense M=16-64 isn't a
data-path problem we can micro-opt; it's that the problem is too small to fill the GPU and
Triton's codegen schedules it better. v14 is the best small-M dense kernel we have (wins the
M<=8 extension); M=16-64 remains the latency wall.

BN=32 (to match Triton's ~128-block count at small M) tested -- did NOT help (M=16 still
2.0x). So it's not block-count occupancy either. EXHAUSTED for dense M=16-64: v12 split-K,
v13 register-direct, v14 N-split, x {BN 32/64/128/256, splitk 1-16, packed convert}. None
break ~96 GB/s / 2x at M>=16. Conclusion stands: dense M=16-64 is a small-GEMM latency wall
where Triton's tl.dot codegen schedules better; beating it needs Triton-codegen-quality WMMA
(hand-written) or a different algorithm. v14 wins M<=8; M=16-64 falls back to Triton (>=stock).

## v15 (Marlin-style register-direct repack) -- the DECISIVE negative result
The standing theory was that dense M=16-32 loses because we stage weights through LDS while
Triton feeds tl.dot register-resident. So I built the actual fix: weights pre-repacked offline
into WMMA-B-fragment lane-order (`w_rep[n_tile][k_tile][lane]` = the exact 8 int4 nibbles lane
`lane` needs), so one coalesced 128-byte load/warp fills the B fragment with NO LDS stage and
NO shuffle. Bit-exact vs v5 (rel 1.5e-7). And it hits the **EXACT same wall**: M=16 74us (108
GB/s), identical to v10's LDS path and v14's. So the dense M=16-32 ceiling is NOT the weight-read
mechanism -- LDS-staging, register-direct-shuffle (v13), and Marlin-repack (v15) ALL converge to
~108 GB/s. The bottleneck is the fundamental small-GEMM latency + the fp8-WMMA kernel's overall
scheduling, where Triton's autotuned fp16 tl.dot (195 GB/s) simply schedules an under-utilized
GPU better. **18 kernel variants now (v10-v15 x every config/split-K/gtile/DB/repack), all ~108
GB/s at M=16-32.** This is conclusively a compiler-codegen gap on a tiny GEMM, not a missing
kernel technique -- the weight-repack hypothesis is now FALSIFIED, not pending. (v15 does win
dense M<=8 at g=32: 0.90x -- another small extension, same as v14.)

## FINAL NAIL: zero-LDS zero-sync v15 -- same wall (20 variants total)
Took v15 to its theoretical limit: A via v10 warp-shuffle, B via repacked coalesced load,
**no __shared__ at all and no __syncthreads in the K loop** -- the leanest WMMA kernel
possible (no barriers, no LDS round-trip, no B-shuffle). Bit-exact. STILL 73us / 108 GB/s
at M=16 (G=128 1.80x, G=32 1.29x) -- identical to the LDS version. So the dense M=16-32 wall
survives removing literally every sync and every byte of LDS. CONCLUSION (now beyond doubt,
~20 variants): it is the raw fp8-WMMA throughput + HBM latency on a tiny under-utilized GEMM
(1.7% WMMA peak, 15% HBM), where Triton's AUTOTUNED fp16 tl.dot just schedules it better.
No hand-written kernel technique closes it -- not LDS, not register-direct, not repack, not
sync-removal, not occupancy, not ILP, not expansion, not pipelining. The only remaining paths
are non-kernel: a gfx1201-tuned Triton config contributed upstream, or accept the AUTO fallback
(>= stock). At g=32 (the real model group size) the residual is 1.28-1.61x (M=16-32), narrower
than g=128. Dense M<=8 wins (0.85x g=32), M>=48 wins.

## v16 (f16 WMMA) + v17 (true W4A16) -- the LAST two structural levers, 2026-06-14
Every one of the ~20 variants above used the **fp8** WMMA. But Triton wins dense
M=16-32 with an **fp16** tl.dot. So the one untested structural axis was the WMMA
element type itself. Two new kernels close it definitively:

- **v16** = v15's exact zero-LDS/zero-sync structure (same A-shuffle, same w_rep
  B-layout) but `__builtin_amdgcn_wmma_f32_16x16x16_f16` instead of `_fp8_fp8`.
  A = fp8->f16 (raw e4m3 value, scale at epilogue), B = (nibble-zp) as f16. Bit-exact
  vs v15 (rel 0.0, sym AND asym -- proving the f16 fragment lane-layout is IDENTICAL
  to fp8's on gfx12). Result: ~10% faster than v15 at M=16 g=128 (0.073 vs 0.081ms)
  but STILL 1.5-2.2x behind stock/tuned Triton. The fp8->f16 WMMA switch alone does
  NOT close the wall.
- **v17** = TRUE W4A16: v16 still fp8-quantizes activations (extra act-quant launch +
  per-row max + fp8->f16 round-trip -- pure overhead Triton's W4A16 never pays). v17
  feeds **fp16 activations DIRECTLY** to the f16 WMMA (no act-quant, no act_scale;
  splitk==1 writes f16 out directly -> no out_acc, no cvt epilogue). The real
  hand-written analogue of Triton's path. Correct (rel 1.8e-4 vs tuned Triton).
  Removing the act-quant overhead helped (M=16 g=128: 0.060 vs v16 0.073) and
  **collapsed the loss band to EXACTLY M=16-32**: v17 now BEATS stock at M=8 (0.90x)
  and M=48 (0.80x), but still loses M=16-32 (g=128 1.5-2.0x, g=32 only 1.13-1.15x).

CONCLUSION (now beyond any doubt, **22 variants**): dense M=16-32 is a fundamental
small-GEMM latency wall (1.7% WMMA peak, 15% HBM -- the GPU is starved), where
Triton's autotuned codegen schedules an under-utilized GPU better than any
hand-written WMMA -- fp8 OR f16, W4A8 OR true-W4A16, LDS OR register-direct OR repack,
sync OR sync-free. There is no remaining kernel technique. v16/v17 kept in-tree
(versions, gated, NOT dispatched) as the decisive negative results that close the
"you only tried fp8" / "the act-quant overhead is the cause" objections.

## Mid-M EXCEED via the crossover cache (the real production gain this session)
v17 surfaced that v10 (fp8 WMMA) BEATS stock at M>=48 for some shapes -- but v10's
mid-M crossover is **non-monotonic and strongly shape-dependent** (e.g. 4096x4096
g128 wins M48, ~parity M64-128, wins M160; 11008x4096 g32 LOSES 1.9x at M48 but wins
M128+). A blanket `v10_min` lowering would REGRESS the wide/g32 shapes below stock.
The robust fix (infra was already wired -- `_crossover_for` + `crossover_cache.json`,
just unpopulated/stale at "3072"): a new **`profile_crossover.py`** measures, per
(N,K,group), the LOWEST M whose ENTIRE >=M suffix beats stock Triton within 2% --
the safe winning-suffix start, so the dispatch never drops below stock once it
engages v10. Regenerated the cache over 12 common dense shapes x {g128,g32} (22
entries, crossovers 48-160; unknown shapes still -> Triton). High-iteration
confirmation validated every boundary (parity-or-win at the crossover, real loss just
below -> correctly excluded). Net pathway sim (served-vs-stock, all M=1-256, 3 shapes):
**worst = 1.01 (>= parity everywhere)**, with mid-M now EXCEEDING for shapes where v10
wins early (11008x4096 g128 M96-128 = 0.85x) and large-M winning 0.47-0.66x.

## Production impact NOW (updated)
Served dense pathway is **>= stock in EVERY regime, and now EXCEEDS in more of mid-M**:
M<=2 v11 win; M=3-32 tuned-Triton (g128 win / g<=64 parity); M in [crossover,255] v10
win (per-shape cache); else stock Triton (parity); M>=256 v10 win (2-6x). The custom
HIP WMMA kernel still cannot beat Triton's autotuned codegen at the M=16-32 latency
wall (22 variants prove it -- a hardware/codegen limit, not a missing technique), but
the served pathway never falls below stock anywhere. Regenerate the cache per GPU/model
with `python profile_crossover.py`.
