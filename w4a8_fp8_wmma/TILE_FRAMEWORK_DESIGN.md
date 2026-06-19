# RDNA4 W4A8 tile-framework — design + perf-candidate audit

Worktree: `feat/w4a8-tile-autotune`. Goal (from the "RDNA4-HipKittens?" thread): factor the
kernel's tile/pipeline/swizzle choices into a reusable, parameterised layer so we can **repeat
the W4A8 exercise for a different quant** cheaply — *and* note any perf wins spotted while
reading. Both deliverables below. Nothing here has been GPU-validated yet (shared box; one
batched window is requested — see §3).

---

## 1. Perf-candidate audit — honest result

I read the full live MoE path (`moe_kernel.hip`, 1409 lines) and grepped the dense kernel +
all design docs. **The cheap micro-optimisations a fresh reader spots are already documented as
tried/neutral/parked.** This is a *good* result: it says the kernels are well-optimised and the
framework's value is cheaper *exploration*, not unharvested wins. Concretely:

| Idea I spotted | Verdict from the repo's own docs |
|---|---|
| Pack 2 nibbles per `cvt_pk_fp8_f32` (B-staging expands 1 nibble/convert, half-wasted) | **TRIED, NEUTRAL** on dense v12 (`PIECE2:57-60`), bit-exact, perf unchanged → not convert-bound. Decode v7 already packs it (`PIECE1:117`). |
| 16-entry int4→fp8 LUT via `v_perm_b32` | Documented as a known alternative (`NOTES.md:104-106`); same bound as above. |
| Kill the B-LDS round-trip (register-direct weights) | v15/16/17 did exactly this → **lost at small M** (the "regdirect wall", `PIECE2`). Parked. |
| Remove A from LDS (warp-shuffle gather) | Already shipped: dense v10, MoE v6. |

### Candidates that survive cross-referencing — ALL ALREADY ACTIVE WORKTREES
Cross-referencing against `git worktree list` shows **every** surviving candidate is already an
in-flight effort. So this deliverable is "**don't duplicate — these are being worked**." Listed
with their owning worktree; all target the measured gemm2 decode-BW wall (gemm1 273 vs gemm2
126–151 GB/s):

| Candidate | Owning worktree | Notes |
|---|---|---|
| N-interleave burst-repack (gemm2) | `feat/w4a8-burst-repack-research` | `RESEARCH_burst_repack.md §2.3`; offline relayout, ~150→200–230 GB/s |
| Decode-BW levers (incl. nt/streaming load hints) | `feat/decode-bw-levers` | the ~1-line `slc`/non-temporal A/B lives here |
| MoE apply-level fusion (drop intermediate HBM) | `feat/w4a8-moe-apply-fusion` | gemm2→gather-reduce fusion |
| Act-quant fusion into prologue | `feat/w4a8-actquant-fusion` | fold `compute_act_fp8` into the gemm prologue |

> Note `v7` already issues **b128** weight loads (`moe_kernel.hip:953`, `v4i_t`), so the Explore
> summary's "v7 currently b64" is **stale** — the b128 double-K item is a *dense v5* idea, not a
> live-path one. Don't fund it for the 35B.

**Conclusion:** there are no un-owned perf wins to add from a fresh read. The framework's payoff is
*cheaper future exploration* + delivering the **grouped-MoE SWMMAC kernel** (below), which is a
genuinely un-owned, live-35B win.

---

## 2. Framework design

### 2.1 What it is NOT
Not a HipKittens clone. HK's marquee primitive (direct global→LDS, `global_load_lds`) **does not
compile on gfx12** ([[rdna4-no-direct-vmem-to-lds]]); and HK has no quant path. So this is a thin,
RDNA4-specific, W-quant-A8 tile layer — the shared scaffolding our kernels already imply, made
explicit and parameterised.

### 2.2 The anchor (decides whether this earns its keep)
A parameterisation that can only re-express v5/v10/v11 is a *knob refactor with one instantiation*.
To deliver the stated "different quants" goal it must be anchored on a concrete **second**
instantiation. Candidate 2nd targets, in order of repo-evidence:

- **W4A8-sparse via SWMMAC** — RDNA4 has `v_swmmac_*_fp8/iu4` (~2× throughput,
  [[rdna4-supports-sparse-fp8-int4-swmmac]]); the PPL gate is **green** at 6:8 / 4:6 sparsity
  ([[slidesparse-ppl-gate-green]]). Shares *everything* with today's kernel except the inner MMA
  (WMMA→SWMMAC) and the weight layout (dense→(2N-2):2N pruned). Cleanest abstraction boundary,
  and it's the documented research direction. **CHOSEN anchor.**

  **The SWMMAC primitive is ALREADY PROVEN — anchor on the sibling worktree, do NOT re-derive.**
  `feat/swmmac-microbench` (`/home/pat/code/vllm-gfx1201-swmmac`, `RESEARCH_swmmac.md`) has, all
  bit-exact-validated on GPU:
  - the full sparse-A operand layout + compression-**index encoding** cracked (the ROCm #6025
    unknown), recipe in `RESEARCH_swmmac.md §3` (`probe_swmmac_idx.hip`, 0/256). *My earlier
    "remaining unknown: idx semantics" was wrong — it's solved.*
  - measured ceiling **1.95×** (sparse-fp8 vs dense-fp8), and a tiled GEMM **1.3–1.6× typical**.
  - **§4d (the number that matters for us): int4-weight→fp8-sparse beats our dense-int4 W4A8 by
    1.17–1.55× AND uses less memory** (~3 bit/wt vs 4). A real win over the live stack.
  - a working torch op (`torch.ops.swmmac.fp8_gemm`), fp8 quant + 2:4→SWMMAC repack, and a vLLM
    plugin (`--quantization swmmac`); dense Sparse-Llama-8B serving is the last (GPU-window) step.

  Signature, for reference:
  `__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(v2i_t a_sparse, v4i_t b_dense, v8f_t c, uint idx) -> v8f_t`
  (A = compressed pruned **weight** fp8, B = dense **activation** fp8, fp32 accum — same acc shape
  as today's dense `wmma_f32_16x16x16_fp8_fp8`).

  **THE GAP (verified un-owned):** all the SWMMAC work above is the **dense** GEMM / dense model.
  `RESEARCH_swmmac.md §4` itself says *"MoE is separate… the production target is the grouped-MoE
  GEMM (`moe_kernel.hip`); these dense shapes characterize the primitive."* There is **no
  grouped-MoE SWMMAC kernel** (confirmed: no moe/expert refs in the swmmac kernels). That is what
  the framework delivers.
- **W4A4** (int4 acts via int4 WMMA) — bigger accuracy risk, no green gate yet.
- **Pure consolidation** (no 2nd target) — just unify the env knobs; honest but doesn't hit the
  stated goal.

### 2.3 Abstraction boundary (what's shared vs per-kernel)
Reading v5/v6/v7 + the fused-silu variants, the **shared** spine is identical across all of them:

- act fp8 quant (`moe_compute_act_fp8_kernel`) — per-token absmax → e4m3
- int4 weight unpack + zero-point + **in-register group-scale fold**
- LDS staging primitives (bank-padded `LDSBK = BK + LDS_PAD`, coalesced uint32/uint128 loads)
- the grouped-MoE contract (sorted_token_ids / expert_ids / padding guards / scatter epilogue)
- epilogue (per-row act-scale, fp16 store **or** atomic scatter)

The **per-kernel** axes (this is exactly what becomes a `TileConfig`):

| Axis | Today (scattered) | Framework form |
|---|---|---|
| Tile width BN | `VLLM_W4A8_MOE_BN` (64/128) | `cfg.BN` |
| Group-stage depth | `VLLM_W4A8_MOE_GTILE` | `cfg.GTILE` |
| block_m / warps | `..._MOE_BLOCK_M`, derived | `cfg.BLOCK_M` |
| A residence | v5=LDS / v6=shuffle | `cfg.A_IN_LDS` |
| Inner MMA | WMMA builtin (hard-coded) | `cfg.MMA` ← **WMMA vs SWMMAC** |
| Decode COLS/NWARPS/BK | 5 env vars in `run_moe_gemm` | `cfg.{COLS,NWARPS,BK}` |
| Weight layout | dense K-major | `cfg.WLAYOUT` ← dense vs sparse/burst-repack |

A `TileConfig` struct (compile-time template params + a runtime launcher table) replaces the
~10 `std::getenv` sites in `run_moe_gemm` / `run_moe_gemm1_silu`, and a single autotune harness
sweeps `TileConfig`s and writes the per-shape winner to a cache (folding the existing
`crossover_cache.json` + `profile_crossover.py` into one mechanism).

> **CORRECTION (2026-06-18, after reading the committed `swmmac_op.hip`).** The earlier claim that
> SWMMAC is "the same kernel with `cfg.MMA=SWMMAC`" — a clean *policy swap* on the `MMA::mma(a,b,c)`
> interface — is **wrong**. SWMMAC has a different GEMM **dataflow**, not just a different builtin:
> - **Operand roles swap**: in `swmmac_op.hip` the **weight is the sparse-A operand** and the
>   **activation is the dense-B** (16x16x**32**); the WMMA grouped kernel has A=activation, B=weight.
> - **No int4→fp8 LDS expansion**: the 2:4-compressed fp8 weight + per-lane index are read straight
>   from DRAM (half the bytes); the whole `int4_signed_to_e4m3` staging loop disappears.
>
> So the honest reusable surface is the **spine** — the grouped-MoE contract, the epilogue
> (act-scale + fp16/scatter), the group-scale fold, the bank-padded LDS/tiling helpers, and the
> autotuner. The **GEMM loop is per-backend.** SWMMAC is therefore a **sibling kernel that shares
> the spine, NOT a `cfg.MMA` instantiation.** That is the accurate statement of "prove the
> abstraction with a 2nd quant": the spine is reused; the inner GEMM is re-authored per backend.
> (The `WmmaFp8` policy stays the right factoring *within* the WMMA family — v5/v6 — which is what
> it was validated for.)

### 2.5 Reconciled deliverable & sequencing
**Step 1 (WMMA consolidation) is DONE + validated** (see §2.6). What remains is the grouped-MoE
**SWMMAC sibling** — re-stated honestly after the §2.3 correction and the committed SWMMAC handoff:

1. ✅ **Consolidate** v5/v6 behind the `WmmaFp8` policy + `TileConfig`. Done, bit-exact + perf-neutral.
2. **Grouped-MoE SWMMAC *sibling*** — a *new* GEMM loop (not a `cfg.MMA` swap): weight = sparse-A
   (2:4-compressed fp8 + per-lane idx, read from DRAM), activation = dense-B; reuse the **spine**
   (MoE contract, group-scale fold, epilogue/scatter, LDS/tiling helpers, autotuner) and the
   **operand/index recipe** from `RESEARCH_swmmac.md §3` — re-implemented in THIS worktree, not by
   editing `feat/swmmac-microbench`'s `swmmac_op.hip` (that is another concern's committed kernel —
   do not touch it, per CLAUDE.md). Building it on the v5/v6 LDS-tiling backbone yields a *tiled*
   SWMMAC incidentally (the handoff's "untuned `swmmac_gemm_k`" concern, addressed within our scope).
3. ✅ **int4-sparse + per-group-scale grouped variant** (~3 bit/wt) — the **production 35B rung**.
   `bench_swmmac_int4.hip` shows 1.17–1.55× over dense W4A8 (the decode-bandwidth win); the grouped
   kernel is now written + self-consistency-validated (see §2.7). The served win still waits on a
   pruned MoE checkpoint existing.

#### Validation ceiling for the SWMMAC sibling (IMPORTANT, honest scope)
**No served grouped-MoE-SWMMAC result is possible** — there is no 2:4-pruned MoE checkpoint (only
dense Sparse-Llama-8B; the 35B AWQ experts are structureless under 4:2 per our Tier-3 screen). So
the deliverable ceiling is:
- **Correctness via self-consistency** (no model needed): synthesise a weight, zero it 2:4 along K,
  compute the GEMM two ways on-device — dense WMMA (4-of-4 with the zeros) vs grouped-SWMMAC (2-of-4
  compressed + idx) — assert **bit-exact (rel 0)**, exactly as `validate_swmmac_real.hip` did for the
  dense case. This proves operand+index construction + the grouped wiring.
- **Achievable speedup via microbench**: grouped-SWMMAC vs grouped-WMMA-dense on the 35B MoE shapes,
  under the gpu-lease — a number, not a served result.

This is the next work item; per the advisor it starts as **design + a self-consistency test plan**,
not a fourth big build, and only after the bench A/B closes step 1 (now closed).

#### Concrete build plan for the grouped SWMMAC sibling (next session, ready to execute)
Mirror the *validated* dense dataflow in `swmmac_op.hip::swmmac_gemm_k` (read 2026-06-18), wrapped in
the grouped-MoE contract. Exact construction to reuse (do NOT re-derive — recipe = `RESEARCH_swmmac.md §3`):
- **operands** (K=32/step): sparse-A weight `a = Wc[nrow*(K/2) + t*16 + (lane>>4)*8]` (v2i, 8 fp8);
  dense-B activation `b = X[mrow*K + t*32 + (lane>>4)*16]` (v4i, 16 fp8); index
  `id = (idx32[nrow*KT + t] >> ((lane>>4)*16)) & 0xFFFF`; `acc = swmmac_f32_16x16x32_fp8_fp8_w32(a,b,acc,id)`.
- **output map** `C[n][m] → Y[m][n]`: `n = blockIdx.x*16 + (lane>>4)*8`, write `acc[0][e]` to col `n+e`.
- **grouped wrapping**: `blockIdx.y` = one moe_align block (one expert via `expert_ids`); per-expert
  weight slab `Wc_e = Wc + e*N*(K/2)`, `idx_e = idx + e*N*KT`; activation row gathered via
  `sorted_token_ids` (`mrow → src = offs/top_k`, with the **`mrow_l` bounds guard** — real M<block);
  epilogue = `act_scale[src] * w_scale[e][n] * acc`, fp16 store or atomic scatter (the SPINE, reused
  verbatim from `moe_gemm_tiled.h`).
- **which variant**: the 35B MoE is **int4 + per-group** scales, so the production target is the
  **int4-sparse** variant (`bench_swmmac_int4.hip`, ~3 bit/wt, 1.17–1.55× over W4A8) — unpack the 2
  kept int4 nonzeros → fp8 + fold per-group scale, then SWMMAC. The fp8/per-channel `swmmac_op` is the
  simpler first rung to get the grouped wiring right, then swap in int4+group-scale.
- **self-consistency test** (`bench_swmmac_grouped.hip`, 1-card lease): synth per-expert weight, zero
  2:4 along K, repack to `(Wc, idx)` via the recipe, grouped-SWMMAC vs a scalar grouped reference on
  the zeroed weights → assert rel 0. No MoE model needed; no served result possible.
- **NB** building on the v5/v6 LDS-tiling backbone (stage activations once, reuse the sparse weight
  across the M-tile) yields a *tiled* SWMMAC — the handoff's "untuned `swmmac_gemm_k`" concern, fixed
  within this worktree's scope (do not edit their kernel).

#### SWMMAC sibling — VALIDATED (2026-06-18): self-consistency PASS, bit-exact at production block_m
**`bench_swmmac_grouped.hip` PASS** (1-card lease): grouped-SWMMAC vs scalar grouped reference on the
same 2:4-zeroed weights —
- gemm1 N=1024 K=2048 bm=64: **mean_rel 0, max_rel 0 (bit-exact)**, 0/524288 bad
- gemm2 N=2048 K=512 bm=64: **mean_rel 0, max_rel 0 (bit-exact)**, 0/1048576 bad
- gemm1 bm=32: mean_rel 1.78e-4 (fp16 output rounding only), 0/262144 bad → PASS

So the operand/index/routing construction is **numerically correct** at the production block_m=64.
(First pass used O(1) synthetic scales → the K-long fp8 sum overflowed the fp16 output to `inf` for
one rng draw; realistic small scales — as real per-token/per-channel W4A8 scales bound the result —
fixed it. Not a kernel bug.) The fp8/per-channel grouped SWMMAC GEMM is done and validated.

#### §2.7 int4-sparse + per-group-scale grouped variant — VALIDATED (2026-06-19): self-consistency PASS
**The production 35B rung.** `moe_gemm_swmmac_int4.h` (`moe_gemm_swmmac_int4_kernel<SCATTER,NWARP>`)
extends the fp8 sibling with the two pieces the real 35B MoE needs, sharing the *entire* SWMMAC
dataflow + grouped-MoE spine otherwise:
- weight = **2:4-compressed INT4** (E, N, K/4 bytes), unpacked to e4m3 with a per-(channel,group)
  **zero-point** (asymmetric `w_zeros`, or symmetric zp=8 when null) — int4 unpack mirrors the
  validated `bench_swmmac_int4.hip::sparse_int4`; ~3 bit/wt (vs the fp8 rung's ~5).
- per-**group** weight scale (E, N, K/group_size) folded **in-register per group**, exactly like the
  WMMA tiled spine (`moe_gemm_tiled.h`), replacing the fp8 rung's single per-channel epilogue scale.
  `group_size` is a multiple of 32 (one SWMMAC K-tile); the 35B g32 → one tile == one group.

**Compile + ISA verified** for gfx1201: emits `v_swmmac_f32_16x16x32_fp8_fp8` (×3 NWARP instances) +
`v_cvt_pk_fp8_f32` (the int4→fp8 unpack), **0 `ds_read`/`ds_write`** (weight straight from DRAM, the
defining no-LDS-expand SWMMAC property).

**`bench_swmmac_grouped_int4.hip` PASS** (1-card lease, 2026-06-19): synth per-(expert,channel,group)
int4 codes + per-group zp + per-group scale, 2:4-prune along K, device-fill the zeroed dense weight
`Wz = e4m3(code−zp)` with the **same** `int4_signed_to_e4m3` the kernel uses (so the two paths' fp8
weight bytes are bit-identical by construction), then grouped-SWMMAC-int4 vs a per-group-scaled dense
reference —
- gemm1 N=1024 K=2048 g32 bm=64: mean_rel 1.75e-4, **max_rel 4.85e-4**, 0/524288 bad → PASS
- gemm2 N=2048 K=512  g32 bm=64: mean_rel 1.77e-4, max_rel 4.86e-4, 0/1048576 bad → PASS
- gemm1 g32 bm=32:               mean_rel 1.76e-4, max_rel 4.85e-4, 0/262144 bad → PASS
- gemm1 **g64** bm=64 (group spans 2 SWMMAC tiles): mean_rel 1.71e-4, max_rel 4.83e-4, 0/524288 → PASS

`max_rel ≈ 4.85e-4` is **exactly one fp16 ULP** (2⁻¹¹) — pure output quantisation (kernel stores fp16,
reference fp32). A wrong int4-unpack / zp / scale-fold / index / routing would be *grossly* off, not at
the fp16 floor; so the int4-sparse + per-group-scale + asymmetric-zp construction is **numerically
correct** at the production block_m=64, including multi-tile groups (g64). (A self-review bug — zp
synthesised per-tile while the kernel applies it per-group — was caught + fixed before the GPU run; it
would have falsely failed only the g64 case.) Validation ceiling reached: no served result is possible
(no pruned MoE checkpoint; 35B AWQ experts are structureless under 4:2).

**Unexercised paths (NOT claimed validated):** the self-consistency bench drives only `SCATTER=false`
and the asymmetric-zp path (`w_zeros != null`). `SCATTER=true` (the atomic-scatter epilogue) and the
symmetric `w_zeros==null` → zp=8 fallback are verbatim-from-the-fp8-spine / trivial-constant and are
*not* independently tested here — same coverage as the fp8 rung. (The optional bit-exact badge — compare
`__float2half(reference)` vs the fp16 kernel output instead of fp32 vs fp16 — would turn the 1-ULP
mean_rel into mean_rel 0; not done, immaterial to the correctness conclusion.)

#### §2.7b Speedup A/B — grouped-SWMMAC-int4 vs grouped-WMMA-dense (2026-06-19, the §2.5 ceiling's 2nd half)
`bench_swmmac_int4_vs_wmma_grouped.hip` (1-card lease, timing-only, synthetic inputs): grouped-SWMMAC-
int4 (~3 bit/wt) vs `moe_gemm_tiled_kernel<WmmaFp8>` (the consolidated grouped WMMA-dense W4A8 GEMM,
4 bit/wt), BN=64 bm=64, 35B shapes, g32, swept over P:

| P (padded rows) | gemm1 (N1024 K2048) | gemm2 (N2048 K512) |
|---|---|---|
| 64   | **17.7×** | 12.0× |
| 256  | 8.9×  | 7.2× |
| 512  | 7.4×  | 2.2× |
| 2048 | **0.91×** | 1.14× |

**Read this honestly — the small-P numbers are NOT the real decode win:**
- At small M the baseline `moe_gemm_tiled` (v5/v6 WMMA) is weight-staging-bound — it dequantises the
  *entire* N×K int4 weight into LDS for very few rows. That is exactly why production **dispatches the
  v7 GEMV at decode, not this WMMA tiled kernel** (TILE_FRAMEWORK_DESIGN §2.6 note). So the 12–18× is
  "SWMMAC-int4 vs a kernel nobody runs at that M" and **overstates the decode win**. The honest decode
  number needs a SWMMAC-int4-vs-**v7-GEMV** A/B (not yet done — the real follow-on).
- The **apples-to-apples** number is **prefill, P=2048: 0.91–1.14× ≈ parity**, where WMMA-tiled *is* the
  production path. Notably SWMMAC-int4 does **not** beat it despite ~half the weight bytes + 2:4-sparse
  MMA: each block's `block_m/16` warps redundantly re-read the same weight rows from DRAM (this first
  rung has **no LDS weight-reuse**), so the sparsity/byte win is eaten by redundant traffic. Building the
  SWMMAC GEMM on the v5/v6 LDS-tiling backbone (stage the sparse weight once, reuse across the M-tile —
  the "tiled SWMMAC" item in §2.5) is the lever that should turn prefill parity into a real win.

**Conclusion (phase 3b):** the int4-sparse + per-group-scale grouped SWMMAC kernel is **written, ISA-
verified, and self-consistency-correct** (the §2.5 ceiling's 1st half, fully met). The speedup half is
**measured but not yet a clean win**: parity at prefill (redundant-weight-read bound), and a large but
baseline-inflated small-M number that needs a v7-GEMV A/B to state honestly. **Next levers:** (1) tiled
SWMMAC (LDS weight-reuse) to convert prefill parity → win; (2) the v7-GEMV decode A/B; (3) fold the
SWMMAC backend into `run_moe_gemm`'s `VLLM_W4A8_MOE_TILED`-style env branch for a torch-level path.

#### §2.7c Levers (1) tiled SWMMAC + (2) the honest v7-GEMV decode A/B (2026-06-19)
Both `§2.7b` follow-ons, run 1-card lease, timing-only, synthetic inputs, bit-exactness re-checked.

**(1) Tiled SWMMAC — `moe_gemm_swmmac_int4_tiled_kernel<SCATTER,NWARP>`** (in `moe_gemm_swmmac_int4.h`).
Stages this block's 16-row weight tile (`Wc_i4` 16×K/4 + `idx` 16×K/32) into LDS **once at entry**
(padded int stride to break the 16-way bank conflict — an unpadded K/16-int stride is a multiple of 32
for these shapes), one `__syncthreads`, then all `NWARP` warps read weight from LDS instead of
re-reading the same rows from DRAM. **Bit-exact vs the non-tiled kernel** (`bench_swmmac_grouped_int4.hip`
extended: fp16-mismatch=0 across gemm1/gemm2, bm32/64, g64 multi-tile). Speedup A/B
(`bench_swmmac_int4_vs_wmma_grouped.hip`, bm=64, **×untiled** / ×WMMA-dense):

| P    | gemm1 ×untiled | gemm1 ×WMMA | gemm2 ×untiled | gemm2 ×WMMA |
|------|---------------|-------------|----------------|-------------|
| 64   | 0.92× | 8.8×  | 0.97× | 8.4× |
| 256  | **1.49×** | 13.6× | 1.05× | 4.8× |
| 512  | 0.94× | 6.9×  | 1.08× | 2.4× |
| 2048 | ~0.94–1.01× | 0.85–0.91× | **1.10×** | **1.25×** |

**Honest read:** LDS weight-reuse helps **only where weight-read is the bottleneck** — gemm2 (short
K=512, weight-BW-bound, the documented decode wall) gains a consistent 1.05–1.10× ×untiled, lifting the
prefill P=2048 case **1.14×→1.25× vs WMMA-dense**; mid-P gemm1 (P=256) gets 1.49×. But the headline
**large-prefill gemm1 P=2048 stays at parity** — weight-reuse is *not* its bottleneck (long-K, activation/
issue-bound at large M), so the "turn prefill parity into a win" hope **only lands for gemm2, not gemm1**.
Tiny P=64 slightly regresses (staging cost, L2 already absorbs the redundancy). Did **not** profile the
gemm1-prefill bottleneck further (advisor: it's a different bottleneck, not LDS — stop tuning LDS).

**(2) Honest decode A/B vs v7-GEMV** — `bench_swmmac_int4_vs_v7_decode.hip` (v7 copied verbatim from
`moe_kernel.hip`, launched in its **production** auto-config nw/bk/cols/MMAX; **realistic moe_align
routing** = few real tokens padded to block_m, so v7 actually compacts — the old bench's `num_valid==P`
hid that). bm=16, ×v7 (>1.0 = SWMMAC beats v7, the production decode path):

| T (real toks) | gemm1 SWMMAC | gemm1 tiled | gemm2 SWMMAC | gemm2 tiled |
|----|-----------|-----------|-----------|-----------|
| 1  | **0.39×** | 0.13× | 1.28× | 0.86× |
| 4  | 1.01× | 0.31× | 2.59× | 1.75× |
| 8  | 1.86× | 0.51× | 4.51× | 3.06× |
| 16 | 4.18× | 1.50× | 11.1× | 7.9× |
| 32 | 9.21× | 2.18× | 6.46× | 16.0× |
| 64 | 9.18× | 2.10× | 10.2× | 15.7× |

**Honest decode conclusion (corrects §2.7b's inflated 12–18×):**
- At **pure decode T=1, v7 WINS gemm1** (SWMMAC 0.39× = v7 2.5× faster) and ~ties gemm2 (SWMMAC 1.28×).
  So SWMMAC does **not** dominate decode — against the kernel production actually runs, it *loses* the
  long-K GEMM at single-stream decode. The old 12–18× was vs WMMA-tiled, "a kernel nobody runs at that M."
- **v7 falls off a cliff with T** (true scalar GEMV: 0.03→1.79 ms gemm1, superlinear). The crossover is
  **2D (which-GEMM × T), not one threshold** — K sets it: short-K **gemm2 favors SWMMAC almost everywhere,
  including T=1 (1.28×)**; long-K **gemm1 favors v7 until ≈T=4–8**, SWMMAC past it. → ideal dispatch:
  **v7 for gemm1 at T≤~4; SWMMAC for gemm2 always and for gemm1 at T≥8**.
- **Tiled SWMMAC is the WRONG choice at decode (bm=16 → NWARP=1, no within-block reuse): strictly worse
  than untiled at small T** (0.13× gemm1 T=1). It only earns its keep at bm≥32 (the §2.7c-(1) regime). The
  bm=16 T=32/64 gemm2 spike (16×) is a cross-block-locality effect, not the within-block reuse it was built
  for — do not over-read it.

**Validation scope of the tiled kernel:** the `SCATTER=false` (non-scatter) path is bit-exact at
NWARP=1/2/4. The **`SCATTER=true` path (gemm2 atomic-scatter epilogue) is NOT yet correctness-checked** —
1+2 are timing/non-scatter complete, but any future torch wiring that routes the scatter epilogue through
the tiled kernel must validate that path first.

**Net (§2.5 ceiling fully characterised):** SWMMAC-int4 is a real win for **batched-decode/small-prefill
gemm2 and mid-M gemm1**; v7 still owns **single-stream decode gemm1**; tiled adds a further 1.1–1.25× **only
for the weight-BW-bound (gemm2 / bm≥32) shapes**. No regime is a universal win, and there is **no served
consumer** (2:4 breaks 35B PPL, no compressed checkpoint — see §2.7b).

#### §2.7d Torch wiring (§2.5 item 3) — DEFERRED, with the structural reason
**Decision: do NOT build the `VLLM_W4A8_MOE_SWMMAC` env-branch in `run_moe_gemm`.** Not "later" —
**structurally blocked**, for two compounding reasons:
1. **The hardware instruction is 2:4, and 2:4 is PPL-hopeless on the 35B** (`[[slidesparse-ppl-gate-green]]`:
   2:4 ≈ +25% PPL; the PPL-viable ratios 6:8/4:6 do **not** map to the `v_swmmac…fp8` 2:4 op). So there is
   no weight source *and the research path to one is blocked at the hardware level*, not merely deferred.
2. **`run_moe_gemm` receives dense `(E,N,K/8)` weights, not compressed `(Wc_i4, idx)`.** A `VLLM_W4A8_MOE_
   SWMMAC` branch there can **never fire** without plumbing compressed tensors through bindings + the python
   adapter — i.e. the exact no-consumer infra the project already declined (`[[layout-share-probe-result]]`).
   A dispatch branch that cannot receive its inputs is **dead code advertising a capability that doesn't
   exist** — worse than absent. The §2.7b/c **benches already are** the exercised dispatch.

**Durable artifact instead of code:** the §2.7c 2D crossover (gemm2 → SWMMAC always; gemm1 → v7 at T≤4,
SWMMAC at T≥8; tiled only at bm≥32 / weight-BW-bound shapes) **is** what a future dispatch *would* encode,
recorded here for when a 2:4-pruned + finetuned checkpoint exists. Build the env-branch (dispatch + bench
only, no bindings/adapter, scatter path validated first) **only on explicit request to keep the plumbing
warm** — otherwise it is parked here.

#### (history) written, compile-verified, construction-reviewed, validation queued
- `moe_gemm_swmmac.h` — `moe_gemm_swmmac_kernel<SCATTER,NWARP>`: the fp8/per-channel first rung,
  mirroring `swmmac_op.hip`'s validated dataflow wrapped in the grouped-MoE spine (per-expert
  Wc/idx slabs, routed activation gather, act-scale + fp16/scatter epilogue). **Compile-verified
  for gfx1201** — emits `v_swmmac_f32_16x16x32_fp8_fp8` (no LDS-expand path).
- `bench_swmmac_grouped.hip` — self-consistency test: synth per-expert 2:4 weights → repack to
  (Wc, idx) per §3 → grouped-SWMMAC vs a scalar grouped reference on the same zeroed weights (both
  HW e4m3 + fp32). **Built.** GPU run **queued behind `moetune`** (shared-box contention held both
  cards); will report PASS/FAIL when a card frees.
- **Construction self-review PASSED** (line-by-line vs the recipe + `swmmac_op`): Wc byte layout,
  idx nibble placement, dense-B offsets, and the SWMMAC `v0·B[p0]+v1·B[p1]` ↔ reference
  `Wz[p0]·X[p0]+Wz[p1]·X[p1]` correspondence all check out. High confidence pending the GPU number.
- NOT yet committed — the GPU self-consistency PASS is the gate before committing the sibling.
- Next after PASS: the **int4-sparse + per-group-scale** variant (the production 35B path), and a
  grouped-SWMMAC-vs-WMMA-dense microbench speedup number.

### 2.4 Acceptance test for the refactor (perf-neutrality, GPU)
The consolidation must be **bit-exact and perf-neutral** vs today's hand-tuned launches:
- For each live shape (gemm1 K=2048→2*inter, gemm2 K=inter→hidden, g32), instantiate the
  `TileConfig` matching today's default dispatch and assert (a) bit-exact output vs current op,
  (b) latency within noise (±2%) of the current launch. This is the gate before the framework
  replaces the current dispatch.

---

## 2.6 Progress (this session — CPU, compile-verified)
- `tile_config.h` — shared device primitives (fp8 cvt, int4→fp8, LDS pad) + `WmmaFp8` MMA policy
  + `TileConfig` + the documented SWMMAC extension point. Compile-verified for `-mcpu=gfx1201`;
  the policy lowers to `v_wmma_f32_16x16x16_fp8_fp8` with **zero overhead** (bare-builtin parity).
- `moe_gemm_tiled.h` — grouped-v5 re-expressed over `moe_gemm_tiled_kernel<MMA, SCATTER, BN>`.
  Compile + ISA verified (BN=64 non-scatter & BN=128 scatter both emit the exact v5 instruction
  mix: wmma + cvt_pk_fp8 + ds_store + scatter atomics). **Bit-exact vs `moe_gemm_v5_kernel` by
  construction** (identical staging/accum/epilogue; only `MMA::mma` replaces the inline builtin).
- Non-destructive: the existing v5/v6/v7 launchers are untouched. Nothing ships until the §2.4
  GPU A/B passes.
- **`__launch_bounds__(MAX_THREADS=256)` matched to v5** (default-1024 target would shrink the
  per-thread register budget → spill → fake regression in the A/B). **Register/occupancy parity
  verified on CPU** (clang `-S`, gfx1201): tiled == v5 exactly — BN64 {VGPR 94, SGPR 63, occ 16,
  0 spill}, BN128-scatter {VGPR 165, SGPR 68, occ 9, 0 spill}. This + bit-exact-by-construction
  means the §2.4 GPU A/B measures the real thing. (Repro: `/tmp/v5_standalone.hip` vs the policy
  kernel; bake `__launch_bounds__` into the v6/v7 templates too — both originals carry it.)

- **§2.4 GPU GATE PASSED (2026-06-18, via gpu-lease 1-card).** `bench_tiled_vs_v5.hip` — standalone
  HIP A/B (no torch), real 35B MoE shapes (gemm1 N=1024/K=2048, gemm2 N=2048/K=512, g=32, top_k=8,
  block_m=64). **Bit-exact: 0 mismatches, maxabs 0.0** on all 6 cases (BN=64/128, P=512/2048), raw
  16-bit compare. **Perf-neutral:** tiled/v5 ratios 0.98–1.06 (one 0.73 noise outlier; identical
  register profiles → no structural diff). The policy-templated grouped GEMM is a proven drop-in
  for `moe_gemm_v5_kernel`.

- **v6 ALSO consolidated + GPU-validated (2026-06-18).** `moe_gemm_tiled_v6_kernel<MMA,SCATTER,BN>`
  reproduces grouped-v6 (A-shuffle, B-only LDS, gtile). Bench extended to A/B it: **bit-exact (0
  mismatch, maxabs 0) on all v6 cases**, perf-neutral (ratios 0.93–1.00). So the **whole WMMA
  grouped path (v5 A-in-LDS + v6 A-shuffle) is now behind the MMA policy + `TileConfig`.** (NB the
  v6 A-shuffle index math is WMMA-fragment-specific — flagged in-code; the SWMMAC port starts from
  the v5 backbone, not v6.)

**Note on v7:** the grouped-v7 GEMV (decode) is a *scalar dot-product* kernel — it does **not** use
WMMA, so it sits outside the MMA-policy abstraction. `TileConfig` selects it at the **dispatch**
level (decode vs prefill), not as a policy. So the policy-consolidation of the WMMA tiled path is
**complete at v5+v6**; v7 stays a dispatch sibling.

- **TileConfig autotune harness DONE + GPU-run (2026-06-18).** `autotune_grouped.hip` sweeps the
  config space {version 5/6 × BN 64/128 × gtile} per shape, times each, emits a winner-cache JSONL
  (`tile_crossover_cache.jsonl`) keyed by (N,K,group,block_m,P) — the single mechanism that folds
  the scattered `VLLM_W4A8_MOE_{BN,GTILE,...}` knobs + `crossover_cache.json`. Coherent result:
  v6 wins all 35B shapes, BN=64, gtile=8 at small P → gtile=4 at P=2048 (group-staging depth vs
  grid-fill tradeoff). This is the framework's "cheaper exploration" payoff, demonstrated.

- **Production wiring DONE (env-gated, non-destructive).** `moe_kernel.hip` now `#include`s
  `moe_gemm_tiled.h`; `run_moe_gemm` routes v5/v6 through the consolidated kernels when
  `VLLM_W4A8_MOE_TILED` is set (default path unchanged). So the existing torch op
  `mmq_fp8_moe_gemm` serves both, and a serving-level A/B is just toggling the env var. The
  `SwmmacFp8` backend slots in at this same branch (step 2). Image build `vllm22-w4a8:tiled`
  (distinct tag, preserves `:combined`) compile-verifies the torch integration in the real env.

- **Image `vllm22-w4a8:tiled` BUILT (2026-06-18) — wiring compile-verified in the real env.** The
  build's `pip install .` compiled `moe_kernel.hip` (with the `VLLM_W4A8_MOE_TILED` branch + the
  new headers) and `w4a8_fp8_wmma._C` imported OK. So the torch integration is confirmed; only a
  *runtime* dispatch A/B remains, and the kernels themselves are already microbench-validated.

- **Runtime dispatch VALIDATED through torch (2026-06-18).** `docker compose --profile moetest`
  (new service, `:tiled`, 1-card lease) ran `test_moe_correctness.py` with `VLLM_W4A8_MOE_TILED=1`:
  **ALL 40 PASSED** — v5/v6 routed through the consolidated kernels match the fp8 reference across
  the shape grid (sym+asym, T=1..128). So the consolidation is validated at **four levels**: kernel
  bit-exact (microbench), register-identical (CPU), wiring compile-verified (image build), and
  runtime-correct through the torch op (this). Correctness is fully proven; only end-to-end serving
  *perf* remains.

- **Serving A/B RUN (2026-06-18, `:tiled`, 2-card lease, offline `bench_tp2.py`, 35B).** No OOM at
  `GPU_MEM_UTIL=0.90`. TILED=0 (originals): decode 920.1 / total 2470.6 — but **cold** boot (1438 s
  init). TILED=1 (consolidated): decode **957.2** / total **2570.0**, **warm**. The 957/2570
  **exactly reproduces the established W4A8 offline number**; the +4% over the TILED=0 run is
  **order-confounded** (cold vs warm + first-batch MoE-autotune primed by RUN A = the documented JIT
  tail), so this is read as **perf-neutral / reproduces baseline, not a win**. Not re-run to
  de-confound — the isolated microbench is the better perf instrument and correctness is proven four
  ways. **Step 1 is now CLOSED.**

**Step 1 DONE.** Remaining: fold `tile_crossover_cache.jsonl` into the python adapter (deferred), and
the **SWMMAC sibling** (§2.5) — the next work item. Superseded note: serving-level perf A/B —
under a 2-card lease, run the 35B with `VLLM_W4A8_MOE_TILED` unset vs =1, confirm equal output +
neutral latency at the vLLM level (use compose, never a hand-rolled `docker run`). Then fold
`tile_crossover_cache.jsonl` into the python adapter's per-shape dispatch. **Step 2** (SWMMAC
backend on the v5 backbone) waits on `feat/swmmac-microbench` being committed — the un-owned
live-35B win and the next big lever.

## 3. One batched GPU-window ask
Per the shared-box protocol, everything GPU is deferred to **one** window covering:
- (a) refactor perf-neutrality + bit-exactness (§2.4),
- (b) candidate #2 (nt-hint A/B — cheap),
- (c) if pursued, candidate #1 (burst-repack) correctness + BW.

CPU work that proceeds now without a window: the `TileConfig` scaffolding (§2.3), the SWMMAC
kernel skeleton (§2.2), and the unified autotune harness.
