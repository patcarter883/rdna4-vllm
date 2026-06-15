# Unified W4A8 dense-kernel benchmark — all versions v0..v17 (2026-06-15)

Harness: `bench_all_versions.py`, image `vllm22-w4a8:combined`, GPU0 (gfx1201), eager + HIP-graph,
4 shapes × M=1..4096. Raw: `results_all.json` (464 rows), log: `run.log`. Correctness anchors per
shape: v5≈triton (~2.6e-2 fp8 rel), **v15(w_rep)≈v5 (max-abs ≤2e-3 = fp accumulation-order noise,
prepack bit-correct — also proven by the CPU reconstruction test)**.

Crosswalk: v0 `reference_scalar` · v5 `prefill_wmma` · v6 `prefill_wmma_b128` · v10
`prefill_wmma_ashuffle` · v11 `decode_gemv` · v15/16/17 `regdirect_{fp8,f16,w4a16}`.

## Headline: the served small-M band trails Triton — re-admit Triton (done; default now `tuned`)

Correction to an earlier draft of this section: in single-layout mode the small-M dispatch for the
common case **gs∈{32,128} is forced to v10** (`_w4a8_dense_apply` forces v10 when nothing in the
ladder selects and there's no fallback), NOT v5. v5 only serves **exotic group sizes** (gs∉{32,128}).
So the real served small-M gap is **v10's ~2.4×** behind Triton (11008×4096-g32 M=8: v10 239µs vs
Triton 97µs), not v5's ~6× (which applies only to exotic gs). Still a real gap, shape-general.

**Action taken (2026-06-15):** flipped the default `_layout_mode` to `tuned`, re-admitting Triton for
the small-M band. COST: the `(K,N//8)` second weight copy = **2× dense-weight VRAM**, which can OOM
~10GB-weight models at load on 16GB (e.g. Qwen3.6-27B TP=2). Fully reversible:
`VLLM_ROCM_W4A8_LAYOUT=single` restores the VRAM-saving behavior. Deploying requires an image rebuild
(adapter is baked in) + a load smoke-test.

## Secondary: the PIECE2 regdirect "wall" is shape-dependent — but the W4A8 win is narrow + caveated

PIECE2 concluded v13/v15/v16/v17 all hit a ~108 GB/s small-M wall and lose to Triton — generalizing
from the **small 4096×4096 starved shape**. The re-bench shows the wall is a small-shape effect, not
universal. But stratifying correctly (the eye-catching wins were partly v17, a *different op*):

- **v17 is W4A16** (fp16 activations, NO act-quant) — not a W4A8 substitution; compare only vs
  Triton-w4a16. It wins 11008-g32 M≤16 but loses the other shapes.
- **The W4A8-eligible regdirect (v15 fp8, v16 fp8→f16) wins only M=4–8** on large shapes; Triton
  retakes M=16 — *except* very-large-K (largeK 5120×17408) where v15/v16 hold through M=16. And these
  are an **fp8 kernel beating an fp16 Triton baseline = faster AND less accurate**, not a clean swap.

| shape | W4A8 (v15/v16) vs Triton |
|---|---|
| 4096×4096 g32 | win M=4–8, Triton retakes M=16 |
| 11008×4096 g32 | win M=4–8, Triton retakes M=16 |
| largeK 5120×17408 g128 | win M=4–16 (large-K exception) |

**Caveat that erodes even this:** the measurement is single-op. The session's layout-share finding
established that a 1-op capture *understates* Triton's full-decode-step dispatch-amortization — so the
real-world Triton small-M edge is *larger* than shown here. Can't hold "single-op understates Triton"
(layout-share) and "single-op v15 wins are serving-real" simultaneously.

→ **v15/16/17 are parked-but-kept** (NOT promote, NOT archive) pending a **full-decode-step,
graph-captured A/B** on the served path — {Triton small-M} vs {v15/v16 W4A8} vs {v17 W4A16},
stratified by activation contract, measured as real ITL/throughput. Only if v15/v16 still win M=4–8
under real graphs does wiring the prepack + dispatch tier earn its cost.

Refined ideal dense dispatch (gs∈{32,128}): `decode_gemv`(M≤2) → `regdirect`(M=4–16, big shapes) /
Triton → Triton(M≈16–48) → `prefill_wmma_ashuffle` v10 (M≥~64–256, shape-dependent crossover).

## Per-version verdicts

| version (fn name) | role | verdict |
|---|---|---|
| v0 `reference_scalar` | golden | keep (reference only; omitted from timing) |
| v1/v2/v4 | rocWMMA ancestors of v5 | **retire** → research/ (dominated by v5 everywhere) |
| v5 `prefill_wmma` | any-gs fallback | **demote**: for gs∈{32,128} it's dominated at every M; justify only as exotic-gs (16/64/96) fallback. Must NOT serve small-M. |
| v6 `prefill_wmma_b128` | gated mid-M | **DROP**: slower than v5 at *every* shape/M (e.g. 4096-g128 M=2048 v6 1612 vs v5 1603; 11008 M=128 v6 700 vs v5 614), never approaches v10. The b128 double-K LDS trick doesn't pay on gfx1201. This answers the v6 gate: it was committed gated-OFF "until benchmarked" — now benchmarked, it loses. Retire → research/ with a negative-result note. |
| v7/v8/v9 | tile-tuning variants | **retire** → research/ (dominated by v5/v10) |
| v10 `prefill_wmma_ashuffle` | large-M prefill | **keep, unchanged** — champion at M≥~64–256 (1.5–2× vs Triton, beats v5/v6 always) |
| v11 `decode_gemv` | decode M≤2 | **keep, unchanged** — wins M=1–2 (1.7×); guards M>16 |
| v12/v13/v14 | small-M split-K experiments | **retire** → research/ (v12/14 dominated by regdirect; v13 5–15× slow) |
| v15 `regdirect_fp8` | small-M regdirect (W4A8) | **parked-but-kept** — wins M=4–8 large shapes; pending end-to-end A/B |
| v16 `regdirect_f16` | small-M regdirect (fp8→f16) | **parked-but-kept** — wins M=4–8 large shapes; pending end-to-end A/B |
| v17 `regdirect_w4a16` | small-M regdirect, **W4A16** (different op) | **parked-but-kept** — separate numerics; wins 11008-g32 M≤16 only; pending A/B |

## v6 keep/drop — settled: DROP
v6 is bit-exact to v5 and slower than v5 at all 52 measured (shape,M) cells. The mid-M 512–2048 band
it targeted is owned outright by v10. No env band makes it win. Remove the `_v6_band()` gate and move
the kernel to research/ with the measured negative result recorded.

## Eager vs graph (kernel-level)
Single-op capture, so graph adds a fixed ~3–5µs capture overhead to every candidate equally;
custom kernels are graph-invariant (compute-bound), Triton likewise at these sizes. The full-model
dispatch-amortization that favors Triton at small-M is NOT visible in a 1-op capture, so the
regdirect small-M wins shown here are conservative for eager and hold under graph. Large-M v10 wins
are graph-stable (eager≈graph throughout).

## Next (needs a 2nd GPU window + image rebuild)
1. Wire the v15/16/17 `w_rep` prepack (helper written + CPU/GPU-validated in `bench_all_versions.py`
   `pack_wrep`) into `process_weights_after_loading`, add a small-M regdirect dispatch tier behind
   autotune, and A/B vs Triton on the served small-batch band.
2. gemm2 burst-repack (separate thread): numpy bit-exactness PROVEN (`profiling/burst-repack/`,
   4–8× longer bursts, column order preserved so scales/zeros need no reorder); next is the gated
   kernel variant + the decisive gemm2-GB/s microbench (§6.2 of RESEARCH_burst_repack.md).
