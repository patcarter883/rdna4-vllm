# W4A8 dense kernel versions — functional crosswalk & status

Kernels are still **numbered internally** (the version int passed to `mmq_fp8_gemm`, the env-var and
`crossover_cache.json` keys, and the DIARY/PIECE notes all key on numbers — those are load-bearing and
unchanged). This table is the **public functional identity** + serving status. Source of the verdicts:
`profiling/all-versions-bench/FINDINGS.md` (unified bench, 2026-06-15, eager+graph).

## Active — dispatched on the served path (`_w4a8_dense_apply` ladder)

| ver | functional name | role | dispatch condition |
|---|---|---|---|
| v0  | `reference_scalar` | scalar fp8 golden | never dispatched; correctness reference only |
| v11 | `decode_gemv` | decode GEMV | `K%1024==0 & gs%32==0 & M<=2` (wins M=1-2 ~1.7x) |
| v10 | `prefill_wmma_ashuffle` | large-M prefill | `gs in {32,128} & M>=prefill_min` (1.5-2x vs Triton at M>=256) |
| v5  | `prefill_wmma` | any-gs fallback | exotic group sizes (gs not in {32,128}); dominated by v10 for gs in {32,128} |
| —   | `triton_w4a16` (stock + gfx1201-tuned) | small/mid-M | when present (layout=tuned): small-M band M<=32; the >=stock floor |

Dispatch order: `decode_gemv`(M≤2) → `prefill_wmma_ashuffle` v10 (M≥crossover) → Triton small/mid-M
(layout=tuned) → `prefill_wmma` v5 (exotic gs / no-fallback). Layout default is **`tuned`** (Triton
re-admitted for small-M; see `_layout_mode`, 2× dense-weight VRAM — set `VLLM_ROCM_W4A8_LAYOUT=single`
to drop the copy).

## Parked-but-kept — in tree, NOT dispatched, pending an end-to-end A/B

The register-direct kernels need an offline `w_rep` (N//16,K//16,32) prepack (helper:
`profiling/all-versions-bench/bench_all_versions.py::pack_wrep`, CPU- + GPU-validated). Single-op bench
shows a **narrow** small-M edge on large shapes; needs a full-decode-step graph A/B before wiring (the
single-op number understates Triton's real small-M advantage).

| ver | functional name | numerics | single-op finding |
|---|---|---|---|
| v15 | `regdirect_fp8`   | W4A8 (fp8 WMMA) | wins M=4-8 large shapes; Triton retakes M=16 (except very-large-K) |
| v16 | `regdirect_f16`   | W4A8 (fp8→f16 WMMA) | ~v15 |
| v17 | `regdirect_w4a16` | **W4A16** (fp16 acts, no act-quant — different op) | wins 11008-g32 M≤16 only |

## Retired → research-only — NOT dispatched (kept for the research trail + git history)

| ver | functional name | why retired (2026-06-15 bench) |
|---|---|---|
| v1  | `rocwmma_v1` | rocWMMA ancestor of v5; dominated everywhere |
| v2  | `rocwmma_tiled` | ancestor of v5; dominated |
| v4  | `rocwmma_pipe` | ancestor of v5; dominated |
| v6  | `prefill_wmma_b128` | **bit-exact to v5 but slower at every (shape,M); never near v10.** b128 double-K LDS trick doesn't pay on gfx1201. `_v6_band` gate removed. |
| v7  | `wmma_tiled_tuned` | tile-tuning variant; dominated by v5/v10 |
| v8  | `wmma_dbuf` | LDS double-buffer variant; dominated |
| v9  | `wmma_dbuf2` | double-buffer variant; dominated |
| v12 | `splitk_smallm` | split-K small-M; stuck ~107 GB/s, dominated by regdirect |
| v13 | `regdirect_shuffle` | register-direct-B-via-shuffle; 5-15× slower (wrong mechanism) |
| v14 | `nsplit_smallm` | N-split small-M; dominated by regdirect |

"Research-only" = the kernel body still exists in `w4a8_fp8_wmma_kernel.hip` (the launcher can still be
asked for it directly for A/B), but **nothing in the served dispatch selects it**. A follow-up
(build-validated) step may physically relocate the retired kernel bodies out of the compiled image.
