# Layout-share probe — Phase 1 findings (2026-06-15)

Tests proposal **(c)**: keep ONE int4 weight layout in VRAM (the Triton `(K, N//8)` packing),
serve small-M decode with native Triton int4→fp16 for free, run large-M prefill on the HIP
fp8-WMMA kernel reading that same layout via an on-the-fly in-register swizzle.

Bench: `bench_layout_share.py`, image `vllm22-w4a8:combined`, GPU0 (gfx1201, 16 GB), eager +
HIP-graph capture, `it=50 warmup=20`. Raw: `results.json`, log: `run.log`.

## Verdict: theory technically verified, but Phase 2 has no live consumer → don't fund it yet

Two separate conclusions, don't conflate them:
- **Technical gate: PASSES.** A large-M HIP win exists (1.5–2×), it survives graph capture, and the
  data is recoverable across layouts bit-identically. Phase 1 measures the components we already
  have — it does NOT confirm the fused kernel (only Phase 2 can prove the win survives reading the
  *transposed* layout), but nothing here kills (c).
- **ROI gate: reframed (2026-06-15).** Original draft said "no served model hits this path → don't
  fund it." That gate is RETRACTED per user direction: W4A8 is a **general universal kernel**, judged
  on its own merits across all shapes/M, NOT on what docker-compose serves
  ([[w4a8-is-a-universal-kernel-not-workload-tuned]]). The dormant-consumer fact (below) changes
  *urgency*, not whether the work is worth doing. The dense large-M win is real and graph-stable, so
  the fused kernel is on the table on technical merit; it competes for priority against the other
  kernel work (full-version bench, v6 gate, gemm2 burst-repack, consolidation).

### 1. The large-M HIP win is real AND survives graph capture  ← headline
On all four dense shapes, `hip_v10` beats stock Triton by **~1.5–2× at M≥256**, and crucially
**v10-graph ≈ v10-eager** (compute-bound, ≫ launch overhead) — the "CUDA-graph illusion" does
**not** erase it at large M. Examples (eager µs):

| shape (N×K, g) | M=256 tri/v10 | M=1024 tri/v10 | M=4096 tri/v10 | crossover |
|---|---|---|---|---|
| 4096×4096 g128   | 146 / 98 (1.5×) | 550 / 346 (1.6×) | 2980 / 1758 (1.7×) | ~M=256 |
| 11008×4096 g128  | 383 / 247 (1.6×) | 1706 / 905 (1.9×) | 7798 / 4061 (1.9×) | ~M=256 |
| 4096×11008 g128  | 397 / 242 (1.6×) | 1735 / 935 (1.9×) | 7581 / 4188 (1.8×) | ~M=128 |
| 14336×4096 g128  | 490 / 335 (1.5×) | 2167 / 1424 (1.5×) | — | ~M=256 |

→ There IS a large-M win worth protecting. The gate to fund the fused kernel passes (dense).

### 2. Small-M Triton win is real (~1.7–2×) — and this bench UNDERSTATES it
At M≤128 Triton wins ~1.7–2× (e.g. 4096×4096: 40 vs 78 µs). Note the bench captures ONE op into
a graph, so it cannot show the dispatch-amortization that favors Triton across a full decode step
(hundreds of ops) — the real-world small-M Triton advantage is *larger* than shown. Serving small-M
with native Triton on the shared layout is the right call.

### 3. The layout share is purely an access-pattern problem (not numeric)
For every shape: repack `(K,N//8)→(N,K//8)` reproduces the native packing **exactly**, and HIP run
on the shared-layout weights is **bit-identical** to HIP on native (max_abs=0). The fp8-vs-fp16
"numeric mismatch" worry was wrong — both layouts store the same int4 nibbles + fp16 scales; only
the pack axis (along-N vs along-K) + scale transpose differ. The fused kernel's job is a strided/
transposed *load*, not a numeric conversion.

### 4. The repack "tax" is M-independent and HBM-bound (the torch number is an artifact)
Weights don't depend on batch, so the translation cost is a per-shape constant. The torch-strided
repack measured 8–61 ms, but that is launch/transpose overhead — the analytic HBM floor (read+write
the int4 weight @ ~1.5 TB/s) is **~11–39 µs** (dense) / **~1.4 µs** (gemm2). Even the floor is
trivially hideable at large M (v10 runs 100s of µs–ms there). And the *fused* (c) kernel does **no**
separate pass at all — it swizzles in-register during the GEMM's existing weight load, so its true
extra cost is VGPR/ALU pressure, which only building it measures.

### 5. Short-K caveat: MoE gemm2 barely benefits
`moe-gemm2 2304×896 g32` (short K): Triton wins all the way to **M=1024**; v10 only leads from
M=2048 and by just **1.3×** at M=4096 — consistent with the documented gemm2 short-K weakness.
`moe-gemm1 1792×2304 g32` crosses over ~M=512, 1.7× at M=4096. So (c)'s payoff is on the dense
FFN/attention shapes (K=4096–11008), not short-K MoE.

## What this does NOT prove (Phase 2 = the build-gate's other half)
All `hip_v10` numbers above are for the **native** `(N,K//8)` layout. The fused (c) kernel must read
the **transposed** `(K,N//8)` Triton layout, whose WMMA B-fragment loads are strided/transposed —
exactly the access pattern that could tank bandwidth and erase the win. Phase 1 confirms: a win
exists (1), it's graph-stable (1), and the data is recoverable bit-identically (3). It does NOT
confirm the fused kernel preserves the win under transposed reads. That needs the kernel.

## Strategic note — three configs, and VRAM is NOT a differentiator
All three are **single-copy**: native `(N,K//8)` and Triton `(K,N//8)` store the same int4 nibbles
once — a transpose doesn't change byte count. The VRAM win (single-layout vs the old two-copy
"tuned" mode) is already banked and is identical across A/B/C. So it's off the scale:

| config | small-M | large-M | build cost |
|---|---|---|---|
| **(A) today** — single HIP layout, v5/v10 everywhere | ~2× slow | fast | free |
| **(B) free win** — single Triton layout, Triton everywhere | fast (~2×) | forfeits 1.5–2× | free |
| **(C) proposal** — single Triton layout, Triton small-M + fused-HIP large-M | fast | fast | multi-day kernel + transposed-read risk |

(C)'s entire value reduces to: **"(B)'s small-M + (A)'s large-M, for a hand-rolled HIP swizzle."**
(B) is a free strict improvement over (A) for decode-heavy serving; (C)'s only extra prize over (B)
is the large-M 1.5–2× (prefill/batch-heavy), gated on Phase 2.

**Share direction is itself a Phase-2 choice** (don't pre-commit): Triton-canonical (HIP pays the
swizzle, as framed here) vs HIP-canonical `(N,K//8)` + a transposed-read *Triton* small-M kernel
(lean on the compiler instead of hand-rolling `v_perm_b32`, but risks losing Triton's pattern-matched
small-M magic). Test both. Also: the small-M Triton numbers here use `triton_w4a16_ref.py`;
production's tuned `triton_w4a16_gfx1201.py` (M≤32) may be faster, so the small-M win is a **floor**.

## Scope: the dense large-M path has NO live consumer — (C) is a solution waiting for a workload
This is the constraint that should gate Phase 2, elevated from footnote to headline:

- **Production `serve` (Qwen3.6-35B-A3B):** quantizes ONLY routed MoE experts (attn/shared/dense all
  in the quant `ignore` list), and its W4A8 MoE kernel doesn't even execute (stock `fused_moe` runs).
  → does not touch the dense v5/v10 path at all.
- **`single` profile (Qwen2.5-Coder-7B-AWQ):** dense layers *are* int4, but it's a smoke-test profile,
  not a production deployment, and its shapes aren't in `crossover_cache.json` (would need load-time
  autotune to engage v10).
- **The +53% prefill win:** measured on `Qwen3.6-27B-AWQ-INT4` (hidden=5120, inter=17408) with
  `VLLM_ROCM_W4A8_FORCE=on` — a benchmark/sweep model, **not in docker-compose**, and large-batch
  still OOMs at load on 16 GB (`profiling/sweep-2026-06-13/SWEEP_FINDINGS.md`).
- **`zaya` profile:** W4A8 explicitly disabled.

→ **No currently-served model runs the dense W4A8 v5/v10 GEMM at large M.** (C) is technically
viable and graph-stable, but funding the multi-day fused kernel only pays off if a dense W4A8 model
(27B, or promoting Qwen2.5-7B) becomes a real served target. The production 35B's actual bottleneck
is the MoE path (where the kernel doesn't execute) — a different problem. **Recommendation: do NOT
fund Phase 2 until a live dense large-M consumer exists;** bank the verified result and revisit when
a dense W4A8 model is promoted to a served profile.
