# CAM / Titans bolt-on memory — CONTINUANCE (self-contained handoff)

**Read this first in any new session, then advance the next incomplete step and update this file.**
Last updated: 2026-06-29 (**STEP-3 READ-BYPASS FIXED — RETRIEVAL NOW PASSES, the canonical-memory
training loop is END-TO-END VALIDATED on the hub (3b done). Dense memory 0.621 / no_memory 0.000 /
ceiling 0.988, ΔNLL +19.17 bits; s24 (2:4-by-design) 0.678 / 0.000 / 0.988, ΔNLL +19.10 — BOTH PASS
the v0 bar (memory ≫ no_memory+0.15 and >0.5; M=8 chance 0.125), FIDELITY DELTA s24−dense +0.057 (2:4
HOLDS). The fix was THREE parts: (1) RMSNorm the per-head read + a final read_out_norm (read norms no
longer swamp the gate); (2) a write→read addressing-supervision InfoNCE (read query→write-address +
retrieved-ctx→stored-value, found the queried binding by matching the QA cargo token) so reads DEPEND
on store content; (3) the ROOT cause — `evaluate()` never re-attached the MAG forward hook after
train_arm removed it, so `set_bank` was a silent no-op and memory==no_memory EXACTLY (logit Δ=0). With
the hook attached the bank now drives the base (logit Δ 16.2, 11/12 argmax flips). 3000 steps/arm @
0.23 (dense)/0.33 (s24) s/step, 12.29 GB peak, ~12 min/arm on one gfx1201 card. Code:
`warmstart/pk_store.py` (RMSNorm + write_addr_val/head_query) + `warmstart/train_mem_canonical.py`
(addr loss + eval-hook fix), logs `warmstart/logs/memc-3b-{dense,s24}.log`. Both GPUs freed, no orphan,
no hang. NEXT = the full knowledge-store-grade step-3 run (3c) at n_sub≈100-320 / N=10k-100k slots —
VRAM-bound on the value bank (bf16 bank + small eval batch to stay local, or 4×3090 cloud DP).** Prior
this date — STEP-3 SMOKE: pipeline PLUMBS (loss 15→4.2, gate opens, gamma-alone zero-init holds) + §6
cost PINNED, but retrieval FAILED (the bypass now fixed above). Prior this date:
CANONICAL_BUILD_PLAN **step 1–2 (ATLAS) — v1-local6 d=4096 canonical-Z atlas BUILT** from the 6 local cards as a validated checkpoint: `ckpt/atlas/canonical_z_v1_local6.pt` (Z[102,4096] near-orthogonal spherical-code keys, min-angle 89.4°/max-cos +0.010; isotropy condition 46.9→6.85; all 6 members map in, worst = Zaya CCA +0.647; base-neutral LOO-rho ≥ +0.908; uniform z-scored relrep; builder `warmstart/build_atlas.py`, CPU-only). The committee→consensus→whiten→spherical-code pipeline is PROVEN end-to-end on the local merge surface. **RECOMMENDED NEXT = step-3 memory training on this hub (1-hr smoke first to pin §6 unknowns); cloud fan-out is the parallel, orthogonal track.** Prior this date: **step 1–2 (d) — 35B-A3B MoE PROBED via minisglang TP=2**: the 6th — and LAST AMD-fixable — local committee card now exists (`ckpt/probe/qwen36_35b_a3b.pt` + `cyankiwi__Qwen3.6-35B-A3B-AWQ-4bit.pt`, sha 28a4acf8, L=27/40, d=2048, geometry=MoE). Probed inside the minisglang engine at TP=2 (`-n 2`, both cards) because the 35B int4 doesn't fit one card and HF mis-loads its experts. KEY FINDING: unlike Zaya, the MoE's `R` is HEALTHY even under plain centering (off-diag std 0.233, 0% pairs |>0.95|, A.norm~3.84 — low-norm like the GDN sibling but well-spread) and under z-score it is **CENTRAL, NOT an outlier**: mean-rho-to-others ≈ +0.79 (Qwen3.5-4B +0.847, gemma +0.846, Qwen3-0.6B +0.833, Llama +0.824, Zaya +0.607). So the GDN+MoE hybrid is mainstream-Platonic while the pure-CCA Zaya sits apart — the central MoE + the Zaya outlier bracket the geometry the atlas is for. **6 local members now in hand; all AMD-fixable members done.** NEXT = cloud fan-out of the un-cached/un-AMD-able roster, THEN build the d=4096 whitened spherical-code canonical-Z atlas (Z-SCORING ALL MEMBERS UNIFORMLY) — or bootstrap a FIRST atlas from the 6 local cards as a checkpoint while cloud arrives. Prior: (c) Zaya DONE (CCA outlier mean-rho +0.470), (b) 4 HF members DONE, bank 16→102 sha 28a4acf8. The v1 build is LOCKED; no open user decision.

## What this project is (the one paragraph)
Bolt a Titans-style long-term **associative memory** onto a **frozen** base LLM (Qwen3.5-4B now)
without touching base weights. The memory is reached via **Memory-as-Gate (MAG)**: a zero-init gated
residual tap injects retrieved memory into the base's residual stream. The north-star product is a
**base-agnostic canonical memory** that any frozen LLM attaches to via a small learned
embedding-space **translator** (RecursiveMAS-style) — "download the memory, fit a tiny translator."
Full design = `titans/CAM_DESIGN.md`. This is the **CAM** (Canonical Associative Memory).

## Where everything lives
- Worktree: `/home/pat/code/vllm-gfx1201-titans` (branch `feat/titans`). Titans code in `titans/`.
- Design (north star + triage + bibliography): `titans/CAM_DESIGN.md`
- V0 build spec: `titans/V0_SPEC.md`
- Phase history: `titans/PHASE0_RESULTS.md`, `PHASE1_RESULTS.md`, `PHASE1_WARMSTART_PLAN.md`,
  `deep_mem/DEEP_MEM_KERNEL.md`
- Memory core: `titans/deep_mem/deep_memory.py` (`DeepMemory`, validated stages 1–3)
- Binding harness (MAC, the wall): `titans/warmstart/recall_boltA.py`
- Adapter + frozen-base loader + runner: `titans/warmstart/m2_adapter.py`,
  `titans/warmstart/run_m2.sh`
- Recall task: `titans/warmstart/recall_deepmem.py` (DocBuilder, single_token_ids, NAME/CARGO)

## What is DONE / validated
- **DeepMemory stages 1–3** — analytic surprise == autograd (1e-16), parallel scan == sequential
  (2e-16), graph-free module GPU-validated; fixes the lucidrains OOM; scales to 64 segments.
- **realemb** — DeepMemory BINDS frozen Qwen embeddings, held-out carry **0.945** (memory is sound).
- **boltA** — MAC (input-embeds prefix) hits the **INJECTION-MECHANISM WALL**: binding+delivery solved
  (direct carry 0.86) but generative-through-frozen-base `memory ≈ no_memory` (0.000 acc, ~22.6 bits).
  ⇒ the **MAC→MAG pivot**.
- **CAM_DESIGN.md** + **V0_SPEC.md** written (this design conversation, 2026-06-28).
- **V0 IMPLEMENTED (2026-06-28):** `titans/warmstart/gated_tap.py` (`GatedMemoryTap` + `MAGInjector`,
  zero-init gated cross-attn, forward-hook injection, robust decoder-layer finder) and
  `titans/warmstart/recall_mag.py` (stage-1 bind via recall_boltA's direct loss → freeze → stage-2
  gate-only LM-loss training → boltA-mirrored eval; default sweeps each tap depth, `--multi` taps all
  together).
- **V0 RUN ON GPU — MAG WORKS, V0 PASSES (2026-06-28).** Log `titans/warmstart/logs/cam-magv0.log`.
  Smoke (40/40 @ L=16) caught one bug: the fp32 tap linears hit the bf16 base hidden →
  `RuntimeError: mat1/mat2 dtype mismatch`. **Fixed** in `gated_tap.py forward()`: compute the whole
  tap in the param dtype (`h.to(wdt)`, fp32), cast only the additive update back to `h.dtype` (keeps
  the gate=0 init an exact bf16 no-op). Re-smoke clean end-to-end. Full run (`--bind-steps 3000
  --steps 3000 --tap-layers 8,16,24`):
  - **Stage-1 binding held-out: carry 0.860 / ablated 0.014 / chance 0.333** — matches boltA; memory
    front-end re-validated.
  - **Stage-2 MAG, generative through the frozen base (memory / no_memory / ceiling):**
    | tap L | memory acc | no_memory acc | ceiling | ΔNLL | verdict |
    |------|-----------|--------------|---------|------|---------|
    | **8**  | **0.885** | 0.014 | 0.973 | +21.84 bits | MAG WORKS |
    | 16 | 0.000 | 0.000 | 0.986 | nan | WALL — **NaN ARTIFACT, not a real depth fail** |
    | **24** | **0.895** | 0.020 | 0.982 | +27.02 bits | MAG WORKS |
  - **Verdict (V0_SPEC §7 rule 1):** at L=8 and L=24, `memory ≫ no_memory + 0.15` and `> 0.5`,
    approaching the in-context ceiling — the **exact opposite of the boltA MAC wall** (memory ≈
    no_memory ≈ 0.000). **The zero-init MAG tap DELIVERS the binding through the frozen base. V0
    PASSES. → greenlight v1.** Gate opened monotonically 0 → ~0.03 at the passing depths (no
    cognitive-bypass / gate-collapse).
  - **L=16 caveat (known, benign):** gamma diverged to NaN at step 2800 (a single late training
    instability), so its eval shows nan/0.000 — a spurious "WALL", NOT evidence L=16 can't inject
    (ceiling 0.986 proves the path is healthy; L=8/L=24 bracket it and both pass). If L=16 is ever
    needed: add gamma/grad guards (clamp gamma, lower lr, or skip-on-nan) — cheap. Does not gate v1.

- **V0 MEMORY CHECKPOINTING + V1 TRANSLATOR — v1 PASSES (2026-06-29).**
  - **Checkpoint save/load** added to `recall_mag.py` (`save_ckpt`/`load_ckpt`, `--save-ckpt`/
    `--load-ckpt`). The frozen tied embed/unembed (~3 GB) are dropped on save and rebuilt from base-1's
    table on load → checkpoint is **78 MB** (`titans/warmstart/ckpt/cam_v0_L24.pt`: BoltAdapter +
    GatedMemoryTap @ **L=24**, carry 0.860). Full bind+tap re-run reproduced V0 (memory 0.893 /
    no_memory 0.014 / ceiling 0.973) before saving.
  - **Base-1 RELOAD SANITY (PASS):** loading the checkpoint and re-running the eval reproduced V0 on
    Qwen3.5-4B: **memory 0.898 / no_memory 0.025 / ceiling 0.975** (matches the original 0.895). The
    memory is now a fixed, reusable asset.
  - **2nd base = `Qwen/Qwen3-0.6B`** (HF-cached, plain `Qwen3ForCausalLM`, **d_base=1024 ≠ 2560**,
    28 layers, tie=True). Smallest convenient cached different-d_base model; NOT a blocker.
    (Other cached candidates with d≠2560: `poolside/Laguna-XS.2` d=2048 but exotic `LagunaForCausalLM`;
    `Zyphra/ZAYA1-8B` d=2048 exotic `ZayaForCausalLM` — both risky in the titans:dev image; Qwen3-0.6B is
    the clean choice.)
  - **Affine translator** (`titans/warmstart/translator.py`, `TranslatedTap`/`TranslatedInjector`):
    the frozen GatedMemoryTap + frozen mem-bank are reused verbatim; a tiny **5.2M-param** affine pair
    stitches base-2's residual into the tap's d_base1 space and back —
    `A:1024→2560` (in), `B:2560→1024` (out), `gamma2∈R^1024` zero-init gate. `h2' = h2 + tanh(gamma2)·B(tap(A h2)−A h2)`.
    Mirrors the fp32-compute/cast-back dtype pattern; has NaN/Inf loss+grad skip guards + a gamma2 clamp.
    Only A/B/gamma2 train (LM-loss through frozen base-2); base-2 + memory + tap all frozen. The mem-bank
    is base-agnostic (DeepMemory's mem_dim retrieval from base-1's own frozen embedding) so the SAME
    memory drives base-2. Recall task built with TWO DocBuilders sharing an rng seed (identical logical
    bindings, each base's own vocab).
  - **Base-2 eval (`recall_v1.py`, 200-step translator fit) — v1 PASSES:**
    | base | memory acc | no_memory acc | ceiling | ΔNLL | verdict |
    |------|-----------|--------------|---------|------|---------|
    | Qwen3-0.6B (d=1024) | **0.604** | **0.000** | 0.961 | +41.4 bits | TRANSLATOR WORKS |
    `memory ≫ no_memory + 0.15` and `> 0.5` on a SECOND base of different hidden dim, with only a tiny
    affine translator fit → **"one memory, two bases" proven** → the Modular Memory Organ product is real.
  - **Two init bugs found+fixed:** (1) double-zero-init dead start — zeroing BOTH B and gamma2 makes
    every gradient zero (grad to A/B flows through tanh(gamma2)=0; grad to gamma2 flows through B=0).
    Fix: zero-init **gamma2 ALONE** (the sole gate), random-init B — mirrors V0's tap (only its gamma is
    zeroed, to_o stays random). Caught by the 40-step smoke (`|g|grad 0.00`). (2) A longer fit was
    converging BETTER (acc 0.812 @ step 1400) but the box threw repeated **transient GPU hangs** this
    session (killed 3 runs); fail-fast-killed each to free the shared lease. lr=1e-3 also tripped the
    NaN-grad guard periodically (guard recovered cleanly, never diverged); lr 5e-4 is calmer. The
    200-step PASS stands as the headline; a clean longer run is a cheap follow-up when the GPU is stable.

- **CROSS-FAMILY FALSIFIER — vocab-family leakage RULED OUT (2026-06-29).** The v1 result was on
  Qwen3-0.6B (SAME tokenizer family as base-1 Qwen3.5-4B) — a possible vocab-leakage confound. This
  increment repeated the translator falsifier on a **genuinely different family**: base-2 =
  **`unsloth/Llama-3.2-3B`** (d_base2=**3072**, **28** layers, `LlamaForCausalLM`, **Llama tiktoken
  BPE vocab** vocab=128256, bos=128000 — a DIFFERENT tokenizer AND architecture; ungated public
  mirror, `snapshot_download`ed CPU-side, ~6GB bf16). Frozen v0 memory (`ckpt/cam_v0_L24.pt`) + the
  affine translator reused verbatim; only A/B/gamma2 trained by LM-loss through the frozen Llama base.
  The single-token NAME/CARGO recall vocab was rebuilt for the Llama tokenizer (43 names / 28 cargo
  single-token; colon `[25]` / nl `[198]` single tokens → dict phrasing intact; BOS=128000 handled by
  the existing `DocBuilder.bos` path). Tap depth mapped proportionally L=24/36 → L=21/28.
  - **Result (3000-step fit, lr 5e-4, batch 8) — v1 PASSES on the cross-family base:**
    | base | family / d_base | memory acc | no_memory acc | ceiling | ΔNLL | verdict |
    |------|-----------------|-----------|--------------|---------|------|---------|
    | Qwen3-0.6B (200-step) | Qwen / 1024 | 0.604 | 0.000 | 0.961 | +41.4 | PASS (same-family) |
    | **Llama-3.2-3B (3000-step)** | **Llama / 3072** | **0.602** | **0.010** | **0.920** | **+22.8** | **PASS (CROSS-family)** |
    memory (0.602) ≫ no_memory (0.010)+0.15 and >0.5, on a genuinely different tokenizer+arch — the
    SAME passing signature as Qwen-0.6B. **⇒ vocab-family leakage is RULED OUT.** The translator is
    NOT exploiting Qwen-family vocab/embedding similarity; the canonical-memory + tiny-affine-translator
    thesis is validated across families. The fitted **translator card is saved**:
    `ckpt/translator_llama32_3b.pt` (63 MB, A 3072→2560 / B 2560→3072 / gamma2 + meta — the §5.5 UMX
    product artifact). This 3000-step run also IS the clean longer fit (priority-2): gate opened
    monotonically 0→0.018, loss 14.5→~0.9, no NaN/grad-guard trips at lr 5e-4.
  - **Plumbing fixes this increment (in `recall_v1.py` / `run_m2.sh` / `translator.py`):** (1) `--base2`
    arg (default Qwen3-0.6B; pass `unsloth/Llama-3.2-3B` for cross-family). (2) **OOM on a 16GB card**
    — base-1 (Qwen-4B ~8GB) + a large cross-family base-2 don't co-fit. Root cause: `MAGInjector.__init__`
    does `self.layers = decoder_layers(base)`, pinning the WHOLE Qwen-4B alive even after `del m1`.
    Fix: after extracting the standalone `frozen_tap`, `injector_tap.layers=None; del injector_tap`,
    then drop `embed_weight` and the adapter's tied `unembed` buffer (~1.5GB fp32, used ONLY by the
    DIRECT bind loss, never at v1). Reorder: build the v0 memory front-end, FREE base-1, THEN load
    base-2. (3) `load_base` now isolates loader-selection (CausalLM vs ImageTextToText) from the device
    move so a real HIP/OOM error surfaces instead of a bogus "unrecognized config" fallback.
    (`expandable_segments` is NOT supported on this ROCm allocator — don't rely on it.) (4)
    `save_translator()` added to `translator.py` + `--save-translator` to `recall_v1.py`.

- **CROSS-FAMILY FALSIFIER — THIRD BASE = Gemma-3-4B (2026-06-29).** Repeated the translator
  falsifier on a SECOND non-Qwen family: base-2 = **`unsloth/gemma-3-4b-pt`** (Gemma SentencePiece
  vocab + Gemma arch, **34 layers**, d_base2=**2560**; tap mapped L=24/36 → L=26/34). Frozen v0
  memory (`ckpt/cam_v0_L24.pt`) + the ~13M affine translator reused verbatim; only A/B/gamma2 trained
  by LM-loss through the frozen Gemma base. Log `titans/warmstart/logs/cam-gemma-v1.log`.
  - **Result (3000-step fit, lr 5e-4, batch 8) — PARTIAL (cross-family transfer real, just under bar):**
    | base | family / d_base | memory acc | no_memory acc | ceiling | ΔNLL | verdict |
    |------|-----------------|-----------|--------------|---------|------|---------|
    | **Gemma-3-4B (3000-step)** | **Gemma / 2560** | **0.488** | **0.000** | **0.998** | **+48.3** | **PARTIAL (CROSS-family)** |
    memory (0.488) ≫ no_memory (0.000) with no_memory pinned at zero on a Gemma tokenizer the memory
    never saw, and ΔNLL +48.3 bits (the largest separation of any base) → cross-family **transfer is
    genuine** (the logged verdict is PARTIAL only because 0.488 sits just below the 0.5 PASS bar — a
    **purely affine** translator doesn't FULLY recover Gemma's residual geometry; this localizes the
    next win to the translator, not the memory). Gate opened monotonically 0→0.263, one early
    NON-FINITE-loss step at 298 cleanly skipped by the guard, no divergence. Translator card saved:
    **`ckpt/translator_gemma3_4b.pt`** (52 MB). **⇒ vocab-family leakage now RULED OUT across THREE
    bases / TWO non-Qwen families (Llama PASS + Gemma strong-PARTIAL + Qwen same-family PASS).**
  - **Citable summary written:** **`titans/RESULTS.md`** — the validated thesis + per-base comparison
    table (Qwen3.5-4B V0 / Qwen3-0.6B / Llama-3.2-3B / Gemma-3-4B), the leakage-ruled-out conclusion,
    and what it proves (download-the-memory + tiny-translator-card is real & cross-family).
  - **Early-return recovery note:** the worker that launched this Gemma fit returned before finalizing
    while the run was still in flight (~step 2400/3000). A recovery iteration (cam-loop-gemma-finalize)
    waited it out and finalized: the run completed cleanly on its own (container `titans-gemma-v1`
    gone, GPU 0 lease freed, no hang — no intervention needed), then captured the eval, confirmed the
    saved card, wrote RESULTS.md, and updated this file. No competing GPU run was launched.

- **STEP 0 — 2:4-BY-DESIGN vs DENSE FIDELITY DE-RISK — 2:4 HOLDS, PASS (2026-06-29).** First build
  step of CANONICAL_BUILD_PLAN §3. Masked **2:4 structured sparsity (SR-STE, by-design from init)**
  into the v0 GatedMemoryTap's four serve-weight projections (to_q/to_k/to_v/to_o — the d×d weights
  SWMMAC accelerates) and measured whether the v0 recall PASS survives vs a dense reference. New code:
  `titans/warmstart/sparse24.py` (`Mask24Linear`: magnitude top-2-of-4 along the input dim + STE +
  SR-STE pruned-weight decay λ=2e-4) and `titans/warmstart/recall_24derisk.py` (loads the FROZEN v0
  memory `ckpt/cam_v0_L24.pt` — does NOT re-bind — and trains a FRESH tap per arm at L=24 from the SAME
  seed: `dense` = nn.Linear control, `s24` = Mask24Linear). Log `titans/warmstart/logs/d24-full.log`.
  - **Result (3000-step fit each arm, lr 1e-3, batch 16, eval n=512):**
    | arm | memory acc | no_memory acc | ceiling | ΔNLL | s/step | conv@ | verdict |
    |-----|-----------|--------------|---------|------|--------|-------|---------|
    | dense | **0.893** | 0.014 | 0.975 | +26.16 | 0.212 | 271 | PASS |
    | **s24 (2:4-by-design)** | **0.887** | 0.012 | 0.975 | +25.94 | 0.233 | **124** | PASS |
    - **FIDELITY DELTA (s24 − dense) = −0.006 memory-acc** — within v0's own reload noise (v0 was
      0.885 / 0.895 / 0.898 across reloads). **BOTH arms PASS the v0 bar** (memory ≫ no_memory+0.15 and
      >0.5), reproducing v0 (dense 0.893 ≈ the saved 0.895). **⇒ 2:4-by-design HOLDS v0 fidelity** —
      masked-from-init training is NOT the +25%-PPL prune-after-dense path; the only residual 2:4 risk
      (fidelity) is **retired**. The remaining 2:4 work is purely serve-side SWMMAC kernel tuning.
    - **All four projections masked to exactly 0.5 sparsity**; gate stayed an exact no-op at init
      (loss 14.115, gate 0.0 for both arms — gamma-alone zero-init verified through the masked weights);
      gate opened monotonically 0→~0.030, no NaN/grad-guard trips. s24 even converged FASTER (step 124
      vs dense's 271 — first 10-step window ≥0.90 acc), consistent with SR-STE's regularizing decay.
  - **1-HR SMOKE / TRAINING-COST PIN (the §6 unknowns, now real numbers):** at this v0 recall scale
    (4B-class frozen base, batch 16, L=24 tap) **step-time ≈ 0.21–0.23 s/step on ONE gfx1201 card**
    (dense 0.212 / s24 0.233 steady-state — 2:4 adds negligible train cost, as expected: SR-STE is plain
    torch). **Steps-to-converge ≈ 120–270** (first ≥0.90-acc window; the toy task is easy — full
    knowledge-store-grade will be more). Both arms' full 3000-step fit + eval finished in **~25 min
    total wall on a single card**, well under the 1-hr budget. Batch sense: acc is stable across
    batch-16 minibatches (per-step acc 0.69–1.0, no batch-size starvation at this scale) — the build
    plan's 4×3090 DP critical-batch question is a knowledge-store-grade concern, not visible on the toy.
    *Caveat:* these s/step are gfx1201 ROCm numbers; the NVIDIA training box (CANONICAL_BUILD_PLAN §4)
    will differ, but the **2:4-adds-~0 train-cost** and **fast-convergence** findings carry over.
  - **One plumbing fix (16GB card):** running BOTH arms in one process OOM'd at the s24 arm's first GDN
    forward (the dense arm's tap + the standalone fp32 embed clone + the adapter's tied unembed buffer
    pinned ~3 GB). Fix in `recall_24derisk.py`: after `load_ckpt`, free `embed_weight` + `adapter.unembed`
    (used only by the never-called direct bind loss) and `gc.collect()/empty_cache()` between arms.
    `expandable_segments` remains unsupported on this ROCm allocator (carried-over caveat).

- **STEP 1–2 (a) — RELATIVE-REP COMMITTEE-PROBE HARNESS up + validated on cached bases (2026-06-29).**
  First sub-step of CANONICAL_BUILD_PLAN §1.1–§1.2 (committee probe → canonical-Z atlas). New file
  `titans/warmstart/probe_relrep.py`: a forward-only, per-model-decoupled extractor that, given a base,
  (1) forwards a FIXED 16-anchor text set (the shared alignment key) through the FROZEN base, (2) taps
  the residual hidden state at the **proportionally-mapped** depth `L = round((24/36)·n_layers)` via a
  forward hook on the decoder-layer ModuleList (reuses `gated_tap.decoder_layers`), (3) mean-pools over
  tokens → one d_base vector per anchor (`A [16,d_base]`), and (4) emits the **relative-representation**
  matrix `R = centered-cosine(A_i, A_j) [16,16]` — tokenizer- AND dim-agnostic, so different-vocab/
  different-d models land in the SAME [16,16] space the atlas fuses. Dumps a per-model card
  `ckpt/probe/<slug>.pt` `{model,n_layers,tap_frac,tap_layer,d_base,n_anchor,tok_counts,A,R,R_raw}`.
  Log `titans/warmstart/logs/probe-smoke.log`.
  - **CRITICAL probe-quality fix found+applied:** raw `cos(A_i,A_j)` on the deep-tap mean-pool COLLAPSES
    toward 1.0 (Qwen-0.6B raw off-diag 0.976–0.996, Gemma 0.994–0.998, A.norm up to ~6e4) because the
    residual is dominated by a few **massive-activation / rogue dimensions** (a near-constant per-model
    bias). The standard relative-rep normalization (Moschella 2209.15430) — **center across anchors
    before cosine** — removes that shared bias and exposes real geometry. `R` now = CENTERED cosine
    (`R_raw` kept as a diagnostic). Sanity bar tightened to require mean-off-diag < 0.9.
  - **Validated end-to-end on 3 already-cached bases (1 leased gfx1201 card, forward-only, exit 0,
    lease freed clean, no hang):**
    | base | family/mech | n_layers | tap L | d_base | centered R off-diag min/mean/max |
    |------|-------------|----------|-------|--------|----------------------------------|
    | Qwen/Qwen3.5-4B | Qwen / GDN | 32 | 21 | 2560 | −0.347 / −0.005 / 0.534 |
    | Qwen/Qwen3-0.6B | Qwen / GQA | 28 | 19 | 1024 | −0.347 / −0.007 / 0.616 |
    | unsloth/gemma-3-4b-pt | Gemma / soft-cap | 34 | 23 | 2560 | −0.518 / −0.006 / 0.591 |
    All cards finite, diag==1, anchors well-distinguished (mean off-diag ≈ 0 after centering, real
    spread). **Cross-model relative-rep correlation (Pearson over off-diagonals) = 0.564–0.620** — the
    Platonic-convergence signal the atlas banks on: high enough to confirm SHARED structure across
    families AND hidden dims (1024 vs 2560 vs 2560), well below 1.0 so model-specific geometry is
    preserved. (Raw-cosine rho was an inflated ~0.82 — centering both spread the matrices AND made the
    cross-model agreement honest.) **⇒ the probe harness is proven; ready to fan out to the committee.**
  - **Note for the atlas builder:** the cached `Qwen/Qwen3.5-4B` config has **32** layers (NOT the 36 an
    earlier note assumed) → tap maps to L=21, not 24; the harness derives L from each model's real
    `n_layers` so this is automatic. Cards stack on `R` (the [16,16] key); `A`/`d_base` differ per model
    and are carried for the whiten→spherical-code step.

- **STEP 1–2 (b) — ANCHOR BANK SCALED + PROBE FANNED across the LOCAL committee (2026-06-29).** Second
  sub-step of CANONICAL_BUILD_PLAN §1.2. (1) **Scaled the anchor bank 16 → 102** content-diverse anchors:
  new `titans/warmstart/anchor_bank.py` — a FIXED, reproducible-by-construction (static list, no RNG)
  curated bank spanning 7 content axes (factual 16 / science-math 18 / code 16 / narrative 16 / dialogue
  12 / structured 12 / multilingual 12), with a stable `anchor_sha` (28a4acf8…) saved into every card +
  to `ckpt/probe/anchor_bank.pt` so EVERY committee member (local + cloud) provably forwards the identical
  ordered anchors before its `R` merges into the atlas. `probe_relrep.py` now imports the bank, stamps
  `anchor_sha`+`categories` into each card, passes `trust_remote_code=True` (for bundled-modeling members
  like Laguna), and **SKIPs-and-records** an unloadable/OOM model instead of aborting the batch. Logs
  `titans/warmstart/logs/probe-{fan,*}.log`, analysis `probe-analysis.log`.
  - **4 LOCAL committee members probed CLEAN (1 leased gfx1201 card, forward-only, all sanity-OK, every
    card sha-identical = same bank):**
    | base | family/mech | n_layers | tap L | d_base | A.norm | centered R off-diag min/mean/max |
    |------|-------------|----------|-------|--------|--------|---------------------------------|
    | Qwen/Qwen3.5-4B | Qwen / **GDN hybrid (SSM)** | 32 | 21 | 2560 | 13.9 | −0.257 / −0.010 / 0.761 |
    | Qwen/Qwen3-0.6B | Qwen / GQA | 28 | 19 | 1024 | 495.5 | −0.968 / −0.007 / 0.992 |
    | unsloth/Llama-3.2-3B | Llama / GQA | 28 | 19 | 3072 | 41.1 | −0.883 / −0.009 / 0.968 |
    | unsloth/gemma-3-4b-pt | Gemma / soft-cap | 34 | 23 | 2560 | 56562 | −0.956 / −0.005 / 0.982 |
    All R finite, diag==1, mean-off-diag ≈ 0 after centering (real spread). **Gemma's massive raw A.norm
    (~5.7e4 massive-activation bias) is fully absorbed by centering** — its rel-rep is healthy, not
    degenerate. **No degenerate probe** in the four.
  - **Cross-model rel-rep structure (102-anchor, Pearson over off-diagonals) — mean +0.539, range
    +0.314…+0.853:** the Platonic-convergence signal holds at scale (consistent with the 16-anchor
    smoke's 0.56–0.62), well below 1.0 so model-specific geometry is preserved. Cluster structure:
    **Llama-3.2-3B is the most central** (mean-rho-to-others +0.643 → the natural atlas anchor);
    **Qwen3-0.6B↔Llama-3.2-3B is the tightest pair (+0.853)** despite different families AND dims
    (1024 vs 3072); **Qwen3.5-4B is the outlier** (mean-rho +0.367, tiny A.norm 13.9, compressed
    off-diag range) — its **GDN-hybrid SSM mechanism genuinely sits apart** in residual geometry (the
    architectural-diversity the committee is FOR, NOT a bad probe — its R still spreads & is distinct).
  - **5 members are MISSING/CLOUD/BLOCKED this increment (recorded, NOT retried in a loop):**
    | member | cached? | local probe result | disposition |
    |--------|---------|--------------------|-------------|
    | **poolside/Laguna-XS.2** | yes | **EXIT 137 (SIGKILL)** in the forward — exotic conv-hybrid remote code (loads via trust_remote_code, weights 100%, then process-killed; uncatchable, took down the batch → re-run isolated) | local, needs debug (OOM vs HIP-fault) — dedicated increment |
    | **TechxGenus/DeepSeek-V2-Lite-AWQ** | yes | SKIP — AWQ needs `autoawq`, which `ImportError`s on `is_torch_fx_available` (transformers 5.5.3 version skew) | cloud (or fix autoawq/transformers pin) |
    | **cyankiwi/Qwen3.6-35B-A3B-AWQ** | yes | EXIT 137 (OOM) — 35B AWQ unpack > 16 GB on one card | cloud, or local TP=2 (`-n 2`) |
    | **Zyphra/ZAYA1-8B** | yes | NOT attempted — `zaya` model_type is **NOT in transformers 5.5.3**, no bundled remote code / auto_map; the only working loader is **minisglang's `models/zaya.py` + `cca_hip` .so** (gfx1201-only) | local via minisglang — own increment (see NEXT) |
    | Ministral-8B / BitNet-2B4T / LFM2-1.2B / GPT-OSS-20b / Llama-3.3-70B | NO | un-cached | cloud fleet (no local download this increment) |
  - **Cards on disk:** `ckpt/probe/{Qwen__Qwen3.5-4B, Qwen__Qwen3-0.6B, unsloth__Llama-3.2-3B,
    unsloth__gemma-3-4b-pt}.pt` (each {model,n_layers,tap_layer,d_base,n_anchor=102,anchor_sha,categories,
    A,R,R_raw}) + the fixed `anchor_bank.pt`. The harness fans more members with just `--models …`.

- **STEP 1–2 (c) — LOCAL ZAYA PROBED via minisglang + cca_hip (2026-06-29).** The 5th local committee
  member. Zaya (`zaya` model_type) is unloadable by pure HF/transformers — the ONLY working loader is
  **minisglang's `models/zaya.py` + the vendored `cca_hip` (torch.ops.zaya_cca) kernel** (gfx1201-only;
  the owner OK'd minisglang where it is the best choice). Stood up a forward-only rel-rep probe INSIDE
  the minisglang engine so the CCA recurrent-state cache + paged KV + ctx are correct, then mirrored the
  HF harness exactly: forward the SAME 102-anchor bank (sha **28a4acf8**, byte-identical), tap the fp32
  merged **residual stream** at the proportionally-mapped depth **L=round(0.667·80)=53**, mean-pool per
  anchor → `A[102, d=2048]`, center, cosine → `R[102,102]`.
  - **Path used (per the minisgl MANDATORY isolation rules):** git worktree
    `/home/pat/code/minisgl-rdna4-cam-zaya` (branch `cam-zaya-probe`, off HEAD be120c5), with all **12
    vendored .so copied in** (incl. `cca_hip/zaya_cca_C…so` — they're gitignored). The probe ran in the
    **`vllm22-w4a8:combined`** image (NOT titans:dev) via the absolute `gpu-lease.sh -n 1`, mounting the
    WORKTREE (not the shared $PWD), the fp8 checkpoint `/home/pat/models/ZAYA1-8B-fp8`→`/models`, the
    warm Triton cache, and the HF cache. Eager (no graph), forward-only, single card; smoke (1 anchor →
    tapped residual `(9,2048)`) passed first, then all 102 anchors, exit 0, lease freed, no hang.
    Probe code: worktree `tools/zaya_relrep_probe.py` + `tools/zaya_relrep_probe.sh`. The
    `ZayaDecoderLayer` is a `BaseOP` (no `register_forward_hook`) — the tap WRAPS the layer instance's
    bound `forward` (the model calls `layer.forward(...)` explicitly), gated on `ctx.batch.is_prefill`
    so a stray decode step can't overwrite the prompt residual.
  - **Geometry: Zaya (L=53/80, d=2048, geometry tag CCA) — A.norm~1629; centered-cosine `R` is
    DEGENERATE.** 99.1% of the centered across-anchor variance is in ONE rogue dimension (massive
    activation far past Gemma's), so plain centering — the existing harness normalization — does NOT
    decorrelate it: |centered-cos|>0.95 for ~85% of off-diagonal pairs (off-diag std 0.93, max +1.000).
    The standard relative-rep fix (Moschella 2209.15430 includes per-feature standardization) — **z-score
    per dimension before cosine** — fully recovers real geometry: off-diag std **0.33**, max +0.975, only
    1 pair >0.95, distinct=True. The card stores **both**: `R` (centered, format-identical to the 4 HF
    cards but degenerate for Zaya), `R_zscore` (the healthy one), `R_raw`, and the raw `A` — so the atlas
    builder can recompute any normalization for EVERY member (each card carries `A`). A
    `normalization_note` documents the collapse.
  - **rho vs the other 4 (recomputed BOTH ways from stored A; all 5 shas == 28a4acf8):** under the
    correct **z-scored** R, the 4 HF members converge MORE tightly than the centered read suggested
    (pairwise +0.85…+0.94, mean-rho +0.77…+0.80 — Platonic signal even stronger), and **Zaya is the clear
    OUTLIER: mean-rho-to-others +0.470** (every Zaya pair +0.45…+0.49). I.e. the **CCA conv-hybrid
    mechanism genuinely sits apart in residual geometry — like the GDN-hybrid Qwen3.5-4B read as the
    outlier in the 16-anchor smoke.** This is the architectural diversity the committee is FOR, NOT a bad
    probe: Zaya's z-scored R is non-degenerate and well-spread. **Implication for the atlas builder:
    Z-SCORE ALL MEMBERS UNIFORMLY** (recompute each member's R from its stored `A` as
    `normalize(z-score(center(A)))·…`) before stacking — it both fixes Zaya AND makes the HF members'
    convergence honest.
  - **Card on disk:** `ckpt/probe/zaya.pt` AND `ckpt/probe/Zyphra__ZAYA1-8B.pt` (identical; the slug-named
    one stacks into the atlas like the HF cards) — {model, geometry=CCA, n_layers=80, tap_layer=53,
    d_base=2048, n_anchor=102, anchor_sha, categories, tok_counts, A, R, R_raw, R_zscore,
    normalization_note}. (Worktree cleaned up after the run.)

- **STEP 1–2 (d) — 35B-A3B MoE PROBED via minisglang TP=2 (2026-06-29).** The 6th local committee
  member, and the LAST AMD-fixable one. `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` (qwen3_5_moe, a GDN+MoE
  hybrid, compressed-tensors int4) does NOT fit one 16 GB card and HF mis-loads its int4 experts
  (random MoE → meaningless probe); minisglang HAS the validated **TP=2** loader, so the rel-rep probe
  ran INSIDE the engine at TP=2 (both cards, the justified `-n 2`), mirroring the Zaya increment.
  Worktree `/home/pat/code/minisgl-rdna4-cam-35b` (branch `cam-qwen35b-probe`, off be120c5, 12 vendored
  .so copied), image `vllm22-w4a8:combined`, forward-only, eager. SAME 102-anchor bank (sha **28a4acf8**,
  asserted byte-identical), tap at the proportionally-mapped depth **L=round(0.667·40)=27**, mean-pool
  per anchor → `A[102, d=2048]`. (A prior agent built this infra + confirmed the TP=2 load; it died on a
  HARNESS error, not a GPU failure — this increment re-ran it clean: smoke OK, then all 102 anchors,
  exit 0, container gone, lease freed, no hang.)
  - **Geometry: 35B-A3B MoE (L=27/40, d=2048, geometry tag MoE) — A.norm~3.84; `R` is HEALTHY even
    under plain centering** (off-diag std 0.233, max +0.968, **0% pairs |>0.95|, diag==1, finite**) —
    NOT the single-rogue-dimension collapse Zaya showed. R_zscore is also clean (off-diag std 0.096,
    max +0.880). The card stores `A`, `R` (centered), `R_zscore`, `R_raw` — same format as the others.
    The tiny A.norm (3.84) echoes the GDN-hybrid Qwen3.5-4B (13.9): the SSM/GDN+MoE residual is
    low-norm but well-spread.
  - **rho vs the existing 6 (z-scored, recomputed from each stored `A`; all shas == 28a4acf8):**
    Qwen3.5-4B **+0.847**, gemma-3-4b **+0.846**, Qwen3-0.6B **+0.833**, Llama-3.2-3B **+0.824**,
    ZAYA1-8B +0.607 → **mean-rho ≈ +0.79.** The **MoE is CENTRAL, NOT an outlier** — it converges
    tightly with the 4 HF members AND partly bridges to the Zaya CCA outlier (+0.607, higher than
    Zaya's mean-to-others +0.470). I.e. the GDN+MoE hybrid's residual geometry is mainstream Platonic,
    unlike the pure-CCA Zaya which genuinely sits apart. (Centered-cosine rho is noisier — +0.64…+0.85 —
    confirming z-score is the correct comparable normalization for the atlas, per the Zaya finding.)
  - **Card on disk:** `ckpt/probe/qwen36_35b_a3b.pt` AND
    `ckpt/probe/cyankiwi__Qwen3.6-35B-A3B-AWQ-4bit.pt` (identical) — {model, geometry=MoE, n_layers=40,
    tap_layer=27, d_base=2048, n_anchor=102, anchor_sha, categories, tok_counts, A, R, R_raw, R_zscore,
    normalization_note}. Worktree removed + git-pruned after the run; both cards free.
  - **Committee now: 6 local members** (Qwen3.5-4B / Qwen3-0.6B / Llama-3.2-3B / Gemma-3-4B / ZAYA1-8B /
    **Qwen3.6-35B-A3B MoE**) spanning GDN-hybrid / GQA / GQA / soft-cap / CCA / **MoE** geometries — the
    central MoE + the Zaya CCA outlier bracket the diversity the atlas is for. ALL AMD-fixable members
    are now in hand; what remains is the cloud-only roster.

- **STEP 1–2 (ATLAS) — v1-local6 d=4096 CANONICAL-Z ATLAS BUILT (2026-06-29).** The FIRST real
  canonical hub — a validated checkpoint built from the 6 in-hand local cards BEFORE any cloud
  fan-out (CANONICAL_BUILD_PLAN §1.1–§1.2, CAM_DESIGN §2.1 dial #1). New builder
  `titans/warmstart/build_atlas.py` (pure CPU linear algebra, NO GPU lease — ran inside `titans:dev`
  `--entrypoint bash` CPU-only; the image torch lacks CPU LAPACK so eigendecomps route through numpy).
  Recipe, exactly per the locked plan: (1) **uniform z-scored relrep recomputed from each card's
  stored `A`** (center → per-dim standardize → cosine — the Zaya-proven normalization, a strict
  superset of plain centering); (2) **RLSA consensus mean-atlas** Rbar = equal-weight mean of the 6
  z-scored R (base-neutral by construction); (3) **classical-MDS embed** of Rbar into the hub
  (eigendecomp → anchor coords); (4) **shrinkage-whiten to isotropy** (alpha=0.5 — λ^(−α/2): full
  whitening alpha=1 was found to DESTROY the consensus geometry, rho→0.12, so a tuned shrinkage
  balances isotropy vs the retrieval-ranking geometry that is a CAM non-negotiable); (5) **spherical-code
  shaping** (Tammes/U-Hop repulsion of the closest pairs + a fidelity anchor to the consensus
  direction). Artifact: **`ckpt/atlas/canonical_z_v1_local6.pt`** (1.8 MB; Z[102,4096] unit-norm
  canonical keys, Z-sha 36ce4f6a, deterministic; + Rbar, whiten transform, mds spectrum, full
  validation report, members, anchor_sha 28a4acf8, build_params).
  - **Validation (the point of the checkpoint) — ALL PASS:**
    - **(a) per-member alignment into Z** (rho of member-R vs the final spherical-code Gram): all 6
      map in cleanly — gemma **+0.872**, MoE +0.870, Qwen3.5-4B +0.860, Llama-3.2-3B +0.860,
      Qwen3-0.6B +0.858, **ZAYA-CCA +0.647 (the WORST-aligning, exactly as predicted** — the CCA
      outlier; but it DOES map in because z-scoring rescued it from the single-rogue-dim collapse
      plain centering would have left).
    - **(b) isotropy after whitening:** active-subspace covariance condition number **46.86 → 6.85**
      (a 6.8× isotropy gain) while geometry is retained (whitened-key Gram vs Rbar rho **+0.965**).
    - **(c) spherical-code quality (102 canonical keys):** min pairwise angle **63.85° → 89.41°**,
      mean angle 90.57°, **max-cos +0.441 → +0.010** — a near-orthogonal, near-optimal spherical code
      (maximal capacity / minimal interference), structure preserved.
    - **(d) base-neutrality (leave-one-out):** dropping any single member moves the consensus by
      LOO-rho **≥ +0.908** — NO member dominates. Most-influential = Zaya (drop→0.908, it carries
      unique outlier geometry); the 5 mainstream members each ≥ +0.9956 (near-zero individual
      influence). Genuinely base-NEUTRAL.
  - **The merge surface is now PROVEN end-to-end** on the 6 local members (GDN-hybrid / GQA / GQA /
    soft-cap / CCA / MoE). Re-whitening when cloud cards land is cheap (re-run `build_atlas.py` with
    the extra slugs in `--members`).

- **STEP 3 — MEMORY-TRAIN SMOKE on the canonical-Z hub: PIPELINE PLUMBS, RETRIEVAL EVAL FAILS (a
  clean, localized finding) (2026-06-29).** First increment of CANONICAL_BUILD_PLAN §3. Stood up the
  step-3 training harness keyed to the v1-local6 canonical-Z hub and ran a 200-step dense+s24 smoke on
  the v0 base (Qwen3.5-4B). New code: `warmstart/pk_store.py` (`ProductKeyStore` — shared product-key
  sparse store in hub space d=4096: two √N sub-codebooks SEEDED from the atlas anchor keys, N=n_sub²
  slots, top-k product addressing, error-correcting delta write per non-negotiable #1, + **multi-head
  reads** factual/positional/recency with per-head query/output projections + learned head biases) and
  `warmstart/train_mem_canonical.py` (loads `canonical_z_v1_local6.pt`; derives hub-space key=cargo /
  value=name / query=QA-cargo from the FROZEN base embeds via a learned into-hub translator; writes the
  episode, reads with the 3 heads, pools to K slots, injects through the frozen base via the **reused
  v0 GatedMemoryTap** (zero-init gamma-alone gate); LM-loss on the answer token; dense + 2:4 SR-STE arms
  via the reused `Mask24Linear`). Generalized DocBuilder to **M=8 bindings** (v0 was 3 → 2.7× capacity
  stress; chance 0.125). Log `warmstart/logs/memc-smoke.log`. **Recorded design choice:** step 3 trains
  a FRESH store (the plan's "reuse v0" = the tap/translator scaffold + the de-risked 2:4 masker, both
  reused; the v0 DeepMemory front-end is a different mechanism, not re-bound — `cam_v0_L24.pt` not loaded
  here).
  - **PLUMBING — PASS.** End-to-end the harness runs (atlas→store write/read→MAG tap→frozen base→LM
    loss): smoke-step0 prints clean (loss 15.15, gate exactly 0 = no-op at init, gamma-alone zero-init
    holds), **training loss falls 15.1 → ~4.2 (−11 nats)**, the **gate opens monotonically 0→0.006-0.007
    (no collapse)**, the **3 read heads strongly differentiate** (factual/positional/recency output norms
    diverge to ~5000/2000/2000 — the multi-head specialisation the plan wanted to observe IS present),
    fp32-compute/cast-back + NaN/grad guards clean (no trips), store write populates ~6% of 1024 slots.
  - **RETRIEVAL — FAILS the eval (the headline §6 finding).** Despite the training-loss drop, the
    held-out eval shows **memory == no_memory EXACTLY (ΔNLL +0.000 bits, 0.000 acc, both arms)** vs
    ceiling 0.979 — i.e. the **store CONTENT has zero causal effect on the output.** Diagnosis (from the
    logs): the read-head output norms **exploded to ~5000 while the gate stayed pinned at ~0.006** — the
    store lowered *training* CE via a large, store-content-**independent** read signal (a read-side
    "cognitive bypass": the head/read projections produce a near-constant huge vector regardless of the
    value bank V), NOT via actual addressing/retrieval. So memory and no_memory (empty V) inject the same
    swamping signal → identical logits. This is overfit noise on the seen batch, not generalizing recall.
    **The fix is localized and cheap for the full run** (none of it touches the atlas or the proven tap):
    (1) **normalize / constrain the read output** (RMSNorm the per-head read or clamp its norm — the
    ~5000-norm blow-up is the proximate cause); (2) **add an addressing-supervision / write-read
    consistency loss** so the store is rewarded for retrieving the *written value*, not for any
    loss-lowering injection (the CPU sanity already showed self-read cos-to-written-value = 0 before
    training — the read_q heads must be trained to align reads with write addresses); (3) raise the gate's
    role (let gamma open further / lower the read scale so the gate, not the read norm, carries signal);
    (4) **more steps** — even v0 at the easier M=3 took 120-270 steps to flip argmax, and M=8 here never
    crossed; the smoke deliberately ran only 200. 2:4 (s24) behaved identically to dense (same plumbing
    PASS, same retrieval FAIL, FIDELITY DELTA +0.000) — consistent with step-0's "2:4 ~0 extra cost",
    so the bypass is an architecture issue, not a sparsity one.
  - **§6 UNKNOWNS PINNED (the point of the smoke — real numbers on ONE gfx1201 card):**
    - **step-time ≈ 0.23 s/step dense / 0.32 s/step s24** (batch 12, M=8, tap L=24, store N=1024, d_hub
      4096). 2:4 adds ~40% here (more than step-0's ~10% — the larger d_hub=4096 Mask24Linear top-2-of-4
      is a bigger masked op than v0's d=2560 tap; still plain torch, no special kernel). 200 steps + eval
      per arm ≈ **3-4 min wall**; both arms + evals **< 10 min total** on one card.
    - **store size:** N=1024 slots (n_sub=32) = the smoke scale; trainable front-end **179 M params**
      (dominated by the 7 d_hub×d_hub fp32 projections: into-hub + 2 write + 3×2 read = ~117 M, + tap
      ~67 M). Per-episode value bank V = batch×N×d_hub×4 B = **201 MB fp32** (batch 12). Knowledge-store
      grade (10k-100k assoc) wants n_sub ≈ 100-320 → N=10k-100k; V scales linearly (a 100k-slot bank at
      batch 12 ≈ 19 GB fp32 — **won't co-fit with the 4B base on a 16 GB card**, so the full run wants
      the value bank in bf16 and/or a smaller eval batch, OR the 4×3090 cloud DP. Capacity-vs-fit is the
      real knob, exactly as §6 flagged).
    - **VRAM headroom:** peak **12.29 GB** of 16 GB at the smoke scale (4B base ~8 GB + tap + the 201 MB
      value bank + activations) → **~3.7 GB headroom** at batch 12 / N=1024. Local-feasible at smoke
      scale; the full knowledge-store-grade run is VRAM-bound on the value bank (see above).
    - **critical batch / multi-head:** per-step acc was 0.000-0.083 (no batch starvation visible, but
      also no convergence to read from) — the multi-head reads DO differentiate (distinct norms per head),
      but whether the F/P/R specialisation is *useful* can't be judged until retrieval works (the bypass
      masks it). Critical-batch remains a full-run question.
  - **Cleanup confirmed:** container `titans-memc` gone, **GPU 0 + 1 both FREE** (foreground `run --rm`
    auto-freed the lease), no orphaned container, no hang. The box was stable this run.

- **STEP 3b — READ-SIDE BYPASS FIXED → RETRIEVAL PASSES; the canonical-memory training loop is now
  END-TO-END VALIDATED on the hub (2026-06-29).** The step-3 smoke's `memory == no_memory` FAIL is
  resolved. Three changes (none touch the proven atlas; localized to the read path, the training
  objective, and the eval harness):
  1. **Read normalization (`pk_store.py`):** added an `RMSNorm` class; each read head now RMSNorm's its
     retrieved value-mix BEFORE `read_o` (`self.read_norm[h]`), and a final `self.read_out_norm`
     RMSNorm's the summed multi-head read before it becomes the MAG bank. The ~5000-norm read blow-up
     that swamped the gate is gone — the store-content DIRECTION drives the injection, the gate (gamma)
     carries the magnitude. (Per-head diagnostic `head_norms` are reported PRE-final-norm so they still
     look large ~1000-3000; the actual bank is bounded ≈√d_hub.)
  2. **Write→read addressing-supervision InfoNCE (`train_mem_canonical.py` `run_step`, `--addr-weight`
     default 1.0):** the store now exposes `write_addr_val()` (→ write address wk = to_wkey(key),
     stored value wv = to_wval(value)) and `head_query()`. The queried binding is found by matching the
     QA cargo token id (at `qa_start`) to the binding cargo ids; two differentiable CE terms force
     (a) `read_q[0](query)` close to the queried binding's write-address wk and (b) the factual head's
     retrieved ctx close to its STORED value wv — far from the other M−1. This closes the addressing
     loop the LM-loss-only bypass shortcut (addr loss drops 8.0→~0.2-0.6; on CPU sanity self-read
     cos-to-written-value was 0 before training).
  3. **THE ROOT-CAUSE EVAL BUG (`evaluate()`):** `train_arm` removes the MAG forward hook before
     returning, and `evaluate()` re-set the bank but **never re-attached a hook** → the tap was a SILENT
     NO-OP at eval, so `set_bank(written)` and `set_bank(empty)` produced BIT-IDENTICAL logits
     (memory==no_memory EXACTLY, ΔNLL +0.000). A targeted diagnostic isolated it (bank_mem norm 513 vs
     bank_emp 0, yet `||logit_mem−logit_emp||_inf = 0.00000`). Fix: `evaluate()` now `attach_tap(...)`
     for the whole eval (the tap self-no-ops on the ceiling pass because its bank is None) and removes
     the hook at the end. WITH the hook: logit Δ jumps to 16.2, 11/12 argmax flips on a held-out batch.
     (The prior increment's read-bypass diagnosis was real and worth fixing — but this missing-hook
     no-op was ALSO masking the eval the whole time; both had to be fixed.)
  - **Result (3000 steps/arm, lr 1e-3, batch 12, M=8 chance 0.125, eval n=512) — BOTH ARMS PASS:**
    | arm | memory | no_memory | ceiling | ΔNLL | s/step | verdict |
    |-----|--------|-----------|---------|------|--------|---------|
    | dense | **0.621** | 0.000 | 0.988 | **+19.17** | 0.230 | **PASS** |
    | **s24 (2:4-by-design)** | **0.678** | 0.000 | 0.988 | **+19.10** | 0.331 | **PASS** |
    memory ≫ no_memory (pinned 0.000) + 0.15 and > 0.5 on HELD-OUT associations; **FIDELITY DELTA
    (s24 − dense) = +0.057** (2:4 HOLDS, even converges faster — train acc 0.83-0.92 by step ~900-1200
    vs dense's slower climb, consistent with step-0's SR-STE regularizing-decay finding). Gate opened
    monotonically 0 → ~0.017, no NaN/grad-guard trips; gamma-alone zero-init held (smoke-step0 gate
    exactly 0, loss 15.15). **⇒ the store CONTENT now causally drives the recall; the read-side bypass
    is closed and the canonical-Z PKM store + multi-head + MAG-tap loop is END-TO-END VALIDATED.**
  - **§6 cost (unchanged from the smoke):** 0.23 s/step dense / 0.33 s/step s24 (batch 12, M=8, N=1024,
    d_hub 4096); **12.29 GB VRAM peak** → ~3.7 GB headroom; ~12 min/arm on ONE gfx1201 card. The toy
    M=8 task takes ~2200-2600 steps for held-out acc to cross 0.5 (the gate is the slow lever; addr
    loss converges much earlier).
  - **Code:** `warmstart/pk_store.py` (RMSNorm + read_norm/read_out_norm + write_addr_val/head_query),
    `warmstart/train_mem_canonical.py` (addr InfoNCE in run_step, `--addr-weight`, eval-hook fix).
    Logs `warmstart/logs/memc-3b-{dense,s24}.log`. **Cleanup confirmed:** containers gone, **GPU 0 + 1
    both FREE**, no orphan, no hang.

## THE CURRENT NEXT STEP
**DECISION MADE (2026-06-29) → the v1 canonical-memory build is LOCKED. See
[`titans/CANONICAL_BUILD_PLAN.md`](CANONICAL_BUILD_PLAN.md).** The owner chose to build the *real*
"perfect memory module" now (executing dial #1 canonical-Z + dial #2 product-key + a 2:4-by-design
SWMMAC serve path, fused). Reuse v0 (`ckpt/cam_v0_L24.pt`) + the translator scaffold; do NOT re-bind.

Headline locked decisions (full rationale in the build plan):
- **Hub:** committee-built canonical-Z, **d=4096** (RDNA4 WMMA/W4A8/SWMMAC-aligned; near-identity for
  7–8B spokes). 11-member committee spanning 8 attention geometries (GDN/GQA/sinks/MLA/soft-cap/
  conv-hybrid/ternary/CCA), skewed small (Titans helps low-capacity models most). Zaya in under an
  owner exception, probed locally (cca_hip is gfx1201-only). Llama-3.3-70B = weak-serving hub ceiling.
- **Memory:** shared product-key sparse store + multi-head reads (factual/positional/recency),
  MAG-tap injection. Product factors as **translator(model) × head(purpose)**.
- **Serve:** 2:4-by-design weights (SR-STE from init) → SWMMAC on RDNA4; dense reference = fidelity
  control. Owner assumption: SWMMAC = dense parity + sparsity gain.
- **Task:** meta-learned leak-free associative recall (generalized DocBuilder), content-diverse,
  capacity-stressed, position-aware. Priority A>C>B (B's consolidation = v2). Target =
  knowledge-store-grade (~10k–100k associations, passage-length).
- **Build order:** 0) de-risk 2:4-by-design vs dense on the v0 harness → 1–2) committee probe →
  canonical-Z atlas → 3) memory training (dense + 2:4 arms) → 4) translator fits.
- **Hardware:** train on 4× RTX 3090 community ($0.88/hr, DP); 70B probe on 4× A40 ($1.76); other
  probes on cheap NVIDIA; Zaya probe local gfx1201. Budget ~$20–50, cap ~$100.

**Immediate next action:** ~~step 0~~ DONE; ~~step 1–2 (a)…(d) probe~~ DONE; **~~step 1–2 (ATLAS) bootstrap
the FIRST d=4096 canonical-Z atlas from the 6 local cards~~ DONE (2026-06-29, see DONE section):
`ckpt/atlas/canonical_z_v1_local6.pt` exists** — Z[102,4096] near-orthogonal spherical-code keys
(min-angle 89.4°, max-cos +0.010), isotropy condition 46.9→6.85, all 6 members map in (worst = Zaya
CCA +0.647), base-neutral (LOO-rho ≥ +0.908), z-scored relrep uniform, builder `build_atlas.py`. The
pipeline is now PROVEN end-to-end on the local merge surface. ~~step 3 memory-train SMOKE~~ DONE
(2026-06-29, see DONE): plumbing PASSES + §6 cost pinned, but retrieval FAILED. ~~step 3b read-fix +
reconverge~~ **DONE (2026-06-29, see DONE): RETRIEVAL NOW PASSES** — dense memory 0.621 / no_memory
0.000 / ΔNLL +19.17, s24 0.678 / 0.000 / +19.10, both > the v0 bar, 2:4 fidelity delta +0.057.
**The canonical-memory training loop is END-TO-END VALIDATED on the hub.** Next = the full
knowledge-store-grade run (3c).

**RECOMMENDED NEXT = (3c) THE FULL KNOWLEDGE-STORE-GRADE STEP-3 RUN.** The smoke-scale loop is now
proven end-to-end (atlas → PKM store write/read with LEARNED addressing → MAG tap → frozen base → held-
out recall PASS, dense + 2:4). Scale it to knowledge-store grade: **n_sub ≈ 100-320 → N = 10k-100k
slots**, passage-length values, more bindings/episode (M ≫ 8), a content-diverse capacity-stressed task.
Keep the 3b fixes verbatim (RMSNorm reads + `--addr-weight` addressing InfoNCE + the eval hook; gamma-
alone zero-init; fp32-compute/cast-back). **Local-feasibility (per the smoke's pinned §6 VRAM):** a
100k-slot value bank @ batch12 ≈ 19 GB fp32 — **WON'T co-fit with the 4B base on a 16 GB gfx1201 card.**
So either (a) **stay local** with the value bank in **bf16** (≈9.5 GB) + a smaller eval batch + a
moderate N (the 12.29 GB peak at N=1024 leaves ~3.7 GB, so bf16 + N≈10-20k is plausibly local; the
exact ceiling needs a quick VRAM probe), or (b) **4×3090 cloud DP** ($0.88/hr, §4; ~$4-20) for the full
100k-slot / large-batch run — the value bank shards over DP ranks. **This is the first step that may
need real cloud spend — surface to the owner IF the chosen N/batch genuinely won't fit local in bf16.**
The bf16-bank-vs-cloud-DP fit question is the one open §6 knob; everything else (addressing, gate,
2:4 fidelity, step-time, tap) is now de-risked at smoke scale. NOTE: step-3 training is pure HF/torch in
THIS worktree but needs working GPU torch — the `titans:dev` image's torch runs on gfx1201 via the GPU
lease (the CPU-only atlas build dodged this; training cannot).

**ALTERNATIVE / parallelizable = (2) CLOUD FAN-OUT to enrich the atlas.** The truly un-cached/un-AMD-able
roster — Ministral-8B, BitNet-2B4T, LFM2-1.2B, GPT-OSS-20b, Llama-3.3-70B — PLUS the demoted
**Laguna-XS.2** (local SIGKILL) and **DeepSeek-V2-Lite-AWQ** (autoawq/transformers skew). Forward-only on
the cheap NVIDIA box (sharded A40 for the 70B), SAME `anchor_bank.py` (carry it over, ASSERT sha ==
28a4acf8), emit the SAME card dict (model,n_layers,tap_layer,d_base,n_anchor,anchor_sha,categories,
tok_counts,A,R,R_raw — the atlas recomputes z-scored R from `A`, so cloud cards need only `A`). Budget
~$5–15 (§4). Each card drops into `ckpt/probe/` and the atlas re-builds in seconds via
`build_atlas.py --members <…all slugs…>`. **This is the bigger lift and is ORTHOGONAL to step 3** — it
can run on the cloud box in parallel while step-3 training proceeds locally; the hub re-whiten when the
cloud cards land is cheap. Recommend running it as a parallel track, not a blocker on step 3.

The v2-dial MENU below is retained for reference (dials #1/#2 are now folded into the build; #4/#5 and
the consolidation tier remain v2).

### v2 MENU (CAM_DESIGN §6 dials — each: what-it-buys / rough cost-risk)
1. **Whitened spherical-code canonical-Z committee** — build Z by a committee relative-rep atlas,
   whiten to isotropy, shape stored keys to a near-optimal spherical code. *Buys:* much higher
   storage capacity + a base-neutral hub that also lets the translator be near-affine again (likely
   fixes Gemma's PARTIAL at the source). *Cost/risk:* HIGH — new canonical-encoder + committee build;
   re-pretrains the memory front-end (the most invasive dial).
2. **Multi-tier store (episodic + PKM bank + consolidation)** — add a fast episodic tier, a paged
   product-key memory bank, and a slow consolidation path. *Buys:* effectively unbounded context /
   long-horizon recall. *Cost/risk:* HIGH — new storage hierarchy + read/write routing; orthogonal to
   the translator question.
3. **GDN/KDA per-channel gate + OSDN 2nd-order write** — replace the scalar write with a per-channel
   (Kimi-Delta) gate and an OSDN diagonal-preconditioned 2nd-order, chunk-parallel write. *Buys:*
   sharper, more selective writes (better signal-to-noise in the store). *Cost/risk:* MEDIUM —
   bounded kernel/write change on the proven core; mind the fp16/fp32 LDS budget (gdn-wmma finding).
4. **Energy + cycle-consistency confidence / abstain** — a retrieval-energy + cycle-consistency
   score that lets the memory abstain instead of hallucinating recall. *Buys:* calibrated reads, no
   confident-wrong recall (productionization safety). *Cost/risk:* MEDIUM — a read-side scorer + gate
   threshold; non-invasive to the store.
5. **Meta-learned / neuromodulatory write rule** — meta-learn the write/forget rule (surprise- and
   context-modulated) instead of the fixed Omega-window rule. *Buys:* highest ceiling — the store
   learns *what* to remember. *Cost/risk:* HIGHEST — meta-training loop, hardest to stabilize; do last.

**Recommended front-runner given the Gemma signal:** **dial #1 (whitened canonical-Z committee)** —
it directly attacks the translator headroom the Gemma PARTIAL exposed (a better-conditioned hub makes
the affine translator sufficient again) and is the §6 capacity lever. If a cheaper first step is
preferred, **dial #3 (GDN/KDA + OSDN)** is the lowest-risk improvement on the proven core.

### Cheap optional polish (NOT load-bearing, not a v2 dial)
- Same-family longer-fit headline + card for Qwen3-0.6B: `recall_v1.py --base2 Qwen/Qwen3-0.6B
  --steps 3000 --lr 5e-4 --save-translator ckpt/translator_qwen3_0p6b.pt` (current headline is the
  200-step 0.604; cross-family already settled the science).
- A wider/nonlinear (MLP) translator on Gemma to test whether 0.488 clears 0.5 with translator
  capacity alone — diagnostic for whether the Gemma gap is translator-class (affine) or geometry
  (which dial #1 fixes at the source).

Operating rules unchanged: every GPU job via the absolute `gpu-lease.sh -n 1`, let it block,
fail-fast Monitor (the box was throwing GPU hangs 2026-06-29 — kill+release, do NOT loop), stay in
THIS worktree, pure HF/torch (no minisglang / no `.so`). Smoke cheap first.

**Reminders for the next builder:** (a) the v0 memory is a fixed 78 MB checkpoint
(`ckpt/cam_v0_L24.pt`) — reload it, never re-bind. (b) Any new trainable gate gets gamma-ALONE
zero-init (NOT also-zero the out-proj → dead start) + the NaN/grad skip guard. (c) The dtype pattern
(compute fp32, cast the additive update back to base dtype) is in both `gated_tap.py` and
`translator.py` — keep it for any new base/tap. (d) **VRAM on the 16GB card is tight for a large
base-2:** you MUST free base-1 before loading base-2 — `injector_tap.layers=None; del injector_tap`
(MAGInjector pins the whole base-1 via `decoder_layers`), `del embed_weight`, drop `adapter.unembed`.
The base-1-free reorder in `recall_v1.py` already does this; reuse it for any 3B+ base-2.
`expandable_segments` is unsupported on this ROCm allocator. (e) Translator cards now exist:
`ckpt/translator_llama32_3b.pt` (Llama-3.2-3B, cross-family). Pass `--base2 <model>` + `--save-translator`.

## Roadmap after V0
- **v1 — DONE (2026-06-29):** froze the v0 memory (78 MB checkpoint), added 2nd base Qwen3-0.6B
  (d=1024≠2560), fit a 5.2M-param affine translator base2↔tap-space; **one memory serves two bases**
  (base-2 memory 0.604 / no_memory 0.000 / ceiling 0.961). See DONE section.
- **v1 CROSS-FAMILY — DONE (2026-06-29):** repeated the falsifier on `unsloth/Llama-3.2-3B` (Llama
  tiktoken vocab + LlamaForCausalLM, d=3072) — memory **0.602** / no_memory 0.010 / ceiling 0.920,
  3000-step fit, translator card `ckpt/translator_llama32_3b.pt`.
- **v1 CROSS-FAMILY 2nd non-Qwen — DONE (2026-06-29):** THIRD base `unsloth/gemma-3-4b-pt` (Gemma
  vocab+arch, d=2560, 34 layers) — memory **0.488** / no_memory 0.000 / ceiling 0.998, ΔNLL +48.3,
  3000-step fit, card `ckpt/translator_gemma3_4b.pt`. PARTIAL (just under 0.5; transfer genuine).
  **Vocab-family leakage RULED OUT across 3 bases / 2 non-Qwen families. Summary: `titans/RESULTS.md`.**
- **v2** (perfection dials): whitened spherical-code canonical Z (committee-built), multi-tier store
  (episodic+PKM+consolidation), GDN per-channel gate + OSDN 2nd-order write, energy+cycle
  confidence/abstain, meta-learned write rule. See CAM_DESIGN §6.

## Operating rules (MANDATORY — do not skip)
- **GPU:** every GPU/torch job via the absolute arbiter
  `/home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh -n 1 -- <cmd>`. Let it block. `-n 1` = one card.
- **Monitor fail-fast:** a crash-looping run must NOT blind-retry — it holds the shared 2-card lease.
  Check the run log; on repeated crash, kill, release lease, diagnose. (See memory
  `monitor-agent-gpu-containers`.)
- **Source isolation:** edits + the run stay in THIS worktree for the whole job (training reads source
  lazily). This is a research/training experiment — pure HF/torch, **no minisglang, no vendored .so**.
- **Train here, serve there:** the memory *training/research* stays in this worktree; only the eventual
  *serving* primitive (the gated residual tap) lands in minisglang later. Don't port the training loop
  into the inference engine.

## Loop discipline — DELEGATE every iteration to a sub-agent (keep the orchestrator lean)
The main loop is a THIN orchestrator. It must NOT read files / write code / run bash / do GPU work
itself — every increment runs inside a spawned **background Agent** so the orchestrator context never
balloons. The durable state is THIS file, which each sub-agent rewrites.
- **Each orchestrator turn:** (1) `TaskList` — if a loop sub-agent (label `cam-loop-*`) is still
  running, just reschedule a ~1800s fallback and end (NO double-spawn); (2) else spawn ONE background
  Agent for the next increment; (3) schedule a ~1800s fallback wakeup; end the turn. On sub-agent
  completion the harness re-invokes the orchestrator with the agent's summary → spawn the next one.
- **Sub-agent contract:** read this CONTINUANCE.md (+ V0_SPEC.md / CAM_DESIGN.md as needed); do
  EXACTLY ONE next increment; follow ALL operating rules above; UPDATE this file's "DONE" + "CURRENT
  NEXT STEP" sections; return a ≤6-line summary (what it did, key numbers, the new next step). Its
  verbose context is discarded — only the summary returns to the orchestrator.
- If a sub-agent reports it is blocked on a real user decision, the orchestrator STOPS the loop
  (omits the wakeup) and surfaces it rather than guessing.
