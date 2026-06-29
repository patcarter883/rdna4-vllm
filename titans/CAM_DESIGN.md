# CAM — Canonical Associative Memory (north-star design + implementability triage)

Status: design (2026-06-28). Supersedes the inputs-embeds brief's injection assumption.
Context: bolt-on Titans memory in front of a frozen base. boltA proved the input-embeds
**Memory-as-Context** prefix does NOT deliver through a frozen base (memory ≈ no_memory).
The pivot is **Memory-as-Gate (MAG)** + a base-agnostic memory accessed via a learned
embedding-space translator (RecursiveMAS-style). This doc designs the maximal version, then
triages it down to a real v0.

---

## 0. The one commitment everything hangs off

**The memory lives in a canonical latent space `Z` and is NEVER trained in any base model's
space.** The base couples in only through a small in/out translator. That single commitment is
what turns "train your own memory model" into "download the memory, fit a tiny translator" —
the product thesis (a hot-swappable Modular Memory Organ).

Everything is specified in the **Miras / test-time-regression coordinates**
`⟨memory architecture, write objective, retention gate, optimizer⟩`
(Behrouz "It's All Connected" 2504.13173; Wang/Shi/Fox "Test-Time Regression" 2501.12352).
Six independent research lanes converged on this frame — the *write mechanism* is effectively
solved; the frontier is **storage geometry**, **multi-tier organization**, and
**base-agnosticism via a canonical space**. CAM is the assembly of all three, which nobody has
yet built together.

## 1. The three entities (corrected data flow)

```
                  ┌───► [ base attention / MoE / GDN stream ] ─────────────┐
   [ base h_t ]   │                                                        ▼
  (d_base) ───────┤                                              h_final = h_t + g ⊙ y_base
                  │                                                        ▲
                  └─► A_i: d_base→Z ─► [ Titans core in Z ] ─► B_i: Z→d_base┘
                       (translator)     (frozen meta-params,    (translator)
                                         live test-time state)
```

- **Base (frozen):** Qwen3.5-4B / Zaya-8B / etc. Hidden dim `d_base`. Untouched weights.
- **Titans core (meta-params frozen; STATE always live):** operates in canonical `Z` (`d_mem`).
  Its trained projections/gates are frozen *after* canonical pretraining; its fast-weights /
  MLP-state still update online at inference — that is the whole point of test-time memory.
  "Frozen core" ≠ "static memory."
- **Translator (trainable, tiny):** `A_i: d_base→Z`, `B_i: Z→d_base`. Affine-first.

**Gate is ADDITIVE + ZERO-INIT, not a convex blend.** `h_final = h_t + g ⊙ y_base`, with `g`
biased to ≈0 at init so the memory starts as an exact no-op and cannot corrupt the frozen base
(2603.16413). Gemini's `g·y_base + (1−g)·h_t` blend is the wrong topology (displaces the base
stream, can't zero-init cleanly). The gate `g` is data-dependent (a function of `[h_t; y_base]`).

## 2. Component spec (the "11")

### 2.1 Canonical space Z (the hub)
- Build `Z` from a **committee** via a relative-rep mean-atlas (RLSA 2311.06547), so Z is no
  single model's home turf → shorter, more faithful average translators. Banks on the Platonic
  convergence prior (2405.07987); respects the partial-convergence caveat (2602.14486).
- **Whiten Z to isotropy; shape stored keys to a near-optimal spherical code.** Capacity ⇔
  optimal hypersphere packing (Hu/Wu/Liu NeurIPS 2024, 2410.23126) → capacity becomes a
  trainable geometric objective on the key map (U-Hop separation loss 2404.03827). Translators
  absorb each base's anisotropy; Z stays clean for retrieval.

### 2.2 Translators (in/out adapters)
- **vec2vec topology `A_i→Z→B_i`** (2505.12540) is literally our architecture, already validated.
- We control both sides ⇒ **paired data for free** (same text through base + canonical encoder)
  ⇒ few-shot supervised alignment (ALGEN 2502.11308). Map is **largely linear** (mini-vec2vec
  2510.02348) ⇒ start **affine**; affine residual-stream stitching transfers *function* across
  LLMs (2506.06609). Add nonlinearity only where a base resists linear fit.
- **Preserve pairwise geometry (VSP loss)** over the memory's stored keys — retrieval ranking
  must survive the round trip.

### 2.3 Memory substrate — a continuum of tiers (Nested-Learning CMS / HOPE 2512.24695)
1. **Working tier — gated-delta fast-weight state.**
   `S_t = S_{t-1}(diag(α) − β k kᵀ) + β v kᵀ`. Per-channel decay (KDA 2510.26692), error-
   correcting delta (additive collapses), reflections allowed (negative eigenvalues) to beat the
   TC⁰ state-tracking ceiling (2508.07395).
2. **Episodic tier — deep-MLP, addressable, deletable.** Superlinear capacity/byte (Titans
   2501.00663); **finite-support (Epanechnikov/LSR) read energy** (2506.10801) not softmax (kills
   crosstalk); sparse retrieval (Hopfield-Fenchel-Young 2411.08590); **Larimar one-shot write +
   selective delete** (2403.11901) — compressive memories cannot forget one fact.
3. **Semantic tier — product-key million-slot bank + recursive-summary consolidation.**
   PKM / Memory-Layers (2412.09764): millions of slots, √N lookup, ~0 added FLOPs; periodic
   "sleep" consolidation compresses old episodes (RAPTOR-style).

### 2.4 Write path
- **Omega rule (window, not last-token)** — optimize memory over a sliding window of recent
  tokens (Atlas 2505.23735); biggest quality lever for online writing (+80% @ 10M ctx).
- **Second-order write that stays chunk-parallel** — OSDN diagonal preconditioner (best
  cost/benefit, +32–39% recall, no extra state); Muon / MesaNet-CG heavier alternatives.
- **Surprise-gated, momentum-smoothed, robust** — write only on high-surprise tokens (sparse,
  less saturation); momentum on surprise (Titans); robust write loss (Huber/Yaad, Moneta) so one
  OOD token from the translator can't corrupt a slot.

### 2.5 Read path + MAG attach
- **MAG injection, zero-init** (§1). Decoupled side-memory + gate (LONGMEM, G-MemLLM, TPTT);
  read/inject projection zero-initialized (2603.16413).
- **The memory knows when it doesn't know.** Two near-free confidence signals gate read-back:
  retrieval **energy at convergence** (Rectified Lagrangian 2502.14003) and translator
  **cycle-consistency** (h→Z→h reconstruction error). Below threshold ⇒ gate stays shut, memory
  abstains instead of hallucinating recall.

### 2.6 Meta-learning (highest ceiling)
- Don't hand-design surprise/momentum/decay — **meta-train the inner write objective + optimizer
  end-to-end through the frozen base's LM loss, across SEVERAL bases at once** so agnosticism is
  trained-in. Add a **neuromodulatory plasticity gate** (Backpropamine 2002.10585): a small net
  emits per-slot, per-step write/decay modulation. Ceiling: self-modifying write rule (HOPE).

## 3. Non-negotiables (multi-lane failure modes)
1. Delta (error-correcting) write, never additive — additive FWP saturates once #keys > dim.
2. Real delete primitive + learned per-slot forget gate — no-forget states degrade
   ("Stuffed Mamba" 2410.07145); compressive-only can't do exact recall *or* targeted forgetting.
3. Chunked-parallel must be parity-checked vs sequential — chunking is itself the accuracy killer
   (TNT 2511.07343; "Titans Revisited" 2510.09551). (Your repo's parity-gate discipline applies.)
4. Translation is lossy and a store/retrieve cycle pays it twice (~0.92 cosine ceiling). Monitor
   round-trip error; never chain translations through Z; acceptance test is **functional** (recall
   improves base output), not cosine, and not naive CKA (use unbiased CKA — biased CKA lies in the
   high-dim/low-sample regime, 2210.16156).
5. Capacity↔generalization is NOT a real tradeoff (Epanechnikov-LSR, saddle-hierarchy) — don't
   sacrifice exact recall for "creative" retrieval; pick an energy that gives both.

## 4. RDNA4 / serving engineering notes
- **Fixed `d_mem` ⇒ compile the memory-update HIP kernel once; hardcode LDS allocation.** The
  translator handles every base's variable `d_base` before data reaches the custom kernel — one
  kernel geometry across the whole model zoo. (Gemini's good catch.)
- **Do NOT W4A8 the online update.** Memory *weights* may quantize, but test-time state + surprise
  gradients need fp16/fp32 (consistent with the GDN LDS-budget finding). Quantize the core's frozen
  projections at most.
- Serving primitive = a **gated residual tap**: run translator→memory retrieve, then inject
  `h_t + g·y_base` at chosen layer(s) during the base forward. More invasive than the inputs-embeds
  brief; bounded and graph-capturable.

### 4.1 The write-rule decision: Gated DeltaNet, not "GLA vs delta"
There is no real fork between "parallel GLA" and "sequential delta rule." **Gated DeltaNet (GDN)**
is chunkwise-parallel via the WY/UT transform AND error-correcting — `S_t = S_{t-1}(diag(α) − β kkᵀ)
+ β v kᵀ`. It dominates both plain GLA (which lacks error-correction → muddy averaged recall, fails
needle) and the naive sequential delta rule (which the chunkwise form already parallelizes). **This
repo already ships `gdn_hip` kernels** — the working tier IS GDN, warm-started from a pretrained
GDN-family donor (see §5 round-2). Plain Hebbian/GLA and Modern Hopfield are alternatives only if a
specific tier needs them (Hopfield for an exact, deletable episodic tier — §2.3 tier 2).

### 4.2 Serving realities the LDS "magic number" ignores
- **Per-sequence state breaks batched GEMM.** Every request carries its own `M_t`, so a 32-way batch
  is 32 distinct evolving state matrices, not one shared weight × 32 — the "continuous-batching
  collapse." This is the dominant production cost and the reason the state must be pinned on-chip and
  the kernel graph-captured at fixed batch shapes (this repo's existing constraint).
- **The 64KB-LDS "sweet spot" is actually the overflow cliff.** A 256×256 FP8 state = exactly 64KB =
  the *entire* RDNA4 LDS budget, leaving **zero** room for compute tiles, double-buffers, or q/k/v —
  the exact wall the `gdn-wmma-lds-budget` work already hit (fp32 128×128 filled the budget → forced
  fp16 state + manual arena reuse). Dimension the state to leave headroom, and **keep state in
  fp16/fp32**, not FP8 — FP8 *state* sacrifices the recall fidelity the memory exists to provide.
  Choose `d_mem`/head-dim from capacity+fidelity needs and the *real* LDS budget, not by pinning to a
  donor model's hidden size.

## 5. Gemini assessment (for the record)
### Round 1
- Right: universal-port framing; compile-kernel-once + hardcoded LDS; memory-as-inheritable-asset
  for SPINE swap; both "catches" (lossy translation, gate calibration).
- Fix: gate must be additive zero-init not convex blend; it chose hidden-keyed coupling and missed
  the canonical-space *geometry* (whitening + spherical code); "predict next base layer" is only an
  auxiliary warm-start — the real objective is functional (LM-loss-through-base).
- Drop: W4A8 on the memory-update kernel.

### Round 2
- **Keep:** the 5 delta-rule drawbacks are mostly correct (esp. continuous-batching/WMMA collapse →
  §4.2); the "database vs brain" reframing is right (the core stores + gates *what/when to write/
  forget*; the base does the thinking — so it need not be "smart"); multi-head decay profiles
  (scratchpad/semantic/bedrock) and **decay-by-utility-density** (retention ∝ retrieval frequency,
  not token-distance) are good instantiations of the continuum tiers; the **UMX ecosystem vision**
  (fixed core + disposable per-model translator cards + open registry) is the right productization
  (→ §5.5), and its two integration gaps (a layer-interception convention + quantization-agnostic
  translator front-end) are real.
- **Fix:** the GLA-vs-delta dichotomy is false → use GDN (§4.1); the delta "sequential bottleneck"
  is overstated (chunkwise-parallel forms solve prefill; the real residual is TNT chunk-boundary
  discontinuity); PRM/GRPO is the wrong *primary* delivery objective (LM-loss-through-base is the
  dense free signal; PRM is a v2 refinement for specific behaviors like RSA-trace hallucination
  suppression); Phase-1 alignment loss as written can drive the gate→0 ("cognitive bypass") — put
  the reconstruction loss on the *translator round-trip with NULL memory*, not on matching the
  no-memory activation with the gate open.
- **Drop (won't work as described):** **"organ harvesting"** — rip out one GLA/recurrent layer of a
  donor and treat it as a perfect standalone frozen memory. A single linear-attention layer is NOT a
  standalone associative memory; its I/O is only meaningful *inside* its native stack, so harvesting
  picks the **most-coupled, least-canonical** space possible — the exact opposite of the whitened
  canonical Z this design needs. "It knows how to do prefix scans" confuses weights with kernels (the
  scan lives in the runtime, not the tensor); RecurrentGemma's per-head state geometry ≠ a 4096 d×d
  fast-weight matrix; and "can't crop a brain" correctly implies you're locked to donors of exactly
  `d_mem` — an arbitrary, severe constraint.
  - **Salvage:** the *good* version of harvesting is **warm-starting** the memory's projections/gates
    from a pretrained GDN-family model (already in the Phase-1 plan), then training + validating it as
    a memory — NOT "extract, freeze, zero further work." Warm-start is free upside; a lone frozen
    layer is not a memory organ.

## 5.5 UMX — productization north star (post-v1)
The "Titans for everyone" framing is the right end-state: a small set of **frozen golden-standard
memory cores** (fixed geometry, AOT-compiled `.so`, no Triton/JIT) + a registry of cheap **per-model
translator cards** (~tens of MB, hours to fit on commodity GPU). Any base downloads its card and
gains the shared memory. Real standardization gaps to design for: (1) a **layer-interception
convention** (a mid-stack tap after the FFN/MoE residual is a sane default, but "geometric midpoint"
is a heuristic, not settled — make the tap layer part of the card metadata); (2) a
**quantization-agnostic translator front-end** that normalizes incoming activations (FP8/INT4/bf16,
arbitrary micro-scales) to a standard precision before the core math. This is a v1+ direction, gated
on the v0 MAG result and the v1 second-base translator working at all — do NOT standardize a core
geometry before the core write-rule + gate are validated.

---

## 6. Perfection → implementable triage

Rank by ceiling (how much it matters) vs cost/risk. Tiers = v0 (prove the bet) / v1 (the product) /
v2 (perfection upgrades).

| Component | Ceiling | Cost/Risk | Tier |
|---|---|---|---|
| **Additive zero-init MAG gate** | decisive (kills/confirms thesis) | low | **v0** |
| Existing DeepMemory (stages 1–3, GPU-validated) | high (already built) | none | **v0** |
| Omega-rule window write | high | low (small change to write) | v0/v1 |
| LM-loss-through-frozen-base objective | decisive | low | **v0** |
| Direct-pretrain memory binding (realemb, done) | high | done | **v0** |
| Affine translator + 2nd base | high (proves agnosticism) | medium | **v1** |
| Per-channel (KDA) gate, OSDN 2nd-order write | medium | medium | v1/v2 |
| Whitened spherical-code canonical Z + committee | high (capacity) | high | **v2** |
| Multi-tier (episodic + PKM bank + consolidation) | high (unbounded ctx) | high | **v2** |
| Energy + cycle-consistency confidence/abstain | medium-high | medium | v2 |
| Meta-learned / neuromodulatory write rule | highest | highest | v2 |

### v0 — the sharpest possible test of the core bet
The bet reduces to ONE question boltA left open: **does additive zero-init MAG deliver the
already-validated DeepMemory binding through the frozen base, where the MAC prefix failed?**
- Base: frozen Qwen3.5-4B (already serving).
- Memory: the existing `DeepMemory` (stages 1–3, GPU-validated; realemb binding 0.94 — already
  trained against Qwen's space, so **no translator needed at v0** — Z = Qwen's space, A/B = identity).
- New code: a single mid-stack **gated residual tap**, `g` zero-init; optional Omega window.
- Train: freeze base + memory; train ONLY the gate on LM-loss over long sequences (two-stage:
  binding pretrain is done, this is the delivery stage).
- Pass/fail: long-context recall (BABILong / needle, multi-fact) with memory beats frozen-base-alone,
  AND no short-context regression. If MAG clears where MAC died, the whole thesis is alive.
- Cost: ~days, one card. This is the cheapest experiment that can falsify the entire program.

### v1 — prove agnosticism (the actual product)
- Add a **second base** with different `d_base` (Zaya-8B, or a different family).
- Freeze the v0 memory; fit an **affine/few-shot translator** base2→Z (paired data, near-trivial).
- Pass/fail: one frozen memory serves two bases via two cheap translators with recall lift on both.
  This is "Modular Memory Organ / hot-swap" demonstrated.

### v2 — perfection upgrades, added only once v0+v1 hold
Whitened spherical-code Z (built from a committee), multi-tier store, OSDN/KDA write upgrades,
energy+cycle confidence, meta-learned write rule. Each is independently ablatable on top of a
working core.

### De-risking order (cheapest-falsifier-first)
1. **MAG gate** (v0) — one tap, zero-init, gate-only training. Confirms/kills the thesis.
2. **Translator + 2nd base** (v1) — proves the canonical-memory product.
3. **Geometry + tiers + meta-rules** (v2) — turn the dials to 11 on a proven core.

### Training-loop decision (answering the Gemini question)
Neither pure next-layer-prediction nor a process-reward model.
- **Primary objective: next-token LM loss through the frozen base** (no teacher needed; the frozen
  base + sequence provide the signal). This is the only objective that is *functional* — it rewards
  memory the base can actually use.
- **Two-stage:** (1) direct-pretrain memory binding with a strong supervised loss [realemb — DONE];
  (2) freeze memory, train gate (+translator) through the frozen base on LM loss.
- **Auxiliaries only:** next-layer-prediction / cycle-consistency as translator warm-start +
  fidelity monitor; VSP to preserve key geometry. PRM/task-success is high-variance and unnecessary
  when LM loss is a dense free signal.

---

## 7. Curated bibliography (by lane)
- Test-time memory / Titans lineage: Titans 2501.00663 · TTT 2407.04620 · Miras 2504.13173 ·
  Atlas 2505.23735 · Nested Learning/HOPE 2512.24695 · TNT 2511.07343 · Titans Revisited 2510.09551 ·
  G-MemLLM 2602.00015 · LONGMEM 2306.07174 · Latent Context Compilation 2602.21221.
- Associative memory: spherical codes 2410.23126 · U-Hop 2404.03827 · Epanechnikov/LSR 2506.10801 ·
  Hopfield-Fenchel-Young 2411.08590 · simplicial 2305.05179 · UHN 2202.04557 · Energy Transformer
  (NeurIPS 2023) · feature-correlation capacity 2508.01395 · Rectified Lagrangian 2502.14003.
- Linear-attn / delta-rule: DeltaNet 2406.06484 · Gated DeltaNet 2412.06464 · RWKV-7 2503.14456 ·
  DeltaProduct 2502.10297 · MesaNet 2506.05233 · Kimi Linear/KDA 2510.26692 · OSDN 2605.13473 ·
  Preconditioned DeltaNet 2604.21100 · Test-Time Regression 2501.12352 · TPTT 2506.17671 ·
  Mamba-2/SSD 2405.21060 · GLA 2312.06635 · TC0 ceiling 2508.07395.
- Fast-weights / meta-plasticity: Linear-Transformers-are-FWP 2102.11174 · Differentiable Plasticity
  1804.02464 · Backpropamine 2002.10585 · meta-plasticity/random-feedback 2210.16414 · three-factor
  2512.09366 · mesa-optimization 2212.07677 · Trained Persistent Memory for Frozen LLMs 2603.16413 ·
  HeLa-Mem 2604.16839 · Stuffed Mamba 2410.07145.
- Hierarchy / compression / episodic: ARMT 2407.04841 · Infini-attention 2404.07143 · Larimar
  2403.11901 · Product-Key Memory 1907.05242 · Memory Layers at Scale 2412.09764 · UltraMem
  2411.12364 · Memory³ 2407.01178 · MemGPT 2310.08560 · episodic-memory position 2502.06975 ·
  sleep-consolidation 2603.14517.
- Cross-model alignment: Platonic 2405.07987 (rebuttal 2602.14486) · relative reps 2209.15430 ·
  vec2vec 2505.12540 · mini-vec2vec 2510.02348 · ALGEN 2502.11308 · model-stitching 2506.06609 ·
  RLSA 2311.06547 · Wasserstein Procrustes 1805.11222 · reliability-of-CKA 2210.16156.
</content>
