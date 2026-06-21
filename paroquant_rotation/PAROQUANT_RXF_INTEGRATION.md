# ParoQuant → RXF integration design

**Date:** 2026-06-21  ·  **Worktree:** `feat/paroquant-rotation` (`paroquant_rotation/`)
**Grounded against** the RXF source extracted fresh from `tcclaviger/vllm22:dev`:
`quantize_rxf.py` (offline), `rxf_kernels.py` (runtime Triton), `rxf.py` (vLLM quant method).
Read [`RXF_FOLD_IN.md`](RXF_FOLD_IN.md) first — it establishes *why* the rotation content is the
surviving ParoQuant lever; this file is the *how*, down to file:function hook points.

---

## 0. The one structural fact everything hangs on

RXF's rotation is a **standalone whole-row pre-pass DECOUPLED from the K=32 int8 GEMM.**
- **Offline** (`quantize_rxf.py:rxf_quantize`): the weight is FWHT-rotated along K *before* the
  size-32 scale search + IQ4-NL codebook assignment. The rotated weight is what gets packed.
- **Runtime** (`rxf_kernels.py:_rxf_rotate_quant_int8_kernel`, int8 path): one program per token
  loads the whole row, applies the **same** normalized FWHT, then fuses per-token int8 quant. The
  GEMM then reads the already-rotated int8 activation.
- The int8 GEMM (`rxf_linear_int8_kernel`, `BLOCK_SIZE_K=32`) **never assumes the rotation span is
  32** — 32 is only the *scale-group granularity of the rotated weight*. Because `R` is orthonormal
  and symmetric (`Rᵀ = R`, `R·R = I`), `(X·R)·(R·W)ᵀ = X·Wᵀ` for **any matched span** — proven on
  toy tensors in `sanity_wider_rotation.py` (rel err ≤ 4e-7 for spans 32…512).

**Therefore a wider / learned / importance-aware rotation is a PRE-PASS-ONLY change. The K=32 int8
GEMM, the packing format, and the per-group fp16 scale are all UNTOUCHED.** This is the entire
integration surface.

RXF ships its real edge **stalled**: `quantize_rxf.py:89  ACT_AWARE_ENABLED = False` ("IN
DEVELOPMENT"), and the quantizer hard-blocks rotation+importance together
(`rxf_quantize`, lines ~285-293) with the note *"importance must be transformed into the rotated
basis first, which is not implemented."* **That is exactly what ParoQuant resolves** — it learns the
rotation jointly with calibration, so importance lives natively in the rotated basis.

---

## 1. Exact hook points (file : function : line, against the extracted source)

### Offline — `quantize_rxf.py`
| concern | location (pre-edit) | what it does |
|---|---|---|
| rotation state | `ROTATION_NAME / APPLY_ROTATION` consts (~122) + `set_rotation()` (~126) | global gate + config tag |
| the FWHT itself | `_fwht32_rows(x)` (~131) | 5-stage butterfly, `*1/sqrt(32)`, fp32 accumulate |
| weight rotation (quantized path) | `rxf_quantize()` (~317) `gw = _fwht32_rows(gw)` | rotate each size-32 group, then scale+codebook |
| weight rotation (protected fp16 expert) | `rotate_fp16_weight()` (~150) | same FWHT for the dense fp16 experts |
| config tag emitted | `build_quant_config()` (~668) `"rotation": ROTATION_NAME if APPLY_ROTATION else None` | written into config.json |
| CLI gate | `--no-rotate` (~827) + pre-flight `rot_bad` / `K % GROUP` (~954-967) | all-or-nothing per checkpoint |
| importance hard-block | `rxf_quantize()` (~285-293) + `--act-aware` yields to rotation (~847-853) | the stalled tension ParoQuant fixes |
| activation collection (gated) | `collect_activations()` (~383), `ACT_AWARE_ENABLED` (~89) | scaffolding retained, disabled |

### Runtime — `rxf_kernels.py`
| concern | location | what it does |
|---|---|---|
| rotation gate | `_APPLY_HADAMARD` + `_ROT_SPAN` + `set_rotation(rotation)` | ✅ generalized: parses `^hadamard(\d+)$`, stores span; `None`/`"off"` → off |
| FWHT stage primitive | `_fwht_stage(x, ROWS, S, NG2H, H)` | span-`S` constexpr butterfly stage (was `_fwht32_stage`) |
| FWHT over a tile | `_fwht(a, BM, BK, S)` | ✅ `log2(S)` stages via `while H<S`, norm `1/sqrt(S)` (was `_fwht32`, 5 fixed) |
| int8 fused rotate+quant | `_rxf_rotate_quant_int8_kernel()`: `NG = BLOCK//S`, `while H<S` stage loop | ✅ span-`S` FWHT, `S` threaded as constexpr |
| fp16 standalone rotate | `_hadamard_rotate_kernel()` / `invoke_hadamard32_rotate()` (passes `S=_ROT_SPAN`) | ✅ generalized W4A16 rotate pass |

### Loader — `rxf.py`
| concern | location | what it does |
|---|---|---|
| config parse + gate | `RXFConfig.from_config()`: `_parse_rotation_span(rotation)` validates | ✅ accepts any `^hadamard\d+$` (power-of-two, %32), rejects unknown |
| field stored | `RXFConfig.__init__` `self.rotation` (string) + `self.rotation_span` (int S) | ✅ both carried to the methods |
| rotation pushed to kernel | `RXFLinearMethod.process_weights_after_loading` (~287) `set_rotation(self.quant_config.rotation)`; same in `RXFFusedMoEMethod` (~492) | the per-load hand-off |
| weight params | `RXFLinearMethod.create_weights` (~252) registers `weight_packed` + `weight_scale` (a `GroupQuantScaleParameter`) | **where a learned-R param would be added** |

---

## 2. Staging — three increments, all pre-pass only

### Stage (a) — WIDER FIXED rotation span  ✅ offline landed this session
**Cheapest discriminating change.** Generalize the FWHT from a hard 32 to any power-of-two span
`S` that is a multiple of 32 (so every size-32 scale group lands inside one rotated block).
- **Storage:** none — a fixed Hadamard is data-blind, reconstructed from `S` alone. Only the config
  tag changes: `rotation = "hadamard{S}"` (`"hadamard32"` stays the exact default).
- **Cancellation:** preserved for any matched span (§0); `sanity_wider_rotation.py` proves it.
- **Why it can help:** span 32 *cannot* move outlier energy across the size-32 boundaries; a wider
  FWHT can. `sanity_wider_rotation.py [4]` shows a single fat outlier's worst per-32 group abs-max
  falling 5.47 → 2.91 → 1.62 as span goes 32 → 128 → 512.
- **Honest caveat (measured):** a data-blind Hadamard spreads the *bulk* energy too, so on a weight
  whose outliers are *not* aligned to the Hadamard basis the wider span can *raise* per-group MSE
  (`sanity_quantize_span.py [3]`: random-position outliers gave 1.15e-3 at S=32 vs 1.84e-3 at
  S=512). **The fixed wider span is a conditional win, not a free one** — which is the entire
  argument for moving to (b)/(c). Treat (a) as the plumbing + A/B harness, not the deliverable.

### Stage (b) — LEARNED rotation matrix
Replace the fixed Hadamard `R` with a learned orthonormal `R` (per layer, or per block of `S`
input channels), fit to *minimize post-quant reconstruction error* on the weight (QuaRot/SpinQuant
style, but here only the weight is needed for a data-blind learned `R`).
- **Representation:** `R` is `S×S` orthonormal. Block-diagonal over K with block size `S`, so the
  per-layer cost is `(K/S) · S² = K·S` floats — or a single shared `S×S` if all blocks reuse it
  (recommended first: one `S×S` shared block → `S²` floats, tiny). ParoQuant's actual form
  (`NOTES.md`, `rotation_hip.hip`) is even cheaper: a product of `KROT` rounds of disjoint
  **Givens** rotations — params `theta [KROT, S/2]` + `idx_ij [KROT, S]` — i.e. `O(KROT·S)` not
  `S²`. **Reuse `rotation_hip.hip`'s validated Givens machinery for the learned form.**
- **Where it goes in the checkpoint:** a new tensor per quantized module, e.g.
  `<module>.weight_rotation` (the `S×S` matrix) **or** the ParoQuant param triple
  `<module>.rot_theta` / `<module>.rot_idx` / `<module>.rot_channel_scale`. Emit it from
  `rxf_quantize` alongside `weight_packed` / `weight_scale`, and set
  `config rotation = "learned"` (+ a `rotation_span` / `rotation_kind` field).
- **Loader:** `RXFConfig.from_config` accepts `"learned"`; `RXFLinearMethod.create_weights`
  registers the rotation param(s) (a plain non-quantized `ModelWeightParameter`);
  `process_weights_after_loading` hands them to a generalized `set_rotation(rotation, params)`.
- **Runtime kernel:** `_rxf_rotate_quant_int8_kernel` must apply the *loaded* `R`. For a dense
  `S×S`, that is an `S`-wide `tl.dot` per block instead of the butterfly; for the Givens form it is
  the `rotation_hip.hip` algorithm transliterated to Triton (KROT rounds of paired rotations). This
  is the **main remaining runtime work** (see §4).
- **Cancellation:** still exact — store `R·W` offline, apply `Rᵀ` (= the loaded `R` transposed, or
  the inverse Givens sequence) to the activation. For Givens, the inverse is the same rounds with
  negated angles in reverse order — already exercised by `rotation_hip.hip`'s forward∘inverse test.

### Stage (c) — IMPORTANCE-AWARE learned rotation  ← the genuine ParoQuant edge
Learn `R` **jointly with calibration**, weighting the reconstruction objective by activation
importance (Hessian diagonal / squared act-magnitude). This is what `ACT_AWARE_ENABLED` was meant
to unlock and what the `rxf_quantize` hard-block forbids today.
- **Re-enabling ACT_AWARE:** the block at `rxf_quantize` lines ~285-293 exists *only* because, with
  a fixed data-blind rotation, importance and rotation are in different bases. With a **learned**
  `R` the importance is folded into the *fitting objective* for `R` itself (`min Σ importance_k ·
  (W − dequant(R·W))²` over `R`), so the two are no longer mutually exclusive — they are co-fit.
  The block becomes: *if rotation is learned-importance-aware, pass the (rotated-basis) importance
  through; only the fixed-Hadamard path keeps the hard-block.*
- **Calibration source:** `collect_activations()` (already present, gated) provides the per-channel
  act statistics; flip `ACT_AWARE_ENABLED = True` behind the new `rotation=learned-aware` path only.
- **Importance in the rotated basis:** because `R` is learned to *include* the importance weighting,
  there is nothing to "transform" — the rotated-basis importance is implied by the fitted `R`. The
  per-group MSE objective in `_quantize_group` / `_pos_exact` already accepts an `importance`
  argument (the `gi` path), so the quant core needs no change once `R` is fit.

---

## 3. What landed (stage (a), offline + RUNTIME)

**`quantize_rxf.py`:**
- `_fwht32_rows` → generalized **`_fwht_rows(x, span)`** (any power-of-two span; the `0.1767766953`
  literal retained for `span==32` so the default is **bit-identical** — proven). `_fwht32_rows` kept
  as a back-compat alias bound to `ROTATION_SPAN`.
- New `ROTATION_SPAN` global; `set_rotation(on, span=GROUP)` validates `span` (power of two, multiple
  of 32) and sets the config tag to `"hadamard32"` or `f"hadamard{span}"`.
- `rxf_quantize` rotates over the **SPAN window** (`Kp//SPAN` blocks) before reshaping to size-32
  scale groups — so a wide rotation genuinely crosses group boundaries; K-divisibility check is now
  `K % SPAN`.
- `rotate_fp16_weight` (protected experts) rotates per `ROTATION_SPAN`.
- New CLI **`--rotation-span S`** (default 32); pre-flight `rot_bad` and the disable message use
  `S`; `set_rotation(rotate, rotation_span)` wires it. `--no-rotate` and the hadamard32 default are
  untouched. `build_quant_config` already emits `ROTATION_NAME`, which now widens automatically.

**Sanity (CPU, no GPU, container venv):**
- `sanity_wider_rotation.py` — orthonormality `R·Rᵀ=I` (≤6e-8), symmetry `R=Rᵀ` (0), cancellation
  `(X·Rᵀ)(R·W)=X·W` (rel ≤4e-7) for spans 32–512, span-32 == shipped (0.0), outlier-spread demo.
- `sanity_quantize_span.py` — default == frozen ref (0.0), widen sets span+tag, bad spans rejected,
  `rxf_quantize` round-trips under spans 32 & 512.

**RUNTIME (`rxf_kernels.py`, `rxf.py`) — landed this session, the matching half of stage (a):**
A `hadamard{S}` checkpoint now serves CORRECTLY (was silently wrong without this). All edits are
pre-pass-only; the K=32 int8 GEMM, pack format, and per-group fp16 scale are UNTOUCHED.

**`rxf_kernels.py`:**
- `set_rotation(rotation)` — was `on = rotation == "hadamard32"`. Now accepts `None`/`"off"` (off) and
  `^hadamard(\d+)$` → on, parsing span `S` (validated power-of-two, multiple of 32). Stores
  `_ROT_SPAN` alongside `_APPLY_HADAMARD`; INFO log reports the span. Unknown strings raise.
- The butterfly generalized. `_fwht32_stage` → **`_fwht_stage(x, ROWS, S, NG2H, H)`** (span `S` is a
  `tl.constexpr`, reshapes to `(ROWS, S)`). `_fwht32` → **`_fwht(a, BM, BK, S)`** running `log2(S)`
  stages via a `while H < S` constexpr loop (H=1,2,…,S/2; `NG2H = S//(2H)`; `NG = BK//S`) with norm
  `0.1767766953 if S==32 else 1/sqrt(S)`. The same loop replaced the 5 hard-coded
  `_fwht32_stage` calls + `0.1767766953` literal + `NG = BLOCK//32` in
  **`_rxf_rotate_quant_int8_kernel`** (the int8 W4A8 fused rotate+quant — the one that matters) and
  in the fp16 path **`_hadamard32_rotate_kernel` → `_hadamard_rotate_kernel`** /
  `invoke_hadamard32_rotate` (now passes `S=_ROT_SPAN`; `BLOCK_K = max(256, S)` so a span never
  straddles a tile). `invoke_rxf_rotate_quant_int8` reads `_ROT_SPAN` and threads `S` to the kernel.
  **span==32 is bit-identical** to the shipped fixed 5-stage code (same order, same literal).

**`rxf.py`:**
- New `_parse_rotation_span(rotation)` helper (`^hadamard(\d+)$`, power-of-two & %32 validation;
  `None` → span 32; raises otherwise). `RXFConfig.from_config` was `raise unless rotation in
  {"hadamard32", None}`; now calls `_parse_rotation_span(rotation)` to validate (accepts any
  `hadamard{S}`, still rejects unknown). `RXFConfig.__init__` keeps `self.rotation` (the string) and
  adds **`self.rotation_span`** (the int S). The `set_rotation(self.quant_config.rotation)` calls in
  `RXFLinearMethod.process_weights_after_loading` (~315) and `RXFFusedMoEMethod` (~520) are unchanged
  — they pass the `hadamard{S}` string straight through to the generalized parser.

**Runtime sanity (CPU, no GPU — `sanity_runtime_span.py`):** a pure-NumPy mirror of the generalized
runtime stage sequence (same H-order, `1/sqrt(S)` norm, per-S group reshape), proven against the
**real offline `_fwht_rows`** (extracted by AST so its heavy imports don't load). For spans
{32,64,128,512}: mirror == reference Hadamard-S (rel ≤ 2e-11), `R·Rᵀ=I` (≤4e-11) & `R=Rᵀ` (0),
**cancellation `(X·Rᵀ)(R·W) ≈ X·W`** rel ≤ 1.04e-7 against the fp32 offline (an fp64 control with the
runtime mirror on both sides drops to ≤1.6e-15, isolating the ~1e-7 as the offline `.float()`
accumulate — same level as `sanity_wider_rotation.py`'s ≤4e-7). **span==32 vs the original fixed
5-stage FWHT-32: abs-diff = 0.0 (bit-identical)** — the regression guard. **ALL PASS.**

**Caveat:** the int8 fused kernel reshapes the loaded row in-register to `(BLOCK//S, S)` with
`BLOCK = next_pow2(K) ≥ K ≥ S`, so `S | BLOCK` always holds — no extra Triton reshape constraint
beyond `S` being a power of two (already required). The fp16 W4A16 path is generalized too (not
deferred). Stage (b)/(c) (learned / importance-aware R) remain TODO — see §4.

**NOT touched (correctly):** the K=32 int8 GEMM, the pack format, the per-group fp16 scale, the
offline `quantize_rxf.py` logic, any other worktree.

---

## 4. Remaining work (next iterations)

1. **Runtime counterpart to stage (a) — ✅ LANDED this session (see §3 "RUNTIME").** A `hadamard{S}`
   checkpoint now serves correctly. `set_rotation` parses `"hadamard{S}"` and stores `_ROT_SPAN`
   (the `== "hadamard32"` literal is gone); the int8 fused kernel and the fp16 rotate kernel both run
   `log2(S)` generalized `_fwht_stage` calls with `NG = BLOCK//S` / norm `1/sqrt(S)`; `from_config`
   accepts `^hadamard\d+$` and carries `rotation_span`. span==32 is bit-identical; proven by
   `sanity_runtime_span.py` (CPU mirror). The only thing gating a served A/B is now GPU time + a real
   `hadamard{S}` checkpoint (item 4), not runtime code.
2. **Stage (b) learned `R`:** emit rotation param(s) from `rxf_quantize`; register them in
   `RXFLinearMethod.create_weights` / `RXFFusedMoEMethod.create_weights`; transliterate
   `rotation_hip.hip`'s Givens forward (and its negated-angle inverse) into a Triton kernel for the
   activation side; offline `R`-fit (start with a per-layer least-squares orthonormal Procrustes, or
   the ParoQuant Givens optimizer).
3. **Stage (c) importance-aware:** flip `ACT_AWARE_ENABLED` behind `rotation=learned-aware`; relax
   the `rxf_quantize` hard-block for that path; fold `collect_activations` importance into the `R`
   objective; the `gi` importance path in `_pos_exact`/`_quantize_group` is already wired.
4. **Evaluation (GPU, separate effort — gpu-lease):** per-group MSE + PPL at iso-bits, span sweep
   {32,64,128,512} fixed, then learned vs fixed, on a real RXF checkpoint (Step-3.7-Flash / Qwen3.x
   MoE). Retune `tuner_rxf.py` for gfx1201 first (RXF_FOLD_IN §"Do NOT quote tok/s yet").
   **Open question the fixed-span caveat (§2a) raises:** does a *fixed* wider Hadamard ever beat
   span-32 on a real model, or does only the *learned* rotation pay? Measure (a) before funding
   (b)/(c) runtime work, but expect (c) to be where the win is.

## 5. Blockers / notes
- No GPU work done or needed for this increment (CPU sanity only) — per task constraints.
- The fixed-wider-span win is **conditional** (§2a). The increment is the right plumbing + A/B
  harness regardless, but the durable contribution is stages (b)/(c). Don't oversell (a).

---

## 6. Stage (b/c) LANDED — the learned, importance-aware Givens rotation

The learned rotation (the real ParoQuant payload) is implemented end-to-end and validated offline +
on CPU. **Key design decision: a SINGLE model-wide shared R** (QuaRot/SpinQuant style), not a
per-module R. Per-module R breaks on **merged linears** (q/k/v share one input activation, gate/up
share one) and **TP shards**: the runtime rotates the *activation* once before a merged/sharded GEMM,
so every sub-weight feeding that activation must be rotated by the SAME R. A shared, block-diagonal
(over span S) R serves any K — dense **and** MoE, merged **and** TP — uniformly. (Per-(layer,
activation-group) R is a possible future increment; it needs a merged-linear-aware loader that writes
the same R to q/k/v and gate/up — noted, not built.)

### What R is, and why it beats Hadamard
R is `span × span` orthonormal, built as a product of Givens rotations and fit by **greedy Givens
coordinate descent INITIALIZED AT THE HADAMARD and refined** (`quantize_rxf.fit_givens_rotation`).
The Hadamard-init is essential: Hadamard is near-optimal for incoherence, and a greedy fit *from
identity* is strictly **worse** (CPU sanity confirmed: 0.66× — i.e. higher MSE — from identity).
Starting at Hadamard and committing only strictly-improving angles makes the learned R **≥ Hadamard
by construction**; a no-improvement fit reproduces Hadamard bit-for-bit. Measured on a synthetic
reasoning-spike weight at iso-bits: **learned-b 1.03× vs Hadamard (uniform), learned-c 1.04× vs
Hadamard (importance-weighted)** — modest but a genuine, guaranteed-non-regressive gain.

### Importance in the rotated basis (stage c — the unblock)
The fixed-Hadamard hard-block existed because importance and a data-blind rotation live in different
bases. With a *learned* R the transform is explicit: the output error injected by a rotated-weight
quant error ε is `(R x)·ε`, so rotated-channel importance = `diag(R·diag(E[x²])·Rᵀ) = (R²)·imp`
(`_rotated_importance`). The fit uses it during the search; `rxf_quantize` threads it into the `gi`
quant objective. So `rotation=givens` is **NOT** hard-blocked with `--act-aware` (only the fixed
Hadamard still is). Calibration is still gated behind `ACT_AWARE_ENABLED=False` (flip + pass
`--act-aware --rotation-kind givens` to run true stage c); stage b (uniform) runs today.

### Where it lives (hook points, all pre-pass only — GEMM/pack/scale untouched)
- **Offline (`quantize_rxf.py`):** `--rotation-kind givens` (default hadamard); `fit_givens_rotation`
  (Hadamard-init Givens descent) + `pool_rotation_blocks` (pools span-blocks from a weight sample) +
  `_apply_rotation_rows` / `_rotated_importance`; `set_givens_rotation` installs ONE model-wide R fit
  in `main()` before the quant loop; `rxf_quantize` rotates each module's weight by it and transforms
  importance; `build_quant_config` writes `rotation="givens{S}"` + `weights.rotation_matrix` (the S×S
  R inline — tiny, no extra checkpoint tensor).
- **Runtime (`rxf_kernels.py`):** `set_rotation` accepts `givens{S}` and turns the **in-kernel FWHT
  OFF** (rotation is external); `invoke_rxf_givens_rotate(x, R)` = block-diagonal `x_block @ Rᵀ` (plain
  torch, CUDA-graph-safe, no Triton); `_GIVENS_R`/`set_givens_rotation` carry R for the **monolithic
  MoE** path (applied inside `invoke_rxf_moe_kernel` where the FWHT used to run).
- **Loader (`rxf.py`):** `_parse_rotation` returns `(kind, span)` for `hadamard{S}`/`givens{S}`;
  `RXFConfig` carries `rotation_kind` + `rotation_matrix`; `givens_R(device)` builds+caches+validates
  R (orthonormality gate); `RXFLinearMethod.apply` pre-rotates x by R then calls the (FWHT-off) GEMM;
  `RXFFusedMoEMethod.process_weights_after_loading` installs the global R for the MoE kernel.

### Validation
- **CPU sanity `sanity_givens_rotation.py` — ALL PASS:** R·Rᵀ−I ≤ 2.4e-7; cancellation
  `(R x)·(R w) = x·w` rel ≤ 3e-7; iso-bits learned-b ≤ Hadamard (1.03×); importance-weighted
  learned-c ≤ Hadamard (1.04×).
- **Offline e2e (Qwen3.5-4B, container venv + GPU):** `--rotation-kind givens` fits the orthonormal R
  on cuda:0, quantizes, and emits `config.json` with `rotation=givens32` + a valid 32×32
  `rotation_matrix` (R·Rᵀ−I = 2.3e-7).
- **The runtime rotation is plain torch** (a small S×S matmul per block) — the CPU sanity exercises
  the *identical* code, so the stage-(a) "CPU mirror misses Triton JIT" caveat does **not** apply.
- **⚠ ROCm GPU-matmul flake in the OFFLINE rotation — FIXED (rotation runs on CPU).** A first GPU
  quant of Qwen3.5-4B `--rotation-kind givens` silently **zeroed 28.9 % of weight groups** →
  broken generation ("hydrogen and" → "a 1000.00 mL"). Root cause **pinpointed**: the offline
  `_apply_rotation_rows` matmul `[N·nB, 32] @ [32, 32]` (large M ≈ 737k) drops ~29 % of output rows to
  ZERO **on GPU** (norm 43.9 → 37.2), while the IDENTICAL matmul is exact on CPU. Hadamard avoids it
  because its FWHT is butterfly add/sub — **no matmul**. The scale search and the runtime rotation are
  clean; this was purely the offline GPU matmul. **Fix (landed):** `rxf_quantize` runs the (cheap)
  Givens rotation on CPU then moves the result back to GPU; the expensive scale search stays on GPU.
  Verified: GPU givens quant now 0 % zero groups. (A defensive non-finite→symmetric-scale guard
  remains in `_exact_group_scale`.) The cancellation convention was independently verified correct
  through a real quantized weight (cancellation Frob err ≈ weight-quant err).
- **GPU serve + coherence DONE:** the Givens W4A8 model generates coherently ("hydrogen and"→"oxygen",
  "France is"→"Paris", "two plus two"→"four"); the rotation cancels through the real int8 GEMM under
  torch.compile + cudagraph.
- **PPL A/B DONE (Qwen3.5-4B) — stage (b) is a NULL result.** Givens (learned, data-blind) vs fixed
  Hadamard, via `eval_ppl_ab.sh` (offline `eval_rxf_ppl.py`, bundled text, ~1.3k tokens):

  | build | W4A8 (int8) PPL | W4A16 (fp16) PPL |
  |---|---|---|
  | Hadamard (fixed)        | **6.5156** | **6.5311** |
  | Givens (learned, b)     | 6.5364 (+0.32%) | 6.5379 (+0.10%) |

  The learned data-blind Givens is **marginally WORSE** on both arms. Minimizing weight-quant MSE (the
  stage-b proxy, where Givens was 1.03×) does not translate to PPL — Hadamard is already near-optimal
  for incoherence — and the int8 activation quant of the learned-rotated activation is slightly worse
  than Hadamard's (the givens−hadamard gap widens from +0.10% fp16 to +0.32% int8). **This is exactly
  what §2(a) and §4.4 predicted: the data-blind rotation is flat; the win, if any, requires stage (c)
  importance-aware calibration** (flip `ACT_AWARE_ENABLED`, `--act-aware --rotation-kind givens` — the
  importance-in-rotated-basis machinery is wired and ready). Stage (b) plumbing is the validated
  substrate; (c) is the experiment that could actually pay.

### Not done (clearly scoped follow-ups)
- **MoE givens served A/B** (Laguna/ZAYA): code path is wired (the global-R MoE kernel hook), but only
  the dense Qwen path is GPU-serve-validated this session.
- **`--rotation-kind givens` + `--fp16-experts`** is guarded OFF (protected experts still take the
  Hadamard FWHT in `rotate_fp16_weight`).
- **Per-(layer, activation-group) R** (the higher-ceiling form) — needs the merged-linear-aware loader.

---

## 7. Stage (c) the STRATEGIC PIVOT — activation conditioning — and the FULL CLOSURE (NULL)

Stage (b)'s PPL null implied the objective was wrong: minimizing **weight-quant MSE** does not move
PPL because Hadamard is already near-optimal for weight incoherence. The pivot (cf. the DFlash result,
where activation conditioning saved INT4 acceptance) was to fit R to the REAL **per-token int8
ACTIVATION-quant error** instead — flatten activation outlier spikes before the int8 cast. This was
implemented and the line was run to a definitive conclusion. **It is a NULL — and, more usefully, the
whole rotation-tuning line is now closed with mechanistic evidence.**

### What was built (stage c, the activation objective)
- `quantize_rxf.py`: `_act_int8_quant_mse` (per-block symmetric int8 quant MSE, mirrors the runtime
  `_rxf_rotate_quant_int8_kernel`); `fit_givens_rotation(score="activation")` (the SAME Hadamard-init
  Givens descent, but the line-search minimizes the activation int8 error on real activation blocks);
  `collect_activations(collect_blocks=True)` now pools SIGNED activation blocks for the fit;
  `ACT_AWARE_ENABLED=True`; `main()` fits R on the activation objective for `givens`+`--act-aware`,
  isolating the variable (weight-side importance OFF, so the ONLY change vs stage b is R's objective).
- `sanity_givens_activation.py` (CPU): R orthonormal, cancellation `(Rx)·(Rw)=x·w` holds, the fit is
  ≤ Hadamard on its own per-block objective (commit-only-improving). **ALL PASS.**

### The DECISIVE measurement — `analyze_act_conditioning.py` (real Qwen3.5-4B activations, 1 GPU pass)
Per-token int8 activation-quant MSE (the EXACT runtime quant, full-row absmax scale), aggregated over
249 Linear modules / 31,872 real activation rows, vs the shipped Hadamard-32:

| rotation | per-token int8 act-MSE | vs Had-32 | learned vs fixed-S |
|---|---|---|---|
| no-rotate | 1.47e-3 | 0.20× | — |
| **Hadamard-32** | 2.88e-4 | 1.00× | — |
| Hadamard-128 | 1.71e-4 | **1.68×** | — |
| Hadamard-256 | 1.25e-4 | **2.30×** | — |
| fitted-256 (learned) | 1.39e-4 | 2.07× | **0.90× (WORSE)** |

Two facts fall straight out: (1) the activation-conditioning lever is **SPAN WIDTH** (a wider FIXED
Hadamard), not learning — wider span monotonically cuts the real activation int8 error up to 2.3×;
(2) the **LEARNED rotation REGRESSES** the faithful per-token metric at EVERY span (0.79–0.94× vs the
fixed Hadamard of the same span) — the greedy per-block descent overshoots the per-token full-row
absmax the runtime actually uses. The learned R loses on both objectives (weight-MSE-PPL in §6, and
the real activation int8 metric here).

### The PPL verdict — wider fixed Hadamard span (the §4.4 open question), answered: NO
Quantized `hadamard128` / `hadamard256` (runtime already span-aware; `quant_qwen_hadamard_span.sh`),
3-way PPL A/B vs the shipped `hadamard32` (`eval_ppl_span.sh`, bundled text, ~1.3k tokens):

| build | W4A8 (int8) PPL | W4A16 (fp16) PPL |
|---|---|---|
| **Hadamard-32 (shipped)** | **6.5156** | **6.5311** |
| Hadamard-128 | 6.5856 (+1.07%) | 6.5795 (+0.74%) |
| Hadamard-256 | 6.5654 (+0.76%) | 6.5521 (+0.32%) |

A wider span is **WORSE on BOTH arms**. fp16 isolates the weight quant, so the regression is
**weight-side**: the wider FWHT spreads bulk weight energy across the size-32 scale-group boundaries,
producing ~86–88k degenerate near-zero scale groups (`fp16-scale-underflow`, vs the per-group MSE
staying ~4.79e-7) that dequant to ~0. That weight-side damage outweighs the activation-conditioning
gain — which is itself **PPL-invisible** here: int8 ≈ fp16 PPL at every span (±0.2%), i.e. the int8
ACTIVATION quant is already near-lossless on this model and has **no PPL headroom** to recover. This is
exactly the §2(a) "conditional, not free" caveat materializing as a loss.

### The synthesis (why the entire rotation-tuning line is a null on W4A8)
On this W4A8 model the PPL bottleneck is the **4-bit WEIGHT**, and the shipped **span-32 Hadamard is
already at/near its optimum** (stage b: learning can't beat it; stage a wider-span: spreading energy
only adds degenerate groups). The **int8 ACTIVATION** path is already near-lossless (int8 ≈ fp16), so
activation conditioning — the entire premise of stage c — has nothing to recover. **The shipped
`hadamard32` default is correct; the learned/wider/importance rotation deltas are all ≤ noise or
negative.** Where stage c *would* pay is a regime where the ACTIVATION quant is the bottleneck (true
4-bit activations / NVFP4 draft acceptance — the DFlash regime), not int8. The stage-c machinery is
kept for that future model, not as a recommended default here.

### Artifacts (this session)
- `sanity_givens_activation.py` — CPU sanity for the activation objective (ALL PASS).
- `analyze_act_conditioning.py` — the decisive real-activation per-token int8 MSE analysis (the
  cheap, pre-PPL gate that should be run before funding any future rotation experiment).
- `quant_qwen_hadamard_span.sh` / `eval_ppl_span.sh` — wider-span quant + 3-way PPL A/B runners.
- Calibration data: `~/.cache/huggingface/calib_wikitext.jsonl` (400 wikitext-2 paragraphs).
- The `hadamard128` / `hadamard256` checkpoints are confirmed-worse throwaways (regenerate from the
  runner if needed).
