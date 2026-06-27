# TiDAR serving path for ZAYA1-8B-Diffusion on RDNA4 — design note

Status: **DESIGN, pre-implementation.** Worktree `feat/tidar-serve` (off `feat/zaya-dflash`,
to inherit the §31g cudagraph overlays + the CCA spec-decode rollback machinery). Owns the
**serving / kernel / cudagraph** side only; AR→TiDAR conversion training is a separate session
(`docs/zaya/TIDAR_CONVERSION_PROMPT.md`, worktree `feat/tidar-convert`).

Source brief: `docs/zaya/TIDAR_SERVING_PROMPT.md`. Papers read for this note: TiDAR
(arXiv 2511.08923v1, "Think in Diffusion, Talk in Autoregression", NVIDIA) and CCA
(arXiv 2510.04476, "Compressed Convolutional Attention", Zyphra). The TiDAR mask shape and the
"+1" factor below are **flagged-uncertain** — see §7.1; they must be re-derived from the paper's
FlexAttention `mask_mod` before the kernel is written, not from the prose summary.

---

## 1. The single most important architectural finding (read first)

CCA on this stack is **NOT** a bespoke attention you must layer a mask *inside*. Tracing the live
code:

`ZayaAttention.forward` (`zaya/overlay/vllm/model_executor/models/zaya.py:203-219`):
```
output_qkv = zeros(...)
self.qkv(hidden_states, output_qkv)     # <- the CCA op (cca.py); produces q|k|v
q, k, v = output_qkv.split([q_dim, k_dim, v_dim])
q, k = self.rotary_emb(position_ids, q, k)
attn_output = self.attn(q, k, v)        # <- STANDARD vLLM Attention, paged KV, triton_attn on RDNA4
attn_output = self.o_proj(attn_output)
```

So CCA decomposes into **two** components, and TiDAR touches each differently:

1. **CCA = a compressed-latent QKV *producer*** (`cca.py`, the `vllm::cca` custom op). It does:
   low-rank `linear_q`/`linear_k` into a shared latent (`num_q_heads*head_dim` / `num_k_heads*head_dim`,
   GQA `gqa_groups = q_heads//k_heads`), a **two-stage causal depthwise+grouped conv** along the
   sequence (`conv_qk`, left-padded by `total_padding = (cca_time0-1)+(cca_time1-1) = 2` by default),
   grouped QK means, per-head RMS-norm, and a two-stream value (`val_proj1` on `hs`, `val_proj2` on
   the *previous-token* hidden `hs2`). Output is **ordinary q, k, v tensors.**
2. **The actual attention is a standard exact softmax** over a paged KV cache — `self.attn` is vanilla
   vLLM `Attention` (triton_attn / rocm_attn backend on RDNA4; some layers are SWA per
   `config.swa_layers`). **This is exact, maskable, score-matrix attention** — there is no SSM /
   linear-attention recurrence in the score computation. CCA's "compression" is entirely on the
   *projection* side; once q/k/v exist, attention is normal.

**Consequence — the TiDAR work splits cleanly into two state domains:**

| TiDAR concept | Lands on | RDNA4 mechanism |
|---|---|---|
| structured causal + block-bidirectional mask | the **standard** `self.attn` paged attention | a custom-mask attention kernel (§3, §4.1) |
| exact-KV "evict on rejection" | (a) the standard **paged KV cache**, and (b) CCA's **conv recurrent state** `conv_states` (2 cols) + `prev_hs` (1 hidden) | reuse the existing spec rollback (§4.2) |
| β rejection sampler | sampler / proposer (logits) | small, host-side or fused (§4.3) |
| single forward/block decode loop | vLLM v1 model-runner / TiDAR proposer | §4.4 |

The recurrent state in (b) is **small and already rollback-instrumented**: the DFlash spec path
already runs candidate tokens through a segmented conv and writes per-position rollback state
(`cca.py:_decode_verify_spec`, the `conv_states`/`prev_hs` per-spec-slot writes). TiDAR's evict-on-
reject is the *same operation* — keep state for accepted tokens, drop the rest — so this is reuse,
not new kernel work. The risky-new part is the **mask** (§3).

### 1.1 Does the causal conv break the bidirectional block? (the CCA-vs-diffusion question)

The `conv_qk` is **causal** (left pad only, `F.pad(..., (total_padding, 0))`, `cca.py:380`). The
mask block in TiDAR is **bidirectional among its own positions.** These do not conflict, because the
conv and the attention are *different operators*: the conv mixes only the **previous `total_padding`=2
positions' pre-projection q|k** into each token (a tiny fixed local window), whereas the mask block's
bidirectionality is a property of the **softmax attention** over q·kᵀ. A mask token attending
bidirectionally to its block-siblings in the attention score matrix is unaffected by the conv being
causal — the conv just means each mask token's *query/key vector* is built from itself + the 2 tokens
to its left (which for the leading mask tokens are the draft tokens / prefix tail, all already
present in the forward's input row order). **No bidirectional conv is needed.** This must be
confirmed against the conversion-trained checkpoint (training may or may not have made the mask block
see future masks *through the conv* — almost certainly not, since the conv stays causal in the AR
parent), but structurally the causal conv is fine. Flag for the conversion session: confirm the
diffusion fine-tune kept `conv_qk` causal.

---

## 2. TiDAR inference algorithm (as implemented for, per the paper)

One model, **one network forward per block**, no iterative denoise (the diffusion half is *all-mask
one-step* — every draft position is a `[mask]` token, decoded in a single pass; multi-step is
unnecessary because training masked the entire diffusion section).

**Per-step input row layout** (one sequence) — **corrected from the paper, supersedes an earlier
`2×block_len` reading**:
```
[ prefix (cached) | S: drafts_from_prev (block_len) | R_0 | R_1 | ... | R_{block_len-1} ]
```
The paper fixes `q_len = block_len · (1 + block_len)`: **1 sampling block `S` + `block_len` replicas
`R_r` of the next-block mask, one replica per possible acceptance length** (replica `R_r` is
conditioned on `r` accepted drafts). The `block_len` replicas computed in one forward ARE the
"pre-draft conditioned on every acceptance length" trick (§2b) made parallel. So the new query region
is `≈ block_len²`, not `2·block_len` (block_len=16 → 272 queries). Prefix is KV-cached, fed only as
keys. Block sizes 4 / 8 / 16. The full predicate is implemented + unit-tested in
`zaya/tidar/tidar_mask.py` (see §3).

**Structured attention mask** (the core new primitive):
- prefix → **causal** (standard AR).
- `drafts_from_prev` (the "sampling block") → **causal** AR-verify: each draft position attends to
  the prefix + earlier draft positions + itself. This produces `p_AR` at every draft position *as if*
  the earlier drafts were already committed — that is what makes one forward verify the whole block.
- `mask_tokens_for_next` → **block-bidirectional**: each mask token attends to the **entire prefix**
  and to **all other mask tokens** in its block, but **not** to the sampling-block (draft) positions.
  This yields the one-step diffusion draft for the *next* block, computed in parallel and (see below)
  conditioned on every possible acceptance outcome.

**Each forward simultaneously:**
- **(a) Rejection-samples the previous step's drafts** against the AR distribution computed *this*
  step. For each sampling-block position left→right: accept the draft token iff it equals the token
  the AR head would pick under the β rule; on the first mismatch, stop (accept count `k ∈ [0,
  block_len]` is variable, prefix-property of speculative decoding). β=1 ⇒ pure-AR accept ⇒ **lossless**
  (output distribution identical to AR). This is what gives the "single model is its own speculator
  *and* verifier" property — no separate drafter, unlike EAGLE/DFlash.
- **(b) Pre-drafts the next block** from the mask positions, **conditioned on all possible accept
  lengths**: because the mask block attends to the prefix (and is bidirectional internally) but the
  AR accept simply *advances the committed prefix boundary by `k`*, the single diffusion forward over
  the mask block already covers every feasible continuation; after `k` is known you **select** the
  matching pre-draft. (Exact selection mechanism is the part to nail from the paper — §7.1.)

**β sampler:** `token = argmax_v ( β · logits_v^AR + (1−β) · logits_v^diffusion )`.
β=1 → lossless (~4.6× per the Zyphra ZAYA blog), mixed β → ~7.7× (blog). NB the *TiDAR paper's* own
numbers are 4.71× @1.5B / 5.91× @8B — different models/configs; treat blog 4.6/7.7 as the ZAYA1-8B-
Diffusion target and the paper numbers as method-level corroboration, **don't** hard-code either as a
gate. Throughput ≈ 7.45–8.25 accepted tokens per network forward.

**Exact KV cache:** prefix + accepted draft tokens are causal ⇒ their KV is computed once, cached,
never recomputed. On rejection, the KV (and CCA conv state) of the `block_len − k` rejected positions
is **evicted**. Mask-token positions are never written to the persistent KV (they are scratch for the
next draft).

---

## 3. The core new primitive: the structured-mask attention (one abstraction, both backends)

This is the real RDNA4 kernel work; the rest is plumbing. **DONE so far:** the backend-neutral mask
is implemented and unit-tested — `zaya/tidar/tidar_mask.py` (+ `test_tidar_mask.py`, **10/10 green**,
CPU, no lease). It emits, from one construction:
1. `build_allow_matrix` — boolean `[q_len, kv_len]` ground truth;
2. `additive_bias` — float `[q_len, kv_len]`, `0`/`-inf`, **added to QK^T pre-softmax**; and
3. `MaskDescriptor` — the compact layout integers for an inline predicate.
The SDPA-equivalence test confirms the additive bias drives real attention identically to a hand-
masked reference; the `_allow_pair` per-pair reference is the exact contract the HIP inline predicate
must reproduce. Full predicate + the (flagged) uncertain choices are documented in the file header
and §7.1.

**Problem:** vLLM's paged backends (triton_attn / rocm_attn) and our `attn_hip` kernel apply *causal*
/ *sliding-window* masks only. TiDAR needs the per-(query,key) structured mask above. The unifying
insight: **for every backend the structured mask is the same additive bias over the QK^T scores
before softmax** — so one tested abstraction (§3.1/§3.2) feeds all of them.

### 3.1 Custom HIP attention (`feat/attn-hip`) — first-class target, and a clean fit
`attn_hip` (`/home/pat/code/vllm-gfx1201-attn-hip/attn_hip/`, `flash_prefill_kernel<HEAD_DIM>`,
rocwmma 16×16×16, bf16 dense prefill, causal+SWA+GQA) is **architecturally ideal for TiDAR** for two
reasons:
- **Its softmax is decoupled from the WMMA fragment layout**: QK^T scores are `store_matrix_sync`'d
  to fp32 `smem_S` and the softmax is a *plain one-thread-per-row smem loop*
  (`attn_kernels.hip:173-180`), exactly where `causal`/`sliding_window` set `s = -1e30f`. So the
  structured mask drops in **right there** — either add our `additive_bias[qr,kpos]` to `smem_S`
  (correctness path, identical numerics to triton/SDPA) or evaluate the `MaskDescriptor` inline
  predicate from `(qr, kpos)` next to the existing causal test (perf path: zero extra memory,
  cudagraph-clean). No WMMA fragment-layout reasoning is touched — the mask lives entirely in the
  fp32 smem softmax. ⚠ Still read `NOTES.md` (gfx12 row = `(lane>>4)*8+e`) before any *other* kernel
  edit.
- **TiDAR turns decode into prefill**, and `attn_hip` is a *prefill* kernel — the workloads match. The
  per-step `block_len²` new queries are precisely a small prefill tile.
  **Gaps to close in `attn_hip` for TiDAR:** (i) GPU-validated functionally (§9 / `gpu_validate.py`);
  the only nit is a pre-existing ~5-ULP ragged/SWA bf16 tail (the attn-hip owner's, not TiDAR-blocking);
  (ii) it is **contiguous (non-paged)** only (paged-KV is item 5 on its perf list) — fine for the
  stub/correctness pass (prefix as a contiguous K/V), but production needs the paged prefix read;
  (iii) **DONE** — the optional `mask_bias` arg is added + validated (§9).

### 3.2 triton_attn additive-bias path (Route B) — stand up first for correctness
If the RDNA4 triton attention kernel accepts an additive `[q_len, kv_len]` bias, the same
`additive_bias` tensor makes the structured mask work with **no new kernel** for a first correctness
pass. Cheaper to stand up; validate triton_attn's bias support first. Numerically identical to §3.1's
bias path (both add the same tensor to the scores), so they cross-check each other.

**Recommended sequence:** SDPA/triton additive-bias correctness stub → `attn_hip` parity green →
`attn_hip` + structured bias in `smem_S` (matches the stub bit-for-tolerance) → `attn_hip` inline
predicate + paged-KV + §31g capture for production.

**SWA interaction:** some ZAYA layers are sliding-window (`config.swa_layers`, `zaya.py:164-179`).
The structured mask must **compose** with the per-layer sliding window (intersection). In both the
bias and the predicate forms this is just an extra AND in the same place the mask is already applied
(`attn_kernels.hip:177-178` already does the SWA test). Benign for block_len ≤ 16.

---

## 4. Where each remaining piece lands

### 4.1 Mask construction
A builder that, given the current `[prefix_len | block_len]` layout, produces the structured mask
(Route B: a bias tensor; Route A: a `mask_mod`-style predicate baked into the kernel). Must be
**cudagraph-static**: fixed `block_len`, fixed capture sizes (§5). Lives next to the CCA attention
metadata builder (`zaya/overlay/vllm/v1/attention/backends/cca_attn.py`, currently a thin
mamba-style metadata holder) or in the attn backend.

### 4.2 Exact-KV evict-on-reject — **reuse the spec rollback**
Two caches to roll back by `block_len − k` on each step:
- **Paged KV** (standard): drop the rejected tail; vLLM's spec-decode machinery already supports
  variable-accept KV truncation via `num_accepted_tokens` + block-table columns.
- **CCA conv state** (`conv_states`, 2 cols) + **`prev_hs`** (1 hidden): `cca.py:_decode_verify_spec`
  already (a) runs `1+num_spec` candidate tokens through the segmented causal conv and (b) writes
  per-candidate-position rollback state to `state_indices_2d[i, j]`, with the next step reading column
  `num_accepted-1`. **TiDAR's evict-on-reject is this exact mechanism** with `block_len` candidates.
  Reuse it; do not rebuild. (The `num_spec` plumbing — `get_current_vllm_config().num_speculative_tokens`,
  `cca.py:191` — becomes `block_len`.)

This reuse is the second-biggest carryover after §31g and de-risks the "diffusion models can't KV-
cache" worry entirely: CCA already cooperates with a variable-accept rollback.

### 4.3 β rejection sampler
Needs access to **both** logit streams at each draft position: `p_AR` (causal rows) and `p_diff` (the
prior step's mask-block output, carried forward). Implement as a small per-step op:
`accept while draft[j] == argmax(β·logit_AR[j] + (1−β)·logit_diff[j])`, stop at first mismatch, emit
`k`. Start host-side/torch for correctness; fuse later if it shows on the profile. β is a serve-time
knob (β=1 default = lossless).

### 4.4 Single-forward decode loop / TiDAR proposer
The v1 model-runner drives: build `[prefix | prev_drafts | mask]`, one forward, β-rejection-sample →
`k`, evict rejected KV+conv (§4.2), select next-block drafts from mask logits, append `k` accepted +
new drafts, loop. This is **simpler than the DFlash AR-spec loop** — one forward, no separate drafter
model, no separate verify pass. Model it as a vLLM v1 *proposer-equivalent* so it slots into the
existing spec-decode runner hooks rather than a forked decode loop.

---

## 5. Cudagraph capture (reuse §31g; simpler here)

TiDAR is **one forward per block** (no drafter, no separate verify), so the §31g FULL-capture story is
strictly simpler than DFlash. Carry over verbatim:
- explicit `cudagraph_capture_sizes = {(2·block_len)·k}` and raise `max_num_seqs ≥ largest size`
  (the uniform-decode FULL-capture bound);
- the persistent static-address buffer pattern (`_cca_seed_buf`-style) for any per-step state a
  captured graph reads (here: the mask/bias tensor and the carried `p_diff`);
- the FULL-capture proposer-style overrides.

**Variable accept-length `k` under a static graph:** the forward's *shape* is fixed (`2·block_len`
queries every step regardless of `k`); only the *post-forward* eviction/selection depends on `k`, and
that is index math (gather by `num_accepted`), already done cudagraph-safely in
`_decode_verify_spec` (fixed width `p = 1+num_spec`, no `.item()` sync, dump-invalid-to-row-`td`
scatter). So `k`-variability does **not** break capture — same trick.

**Measurement discipline (CRITICAL):** `torch.profiler` bypasses cudagraph replay, so launch-COUNT
under the profiler cannot measure capture effectiveness — use the DEBUG dispatch probe
(`gpu_model_runner.py:4131` "Running batch with cudagraph_mode") + real throughput. (memory:
[[profiler-bypasses-cudagraph-replay]].) Tooling carried in `zaya/dflash/`: `dispatch_probe.sh`,
`capture_mode_test.sh`, `throughput_capsizes_ab.sh`, `analyze_launch_count.py`.

---

## 6. Progress without weights (no public ZAYA1-8B-Diffusion checkpoint)

The mask + loop + KV-evict + sampler are **weight-independent for correctness** (coherence is not).
Plan:
1. **Stub / random-weight ZAYA-CCA**: build the structured-mask attention (Route B bias first) and
   the decode loop; validate against a reference dense FlexAttention-style mask in torch — assert the
   paged structured-mask output equals a dense masked-softmax reference bit-for-(tolerance); assert
   KV+conv evict leaves state identical to a from-scratch recompute of the accepted prefix; assert the
   β=1 sampler accept/reject matches a reference loop.
2. **Loader designed to swap** either the conversion-session checkpoint or a future Zyphra release.
3. Coherence + throughput gates only once a real checkpoint exists.

---

## 7. Open questions / risks

### 7.1 Exact mask predicate — **partially resolved + encoded; two off-by-ones still to confirm**
Re-derived from the paper: `q_len = block_len·(1+block_len)` = 1 sampling block + `block_len` mask
replicas (one per acceptance length), key axis `q_len + max_seq_len`. The full predicate is now
**implemented and unit-tested** in `zaya/tidar/tidar_mask.py` (S causal + blind to replicas; replica
`R_r` bidirectional within itself, isolated from other replicas, conditioned on the first `r` drafts,
sees all prefix). The paper gives **no reproducible `mask_mod`**, so two choices remain genuinely
uncertain and are exposed as flags + pinned by tests, to confirm against the conversion checkpoint:
- `replica_offset` — is `R_r` conditioned on `r` or `r+1` accepted drafts? (acceptance is `0..block_len`
  = `block_len+1` outcomes but there are only `block_len` replicas → one boundary is folded; default
  offset 0 covers `0..block_len-1`). Affects which replica the runner selects post-accept
  (`select_next_drafts_row_range`).
- `sampling_causal` — S self-attention causal (paper: "clean tokens … causally") vs bidirectional.
Getting these wrong silently breaks **losslessness**, not just speed — re-check against the paper's
Figure 3 and the conversion-trained checkpoint's behaviour. The construction is otherwise locked.

### 7.2 triton_attn custom-mask/bias support (gates Route A vs B) — **RESOLVED (2026-06-27): gate landed, GPU-validated. See §9.**
The RDNA4 triton unified-attention kernel already has a **native query-query additive-bias hook**:
`unified_attention(..., qq_bias=...)` (`vllm/v1/attention/ops/triton_unified_attention.py`), a
`[q_len, q_len]` `0`/`-inf` tensor added to the QK^T scores via `load_qq_bias_tile`
(`triton_attention_helpers.py:344`) — indexed `[query_pos, key_rel_pos]` where
`key_rel_pos = seq_offset − context_len` is the key's position *within the new-token region*. This
is **exactly the form `tidar_mask.additive_bias` already emits** (restricted to the query-query
columns). It is real and in-use: the spec-decode **tree-attention** backend (`tree_attn.py`,
`_prepare_tree_attn_bias` → `qq_bias=decode_meta.tree_attn_bias`) passes precisely such a `0`/`-inf`
`[q_len, q_len]` mask. So TiDAR's **causal** mask parts (S causal-verify; replica-sees-first-r-drafts;
prefix causal) map onto `qq_bias` with **no new kernel**.
- **THE ONE GAP:** the kernel applies the causal `seq_mask` (`compute_kv_seq_mask`,
  `seq_offset <= query_abs_pos`) at `triton_unified_attention.py:528` and **adds `qq_bias` *after***
  (line 538), and hard-asserts `causal=True` (line 838). So `qq_bias` can only *further-restrict*
  within the causal cone — it **cannot** grant the replica block's **bidirectional (anti-causal)**
  attention (a mask token attending its block-siblings to the right). Tree masks never need this
  (a node attends only ancestors ⊂ causal), which is why the stock ordering suffices for them.
- **Fix (small, well-scoped, the Route-B analogue of the done attn_hip `mask_bias` change):** gate the
  causal `seq_mask` so that for query-query keys (`key_rel_pos ≥ 0`) the causal restriction is dropped
  and `qq_bias` alone defines allow/deny, while **prefix keys stay causal**. (Alternative if a kernel
  edit is unwanted: run the small bidirectional mask block — `≤ block_len²` queries — as a *separate*
  non-causal attention via attn_hip's already-validated square `mask_bias`, and keep the causal
  S-verify on the paged `qq_bias` path; less clean, splits the forward.) Either way the `additive_bias`
  tensor is reused verbatim. **DONE (2026-06-27):** the `seq_mask` gate landed in the overlay
  `zaya/tidar/triton_overlay/triton_unified_attention.py` and passed the GPU SDPA-equivalence gate
  (`gpu_validate.py` Part D) — Route B is now a working PAGED structured-mask path (full results in §9).
### 7.3 Access to both logit streams (`p_AR`, `p_diff`) at the sampler — plumb the mask-block logits
forward across steps as a captured static buffer.
### 7.4 SWA × structured-mask composition on sliding-window layers (§3).
### 7.5 Whether the diffusion fine-tune kept `conv_qk` causal (§1.1) — confirm with the conversion
session; if it made the conv non-causal for mask positions, the CCA op needs a branch.
### 7.6 RoPE on mask tokens — mask positions sit at `prefix_len + block_len + j`; confirm position_ids
for the mask block match what training used (else the draft logits are off-distribution).

---

## 8. Carry-over ledger (reuse, don't rebuild)

| Reuse | From | For |
|---|---|---|
| §31g FULL-cudagraph-capture (capture sizes, static buffers, dispatch probe) | `feat/zaya-dflash`, `zaya/dflash/` | §5 |
| spec-decode variable-accept rollback (`_decode_verify_spec`, conv/prev_hs per-slot writes, cudagraph-safe scatter) | `cca.py` | §4.2 evict-on-reject |
| CCA HIP conv kernels (`cca_prefill_qk`, `cca_decode_qk`) | `cca.py` + `cca_hip/` | unchanged QKV producer |
| rocwmma flash-attn — softmax decoupled into fp32 `smem_S` (mask hook at `attn_kernels.hip:173-180`) | `feat/attn-hip` `attn_hip/` (+`NOTES.md`) | §3.1 structured-mask kernel |
| backend-neutral TiDAR mask (bias + predicate + descriptor), 10/10 tests | `zaya/tidar/tidar_mask.py` (NEW) | §3 — feeds attn_hip AND triton/SDPA |
| profiler-bypasses-replay methodology | memory | §5 measurement |

**OBSOLETE — do not carry:** the DFlash drafter (`cca_drafter_model.py`), the separate-verify HIP
path, the M5 'all'-mode rollback-as-spec experiment, OPD drafter training. TiDAR is single-model,
single-forward — most of the DFlash complexity dissolves.

---

## 9. Implementation progress + next steps

- [x] **Mask pinned + encoded (§3, §7.1):** `zaya/tidar/tidar_mask.py` — backend-neutral allow-matrix
  + additive bias + `MaskDescriptor`; `test_tidar_mask.py` **10/10 green** (CPU, no lease), incl. the
  SDPA-equivalence gate and the per-pair reference contract for the inline kernel predicate.
- [~] **attn_hip GPU-validated functionally; one tail-rounding nit left to the attn-hip owner
  (§3.1):** ran `attn_hip_parity.py` + `zaya/tidar/gpu_validate.py` (Part A) under a 1-card lease in
  `vllm22-w4a8:combined`. Kernel is **functionally correct — cosine 0.999999** on all 6 geometries.
  The flat `max|Δ|≤5e-3` parity gate is mis-calibrated (a single bf16 ULP on a peaked causal row,
  |out|~2-4, is ~1.5e-2). At a bf16-ULP-aware bound, the clean multiple-of-tile causal case and the
  noncausal case pass cleanly; **ragged (S=100), SWA=64, S=64, D=64 causal each have ONE element off
  by 4-5 ULP** — exactly the tail/ragged-masking failure `NOTES.md` predicted. Does **not** block
  TiDAR (won't flip argmax tokens), but the `feat/attn-hip` owner should tighten the ragged/SWA tail
  before the kernel is the production attention. (Note: a pure per-element ULP-ratio metric explodes
  on near-zero outputs — use `atol + rtol·|ref|`, not raw ULP ratio.)
- [x] **TiDAR mask validated ON DEVICE (§3, `gpu_validate.py` Part B):** the `additive_bias` drives a
  real bf16 attention identically (max|Δ|=0.0) to a boolean-masked reference for block_len ∈ {4,8,16}
  × prefix ∈ {0,64,512}. The on-GPU correctness gate for the structured-mask path both backends share.
- [x] **Structured mask wired into the custom HIP attention + GPU-validated (§3.1):** added an
  OPTIONAL `mask_bias` (square `[seq,seq]` fp32 additive bias) to `attn_hip.flash_prefill`
  (`feat/attn-hip`: `attn_kernels.hip` kernel sig + the `smem_S` softmax loop, `bindings.cpp` schema
  `Tensor? mask_bias=None`, `op.py` fake; **null ⇒ byte-identical to before**, existing callers
  unaffected). `gpu_validate.py` Part C: the **real kernel driven by the square TiDAR mask matches a
  boolean-masked SDPA reference** — cosine 0.999999, ≤4 bf16-ULP, B∈{4,8}×P∈{0,64,200}. `tidar_mask`
  gained `build_square_allow_matrix` / `square_additive_bias` for self-attention kernels.
  *(Edit is uncommitted in `feat/attn-hip` per protocol; it's that effort's branch — coordinate
  before committing. The kernel's pre-existing ~5-ULP ragged/SWA tail nit is unrelated and theirs.)*
- [x] **Decode loop + evict stub (§4.2, §4.4) — DONE, CPU-validated:** `zaya/tidar/tidar_loop.py`
  (+ `test_tidar_loop.py`, **6/6 green**, CPU float64, no lease). A random-weight CCA-shaped stub
  (`StubCCALM`: causal depthwise conv = `conv_states`, previous-token v term = `prev_hs`) drives the
  single fused forward over `[prefix | S | R_0..R_{B-1}]` (one `tidar_forward`, structured
  `additive_bias`, per-replica **segmented conv**). The loop verifies → commits k accepted + 1 bonus
  → evicts the B−k rejected positions → re-derives next drafts. Pinned: **evict-on-reject == a
  from-scratch recompute** of the committed KV+conv state every step (`IncrementalKVConv`, test B);
  the per-replica segmented conv == fresh recompute (test C, the §1.1 causal-conv reuse); the
  in-forward replica `R_k` == a fresh predraft after k accepts (test E — production may take the
  one-forward shortcut). *Note:* the runtime models the `_decode_verify_spec` rollback **logic**
  standalone (no vLLM model-runner / weights exist yet); folding onto the real `cca.py` path is a
  later checkpoint-stage step.
- [x] **β=1 sampler (§4.3) — DONE:** `beta_verify` (accept while `draft==argmax(β·p_AR+(1−β)·p_diff)`,
  stop at first mismatch, emit `k` + bonus). Tests: β=1 accept/reject == greedy AR (test A), the
  end-to-end loop with β=1 reproduces greedy AR **exactly** for B∈{1,4,8} × 3 seeds (test D, the
  headline lossless property), and β<1 mixes both logit streams (smoke). `p_diff` plumbing is wired
  (carried prior-step mask logits); β=1 default ignores it = lossless.
- [x] **triton_attn additive-bias path (§3.2/§7.2) — DONE, GPU-validated (Route B now works):** the
  RDNA4 triton unified-attention kernel exposes a query-query additive-bias hook
  (`unified_attention(qq_bias=...)`, a `[q_len,q_len]` `0`/`-inf` tensor added post-causal via
  `load_qq_bias_tile`), already used by `tree_attn.py` for spec-decode tree masks — it accepts
  `tidar_mask.additive_bias` verbatim for the **causal** parts. The **gap** (stock applies the causal
  `seq_mask` then *adds* `qq_bias`, so `-inf+finite=-inf` can't grant the replica block's
  **bidirectional** attention) is closed by a small `seq_mask` gate in the overlay
  `zaya/tidar/triton_overlay/triton_unified_attention.py`: when `USE_QQ_BIAS`, OR the causal mask
  back to True for keys in the query-query region (`key_rel_pos = seq_offset − context_len ∈
  [0, qq_bias_stride_0)` — the exact predicate `load_qq_bias_tile` already uses), so `qq_bias` alone
  defines allow/deny there while **prefix keys (`key_rel_pos < 0`) stay strictly causal**. The
  `causal=True` assert is unchanged (bidirectionality comes entirely from the qq-region gate). When
  `USE_QQ_BIAS=False` the block is dead-code-eliminated → stock path byte-identical. **GPU
  SDPA-equivalence gate** (`gpu_validate.py` Part D, weight-independent, 1-card lease,
  `vllm22-w4a8:combined`): the patched `unified_attention` over a paged single-sequence KV cache
  matches a boolean-masked fp32 SDPA over `build_allow_matrix` for block_len ∈ {4,8,16} × prefix ∈
  {0,64,512} — **cos 0.999997–0.999999, worst 1–7 bf16-ULP, ≤1 tail outlier** (same `atol+rtol·|ref|`
  bf16 bar as Part C; a pure ULP-ratio explodes near zero so it's gated on cosine + a >6-ULP outlier
  COUNT). **Tank-check:** the UNPATCHED kernel FAILS every bidirectional case (cos 0.756–0.997,
  32–5253 ULP, thousands of >6-ULP outliers) — proving the gate is what fixes it. **Regressions:**
  `qq_bias=None` and a strictly-causal (tree-attn-style) `qq_bias` are both **byte-identical**
  patched vs stock (max|Δ|=0.0) — the OR-back-on is a no-op outside the bidirectional region, so
  existing tree-attn callers are unaffected. *(Overlay is uncommitted per protocol; folding onto the
  installed kernel / real `cca.py` is a later checkpoint-stage step. The two mask off-by-one flags —
  `replica_offset`, `sampling_causal` — still await the conversion checkpoint.)*
- [ ] then §31g capture (§5), paged-KV in attn_hip, coherence/throughput once a checkpoint lands.

GPU work: every job via `scripts/gpu-lease.sh -n 1 -- …` (TP=1); container `vllm22-w4a8:dflash-rxf`;
warm Triton cache + `.env` (`HF_HOME` + `VLLM_HOST_TRITON_CACHE=/home/pat/code/.triton-cache-zaya-dflash`).
Don't commit (uncommitted overlays inherited from `feat/zaya-dflash`; fold at a future M-stage when
asked). Reference vLLM (read-only): `/home/pat/code/_vllm_ref_combined`.
