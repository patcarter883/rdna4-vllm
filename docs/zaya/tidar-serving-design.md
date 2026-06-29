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

> ⚠ **CORRECTION (2026-06-29, `replica_diag.py`) — this "no bidirectional conv needed" holds only for
> the TWO-FORWARD path.** In the FUSED single forward `[committed | S | R_0..R_{B-1}]` the replicas sit
> AFTER the S drafts in the sequence, so each replica's leading `total_padding=2` tokens read the S
> draft tokens through the causal conv (the conv ignores the attention mask) → corrupted leading drafts
> (token-level match `[0,0,1,1]`: only the conv-window tokens wrong). The FUSED path therefore needs a
> **SEGMENTED conv** (each R_r convolved over its own `[committed+first-r-drafts | mask*B]` segment) —
> what `cca.py:_decode_verify_spec` already does. The two-forward `block_predict` is unaffected because
> its mask block immediately follows the committed tokens. See §9 (single-forward entry).

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
### 7.5 Whether the diffusion fine-tune kept `conv_qk` causal (§1.1) — **RESOLVED (2026-06-28, STEP 4): it stayed CAUSAL. See §9.**
Confirmed on the checkpoint's real layer-0 `ZayaCCAProjection` (`cca_evict_gate.py`): appending tokens
after position `p` does not change `p`'s q/k/v (max|Δ|=0.0); the FT kept `conv_qk` left-pad-only at the
K=2 boundary (the mask patch only replaces `create_causal_mask`, never the conv). **No CCA non-causal
branch needed.** (Earlier worry: if the FT made the conv non-causal for mask positions, the CCA op would
have needed a branch — it didn't.)
### 7.6 RoPE on mask tokens — **RESOLVED (2026-06-29, `bisect_fusion.py`).** Mask positions sit at
`prefix_len + block_len + j` and all `block_len` replicas share the next-block positions
`L+B..L+2B-1` (GATE A's scheme). The fused forward under this scheme is **fp32 bit-identical** to the
causal verify forward on the real checkpoint, so the convention is correct. (Was: confirm the mask-block
position_ids match training, else draft logits go off-distribution — now pinned. See §9.)

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
- [x] **β=1 COHERENCE GATE PASSED on the real checkpoint (§9 NEXT step 2) — 2026-06-28.** Checkpoint
  `pat883/zaya1-8b-tidar-experts` (full-ft-all ZAYA1-8B-Diffusion, block_size=4, step 999) loaded via
  the Zyphra fork (`serve_loader.py`, build-from-config + `load_state_dict`, 0 missing/0 unexpected)
  and run on CPU (no lease — 17.7 GB bf16 fits host RAM, not a 16 GB card). **β=1 TiDAR decode ==
  AR-greedy token-for-token on all 4 prompts** (incl. varied-token cases, not just the degenerate
  `= = =` collapse), `coherence_gate.py` GATE B. This empirically pins **`replica_offset=0` +
  `sampling_causal=True` + the verify/commit bookkeeping** on a genuinely TiDAR-trained model. The
  lossless loop is the **two-forward** form (verify against a causal forward over `[committed|drafts]`;
  diffusion drafts from a separate `[committed|mask*B]` block forward) — exactly what `tidar_loop.py`
  uses. Acceptance avg 0.8–2.0 / 4 (drafts are hints; β=1 lossless regardless — throughput lever, not
  a losslessness concern). Reusable artifacts: `serve_loader.py`, `coherence_gate.py`,
  `diag_causality.py`, `diag_ar.py`, `zaya_mask_patch.py` (copied from convert), all in `zaya/tidar/`.
  - **CRITICAL LOADER REQUIREMENT:** `attn_implementation="eager"`. The `zaya_mask_patch`
    `create_causal_mask` monkeypatch is consumed ONLY by ZAYA's eager attention; under the default
    **SDPA the injected bias is silently dropped** → the model is non-causal under appended tokens →
    the gate diverges. (Confirmed: full-recompute AR under eager == native cached `generate()` AR,
    `diag_ar.py` MATCH=True — methodology is sound once eager.)
- [x] **§7.5/§1.1 — the fused single-forward "contamination" was a MISDIAGNOSIS: it is bf16 numerical
  noise, NOT a sequence-global op. The FUSED single forward IS mathematically lossless → the ~2.4×
  lever is UNLOCKED — 2026-06-29 (`bisect_fusion.py`).** GATE A (bf16) saw the S verify rows shift when
  the `block_len²` mask replicas are present (max|Δlogit|≈1.8), and §7.5 *attributed* it to a
  sequence-global op (MoE load-balance / global norm). **That attribution was wrong.** A layer-by-layer
  S-row bisection (fused vs causal forward, real checkpoint) shows: (a) **fp32 ⇒ BIT-IDENTICAL at every
  one of the 40 layers AND in the final logits (max|Δ|=0.0, no diverging layer)**; (b) bf16 ⇒ layers
  0–3 bit-identical, then a *tiny* divergence (mean 4e-3) that grows gradually with depth (ballooning
  only where ‖hidden‖ blows up, layers 35–38) — the signature of floating-point accumulation, not a
  structural global op (which would jump sharply at one MoE layer from its first occurrence). A code
  audit independently confirms ZAYA MoE routing + `ResidualScaling` are strictly **per-token** (no
  capacity drop, no in-forward load-balance update, no cross-token norm). The bf16 seed is the
  **grouped-MoE-GEMM batching** (S tokens sharing an expert with R tokens round differently in the
  batched bf16 matmul — per-token in math, not bit-exact in bf16) + residual amplification. **Two
  consequences:** (1) **§7.5 dissolved** — verify is bit-exact with the scratch present, so the
  production path CAN be ONE fused forward (verify S + pre-draft R together) — 1 fwd/step instead of 2.
  (MEASURED single-forward speedup is **~1.52× aggregate**, NOT the naive `(1.40+1)=2.4×` — the replicas
  draft WEAKLY, see the single-forward entry below; 2.4–2.7× IS hit on prompts where drafts land.)
  (2) **§7.6 pinned** — the bisection
  used GATE A's replica-position scheme (all replicas at the next-block positions `L+B..L+2B-1`) and is
  fp32 bit-exact, so that `position_ids` convention is VALIDATED on the real checkpoint. **bf16 caveat:**
  β=1 isn't strictly *bit*-lossless under bf16 (rare borderline-argmax flips on low-entropy prompts —
  same class as the two-forward divergence); mitigate by computing the few S verify-row logits in fp32,
  or accept the noise-level flip. The "one-forward shortcut" (`tidar_loop.py` test E) is **REINSTATED.**
- [x] **SINGLE-FORWARD TiDAR — implemented + measured: ~1.52× aggregate, lossless (fp32); the limiter
  is replica DRAFT quality, not the architecture — 2026-06-29 (`single_forward_tidar.py`).** The loop:
  ONE fused forward/step over `[committed | S=prev_drafts | R_0..R_{B-1}]` → verify S rows (β=1 accept k
  + bonus) AND read replica `R_k` as the next block's drafts (`select_next_drafts_row_range`). Results
  (real checkpoint, 4 prompts):
  - **LOSSLESS in fp32** (committed == AR-greedy on all 4); **bf16 diverges on 2/4** (borderline-argmax
    flips — confirms the bf16 caveat; strict losslessness needs fp32 verify-row logits).
  - **Forward count: 0.656 fwd/token ⇒ 1.52× aggregate (≈41 tok/s)** vs AR; beats the two-forward 1.12×.
    PER-PROMPT spread is wide: 0.375–0.417 fwd/tok (**2.4–2.7×**) when drafts land (avg_accept 2.0–2.5),
    but **1.0 fwd/tok (1.0×)** on low-entropy/factual prompts ("2+2", "mitochondria": avg_accept 0.04–0.26).
  - **Why not the naive 2.4× — ROOT-CAUSED to the §1.1 causal CONV (`replica_diag.py`, token-level):**
    in the fused sequence `[committed | S=drafts | R_0..R_{B-1}]`, each replica's leading `total_padding=2`
    tokens read their 2 *sequence-left* neighbors through the causal `conv_qk` — which are the **S draft
    tokens**, NOT the committed/accepted context. The conv is a separate operator from attention and does
    **not** respect the structured mask (so even though R is masked from S in *attention*, the *conv*
    pulls S in). Token-level proof: R_k vs `bp_k = block_predict(committed+drafts[:k])` matches as
    **`[0,0,1,1]`** — tokens 2–3 (conv window entirely inside the replica) are EXACT; only tokens 0–1
    (conv window straddling the drafts) are wrong. **This is §1.1 materialized for the FUSED path** (the
    "no bidirectional conv needed" note held only for the TWO-FORWARD path, where the mask block
    immediately follows the committed tokens so the conv reads the right neighbors). Two construction
    issues, both identified: (i) replica RoPE positions must be `[L+r, L+r+B-1]` not `[L+B, L+2B-1]`
    (fixed in `single_forward_tidar.py`/`replica_diag.py` — necessary but not sufficient), and (ii) the
    conv must be **SEGMENTED per replica** (each R_r sees `[committed+first-r-drafts | mask*B]` as its own
    conv segment) — which is EXACTLY what `cca.py:_decode_verify_spec` already does for spec decode (the
    "cca.py KV-cached path" the design anticipated). The post-forward bonus token is a separate, smaller
    residual (k vs k+1 context), secondary to the conv.
  - **Verdict:** the fused single forward is the RIGHT serving architecture (verify lossless, fewest
    forwards, best aggregate so far, ~1.52×). The steady ~2.4× IS reachable but requires the
    **segmented-conv serving path** (the real `cca.py` `_decode_verify_spec`-style conv, not the naive HF
    dense flat-conv forward used in these CPU harnesses) + the replica-position fix; on the naive dense
    forward the conv corrupts the replicas' leading 2 tokens → capped acceptance. Plus the standing lever
    of a better-trained checkpoint. Realizing it = the segmented-conv fused forward on the live cca.py
    runner (TP=2) — a real but well-scoped follow-on, no remaining unknowns.
  - **Segmented-conv CONFIRMED as the cause; CPU construction-trick is too finicky — 2026-06-29
    (`segmented_fused_tidar.py`).** Tried to realize the segmented conv WITHOUT the cca.py runner, via a
    construction trick: insert each replica's correct 2-token conv context `ctx_r` before R_r in the
    sequence, masked from attention so it only feeds the causal conv. Result: the replica's leading 2
    tokens FLIPPED from wrong→correct (token-match went `[0,0,1,1]` flat → **`[1,1,0,0]`** segmented),
    **proving the conv is genuinely the cause** — but the trick introduced a residual mismatch on the
    *trailing* 2 tokens (a standalone-layout bug), so net acceptance didn't recover (0.52). Verify stayed
    lossless throughout. **Takeaway:** the §1.1 conv is the confirmed root cause; the CLEAN fix is the
    real `cca.py:_decode_verify_spec` segmented conv (native conv-state handling) on the live runner, NOT
    a CPU-harness reconstruction. The algorithm is fully de-risked; only the (substantial) live cca.py
    TP=2 integration remains. Diagnostic scripts: `bisect_fusion.py`, `single_forward_tidar.py`,
    `replica_diag.py`, `segmented_fused_tidar.py`.
- [x] **STEP 3 — structured mask WIRED into the real standard-attention backend + GPU-validated
  (`gpu_validate.py` Part E) — 2026-06-28.** The kernel hooks (Route A `mask_bias`, Route B `qq_bias`)
  were already validated (Parts C/D); step 3 adds the missing connective tissue so the per-step TiDAR
  mask reaches the *standard* `self.attn` call (zaya.py:216 — CCA is only a QKV producer, §1, so the
  mask lands on `self.attn`, NOT inside CCA):
  - **`zaya/tidar/tidar_attn_metadata.py`** (NEW): `build_tidar_mask_meta(prefix_len, block_len, …)`
    builds `TiDARMaskMeta` (Route-B `qq_bias` = the `additive_bias(d)[:, prefix_len:]` new-token slice
    Part D drives the kernel with, verbatim; + optional Route-A square `mask_bias`), cudagraph-static
    for a fixed block_len. A module-level **active-mask carrier** (`set_/get_/clear_active_tidar_mask`,
    `update_active_tidar_mask_` for the §31g static-address in-place case) bridges the builder to the
    backend without patching vLLM's frozen `ForwardContext`. A **backend hook** (`wrap_unified_attention`
    / `install_tidar_attn_hook`) wraps triton_attn's module-bound `unified_attention` so it injects the
    active `qq_bias`; null-safe (no active mask, or a caller that already passed `qq_bias` like
    tree_attn, ⇒ byte-identical passthrough).
  - **`cca_attn.py`**: `CCAAttentionMetadata` gains an optional `tidar_mask: TiDARMaskMeta | None = None`
    field so the mask travels alongside the CCA conv-state metadata (§4.1); default None ⇒ zero change.
  - **GPU gate (`gpu_validate.py` Part E, 1-card lease, `vllm22-w4a8:combined`):** mask built via the
    serving builder, installed on the carrier, then the HOOK-WRAPPED `unified_attention` called WITHOUT
    an explicit qq_bias (exactly what the stock backend does) over paged single-seq KV. RESULT for
    block_len {4,8,16} × prefix {0,64,512}: **== boolean-masked fp32 SDPA** (cos 0.999997–0.999999,
    ≤4 bf16-ULP, 0 outliers >6 ULP) AND **byte-identical to the explicit-qq_bias path** (max|Δ|=0.0 —
    the wrap only changed where qq_bias came from). **Regression:** carrier cleared ⇒ wrapped ==
    stock byte-identical (a plain decode step is untouched). Full A–E run: ALL PASS.
  - *Single-sequence for now* (matches the β=1 coherence gate + the validation); batched-decode (one
    qq_bias per sequence) + the Route-A attn_hip backend wiring are step-4 concerns once the loop is on
    the real `cca.py` KV path. Overlay uncommitted per protocol.
- [x] **STEP 4 — decode loop / β sampler / evict-on-reject FOLDED onto the real
  `cca.py:_decode_verify_spec` KV+conv-state rollback — 2026-06-28.** The cca.py `(1 + num_spec)`
  candidate-window conv + per-spec-position rollback IS the TiDAR evict-on-reject path: `num_spec`
  maps to the TiDAR `block_len`, the verify processes the whole `[current state | block_len
  candidates]` block writing the conv window + `prev_hs` ENDING at each candidate `j` to slot
  `state_indices_2d[i, write_col[i,j]]`, and the next step reads `blk_scheduled_prev +
  (num_accepted-1)` = the accepted-prefix end — appending the rejected tail then reading the accepted
  column IS the evict (no physical truncation). This is the exact `IncrementalKVConv.commit_block`
  contract.
  - **REAL-MODEL equivalence gate (`zaya/tidar/cca_evict_gate.py`, CPU, no lease):** drives the
    checkpoint's actual layer-0 `ZayaCCAProjection` (the conv producer cca.py caches as
    `conv_states`/`prev_hs`) over `[committed | block_len-draft block]`, evicts the rejected tail, and
    asserts the committed conv/`prev_hs` state == a from-scratch recompute of the accepted token
    stream. **PASS, max|Δ|=0.0** (bit-identical) for `k_accept` ∈ {0..block_len} × prefix ∈ {4,16}.
    Equivalence holds because the conv is CAUSAL — position `p`'s q/k/v + conv-window + recurrent
    state are independent of any token appended after `p`. Verification is kept ISOLATED from the
    `B*B` mask-replica scratch (the §7.5 fusion-contamination finding): the gate drives ONLY the conv
    producer, never the replicas; the TiDAR structured mask rides the SEPARATE standard-attention
    backend (active-mask carrier + `CCAAttentionMetadata.tidar_mask`, step 3), not this producer.
  - **CONV-CAUSALITY confirmed (§1.1/§7.5) on the real conv:** appending tokens after position `p`
    does NOT change `p`'s q/k/v (max|Δ|=0.0, prefix/tail ∈ {(4,4),(8,8),(16,4)}). The diffusion FT
    kept `conv_qk` CAUSAL (left-pad only, `F.pad(.., (total_padding, 0))`, cca.py:380) at the K=2
    (`cca_time0`/`cca_time1`) boundary — the mask patch only replaces `create_causal_mask` (the
    *attention* mask), never the CCA conv. **No cca.py non-causal branch needed.**
  - **β=1 losslessness re-confirmed after the fold:** `coherence_gate.py` GATE B still PASS (β=1
    TiDAR == AR-greedy token-for-token, all 4 prompts). GATE A still reports fusion NOT viable
    (max|Δlogit|≈1.84) — corroborating the isolation requirement the fold satisfies.
  - **Additive cca.py wiring:** a null-safe `num_spec → block_len` doc anchor in
    `_decode_verify_spec` (no code change to the rollback math — it was already generic over
    `1+num_spec`); `num_spec==0` ⇒ the conv producer is byte-identical (non-TiDAR path untouched).
    `test_tidar_loop.py` + `test_tidar_mask.py` 16/16 green (no stub regression).
- [x] **SERVABLE BUILD — the TiDAR checkpoint now SERVES on vLLM TP=2 (gfx1201) — 2026-06-28.**
  Unblocks throughput/§31g (a serve loop now exists). The checkpoint `model_latest.pt` is in the
  **Zyphra HF-transformers-fork naming** (`qkv_proj.q_proj`, `conv_qk_depthwise/grouped`,
  `qk_norm.temp`, fused `mlp.experts.gate_up_proj/down_proj`, `input_layernorm`); the vLLM `zaya`
  overlay's `load_weights` is exact-match and needs **vLLM naming** (`self_attn.qkv.linear_q`,
  `conv_qk.0/1`, `qkv.temp`, per-expert `zaya_block.experts.local_experts.N.linear_fc{1,2}`,
  `input_norm`/`res_scale`, 40→80 split layers). The fp8 build serves because it was made from an
  already-converted repr — **Zyphra ships BOTH formats as parallel snapshots** of `Zyphra/ZAYA1-8B`
  (`67d34da`=HF-fork, `970cfc9f`=vLLM).
  - **Converter (`zaya/tidar/hf2vllm_map.json` + derive/apply scripts):** used the two parallel
    snapshots as a **byte-provenance oracle** — every one of the **2483 vLLM tensors matched an
    HF-fork source byte-for-byte (0 ambiguous, 0 unmatched)**, yielding the exact name map incl. the
    layer split (vLLM `2L`=ATT ← HF `L` self_attn/input_layernorm; `2L+1`=MoE ← post_attention_layernorm
    /experts/router), expert de-fusion (`gate_up_proj[e]`→`linear_fc1`, `down_proj[e]`→`linear_fc2`),
    router renames (`router_mlp.fc1`→`router_mlp.0`, `out_proj`→`router_mlp.4`, `norm`→`rmsnorm_eda`),
    and the residual-scale chain (last layer's `post_mlp_residual_scale` → top-level `model.res_scale`).
    Replicates the original converter's wiring from provenance — no reverse-engineering needed.
  - **Servable dir** `/home/pat/code/zaya1-8b-tidar-serve`: map applied to `model_latest.pt` → 2483
    vLLM-naming safetensors (16.5 GB) + the bf16 vLLM config (`970cfc9f`, 80 layers, 16 experts,
    tie=None→overlay ties, no lm_head) + the TiDAR tokenizer (`<|tidar_mask|>` 262147).
  - **Serve (compose profile `zaya-tidar`, TP=2 bf16, `gpu-lease -n 2`):** all **2483 weights loaded
    0 missing / 0 unexpected** (the converter's proof — the HF-fork dir had failed here with a
    `not initialized from checkpoint` ValueError), 8.48 GiB/card, KV 90k tok @ 2.75x, graphs captured,
    **Application startup complete**. **Coherence:** "capital of France is" → " Paris" ✓ (then the
    known overfit `= = =` tail), correct train-speed reasoning, coherent chat — proving the weights
    are structurally correct, not just shape-loadable. **Baseline AR decode ≈ 27 tok/s** single-stream
    (the eventual TiDAR-speedup denominator). Serve then torn down (cards freed); re-launch:
    `ZAYA_MODELS_DIR=/home/pat/code gpu-lease -n 2 --detach --name zaya-tidar -- docker compose --profile zaya-tidar up -d`.
- [~] **TiDAR proposer / model-runner integration (DONE for the per-step wiring; the live-runner
  fused forward is §7.6-gated) — 2026-06-28:** the step-3 carrier+hook + step-4 evict are now wired
  into a real per-step β=1 TiDAR decode loop via the NEW orchestration module
  `zaya/tidar/tidar_proposer.py` — which implements **no** new mask/rollback math, only routes the
  pinned primitives:
  - `maybe_install_tidar_hook()` installs `install_tidar_attn_hook` once at model load; wired
    ADDITIVELY into `ZayaForCausalLM.__init__` (overlay `zaya.py`), guarded so an import/install
    failure is a silent no-op and an installed-but-inert hook (no carrier) is byte-identical to stock.
  - `TidarProposer.run_block(prefix_len)` is the per-step set-before / clear-after carrier boundary
    (fresh-alloc `build_tidar_mask_meta`; the in-place `update_active_tidar_mask_` is reserved for the
    §31g capture step). It ALWAYS clears the carrier on exit (even on exception) so no mask leaks into
    the next forward. `verify_commit` forwards to `tidar_loop.beta_verify` (β=1 lossless);
    `evict_contract(num_accepted) → num_accepted-1` names the `cca.py:_decode_verify_spec` rollback
    column (the step-4 fold; no new rollback math). Serve-enable flag: `VLLM_TIDAR_BLOCK_LEN` (unset ⇒
    plain decode, hook inert).
  - **CPU gate `test_tidar_proposer.py` (no lease, 7/7; full suite 23/23):** env-flag parsing,
    carrier set/clear discipline (incl. clear-on-exception), `verify_commit==beta_verify`,
    evict-column contract, and an end-to-end β=1==greedy-AR through the proposer surface (B∈{1,4,8}×3
    seeds) on the `StubCCALM`.
  - **GPU gate `gpu_validate.py` Part F (1-card lease, `vllm22-w4a8:combined`): RUNNER-PATH β=1 decode
    loop == AR-greedy, token-for-token** — a full β=1 TiDAR decode driven through `TidarProposer`
    (hook installed; carrier set BEFORE / cleared AFTER each block forward) over the **real RDNA4
    triton_attn kernel** (the exact kernel `ZayaAttention.forward → self.attn → triton_attn`
    dispatches to) commits the SAME stream as plain AR-greedy through that kernel — IDENTICAL for
    B∈{1,4,8}×2 seeds, carrier-clean after every loop; PLUS the null-safety regression (carrier
    cleared ⇒ byte-identical to stock, max|Δ|=0). RESULT: **ALL PASS** (A–F). This pins that the
    *wiring* (proposer carrier + hooked real kernel) is itself lossless, complementing the standalone
    `coherence_gate.py` GATE B that pins losslessness on the real **weights**.
  - **§7.6 BOUNDARY (the explicit blocker, NOT this item):** driving the converted checkpoint through
    this same triton_attn carrier on the live **vLLM runner** needs (a) TP=2 / a >16 GB fit (this item
    is capped at `-n 1`) and (b) `position_ids` for the fused `[S | R_0..R_{B-1}]` replica block —
    **§7.6, still un-pinned on-device**. The standalone β=1 coherence gate (real checkpoint, CPU)
    sidesteps §7.6 by re-deriving R_0 from a fresh causal forward each step; Part F holds weights fixed
    and exercises the wiring. The proposer deliberately emits NO replica position_ids (`run_block`
    carries only the contiguous `[prefix | S]`/`[prefix | mask*B]` block); the live-runner fused
    single forward is the §7.6 follow-on, the per-step proposer/runner wiring itself is DONE + lossless.
- [x] **§31g FULL-cudagraph CAPTURE of the carrier + hooked block forward — DONE + GPU-VALIDATED
  weight-independently at -n 1 (`gpu_validate.py` Part G) — 2026-06-28.** This is the §5 / step-6 capture
  step: it exercises the one piece the eager Part D/E/F path skips and that capture REQUIRES — the
  in-place, static-ADDRESS active-mask carrier (`update_active_tidar_mask_`) — and gates capture by
  eager==replay BIT-EQUALITY (the honest weight-independent signal at -n 1; NOT torch.profiler
  launch-count, which bypasses replay — memory [[profiler-bypasses-cudagraph-replay]]). **Evidence:
  GPU-validated (1-card lease, real triton_attn kernel), reproduced across two independent runs — full
  A–G ALL PASS; NOT checkpoint-driven (the live-runner fused single forward on real weights is the
  §7.6-gated step-6 follow-on below, out of this `-n 1` item's scope).**
  - **G1 (static-address carrier, OUTSIDE any graph):** after `update_active_tidar_mask_` copies a new
    step's qq_bias INTO the already-active carrier buffer, the HOOK-wrapped `unified_attention` output
    == the fresh-alloc `build_tidar_mask_meta` path (**max|Δ|=0.0**), AND the carrier buffer's
    `id()`/`data_ptr()` are UNCHANGED across two in-place updates (**addr-stable=True**) for block_len
    {4,8} × prefix {0,64,512}. This is the first exercise of the §5 "persistent static-address buffer"
    contract the design names but never drove.
  - **G2 (capture==replay, the core gate):** the carrier+hooked block forward is captured under
    `torch.cuda.graph` at a FIXED capture size `q_len = block_len·(1+block_len)` (static q/KV/out
    buffers + the static-address carrier); for a new step, new q/k/v are copied into the static buffers
    + `update_active_tidar_mask_` updates the carrier in place, `g.replay()`, read static out. Replayed
    out == the same EAGER hooked call **bit-equal (max|Δ|=0.0)** for block_len ∈ {4,8,16} × prefix
    {0,64}, ≥2 distinct mask/input fills each (the carrier's `data_ptr()` asserted unmoved across every
    replay). This pins the §5 "k-variability does **not** break capture" property: the forward's shape
    is fixed at `block_len·(1+block_len)` regardless of accept-length k; only the post-forward
    index/eviction math (NOT in the captured region) varies. **Full A–G run: ALL PASS** (1-card lease,
    `vllm22-w4a8:combined`; attn_hip rebuilt; `attn_hip` NOT touched — the mask rides the standard
    triton_attn carrier). Overlay uncommitted per protocol.
  - **STILL-OPEN GATED FOLLOW-ONS (NOT this item):** the live-runner FULL_DECODE_ONLY *dispatch* probe
    (launch-count) needs the §7.6 fused single forward + TP=2; paged-KV in attn_hip (step 7) and β<1
    (step 8) remain.
- [x] **THROUGHPUT measured — two-forward TiDAR ≈ 1.12× on this checkpoint (step 5) — 2026-06-29.**
  `zaya/tidar/throughput_tidar.py`. The lossless production path is **two-forward** (a `[committed|mask*B]`
  diffusion-draft forward + a `[committed|drafts]` causal-verify forward per step; CONTIGUOUS positions
  ⇒ the §7.6 replica-`position_ids` unknown — which only blocks the §7.5-contaminated FUSED forward —
  does NOT apply). Speedup is **cache-independent + exact via FORWARD COUNT**: AR = 1 fwd/token; TiDAR =
  2 fwd/step → (k+1) committed tokens. Each forward costs ~the same (batch-1 8B decode is
  **memory-bandwidth bound**: the 27 tok/s baseline = 16 GB/37 ms ≈ 430 GB/s ≈ card BW ⇒ a B-query block
  forward ≈ a 1-token decode), so forward-count ratio = throughput ratio. **Measured on the real
  checkpoint** (4 prompts, n_new=24, run on CPU — forward count + acceptance are hardware-independent;
  the 2-GPU `device_map` run loaded+dispatched fine but truncated at the first PART-1 GPU forward, so the
  device-independent CPU count is the reported signal): **avg accept 1.40/4, forward-count speedup 1.12×
  (≈ 30 vs 27 tok/s)**, predicted `(avg+1)/2 = 1.20×`. Per-prompt 1.71× (accept 2.86/4) down to 0.86×.
  - **Caveat (honest):** 1 of 4 prompts ("Q: What is 2+2? A:") **DIVERGED** from AR-greedy. β=1 is
    lossless *by construction* (verify IS the AR argmax), but here AR and TiDAR are TWO different-shaped
    bf16 forwards (`[committed]` vs `[committed+drafts]`/`[committed|mask*B]`), so a borderline argmax at
    a low-entropy position can flip — an fp-nondeterminism artifact, not a logic bug (a real fused
    single-model deploy verifies from its OWN forward, no separate baseline to diverge from). More likely
    on this overfit, low-entropy checkpoint.
  - **Verdict:** TiDAR serving works and gives a **modest** win on this checkpoint, gated by (i) the
    two-forward constraint (the fused single forward would ≈ double it to `(k+1)/1 ≈ 2.4×` but is
    §7.5-contaminated + §7.6-`position_ids`-gated) and (ii) low acceptance (1.4/4 — this checkpoint's
    diffusion drafts are weak/overfit; a better-trained checkpoint is the real lever). The 4.6× blog
    number assumes the fused forward + a clean model — neither holds here yet.

GPU work: every job via `scripts/gpu-lease.sh -n 1 -- …` (TP=1); container `vllm22-w4a8:dflash-rxf`;
warm Triton cache + `.env` (`HF_HOME` + `VLLM_HOST_TRITON_CACHE=/home/pat/code/.triton-cache-zaya-dflash`).
Don't commit (uncommitted overlays inherited from `feat/zaya-dflash`; fold at a future M-stage when
asked). Reference vLLM (read-only): `/home/pat/code/_vllm_ref_combined`.
