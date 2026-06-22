# ZAYA DFlash — CCA-aware speculative-decoding drafter (design plan)

Status: **DESIGN — no code yet.** This is the design deliverable; implementation
is phased (M0–M6 below). Decisions locked with the user 2026-06-21: **train our
own drafter** (no off-the-shelf speculator), **CCA-aware drafter architecture**,
**design doc before code**.

Companion docs:
- `docs/zaya/cca-all-mode-spec-plan.md` — the bit-lossless CCA rollback work. It
  is a **hard dependency** of any spec-decode-with-verification path on ZAYA and
  is referenced throughout.
- `patches/dflash_triton_noncausal/FINDINGS.md` (worktree `feat/dflash-spec`) —
  the existing DFlash bring-up on Qwen/Laguna and the **INT4 acceptance wall**
  that motivates "train our own."

---

## 1. Goal

Make ZAYA1-8B (RXF/int8, CCA hybrid, TP=1) serve faster with **DFlash
speculative decoding**, using a **drafter whose token-mixing is CCA-native**
(matched to the target) rather than a stock softmax-attention Eagle/DFlash head.
"Develop a ZAYA DFlash model" = **train and serve that drafter**.

Target outcome: a measurable wall-clock decode speedup at acceptance high enough
to beat the verify overhead (rule of thumb: mean accepted length > ~1.3 to win on
a single card), with greedy output **bit-identical** to non-spec ZAYA.

---

## 2. How DFlash actually works (ground truth from the patched source)

Read from `patches/dflash_triton_noncausal/patched/{qwen3_dflash.py,dflash.py,algos.py}`.
DFlash is **not** an autoregressive draft model. It is a **parallel, cross-attention
block drafter**:

1. **Target emits auxiliary hidden states.** `algos.py::update_dflash` maps the
   speculator card's `aux_hidden_state_layer_ids` to vLLM's
   `eagle_aux_hidden_state_layer_ids` (extraction layers) and to
   `dflash_config.target_layer_ids`. The target model must expose those
   intermediate-layer hidden states to the runner (the Eagle3 interface).
2. **Combine → context states.** `DFlashQwen3ForCausalLM.combine_hidden_states`
   (`qwen3_dflash.py:658`) projects the concatenated aux states through `fc`
   (`[hidden, target_hidden*num_aux]`) into drafter-hidden width — one context
   vector per already-decoded ("context") token.
3. **Context K/V are pre-inserted into the drafter's KV cache.**
   `precompute_and_store_context_kv` (`qwen3_dflash.py:418`) projects the context
   states to K/V for **every** drafter layer (one fused GEMM), RMS-norms,
   RoPEs, and writes them straight into each layer's paged KV cache. This runs
   **outside** CUDA graphs (`dflash.py:266`).
4. **One parallel forward over a `(1 + num_spec)` query block.** The query tokens
   are the **bonus token** + `num_spec` **mask tokens** (`mask_token_id`). They
   attend **bidirectionally** over the pre-inserted context K/V (and each other),
   predicting all `num_spec` draft tokens at once. This bidirectionality is why
   the project needed the **non-causal `triton_attn`** patch (`FINDINGS.md`
   "What was built"); `dflash.py:189` sets `causal=self.dflash_causal` (default
   `False`).
5. **Verify.** The target runs the `(1 + num_spec)` candidates in one step and
   accepts the longest matching prefix; rejected tokens are dropped and state
   rolls back.

Key consequence for us: **DFlash's drafter is a softmax cross-attention model
with a full per-layer KV cache and a bidirectional mask block.** A "CCA-aware"
drafter must reproduce DFlash's *parallel block drafting* and *aux-state
conditioning* while replacing the softmax cross-attention with CCA mixing.

### Why "train our own" is the right call (it dissolves the #1 DFlash blocker)

`FINDINGS.md` is unambiguous: on this box, the *plumbing* works but acceptance
collapsed to ~0.8% because the off-the-shelf speculator was **distilled against a
higher-precision target** than the INT4 one we can fit — a hidden-state
distribution-shift wall, not a bug, and unrecoverable without a bigger card.
Training our drafter **against the exact on-device RXF/int8 ZAYA hidden states**
removes that mismatch by construction. This is the single biggest reason ZAYA
DFlash can succeed where Laguna-INT4 DFlash could not. **It must be preserved:
all hidden-state capture for distillation runs on the served quantized model, not
a bf16 reference.**

---

## 3. ZAYA / CCA — the relevant architecture

From `zaya/overlay/vllm/model_executor/models/zaya.py` and
`.../layers/mamba/cca.py`:

- **Stack:** 80 layers, `layer_n` even → `ZayaDecoderATTLayer` (CCA attention),
  odd → `ZayaDecoderMLPLayer` (top-1 MoE). `hidden_size=2048`, `head_dim=128`,
  8 q-heads / 2 k-heads, `vocab_size=262272`, fp32 residual stream, fp32 logits
  (`_FP32EmbeddingMethod`). `moe_router_topk == 1` (asserted).
- **CCA is a *recurrent* mixer, not softmax-over-full-context.** State per
  sequence = a **conv window of `total_padding = (cca_time0-1)+(cca_time1-1) = 2`
  columns** of packed q|k, plus a **running `prev_hs`** (one hidden vector). The
  `CCA` op emits per-token q|k|v (`forward_cuda`), and the `cca_attn`
  (Mamba-style) backend does the actual attention. There is **no big bidirectional
  KV cache** to pre-fill the way DFlash does — the "context" is compressed into
  that tiny recurrent state. TP=1 only (per-head RMSNorm + grouped-mean state
  don't column-split).
- **ZAYA already has a spec-decode verify path.** `cca.py::_decode_verify_spec`
  (line 827) runs `(1 + num_spec)` candidate tokens through the causal conv as a
  segment `[current conv state | candidates]` and writes **per-spec-position
  rollback state**. The conv-state block is pre-widened by `num_spec`
  (`get_state_shape` → `cca_state_shape(num_spec=...)`). **The target side of
  spec-decode verification is therefore already built** — the open issue is
  bit-lossless rollback metadata threading (§6.3, the companion doc).

This is the asymmetry that makes a CCA-aware drafter attractive: the **expensive,
already-solved** part (target verification of a candidate block) exists; we are
adding the **cheap** part (proposing the block).

---

## 4. The central design tension (state it honestly)

| | DFlash drafter | CCA |
|---|---|---|
| Mixing | softmax cross-attention | depthwise+grouped causal conv (RF=2) + tiny recurrent state |
| Context | full per-layer KV cache, pre-inserted from target aux states | compressed 2-col conv window + `prev_hs` |
| Drafting | **parallel**, bidirectional over the `(1+num_spec)` mask block | **sequential/causal** by nature |
| Cost | a small but real transformer + KV cache | minimal state, reuses ZAYA's fused HIP conv kernels |

A literal "CCA drafter" is **causal** with a **2-token receptive field** — far too
myopic to draft a `num_spec`-long block in one shot. So the design must **widen
and de-causalize the mask block** the same way DFlash de-causalized `triton_attn`.

**Proposed resolution (the architecture in §5):** the drafter seeds its CCA
recurrent state from the **committed prefix** (via target aux states), then mixes
the `(1+num_spec)` bonus+mask block with a **small bidirectional CCA conv** (or a
single bidirectional CCA "attention" over the block). num_spec is small (~4–7),
so a full bidirectional mix over the block is cheap and keeps the drafter's
parameter footprint tiny.

---

## 5. Proposed architecture — CCA-aware DFlash drafter

Two variants; **recommend starting with Variant A, keeping B as the optimization.**

### Variant A — independent CCA drafter conditioned on target aux states (recommended)

A 1–2 layer drafter that *is* a CCA model, with DFlash's conditioning and parallel
block drafting bolted on:

```
target aux hidden states (layers L_a … from served RXF ZAYA)
        │  concat over aux layers
        ▼
   fc:  [hidden, target_hidden * num_aux]  ──► context states  (per context token)
        │
        ├─►  "context compression": fold context states into the drafter's CCA
        │     recurrent seed-state (conv window + prev_hs) at the last context
        │     position.  This replaces DFlash's precompute_and_store_context_kv —
        │     instead of a per-layer KV cache, we keep CCA's compressed state.
        ▼
   block input = [embed(bonus_token), embed(mask)×num_spec]  (the (1+num_spec) block)
        │
        ▼
   N × CCA-drafter layer:
        - linear_q/k, val_proj1/2  (as in cca.py)
        - BIDIRECTIONAL conv/mix over the (1+num_spec) block, seeded by the
          context state  → de-causalized, RF spans the whole block
        - grouped means + per-head RMSNorm  (reuse cca.py math)
        - SwiGLU MLP (or top-1 MoE-lite — start dense)
        ▼
   norm → lm_head  → num_spec draft logits  (one parallel forward)
```

- **Reuse:** the per-head RMSNorm, grouped-means, and conv math are lifted from
  `cca.py` (and its fused HIP kernels where shapes allow). The bidirectional block
  conv is a new, small kernel/path — but operates on a `≤8`-wide block, so an
  eager torch impl is fine for M1–M4.
- **No non-causal `triton_attn` dependency.** CCA doesn't go through
  `triton_attn`; the bidirectionality lives in the drafter's own block conv. (The
  DFlash triton patch is irrelevant to the CCA drafter — one fewer moving part.)
- **Vocab:** 262272 is large; an `lm_head` at full vocab is ~0.5 GB at bf16. Adopt
  DFlash's **draft-vocab reduction** (`d2t`/`draft_id_to_target_id`,
  `qwen3_dflash.py:601-641`) — train on a reduced draft vocab (e.g. 32k–64k most
  frequent ZAYA tokens) and remap to full vocab at verify. Cuts drafter size and
  speeds training.

### Variant B — state-sharing drafter (reuse ZAYA's own committed CCA state)

Since ZAYA already maintains the committed conv window + `prev_hs` per sequence,
the drafter could **read the target's final-CCA-layer recurrent state directly**
as its seed (no separate context projection / `fc` at all, or a thin adapter),
and only learn the bidirectional block head. This is the most "CCA-native" option
and the cheapest at inference, but couples the drafter tightly to the target's
internal state layout and to the bit-lossless rollback machinery. **Defer to after
A proves acceptance is reachable** — it's an inference-cost optimization, not a
prerequisite for the acceptance question.

### Open architectural parameters (to settle in M2 with small ablations)

- Number of drafter layers (1 vs 2 vs 3).
- Which target aux layers to condition on. DFlash uses a handful spread through
  depth (Laguna: 5 layers). For ZAYA start with ~3–5 spread across the 80 (e.g.
  the outputs of CCA layers at depths ~1/4, 1/2, 3/4, last) — `VLLM_DFLASH_AUX_SHIFT`
  exists as an off-by-one debug knob and we should keep an equivalent.
- Whether the block mix is "bidirectional conv" or "tiny full-attention over the
  ≤8 block" (the block is small enough that softmax attention over it is also
  cheap — but conv keeps it CCA-native; A/B them).
- Dense MLP vs top-1 MoE-lite in the drafter (start dense).

---

## 6. vLLM integration points

### 6.1 Target side — expose aux hidden states (new, required)

`ZayaForCausalLM`/`ZayaModel` currently loop layers without tapping
intermediates (`zaya.py:714-721`). Add the **Eagle3 aux-hidden-state interface**
the runner already drives for DFlash:
- implement `SupportsEagle3` (or the equivalent `set_aux_hidden_state_layers` /
  `get_eagle3_aux_hidden_state_layers` hooks vLLM uses) on `ZayaForCausalLM`;
- in `ZayaModel.forward`, capture the residual/hidden at the configured aux
  `layer_n`s and return them alongside the final hidden state.
- Mind ZAYA's **fp32 residual stream** and `ResidualScaling`: decide whether aux
  states are taken pre- or post-`res_scale`, and cast consistently. The drafter's
  `fc` must be trained on whatever convention we pick (capture-time == serve-time).

This is additive and gated; it must be a **no-op when spec is off** (no aux capture
cost on the normal serving path).

### 6.2 Drafter model + proposer (new, required)

- Register a `DFlashCCADraftModel` (config transform in `algos.py`-style:
  `update_dflash_cca`, or reuse `update_dflash` with an arch flag) and a model
  class under `zaya/overlay/...`.
- The proposer: `DFlashProposer` (`dflash.py`) is built around
  precompute-context-KV + a bidirectional mask block. For Variant A we either (a)
  subclass it and override `precompute_and_store_context_kv` to fold context into
  CCA seed-state + `build_model_inputs_first_pass` to run the CCA block, or (b)
  write a sibling `CCADFlashProposer`. Lean to (a) to inherit the fused
  input-expansion kernel (`copy_and_expand_dflash_inputs_kernel`) and the
  bonus+mask token machinery.

### 6.3 Verification + bit-lossless rollback (shared dependency)

The **target** verifies the candidate block through `cca.py::_decode_verify_spec`.
Per `cca-all-mode-spec-plan.md`, bit-lossless partial-acceptance requires
`--mamba-cache-mode all` (+ `--mamba-block-size 16`) **and** a core-vLLM runner
change to thread `num_accepted_tokens` / `block_idx_last_scheduled_token_prev_step`
to the CCA backend (the 2026-06-13 GPU probe found the runner doesn't pass them
even in 'all' mode). **Coherent-but-not-bit-lossless (align mode, ~1.76x ngram
today) is enough to first measure drafter acceptance**; bit-losslessness is the
ship gate, not the bring-up gate. Sequence the work so we can measure acceptance
on the align path before paying for the 'all'-mode runner change.

### 6.4 Memory / placement

8B RXF target + a 1–2 layer drafter (reduced vocab) fits comfortably on one 16 GB
card — none of Laguna's TP=2/OOM constraints apply. TP=1 throughout (CCA can't
shard). Use the `zaya` compose profile as the base; add a `zaya-dflash` profile
with `--speculative-config {... "method":"dflash-cca"}` once the proposer exists.

---

## 7. Training / distillation pipeline (we train our own)

This is the bulk of the effort and the part with real compute cost.

1. **Corpus + rollouts.** Pick a representative instruction/codeⁿ corpus. Run the
   **served RXF/int8 ZAYA** (greedy) to generate continuations; this fixes the
   target-token labels *and* the on-device hidden-state distribution (§2 — must be
   the quantized model).
2. **Capture.** During those rollouts, dump per-token: the configured aux hidden
   states (§6.1), the committed CCA seed-state (for Variant B), and the next
   token id (label). Store to disk (real disk under `/home/pat/code`, never
   tmpfs — `[[never-work-in-tmpfs]]`). Estimate size: ~num_aux × 2048 × 2 B per
   token; budget the corpus accordingly.
3. **Train.** Offline (can use the second card via lease). Loss = cross-entropy of
   drafter's `num_spec` predictions vs the target's next-N tokens, optionally + KL
   to the target's logits for the bonus position (EAGLE/DFlash-style). Train the
   reduced draft vocab + `d2t` map. The drafter is tiny so this is hours, not
   days — but the **capture** pass over a large corpus is the GPU cost; size it
   deliberately.
4. **Iterate on acceptance**, not loss: measure mean-accepted-length on held-out
   prompts via `test/dflash/check_dflash.py metrics`.

Open: do we have a license/corpus for ZAYA-matched distillation? (decision Q1
below). A pragmatic first pass: distill on a few hundred MB of self-generated
greedy rollouts — enough to learn the easy, high-frequency continuations that
dominate acceptance.

---

## 8. Feasibility risks (ranked)

1. **Does a CCA-shaped, aux-conditioned drafter draft well at all?** This is the
   real research risk and the reason for the design-first sequencing. CCA's native
   RF is 2 tokens; we're betting that *aux-state conditioning + a bidirectional
   block mix* gives the drafter enough signal to propose a `num_spec` block. No
   precedent exists. **M2/M4 settle this empirically; if acceptance is hopeless,
   fall back to a stock softmax DFlash head trained on ZAYA (still "our own,"
   still dissolves the INT4 wall) — keep that as the safety net.**
2. **Bit-lossless rollback is a core-vLLM runner change** (§6.3, companion doc) —
   multi-cycle GPU effort. Mitigated by measuring acceptance on the align path
   first; only pay it for the ship gate.
3. **Distillation data/compute.** The capture pass is the GPU cost; the train is
   cheap. Sizeable but bounded.
4. **fp32 residual / aux-state convention** mismatches between capture and serve
   would silently tank acceptance (the §2 distribution-shift failure mode, in
   miniature). Pin the convention and assert it.

---

## 9. Phased milestones

- **M0 — Design (this doc).** ✅ deliverable.
- **M1 — Target aux-hidden-state exposure + plumbing.** Implement §6.1 on
  `ZayaForCausalLM`; register a stub `DFlashCCADraftModel` with **random weights**;
  boot the `zaya-dflash` profile and confirm the spec path engages end-to-end
  (verify runs, output coherent, acceptance ≈ random). No training. Proves §6
  wiring. (CPU/static where possible; one short GPU boot.)
- **M2 — Drafter architecture + ablation harness.** Build Variant A (eager torch),
  finalize layer count / aux-layer set / block-mix on tiny ablations against a few
  prompts (using *quickly* distilled weights or teacher-forced acceptance proxy).
- **M3 — Distillation harness.** §7 capture + train scripts; capture a small
  corpus from the served RXF ZAYA; train a first real drafter.
- **M4 — Acceptance measurement.** Serve on align path; measure mean accepted
  length vs the verify overhead. **Go/no-go on the CCA-aware bet** (risk #1).
- **M5 — Bit-lossless ship gate.** If M4 wins, land §6.3 ('all'-mode runner
  threading + the cca.py rollback draft); greedy bit-identity gate.
- **M6 — Bench + serve.** Wall-clock decode speedup on a single card; bake into a
  `vllm22-w4a8:zaya-dflash` image; document.

---

## 11. M1 results (2026-06-21) — plumbing validated; a reprioritizing finding

Implemented the target-side aux exposure (`zaya/overlay/.../models/zaya.py`:
`ZayaModel(EagleModelMixin)` + `ZayaForCausalLM(SupportsEagle3)`, aux collected at
the tapped depths with Llama's `_maybe_add_hidden_state` semantics — side-effect-free
on the normal path), a random-weight stub drafter (`zaya/dflash/make_stub_drafter.py`,
matched to z-lab's DFlash format: `tie_word_embeddings` so it borrows ZAYA's
embed/lm_head ⇒ drafter `hidden_size=2048`), and a `zaya-dflash` compose profile
(bind-mounts the edited `zaya.py` + stub onto `vllm22-w4a8:dflash-rxf`).

**Booted clean on 1× gfx1201, eager, TP=1. The DFlash spec path engages end-to-end:**
- `Sharing target model embedding/lm_head weights with the draft model` ✓ (borrow works).
- `Using auxiliary layers from speculative config: (2, 40, 77)` ✓ — `SupportsEagle3`
  recognized; `dflash_config.target_layer_ids=[1,39,76]` → ZAYA depths {2,40,77}.
- 99 drafts / 396 draft tokens created, verify ran, **0 accepted** (correct for a
  random drafter). No exceptions; `Application startup complete`.

**Finding (reprioritizes M4/M5): the CCA target *verify* path is incoherent under
constant full-rejection.** Output was garbage ("compute compute compute…") even
though `num_accepted=0` means generation should equal ZAYA greedy. The edit is
side-effect-free (verified), so this is the **known CCA align-mode rollback
limitation** (`cca-all-mode-spec-plan.md`): on full rejection the verify reads the
wrong state slot → compounding divergence. ngram spec rarely tripped it (acceptance
usually >0); a weak drafter trips it *every step*.

**Consequence:** bit-lossless CCA rollback (was M5, "ship gate") is actually a
**prerequisite for clean acceptance measurement (M4)** — an early, weakly-trained
drafter will have low acceptance and hit the same corruption, confounding the signal.
Two ways forward (decide at M4): (a) land the 'all'-mode rollback first (the M5
core-vLLM runner change), or (b) measure acceptance corruption-robustly — single
decode step / position-0 only, resetting state between steps. **(b) is the cheap
first probe; (a) is unavoidable for a real speedup.** M5 moves before M4.

## 12. M5 results (2026-06-21) — the rollback enabler landed; full-rejection coherence fixed

The cca-all-mode-spec-plan's real blocker (2026-06-13: *"the RUNNER doesn't pass
num_accepted_tokens / prev_last_scheduled_idx to the CCA backend's build()"*) is
**resolved with a 2-hunk runner change** (`zaya/dflash/cca_all_mode_runner.patch`):
`gpu_model_runner.py` only threaded the spec rollback fields to `Mamba2`/`GDN`
metadata builders; `CCAAttentionMetadataBuilder` was excluded → its inherited base
`build()` got `None`. Adding CCA to both `isinstance` gates is the entire core
change — `CCAAttentionMetadata` already inherits the fields and the base builder
already has the full 'all'-mode block_idx logic. (No `mamba_attn.py`/`cca_attn.py`
change needed, exactly as the plan predicted.)

**Validated on 1× gfx1201, 'all' mode (block-16), eager, TP=1, stub drafter:**
- Boots clean: `mamba_cache_mode=all`, prefix caching on, aux layers `(2,40,77)`.
- `ZAYA CCA spec-verify REACHED: all_avail=True (blk_scheduled, blk_scheduled_prev,
  num_accepted all non-None)` — the fields now flow **and** verify is correctly
  classified as decode (reaches `_decode_verify_spec`; pre-fix it was treated as
  prefill because `num_accepted` was None).
- **Coherence restored under constant full-rejection.** The M1 garbage
  ("compute compute compute…") is gone — coherent, correct output across short,
  long, and arithmetic prompts ("the first five primes are 2, 3, 5, 7, 11" + a
  correct primality definition; no late divergence). With 0% acceptance every step
  keeps the bonus token, so coherence here *is* the proof the full-rejection
  rollback works.

**What M5 does NOT yet prove (gated on an *accepting* drafter):**
- **Partial-acceptance bit-identity.** The stub drafter never accepts (0%), so the
  rollback was exercised only in the full-rejection regime. The case that the old
  align path got wrong — *partial* acceptance (accept k of num_spec, roll to token
  k) — needs a drafter with >0 acceptance. Cheapest way to close it now without
  training: an **ngram drafter in 'all' mode** (accepts on repetitive text) +
  greedy bit-identity diff (num_spec=0 vs >0, equal token ids). Otherwise it
  co-validates with the trained drafter at M2/M4.
- **Prefill 'all'-mode slot write** (`cca.py` still collapses `state_indices_p` to
  column 0): coherent output suggests it's adequate, but a rigorous bit-identity
  gate may expose a first-token offset — re-check when running the bit-identity diff.

Net: the **hard core-vLLM enabler is done and de-risked**, and the M1 reprioritization
is satisfied — acceptance can now be measured on a coherent path (M4). The remaining
bit-identity gate is cheap once any accepting drafter exists.

## 13. Bit-identity gate RESULT (2026-06-22) — M5 NOT closed: a real verify-path bug

Ran the partial-acceptance bit-identity gate with an **ngram** drafter (cheap source
of real acceptance; the M5 stub never accepted). Tooling (committed in the worktree):
`zaya/dflash/bitident_client.py` (greedy temp=0 token capture via completions logprobs),
`bitident_diff.py`, `run_bitident.sh`; baseline compose profiles `zaya-bi-all-nospec`
(all/block-16, num_spec=0) and `zaya-bi-align-nospec` (align, num_spec=0). All eager,
TP=1, same model/len. Token dumps in `/home/pat/code/_bitident/`.

ngram gave **genuine partial acceptance** (340/720 accepted, per-pos 124/81/69/66,
mean accepted length up to ~4.0) — the regime the 0%-accept stub never exercised.

**Results (6 greedy prompts, token-exact diff):**
- **G (align/nospec) == B (all/nospec): BIT-IDENTICAL.** ⇒ 'all'/block-16 mode, prefix
  caching, and the prefill column-0 collapse (cca.py:319) are all bit-clean. Closes
  the cca-all-mode-spec-plan "prereq decision 1" AND the prefill column-0 re-check.
- **Rollback fields flow correctly.** Per-step debug (now in cca.py, gated by
  `ZAYA_SPEC_DEBUG`/`ZAYA_SPEC_DEBUG_N`): `all_avail=True` on every real verify step
  (`seq_lens=[5]`), with partial `num_accepted` (5/1/2/3) and `blk_sched_prev` present.
  The M5 runner patch works. `all_avail=False` only on `seq_lens=[1]` steps (ngram
  proposed nothing). So the M1 "all_avail=False" was a first-step/single-token artifact.
- **G != S (all/ngram): DIVERGES at token 1–3 on every prompt.** Since G==B, the
  divergence is the **spec verify path**, not cache mode.
- **Confound ruled out — fused-vs-eager is NOT the cause.** num_spec=0 uses the fused
  HIP decode kernel (`cca_decode_qk`); num_spec>0 is forced eager. The fused kernel is
  itself NOT bit-identical to eager (`B_eager` vs `B` diverges, but *slowly* — token
  4–78, fp-noise accumulation). Critically, **`B_eager` (eager, nospec) vs `S` (eager,
  ngram) STILL diverges at token 1–3** — same as G vs S. Removing the kernel confound
  did not fix it. The *speed* of divergence is the tell: spec = structural (token 1),
  fused-kernel = noise (token 40+).
- **Within-block conv is provably correct.** `_conv_qk_decode` is a valid stride-1
  conv: `tp_pad+L` cols → `L` outputs ending at the right token; `[init_state|toks]`
  reproduces sequential decode position-by-position. Full-rejection is coherent. So the
  bug is the **accepted-path cross-step state handoff** — the `_decode_verify_spec`
  `all_avail` rollback column math (the "DRAFT — GPU-unvalidated" code, never
  token-verified). Prime suspect: `read_col = blk_scheduled_prev + (num_accepted-1)` /
  `write_col = blk_scheduled + j` vs `clamp(max=max_col)` — observed `blk_sched≈1–3`,
  `num_accepted` up to 5 ⇒ `read_col`/`write_col` up to ~7 against a small `max_col`
  (block-size 16, short seq) → clamp collisions. Needs GPU-instrumented confirmation
  (log `max_col`, `write_col`, the block table; correlate divergence with acceptance).

**⇒ M5 is NOT closed. "Coherence under full rejection" (the prior M5 claim) is necessary
but NOT sufficient — it never token-checked partial acceptance, which is where the bug
lives.** This also matters for M4: a drafter is scored against the target's verify
logits; if accepted-path state is corrupted, acceptance is mismeasured. **Next: fix the
`_decode_verify_spec` rollback addressing against mamba2 semantics (mamba_mixer2.py
962–984), re-run the bit-identity gate (use `B_eager` as the eager baseline so the fused
kernel delta doesn't mask the result).**

### 13a. Fix work (2026-06-22) — two bugs fixed, divergence token 1 → 9–70; residual is ngram-only

Diffed against `B_eager` (all/nospec, `ZAYA_CCA_HIP=0`) to remove the fused-kernel confound.
Progression: **token 1–3 → 3 → 9–70.** Two surgical cca.py fixes (both gated, align/none
byte-unchanged), each verified to move the divergence later:

1. **Prefill 'all'-mode slot write** (token 1 → 3). The prefill wrote its end-of-sequence
   conv/`prev_hs` state to the collapsed **column 0**, but the first verify seeds from
   `state_indices_2d[row, blk_scheduled_prev + (num_accepted-1)]` = the prefill's *last*
   block. num_spec=0 reads col 0 too (why G==B passed); num_spec>0 reads the computed
   column → garbage seed. Fix: `prefill_write_slots = state_indices_p_2d[row, blk_sched_p]`
   (gated on `num_spec>0`; col-0 retained for num_spec=0).
2. **Single-token 'all'-mode addressing** (token 3 → 9–70). Steps where ngram proposes
   nothing (`seq_len=1`, `num_accepted=None`) hit the `elif blk_computed` branch, which
   read `blk_computed` but wrote `blk_computed+1` (compact-layout `read_col+1`) → the next
   single-token step re-read the stale `blk_computed` slot → conv-state staleness (RF=2 so
   ~2-3 tokens to flip argmax). Fix: in 'all' mode (`blk_scheduled` present) single-token
   steps read `blk_computed` / write `blk_scheduled` in place (mamba2 num_spec==0 path,
   mamba_mixer2.py:978-984), clamped to `max_col` not `p-1`.

**Residual (token 9–70): the verify→single-token transition after PARTIAL acceptance
(num_acc>1).** A verify writes per-position checkpoints to spec slots `blk_scheduled+[0..num_spec]`
(committed at offset `num_acc-1`); a following single-token step reads the *coarse*
`blk_computed`, which only equals the committed spec slot when `num_acc==1`. **Unfixable
surgically in CCA's design:** CCA stores per-position state in *separate blocks* (cca_state_shape
`del num_spec`, mamba_utils.py:224) and vLLM gives single-token steps no `num_accepted`
(treats them non-spec), so the layer can't locate the committed spec slot. mamba2 avoids
this entirely by storing **one wide conv window** (`conv_kernel_size-1+num_spec`) in a single
coarse block and rolling back via **token-offset `num_accepted-1`** (causal_conv1d.py:850,864),
never per-position blocks.

**Why this is fine for the project: single-token steps only exist because ngram proposes
0 tokens on no-match. The DFlash drafter ALWAYS proposes a full `num_spec` block → every
step is a full verify → the residual never triggers.** And the verify→verify path is now
bit-identical *across block boundaries* — the counting prompt stayed identical through
token 70 (~4 block-size-16 boundaries of pure verify→verify). So **M5 is effectively closed
for the real (DFlash) drafter**; only the ngram convenience-probe hits the residual.

To fully close the ngram path (optional, not needed for dflash): either adopt mamba2's
wide-conv-window + token-offset rollback (state-shape change + rollback rewrite) or hook the
post-sampling consolidation copy (`get_cca_conv_copy_spec`) for 'all' mode. Re-verify
bit-identity with the trained dflash drafter at M4. Tooling: per-step + addressing traces
are in cca.py gated by `ZAYA_SPEC_DEBUG`/`ZAYA_SPEC_DEBUG_N`.

## 10. Decisions for the user (before M1/M3)

- **Q1 (M3 blocker): distillation corpus.** What data may we use to generate
  ZAYA greedy rollouts for distillation (a license-clear instruction/code mix,
  or a specific dataset)? Self-generated rollouts from a small seed-prompt set is
  the zero-dependency fallback.
- **Q2 (architecture, M2): block mixer** — bidirectional **CCA conv** (most
  CCA-native) vs **tiny full-attention over the ≤8 block** (simpler, proven math).
  Default: A/B both in M2, ship the better.
- **Q3 (scope): safety net.** If the CCA-aware drafter underperforms in M4, do we
  pivot to a stock softmax DFlash head trained on ZAYA (still our-own, still
  resolves the INT4 wall)? Default: yes, keep as fallback.
</content>
</invoke>
