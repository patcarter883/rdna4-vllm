# Session prompt — TiDAR serving, CHECKPOINT phase (start HERE once a fine-tuned TiDAR model exists)

You are resuming the **serving / kernel / cudagraph** side of the TiDAR pivot (Zyphra
ZAYA1-8B-Diffusion) for the FIRST time with a real trained checkpoint. The pivot rationale,
algorithm, and architecture are settled — do **not** re-litigate them or redo the paper reads. The
weight-independent serving primitives are **built, GPU-validated, and committed**; this phase wires
them onto a real model and proves losslessness + throughput.

This session owns serving only. AR→TiDAR conversion training is the SEPARATE `feat/tidar-convert`
effort — its worktree is **READ-ONLY cross-reference**, never edit it, never run training there.

## Preconditions (the trigger for this prompt)
A TiDAR fine-tuned checkpoint is available from `feat/tidar-convert`. There are **two flavors** and
they validate different things — check which exists (`ls /home/pat/code/vllm-gfx1201-tidar-convert/zaya/tidar/out/`):
- **Qwen-LoRA proxy** (`out/tidar-qwen-lora/adapter_model.safetensors`) — a *standard transformer*
  with the TiDAR objective LoRA'd in. Exercises the **mask + decode loop + β sampler + coherence**
  end-to-end on a genuinely TiDAR-trained model, **cheaply**. Does NOT exercise CCA / the `cca.py`
  conv-evict fold (no CCA in Qwen). Use it FIRST to pin the mask off-by-ones and the loop.
- **ZAYA1-8B-Diffusion TiDAR checkpoint** (output of `train_tidar_zaya.py`) — the production target;
  the only flavor that exercises the full CCA QKV-producer + conv/`prev_hs` evict path.

## READ FIRST, in order
1. `docs/zaya/tidar-serving-design.md` — §9 is the progress checklist (source of truth); §1 the CCA
   architecture finding, §2 the algorithm, §3 the mask/two routes, §4 the runtime, §7 open questions.
2. This session's committed work: **branch `feat/tidar-serve`, commit `ca4da6b`** —
   `zaya/tidar/{tidar_mask,tidar_loop,gpu_validate}.py`, `zaya/tidar/triton_overlay/`, and the design
   docs. Read `tidar_loop.py` (the decode loop you will port) and its header.
3. Memory `[[tidar-diffusion-pivot]]`.

## Already DONE + committed (do NOT rebuild — `ca4da6b`)
- **Structured TiDAR mask** (`tidar_mask.py`): allow-matrix + `additive_bias` + `MaskDescriptor` +
  square self-attn forms. 10/10 CPU tests.
- **Decode loop + β sampler + exact-KV/conv evict** (`tidar_loop.py`): `StubCCALM` (causal-conv
  `conv_states` + `prev_hs` state), fused single forward over `[prefix | S | R_0..R_{B-1}]` with
  per-replica **segmented** conv, `beta_verify`, `tidar_decode` (verify → commit k+bonus → evict
  B−k → re-draft). 6/6 CPU tests incl. **β=1 loop == greedy AR exactly** and **evict ==
  from-scratch recompute**.
- **Route B (triton paged) structured-mask path** (`triton_overlay/triton_unified_attention.py`): a
  `seq_mask` gate that un-masks query-query keys (`key_rel_pos ∈ [0, qq_bias_stride_0)`) so
  `additive_bias` defines the replica block's **bidirectional** attention; prefix stays causal;
  `USE_QQ_BIAS=False` byte-identical to stock. GPU SDPA-equivalence + tank-check ALL PASS.
- **Route A (attn_hip contiguous)**: the optional square `mask_bias` arg on `feat/attn-hip`'s
  `flash_prefill` (that effort's branch; additive/null-safe). GPU-validated (`gpu_validate.py` Part C).
- **GPU gate** `gpu_validate.py` Parts A–D: ALL PASS, 1-card lease, cos 0.999997–0.999999, ≤7 bf16-ULP.

## Settled findings — do NOT rediscover (from the 2026-06-27/28 session)
- **`sampling_causal = True` is CONFIRMED.** The convert session's authoritative training mask
  (`/home/pat/code/vllm-gfx1201-tidar-convert/zaya/tidar/tidar_objective.py`, `build_tidar_mask`)
  attends clean/AR rows strictly causally (`bias[:S,:S]`, `j<=i`). Matches our serving default.
- **`replica_offset` is NOT pinned by training, and that is expected.** The training mask is
  **block-granular** (a mask block sees clean tokens of *strictly-earlier whole blocks*, bidirectional
  within its own block) and has **no replicas, no partial within-block acceptance, no bonus token** —
  those are **inference-only** speculative-decoding constructs. So you cannot "confirm" `replica_offset`
  by reading training source. **Pin it EMPIRICALLY via the β=1 coherence gate (NEXT step 2):** β=1
  TiDAR decode must equal plain AR-greedy token-for-token; a wrong offset / bonus handling **diverges
  there**. Default `replica_offset=0`; if coherence diverges, try the bonus-fold variants in §7.1.
  ⚠ This is **losslessness-critical** (§7.1): do not trust ANY throughput number before β=1 coherence
  is token-identical.
- The **training layout** (doubled `[clean | mask]`) ≠ the **serving layout** (`[prefix | S |
  R_0..R_{B-1}]`). They are different constructions for the same model — don't conflate.
- CCA is a **QKV producer** feeding a STANDARD vLLM attention (`self.attn`, triton_attn on RDNA4);
  the mask goes in the attention backend, the evict reuses `cca.py:_decode_verify_spec`. See §1.

## NEXT (in order)
1. **[DONE 2026-06-28] Loader.** `zaya/tidar/serve_loader.py` loads the real ZAYA checkpoint
   (`pat883/zaya1-8b-tidar-experts`, full-ft-all, block_size=4) via the Zyphra fork
   (build-from-config + `load_state_dict`, 0 missing/0 unexpected). The Qwen-LoRA proxy was SKIPPED —
   the real ZAYA checkpoint exists and loads cleanly, so the full CCA path was exercised directly.
   ⚠ Loader MUST pass `attn_implementation="eager"` (the mask patch only works under eager; SDPA drops
   the injected bias). Runs on CPU (17.7 GB bf16 fits host RAM, not a 16 GB card); fork venv at
   `/home/pat/code/.venv-zaya-fork` ([[zaya-transformers-fork-reference]]).
2. **[DONE 2026-06-28] β=1 COHERENCE GATE PASSED.** `zaya/tidar/coherence_gate.py` GATE B: β=1 TiDAR
   decode == AR-greedy token-for-token on 4 prompts (varied tokens, not just `= = =`). Pins
   **`replica_offset=0` + `sampling_causal=True`**. The lossless loop is **two-forward** (verify vs a
   causal forward; diffusion drafts from a separate block forward) — the design's own `tidar_loop.py`
   form. **NEW FINDING (GATE A):** the *fully-fused* single-forward (verify + B² mask replicas in one
   pass) **contaminates the verify rows via a sequence-global op** (NOT attention) and is NOT viable on
   ZAYA — production must keep verify isolated from the mask-replica scratch (refutes the one-forward
   shortcut, `tidar_loop.py` test E). See design §9 + §7.5/§1.1.
3. **[NEXT] Wire the mask into the real attention backend.** Route A (`attn_hip` square `mask_bias`,
   contiguous) for the first real run; **Route B** (`triton_overlay` `qq_bias` gate, paged) for
   production. Land the mask/metadata in `cca_attn.py` (the CCA attention metadata builder).
4. **Fold the loop onto the real CCA path (ZAYA only).** Decode loop / β sampler / evict onto
   `cca.py:_decode_verify_spec` (`num_spec` → `block_len`), rolling back `conv_states` + `prev_hs` by
   `block_len − k`. Assert accepted-prefix state == a from-scratch recompute (the real-model analogue
   of `IncrementalKVConv`). Confirm §1.1 / §7.5: the diffusion fine-tune kept `conv_qk` causal.
5. **Throughput.** Accepted-tokens-per-forward + tokens/s; targets ≈ **4.6× (β=1) / 7.7× (mixed-β)**
   per the Zyphra ZAYA blog (≈7.45–8.25 accepted/forward). Treat as corroboration, NOT a hard gate.
6. **§31g FULL-cudagraph-capture** the single forward (reuse `zaya/dflash/` tooling +
   `[[profiler-bypasses-cudagraph-replay]]`: measure with the DEBUG dispatch probe + real throughput,
   NOT launch-count under the profiler). Capture sizes `{block_len·(1+block_len)}`; carried `p_diff`
   and the bias as persistent static buffers (§5).
7. **paged-KV in `attn_hip`** (Route A production) — coordinate with the `feat/attn-hip` owner; until
   then Route B (triton paged `qq_bias` gate) is the paged route.
8. **β<1 mixed sampling** for extra speed once β=1 is proven lossless — needs the carried `p_diff`
   (prior step's mask-block logits) plumbed forward (§4.3/§7.3); `beta_verify` already accepts it.

## How to run (host torch is broken — missing libmpi; use the container venv)
- CPU loop/mask tests (no lease):
  ```
  docker run --rm -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= \
    -v /home/pat/code/vllm-gfx1201-tidar-serve/zaya/tidar:/work -w /work \
    --entrypoint bash vllm22-w4a8:dflash-rxf -lc \
    'source /app/.venv/bin/activate && python -m pytest -q test_tidar_mask.py test_tidar_loop.py'
  ```
- GPU work (mask gate, real-model load, coherence): every job via `scripts/gpu-lease.sh -n 1 -- …`
  (TP=1, ONE card). Container `vllm22-w4a8:dflash-rxf` (stub/loop) or `:combined` (attn_hip rebuild +
  `gpu_validate.py`). Mount the warm Triton cache
  `/home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton`. The existing GPU gate
  `zaya/tidar/gpu_validate.py` documents the exact container invocation in its header.
- Reference vLLM (READ-ONLY): `/home/pat/code/_vllm_ref_combined`. The Route-B kernel hook lives at
  `vllm/v1/attention/ops/triton_unified_attention.py` (`qq_bias`) — our gated copy is
  `zaya/tidar/triton_overlay/`.

## Protocol (CLAUDE.md — MANDATORY)
- Work in worktree `feat/tidar-serve` (`/home/pat/code/vllm-gfx1201-tidar-serve`). Commit on this
  branch only, **never `main`**; coordinate any `attn_hip` edits with the `feat/attn-hip` owner.
- The `feat/tidar-convert` worktree is a separate effort: **READ-ONLY** (e.g. consult
  `tidar_objective.py`, `TIDAR_LOOP.md`, `tidar_conversion_design.md` for how training built the mask
  / how the checkpoint expects positions) — do NOT edit it or do training/data work.
- EVERY GPU job through `gpu-lease.sh -n 1`. Never ask a human for a card / never poll rocm-smi.

## DON'T
- Don't trust throughput before the β=1 coherence gate is **token-identical** to AR greedy.
- Don't "confirm" `replica_offset` from training source — pin it empirically (step 2).
- Don't redo the paper reads / re-derive the mask; don't reopen DFlash AR-spec (dead end); don't do
  conversion training (other session); don't edit the `feat/attn-hip` kernel's existing behaviour.
