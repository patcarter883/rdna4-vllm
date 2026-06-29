# Session prompt — TiDAR serving, STEP 3+ (β=1 coherence gate already PASSED; now wire the real path)

You are resuming the **serving / kernel / cudagraph** side of the TiDAR pivot (Zyphra
ZAYA1-8B-Diffusion). A real TiDAR checkpoint exists, it loads, and **β=1 losslessness is PROVEN on
it**. Do NOT re-litigate the pivot, re-read the papers, or redo the loader / β=1 gate. This phase
wires the proven loop onto the real attention backend + CCA KV path, then measures throughput.

This session owns serving only. AR→TiDAR conversion training is the SEPARATE `feat/tidar-convert`
effort — its worktree is **READ-ONLY cross-reference**, never edit it, never train there.

## Where we are (DONE — do NOT rebuild)
The checkpoint **`pat883/zaya1-8b-tidar-experts`** (full-ft-all ZAYA1-8B-Diffusion, `block_size=4`,
mask_id 262147, step 999) is downloaded to the warm HF cache + loads cleanly.
- **Loader** `zaya/tidar/serve_loader.py`: builds ZAYA from the base config + `load_state_dict`
  (0 missing / 0 unexpected). Runs on CPU (17.7 GB bf16 fits the 91 GB host; will NOT fit a 16 GB
  card). ⚠ **MUST pass `attn_implementation="eager"`** — the mask patch is dropped under SDPA.
- **β=1 COHERENCE GATE PASSED** `zaya/tidar/coherence_gate.py` (GATE B): β=1 TiDAR decode ==
  AR-greedy token-for-token on 4 prompts (varied tokens, not just `= = =`). This **pinned
  `replica_offset=0` + `sampling_causal=True`** — the two §7.1 flags are now closed. The lossless
  loop is the **two-forward** form (verify vs a causal forward over `[committed|drafts]`; diffusion
  drafts from a separate `[committed|mask*B]` block forward, trainer's block bias).
- All the weight-independent primitives (mask, loop, β sampler, Route A attn_hip `mask_bias`,
  Route B triton `qq_bias` gate) were built + GPU-validated earlier (commit `ca4da6b`).

## Settled findings — do NOT rediscover
- **`attn_implementation="eager"` is mandatory** for the `zaya_mask_patch` create_causal_mask
  monkeypatch (ZAYA's SDPA path silently ignores the injected bias → model goes non-causal under
  appended tokens → the gate diverges). Confirmed: eager full-recompute AR == native cached
  `generate()` AR (`diag_ar.py` MATCH=True).
- **`replica_offset=0`, `sampling_causal=True`** — empirically pinned by the passing β=1 gate.
- **The fully-FUSED single-forward does NOT hold on ZAYA (§7.5/§1.1 materialized).** Folding verify
  (S rows) + the `block_len²` mask replicas into ONE forward CONTAMINATES the S verify rows
  (max|Δlogit|≈1.8; flips the argmax on some prompts). It is **NOT attention** (S is masked from the
  replicas) — it is a sequence-GLOBAL op (MoE aux-loss-free load-balancing or a global norm): the
  mere PRESENCE of the mask-replica scratch tokens shifts the real tokens' logits. The one-forward
  shortcut (`tidar_loop.py` test E) is **REFUTED for this checkpoint**. ⇒ **The production path must
  keep verification ISOLATED from the mask-replica scratch** — i.e. the cca.py KV-cached path where
  mask positions are never written to KV and the global op is per-token at compute time. (This is
  exactly what step 4 builds, and it is also what fixes the fusion contamination.)
- CCA is a QKV producer feeding a STANDARD vLLM attention (`self.attn`, triton_attn on RDNA4); the
  mask goes in the attention backend, the evict reuses `cca.py:_decode_verify_spec`. See design §1.

## READ FIRST, in order
1. `docs/zaya/tidar-serving-design.md` — **§9** (progress + the two findings above, source of truth),
   §1 (CCA architecture), §2 (algorithm), §3 (mask / two routes), §4 (runtime), §7 (open questions).
2. This session's code in `zaya/tidar/`: `serve_loader.py`, `coherence_gate.py`, `tidar_loop.py`,
   `tidar_mask.py`, `zaya_mask_patch.py`, and the diagnostics `diag_causality.py` / `diag_ar.py`.
3. Memory `[[tidar-diffusion-pivot]]` and `[[zaya-transformers-fork-reference]]` (fork venv recipe).

## NEXT (in order)
3. ✅ **DONE (2026-06-28).** Structured mask WIRED into the real standard-attention backend +
   GPU-validated (`gpu_validate.py` Part E, A–E ALL PASS). New `zaya/tidar/tidar_attn_metadata.py`
   (`build_tidar_mask_meta` → `TiDARMaskMeta` with Route-B `qq_bias` + optional Route-A square
   `mask_bias`; module-level active-mask carrier; `wrap_unified_attention`/`install_tidar_attn_hook`
   backend hook). `cca_attn.py` `CCAAttentionMetadata` gained `tidar_mask` (null-safe). The wired path
   (builder→carrier→hook→`unified_attention`, NO explicit qq_bias) == boolean-masked SDPA (cos
   0.999997–0.999999, ≤4 ULP) AND byte-identical to the explicit-qq_bias path; carrier-cleared ==
   stock byte-identical. Single-sequence (batched-decode + Route-A attn_hip wiring deferred to step 4).
   See design §9. *Was:* Land the TiDAR mask/metadata in the CCA attention metadata builder
   (`cca_attn.py`); Route A (`attn_hip` square `mask_bias`) / Route B (`triton_overlay` `qq_bias`).
4. ✅ **DONE (2026-06-28).** Folded the decode loop / β sampler / evict-on-reject onto the real
   `cca.py:_decode_verify_spec` KV+conv-state rollback. `num_spec` maps to TiDAR `block_len`; the
   `(1+num_spec)` candidate-window conv + per-spec-position rollback IS the evict (append the whole
   block, next step reads the accepted-prefix column `num_accepted-1` — no truncation), == the
   `IncrementalKVConv.commit_block` contract. New **`zaya/tidar/cca_evict_gate.py`** drives the
   checkpoint's real layer-0 `ZayaCCAProjection` and asserts **evict-on-reject == from-scratch
   recompute of the accepted token stream, max|Δ|=0.0** on the real ZAYA conv/`prev_hs`
   (`k_accept` 0..block_len × prefix {4,16}), kept ISOLATED from the mask-replica scratch (§7.5).
   **Conv-causality confirmed** on the real conv (appending tokens doesn't change earlier positions,
   max|Δ|=0.0): the diffusion FT kept `conv_qk` CAUSAL at the K=2 boundary (mask patch only touches
   `create_causal_mask`, never the CCA conv) — **no cca.py non-causal branch needed**. `coherence_gate.py`
   GATE B β=1==AR-greedy still PASS after the fold; `test_tidar_{loop,mask}.py` 16/16 green. Additive
   null-safe `num_spec→block_len` doc anchor in `_decode_verify_spec` (rollback math unchanged;
   `num_spec==0` byte-identical). *Was:* Decode loop / β sampler / evict onto `_decode_verify_spec`;
   assert accepted-prefix state == from-scratch recompute; confirm conv_qk causal at the K=2 boundary.
5. **Throughput.** Accepted-tokens-per-forward + tokens/s. Targets ≈ 4.6× (β=1) / 7.7× (mixed-β) per
   the Zyphra blog — corroboration, NOT a hard gate. NB current draft acceptance is modest (avg
   0.8–2.0 / 4) because this overfit checkpoint's diffusion drafts are weak; that's the throughput
   lever, not a losslessness concern.
6. **§31g FULL-cudagraph-capture** the single forward (reuse `zaya/dflash/` +
   `[[profiler-bypasses-cudagraph-replay]]`: DEBUG dispatch probe + real throughput, NOT launch-count
   under the profiler). Capture sizes `{block_len·(1+block_len)}`; carried `p_diff` + bias as static.
7. **paged-KV in attn_hip** (Route A production) — coordinate with the `feat/attn-hip` owner; until
   then Route B (triton paged `qq_bias`) is the paged route.
8. **β<1 mixed sampling** for extra speed once β=1 is proven lossless on the wired path — needs the
   carried `p_diff` (prior step's mask-block logits) plumbed forward (§4.3/§7.3). *(If draft acceptance
   needs lifting for the β<1 regime, an OPD-style on-policy distillation of the draft distribution is a
   possible lever here — but it is NOT a current step; flag the user before any such training run.)*

## How to run (host torch is broken — use the fork venv inside the container)
- Fork venv (Zyphra `transformers` + container torch 2.10, built per the prep) at host
  `/home/pat/code/.venv-zaya-fork`, mount at `/opt/zaya-fork-venv`. CPU coherence/diag (no lease):
  ```
  docker run --rm -e HF_HOME=/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
    -v /home/pat/code/.venv-zaya-fork:/opt/zaya-fork-venv \
    -v /home/pat/code/vllm-gfx1201-tidar-serve/zaya/tidar:/work -w /work \
    --entrypoint bash vllm22-w4a8:dflash-rxf -lc '/opt/zaya-fork-venv/bin/python coherence_gate.py'
  ```
- GPU work (mask wiring, attn backends, throughput): every job via `scripts/gpu-lease.sh -n 1 -- …`
  (TP=1). The 17.7 GB model does NOT fit one 16 GB card for a plain HF load — the **serving** path is
  paged-KV via vLLM (cca.py), which is the point of steps 3-4; for HF-forward checks stay on CPU or
  use 2 cards (`-n 2`, device_map — note the mask-patch global bias is single-device, so the simple
  monkeypatch does NOT compose with a 2-card device_map; CPU is the clean HF-forward path).
- Reference vLLM (READ-ONLY): `/home/pat/code/_vllm_ref_combined`; Route-B kernel hook at
  `vllm/v1/attention/ops/triton_unified_attention.py` (`qq_bias`); our gated copy in
  `zaya/tidar/triton_overlay/`.

## Protocol (CLAUDE.md — MANDATORY)
- Work in worktree `feat/tidar-serve` (`/home/pat/code/vllm-gfx1201-tidar-serve`). Commit on this
  branch only, **never `main`**; coordinate any `attn_hip` edits with the `feat/attn-hip` owner.
- The `feat/tidar-convert` worktree is READ-ONLY (consult `tidar_objective.py`, `train_tidar_zaya.py`,
  `check_zaya_mask.py` for how training built the mask / conv mode) — do NOT edit or train there.
- EVERY GPU job through `gpu-lease.sh -n 1`. Never ask a human for a card / never poll rocm-smi.

## DON'T
- Don't try to verify from the fully-fused single-forward — it's contaminated on ZAYA (use the
  two-forward / KV-cached split; see the settled finding).
- Don't load the model under SDPA (mask patch dropped) — `attn_implementation="eager"`.
- Don't re-pin `replica_offset` / `sampling_causal` (closed = 0 / True) or redo the β=1 gate / paper
  reads; don't reopen DFlash AR-spec; don't do conversion training (other session).
