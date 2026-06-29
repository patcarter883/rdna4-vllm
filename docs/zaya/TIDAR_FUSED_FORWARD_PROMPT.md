# Session prompt — TiDAR serving, FUSED single forward on the live cca.py runner (the ~2.4× lever)

You are resuming the **serving** side of the TiDAR pivot (Zyphra ZAYA1-8B-Diffusion on RDNA4 / gfx1201).
The checkpoint serves, the whole weight-independent stack is GExoneU-validated, throughput is measured, and the
remaining speedup lever is **fully root-caused and de-risked**. Your job is the ONE remaining build:
realize the **fused single forward with a segmented conv** on the live `cca.py` runner to turn the
measured **1.52× single-forward** into the steady **~2.4×**. Do NOT re-litigate the pivot, re-read the
papers, re-derive the converter, or re-run the fused-forward diagnostics — they're done (see below).

## Where we are (DONE — do NOT rebuild) — all on branch `feat/tidar-serve`, committed @ `8b5a466`, PR #4
- **Servable build:** `pat883/zaya1-8b-tidar-experts` serves on **vLLM TP=2 bf16** via a byte-verified
  HF-fork→vLLM weight converter (`zaya/tidar/hf2vllm_map.json`; 2483 tensors matched). Servable dir
  `/home/pat/code/zaya1-8b-tidar-serve`; compose profile `zaya-tidar`. 0 missing/0 unexpected, coherent,
  **~27 tok/s AR baseline** (the speedup denominator). Memory: `[[tidar-checkpoint-servable-on-vllm]]`.
- **Steps 3–4, proposer, §31g** — all GPU-validated (`gpu_validate.py` Parts E/F/G): mask→triton_attn
  carrier+hook (`tidar_attn_metadata.py`, `cca_attn.py`), evict folded onto `cca.py` (`cca_evict_gate.py`,
  evict==recompute max|Δ|=0), proposer/runner wiring (`tidar_proposer.py` + guarded hook in `zaya.py`),
  cudagraph capture.
- **Throughput measured:** two-forward TiDAR **1.12×**; single-forward **1.52×** (up to 2.4–2.7× when
  drafts land), lossless in fp32. Scripts: `throughput_tidar.py`, `single_forward_tidar.py`.

## Settled findings — do NOT rediscover (design §9, §1.1, §7.5, §7.6)
- **§7.5 was a MISDIAGNOSIS.** The fused single forward over `[committed | S=prev_drafts | R_0..R_{B-1}]`
  is **mathematically lossless** — fp32 is bit-identical fused-vs-causal at all 40 layers
  (`bisect_fusion.py`). The bf16 "contamination" was grouped-MoE-GEMM batching noise; ZAYA MoE routing is
  strictly per-token. **Verify (S rows) needs NO isolation from the mask-replica scratch.**
- **bf16 caveat:** β=1 isn't strictly bit-lossless in bf16 (rare borderline-argmax flips on low-entropy
  prompts) — compute the few **S verify-row logits in fp32** (cheap) for strict losslessness.
- **§7.6 replica positions pinned:** replica `R_r` predicts at positions **`[L+r, L+r+B-1]`** (matching
  `block_predict` after r accepts), NOT all at `[L+B, L+2B-1]` (that's correct only for r=B). Already
  applied in `single_forward_tidar.py`/`replica_diag.py`.
- **THE ROOT CAUSE of low single-forward acceptance = §1.1 causal conv (`replica_diag.py`, token-level).**
  In the flat fused sequence each replica's leading `total_padding=2` tokens read the **S draft tokens**
  through `conv_qk` (the conv is causal over sequence order and IGNORES the attention mask). Confirmed:
  segmenting the conv flips the token-match pattern `[0,0,1,1]`→`[1,1,0,0]` (`segmented_fused_tidar.py`).
  The **bonus token** (k vs k+1 context) is a smaller secondary residual, not the main issue.
- **A CPU construction-trick to fake the segmented conv is too finicky** (`segmented_fused_tidar.py`
  fixed the leading 2 tokens but broke the trailing 2). The CLEAN fix is the real segmented conv that
  **`cca.py:_decode_verify_spec` already implements** for spec decode (per-segment conv with cached state).

## NEXT (the build — algorithm has NO remaining unknowns)
1. **Wire the fused single forward onto the live cca.py runner (TP=2).** Per TiDAR step, run ONE forward
   over `[committed(KV-cached) | S=prev_drafts(block_len) | R_0..R_{B-1}]` where:
   - the **structured mask** rides the standard `self.attn` via the step-3 carrier+hook
     (`tidar_attn_metadata.build_tidar_mask_meta` + `set_active_tidar_mask`);
   - **`block_len` maps to `cca.py`'s `num_spec`** (`get_current_vllm_config().num_speculative_tokens`);
   - the **conv is SEGMENTED**: each replica `R_r`'s conv segment is `[committed + first-r-drafts | mask*block_len]`
     — extend/reuse `cca.py:_decode_verify_spec`'s per-segment conv+state machinery (it already does exactly
     this for the verify candidates; the replicas are additional segments conditioned on r accepts);
   - replica **positions `[L+r, L+r+B-1]`**; verify **S-row logits in fp32**.
2. **Verify (S rows) + select next drafts (R_k):** β-sample the S rows (`tidar_proposer.verify_commit`
   → `beta_verify`, β=1), commit k+1 (k accepted + bonus), then read replica `R_k`
   (`select_next_drafts_row_range`) as the next block's drafts. Evict the rejected KV+conv via the
   existing `_decode_verify_spec` rollback (step-4 path).
3. **GATES (must pass):**
   - **Lossless:** the served committed stream == plain AR-greedy token-for-token (fp32 verify rows).
   - **Drafts correct:** the live R_k == a fresh `block_predict([committed+k], mask*B)` (the segmented conv
     should now make this match — the thing the CPU trick couldn't).
   - **Throughput:** ~1 forward/step → measure tok/s vs the **27 baseline**; expect acceptance to recover
     toward the two-forward's ~1.40 ⇒ **~2.4× (≈65 tok/s)**. Use the `[[profiler-bypasses-cudagraph-replay]]`
     dispatch-probe + real throughput, NOT launch-count under the profiler.
4. **Then:** §31g FULL-capture the fused forward at fixed `q_len=block_len·(1+block_len)` (Part G already
   validated the carrier/capture mechanics); β<1 mixed sampling for extra speed (needs the carried `p_diff`).

## How to run
- **Live serve (TP=2):** `ZAYA_MODELS_DIR=/home/pat/code HF_HOME=/home/pat/.cache/huggingface
  VLLM_HOST_TRITON_CACHE=/home/pat/code/vllm-gfx1201/.triton-cache-combined
  scripts/gpu-lease.sh -n 2 --detach --name zaya-tidar -- docker compose --profile zaya-tidar up -d`
  (warm cache reused; both cards leased so no concurrent-compile race). Tear down: `docker compose -p
  lease-zaya-tidar down`.
- **CPU losslessness/draft gates** (HF fork model on `model_latest.pt`, no lease): the
  `zaya/tidar/*.py` harnesses run via `vllm22-w4a8:dflash-rxf` + the fork venv `/opt/zaya-fork-venv`
  (`/home/pat/code/.venv-zaya-fork`) — see any of `coherence_gate.py` / `single_forward_tidar.py` headers
  for the exact `docker run` line. `--ckpt` = the HF snapshot dir under the HF cache (has `model_latest.pt`).
- **GPU validation:** `gpu_validate.py` under `scripts/gpu-lease.sh -n 1` (`vllm22-w4a8:combined`).

## DON'T
- Don't re-attempt the CPU construction-trick for the segmented conv (`segmented_fused_tidar.py` proved
  it's the wrong vehicle) — use the real `cca.py:_decode_verify_spec` segmented conv on the live runner.
- Don't isolate verify from the scratch (the §7.5 reason is void — verify is bit-exact). Don't re-pin
  `replica_offset=0`/`sampling_causal=True`/the replica positions/the §7.6 convention (all settled).
- Don't trust bf16 for the strict-lossless gate (use fp32 verify rows). Don't measure cudagraph
  effectiveness with profiler launch-count.
- Protocol: work in worktree `feat/tidar-serve`; commit on that branch only, never `main`; EVERY GPU job
  via `scripts/gpu-lease.sh`. The fused-forward analysis + servable build are committed @ `8b5a466` / PR #4.
