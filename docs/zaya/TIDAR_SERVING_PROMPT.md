# Session prompt — build the TiDAR serving path for ZAYA1-8B-Diffusion on RDNA4 (gfx1201)

You are continuing the Zyphra/ZAYA inference work on AMD RDNA4 (gfx1201, TP=1). We are PIVOTING
off autoregressive speculative decoding (the DFlash effort) and onto **serving a TiDAR-style
diffusion/AR-hybrid model** — Zyphra's ZAYA1-8B-Diffusion. THIS session owns the **serving / kernel /
cudagraph path** (the inference engine side). A SEPARATE session owns the **AR→TiDAR conversion
training** (producing the checkpoint) — do NOT do training/data work here; stay in the vLLM
attention-backend + decode-loop + cudagraph layer.

## Why the pivot (read first)
We spent a long campaign trying to make train-our-own DFlash speculative decoding win on ZAYA1-8B
(CCA hybrid). It does not, and the reason is structural, now confirmed both empirically and by Zyphra:
- AR speculative decoding on a CCA/recurrent-state target is taxed by a serialized ~15 steps/s decode
  ceiling + per-position state rollback on the verify path. Even with the ENTIRE spec forward
  FULL-cudagraph-captured this session (verify capture −25% wall, drafter FULL capture −16% more,
  acceptance held at 1.357), AR-spec still loses to no-spec on wall-clock (3-way: no-spec ~261s vs
  verify-only ~365s same-window). Held-out acceptance is capped ~1.30 (drafter only generalises pos0).
- Zyphra shipped **ZAYA1-8B-Diffusion-Preview** (https://www.zyphra.com/our-work/zaya1-8b-diffusion-preview):
  a discrete-diffusion conversion of AR ZAYA1-8B that KEEPS CCA, generates 16 tokens/step, 4.6×
  (lossless) / 7.7× (mixed-logits) over AR — and EXPLICITLY frames it as "single model speculator and
  verifier ... speculation in the same forward pass as verification, reducing overhead vs ... EAGLE or
  dFlash." i.e. they obsoleted our exact bet with the fused single-model version.

The two technical references the blog gives (all we have — there is no public checkpoint yet):
- **TiDAR** (the method): arXiv 2511.08923 "Think in Diffusion, Talk in Autoregression" (NVIDIA).
- **CCA** (the attention): arXiv 2510.04476 "Compressed Convolutional Attention" (Zyphra). NB: CCA is
  compressed-latent ATTENTION + conv (8× KV / 1.7× prefill), NOT a pure SSM — it has an exact-ish KV
  cache + a small conv recurrent component. CCA suits TiDAR because TiDAR turns decode into prefill and
  CCA is cheap at prefill.

## TiDAR inference algorithm (what you are building a server for)
One model, ONE forward pass per block. Sequence laid out `[prefix | drafts_from_last_step |
mask_tokens_for_next]` with a STRUCTURED attention mask:
- prefix + drafts_from_last → **causal** (AR verify),
- mask tokens → **block-bidirectional** (one-step masked diffusion draft; all-mask, NO iterative denoise).
Each forward pass simultaneously (a) rejection-samples the previous step's drafts against the AR joint
dist (lossless) — variable #accepted, (b) pre-drafts the next block in parallel conditioned on EVERY
possible acceptance length, (c) selects the pre-draft matching the accepted length. **Exact KV cache**
(store causal tokens, evict on rejection, never recompute). β-sampler: `argmax(β·AR + (1−β)·diff)` →
β=1 lossless (4.6×), mixed (7.7×). Block size 4/8/16. ~7.45–8.25 tokens / network-forward-eval.
READ THE FULL TiDAR PAPER (arxiv.org/html/2511.08923v1) for the exact mask shape
`(block_len × (1+block_len), max_seq_len+block_len)`, the rejection-sampling math, and the
pre-draft-conditioned-on-all-outcomes trick before implementing.

## What TRANSFERS from this session (reuse, don't rebuild)
- **§31g cudagraph capture is the high-value carryover and is SIMPLER here** (TiDAR is one forward/block,
  no separate drafter or rollback). Reuse: explicit `cudagraph_capture_sizes={(qlen)*k}` + raise
  `max_num_seqs ≥ largest size` (the uniform-decode FULL-capture bound); the FULL-capture proposer-style
  overrides; the persistent static-address buffer pattern (`_cca_seed_buf`) for any per-step state a
  captured graph reads. Tooling in `zaya/dflash/`: profile_spec_step.sh, dispatch_probe.sh,
  capture_mode_test.sh, throughput_capsizes_ab.sh, analyze_launch_count.py, attribute_launches.py.
- **Methodology memories** (CRITICAL): [[profiler-bypasses-cudagraph-replay]] — torch.profiler bypasses
  cudagraph replay, so launch-COUNT under the profiler CANNOT measure cudagraph effectiveness; use a
  DEBUG dispatch probe (gpu_model_runner.py:4131 "Running batch with cudagraph_mode") + real throughput.
- The CCA HIP path (`zaya/overlay/.../mamba/cca.py`, `cca_prefill_qk`) and the structured-grid verify
  rewrite (§31b) are reference for driving CCA over a static grid under cudagraph.
- OBSOLETE (do not carry): the DFlash drafter (`cca_drafter_model.py`), separate-verify HIP path, the
  M5 'all'-mode rollback-as-spec machinery, the OPD drafter training.

## What is NEW (the serving work)
1. **Structured attention mask** in the attention backend: the `[prefix | drafts | mask]` causal +
   block-bidirectional pattern, over CCA's compressed-latent attention. This is the core new primitive.
   Coordinate with the `feat/attn-hip` worktree (/home/pat/code/vllm-gfx1201-attn-hip/attn_hip/) — a
   rocwmma flash-attention kernel; a structured-mask variant may belong there. Read its NOTES.md for the
   gfx11→gfx12 WMMA fragment-layout trap before writing any WMMA (use rocwmma; gfx12 layout is
   row=(lane>>4)*8+e).
2. **The single-pass decode loop**: draft-verify-via-rejection-sampling + pre-draft-next-block, exact-KV
   evict-on-reject, in the vLLM v1 model-runner / a TiDAR proposer-equivalent.
3. **The β rejection sampler** (lossless vs mixed-logits).
4. **FULL-cudagraph-capture the single TiDAR forward** (reuse §31g).

## Blocker / how to make progress without weights
There is NO public ZAYA1-8B-Diffusion checkpoint (preview only). Options, pursue in parallel:
- Build + unit-test the structured-mask attention and the decode loop against a STUB / random-weight
  ZAYA-CCA (correctness of masks/loop/KV-evict/sampler is weight-independent; coherence is not).
- A real checkpoint comes from the **conversion-training session** (see TIDAR_CONVERSION_PROMPT.md) or a
  future Zyphra release — design the loader to swap either in.

## Protocol (CLAUDE.md — MANDATORY)
- Dedicated feature-branch WORKTREE for edits (never shared main, never `git switch` shared checkout).
  Suggest `git worktree add -b feat/tidar-serve ../vllm-gfx1201-tidar-serve main` (or branch off
  feat/zaya-dflash to inherit the §31g overlays + tooling).
- EVERY GPU job via `scripts/gpu-lease.sh -n 1 -- …` (TP=1); never ask a human / never poll rocm-smi;
  queue behind other leases. ⚠ Booting concurrently with another agent's GPU job has caused a "GPU Hang
  HW exception" mid-weight-load — queue into genuine gaps.
- Container = `vllm22-w4a8:dflash-rxf` (has ZAYA+CCA+overlays); warm cache + .env in the worktree
  (HF_HOME + VLLM_HOST_TRITON_CACHE=/home/pat/code/.triton-cache-zaya-dflash). Test the LATEST image,
  mount the warm Triton cache. Full container protocol in CLAUDE.md.
- Reference vLLM source (read-only): /home/pat/code/_vllm_ref_combined.

## First steps
1. Read the full TiDAR paper (2511.08923) — nail the structured mask shape, the rejection-sampling +
   pre-draft-conditioned-on-all-outcomes algorithm, and the exact-KV evict logic.
2. Read the CCA paper (2510.04476) + the existing cca.py to know CCA's attention/KV structure you must
   layer the structured mask onto.
3. Design the serving path (write a design note in docs/zaya/): mask construction, decode loop, sampler,
   KV-evict, and where each lives in the vLLM v1 stack; what reuses §31g capture.
4. Stand up the structured-mask attention against the CCA path; validate the mask + decode loop on a stub.
5. FULL-cudagraph-capture the single forward (reuse §31g); measure launches/step + throughput vs the AR
   baseline once a real checkpoint exists.

## DON'T
- Don't do AR→TiDAR conversion training / data work (other session owns it).
- Don't reopen DFlash AR-spec (confirmed dead end — §31e/§31g in docs/zaya/zaya-dflash-plan.md).
- Don't commit (uncommitted overlays; fold at a future M-stage when asked).
- Don't web-search-substitute the papers without reading them; don't guess the TiDAR algorithm.
