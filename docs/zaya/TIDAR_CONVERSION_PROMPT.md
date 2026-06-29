# Session prompt — convert AR ZAYA1-8B into a TiDAR diffusion/AR-hybrid model (training)

You are continuing the Zyphra/ZAYA work on AMD RDNA4 (gfx1201). We are PIVOTING off train-our-own
DFlash speculative decoding (a confirmed dead end on this architecture) and onto **TiDAR** — a fused
single-model diffusion-draft + AR-talk approach. THIS session owns the **AR→TiDAR conversion training**:
fine-tune the existing autoregressive ZAYA1-8B into the TiDAR form, producing a checkpoint. A SEPARATE
session owns the **serving / kernel / cudagraph path** (TIDAR_SERVING_PROMPT.md) — do NOT do
attention-backend / decode-loop / cudagraph work here; stay in the training/data/recipe layer.

## Why the pivot (read first)
We tried for a long campaign to make a separate DFlash drafter + AR verify win on ZAYA1-8B (CCA hybrid).
It does not: held-out acceptance plateaus ~1.30 (the drafter only generalises pos0 — a structural
single-pass-mask cap, NOT exposure bias; Draft-OPD and dual-pass T=2 both moved it <0.01), and AR-spec
loses to no-spec on wall-clock even with the entire forward FULL-cudagraph-captured. Zyphra then shipped
**ZAYA1-8B-Diffusion-Preview** (https://www.zyphra.com/our-work/zaya1-8b-diffusion-preview): a
discrete-diffusion CONVERSION of AR ZAYA1-8B that KEEPS CCA, does 16 tokens/step, 4.6× (lossless) /
7.7× (mixed-logits), and explicitly beats "EAGLE or dFlash" by fusing speculation+verification into one
forward pass. The insight our own diagnosis kept circling — "the DFlash drafter is really a single-pass
masked-diffusion head; the separate drafter + separate verify is the overhead" — is exactly what TiDAR
removes by making the BASE model do both. So instead of training a separate drafter, we convert ZAYA
itself.

References (all we have — no public diffusion checkpoint or training code):
- **TiDAR** (the method+training): arXiv 2511.08923 "Think in Diffusion, Talk in Autoregression" (NVIDIA).
- **CCA** (the attention ZAYA keeps): arXiv 2510.04476 "Compressed Convolutional Attention" (Zyphra) —
  compressed-latent attention + conv; cheap at prefill, which is why it suits TiDAR ("decode becomes
  prefill"). It is NOT a pure SSM.
- Blog confirms it is a **conversion** (fine-tune of an AR LLM), the first MoE diffusion model converted
  from an AR LLM, **trained on AMD** — i.e. feasible on this hardware.

## TiDAR training recipe (what to reproduce) — READ THE PAPER for specifics
TiDAR fine-tunes an AR model with a STRUCTURED attention mask so one forward pass does both:
- **AR "talk"**: prefix + draft tokens attend **causally**; standard next-token AR loss (this preserves
  the AR quality — the "talk in autoregression" half).
- **Diffusion "think"**: a block of **mask tokens** attends **block-bidirectionally** to the prefix;
  trained as **one-step** masked diffusion (all positions = [mask], single denoise step — NOT iterative)
  to predict the block in parallel (the draft).
The joint objective trains both heads in one pass over the structured mask. At inference the diffusion
half drafts and the AR half verifies via lossless rejection sampling (β-sampler; β=1 lossless 4.6×,
mixed 7.7×). Block size 4/8/16; ~7.45–8.25 accepted tokens / forward eval. READ arxiv.org/html/2511.08923v1
for the exact mask construction, loss weighting, the all-mask one-step diffusion training detail, and any
warmup/curriculum before designing the fine-tune.

## What TRANSFERS from the DFlash effort (reuse, don't rebuild)
- The training pipeline + infra: `zaya/dflash/train_cca_drafter.py` (pure-torch ZAYA/CCA training mirror,
  bf16 vocab matmul, save-epochs), the capture/corpus tooling (build_curated_corpus.py, build_eval_splits.py,
  capture_drafter_data.py), `measure_acceptance.py`, AMD training setup (container, gpu-lease).
- The curated corpus work (Dolly+OASST+CodeAlpaca+GSM8K+WritingPrompts) and held-out eval splits.
- Access to AR ZAYA1-8B weights (/models/ZAYA1-8B-fp8 in-container; bf16 embed/lm_head) and the CCA model
  code (zaya.py, cca.py overlays).
- OBSOLETE: the separate-drafter architecture (cca_drafter_model.py), the seed-fold / serve-convention
  capture, Draft-OPD — TiDAR has no separate drafter, so the off-policy/seed machinery goes away. Keep the
  data + the ZAYA training mirror; drop the drafter-specific parts.

## What is NEW (the conversion work)
1. The **structured-mask training objective** (causal AR-talk + block-bidirectional one-step-diffusion
   think) applied to the FULL ZAYA1-8B (8B MoE), not a small drafter head.
2. The mask-token / vocabulary handling (a [mask] token; possibly a draft-vocab as in DFlash).
3. A **conversion fine-tune** (start from AR ZAYA weights, not from scratch) — scope compute carefully:
   full-8B-MoE fine-tune on AMD is a large job; validate the recipe on a SHORT/small run first.
4. Produce a checkpoint the serving session can load (define the format with that session).

## Risks / open questions to resolve from the paper
- Compute cost of an 8B-MoE conversion fine-tune on the available AMD cards — budget + checkpoint/resume.
- Whether CCA needs any change for the bidirectional block (CCA is causal-ish; the block-bidirectional
  mask over a compressed-latent + conv attention may need care — coordinate with the serving session,
  which owns the inference mask).
- Data: does conversion need the original pretraining mixture, or does a SFT-scale corpus suffice? (The
  paper's conversion data scale is the key unknown — read it.)

## Protocol (CLAUDE.md — MANDATORY)
- Dedicated feature-branch WORKTREE (never shared main, never `git switch` shared checkout). Suggest
  `git worktree add -b feat/tidar-convert ../vllm-gfx1201-tidar-convert main`.
- EVERY GPU job via `scripts/gpu-lease.sh` (training may want `-n 2`); never ask a human / never poll
  rocm-smi; queue behind other leases. ⚠ concurrent heavy GPU loads have caused GPU-hang HW exceptions —
  queue into genuine gaps. Use real disk under /home/pat/code (never tmpfs) for checkpoints.
- Training container = `vllm22-w4a8:dflash-rxf` (entrypoint bash + venv) or the existing training setup;
  warm cache + .env (HF_HOME). Full container protocol in CLAUDE.md.

## First steps
1. Read the full TiDAR paper (2511.08923), especially the TRAINING section: the structured-mask loss,
   the all-mask one-step-diffusion objective, conversion data scale, and any curriculum.
2. Read the CCA paper (2510.04476) + cca.py to understand what the block-bidirectional mask means for
   CCA's compressed-latent + conv attention (coordinate the mask semantics with the serving session).
3. Write a conversion design note (docs/zaya/): objective, mask, data, compute budget, checkpoint format.
4. Adapt the ZAYA training mirror (train_cca_drafter.py) into a full-model TiDAR conversion fine-tune;
   strip the separate-drafter parts.
5. Run a SHORT validation conversion (few steps / small data) to prove the loss + mask wire up and the
   one-step-diffusion head learns; then scope the full run.
6. Hand a checkpoint to the serving session for end-to-end throughput/quality measurement.

## DON'T
- Don't build the serving attention-mask / decode-loop / cudagraph path (other session owns it).
- Don't reopen the DFlash separate-drafter / Draft-OPD direction (superseded).
- Don't launch a giant full-8B conversion before validating the recipe on a short run.
- Don't guess the TiDAR training recipe — read the paper.
