# Contributing

Thanks for helping improve the gfx1201 (RDNA4) vLLM stack. This repo is mostly
**packaging** around upstream projects — read [`DIARY.md`](DIARY.md) first; it
explains why almost every odd-looking decision exists, and most debugging walls
already have a documented door.

## What lives where

| If you want to change… | Edit it… |
|---|---|
| how the image is assembled (wheel install, patches, W4A8 build) | `Dockerfile`, `Dockerfile.fromsource` |
| how it's served (profiles, flags, mounts, env) | `docker-compose.yml`, `.env.template` |
| the WNA16-MoE `tp_size` source fix | `patches/moe_wna16.py` |
| the W4A8-FP8-WMMA MoE kernel | `w4a8_fp8_wmma/` |
| the kernels/engine themselves (vllm / aiter / flash-attention) | **not here** — see "Upstream" below |

The three wheels are built from **separate forks**, not from this repo:
`patcarter883/vllm-gfx1201`, `patcarter883/aiter-gfx1201`,
`patcarter883/flash-attention-gfx1201` (each on a `gfx1201` branch). A change to a
kernel or to vLLM core goes there; this repo only *consumes* the resulting wheels
(via a GitHub Release) and bakes the small source patches.

## The one hard rule: ABI

Everything here is pinned to **gfx1201 + Python 3.12 + torch 2.10 + ROCm 7.14**
(the `rocm/vllm-dev:nightly-therock714` base). Native artifacts built against any
other torch/arch will fail to load (`ImportError: undefined symbol _ZN3c10...`).
Consequences for contributors:

- **Never commit a prebuilt `.so`/`.whl`.** Wheels live in a GitHub Release; the
  W4A8 kernel is compiled *inside* the image. `.gitignore` already blocks them.
- If you bump the base image or any pinned version, you must rebuild **all three
  wheels** from the forks and the W4A8 kernel — they are not independent.

## Dev workflow

1. **Fork + branch.** Branch from `main`; keep one logical change per PR.
2. **Validate compose** for any `docker-compose*.yml` / `Dockerfile*` change:
   ```bash
   docker compose config -q
   docker compose -f docker-compose.yml -f docker-compose.fromsource.yml config -q
   ```
3. **Build + smoke** on real gfx1201 hardware when you touch the image or runtime:
   ```bash
   docker compose --profile tp2-baseline up --build
   ./scripts/smoke.sh            # /v1/models + one completion
   ```
   The first boot pays the 15–30 min cold Triton compile (see DIARY "Act II").
4. **W4A8 changes** must pass the in-container kernel tests and an e2e generate;
   bisect with `VLLM_ROCM_USE_W4A8_FP8_WMMA=0` / `..._MOE=0` / `..._MOE_VERSION=0`
   to localize a regression (kernel numerics vs wiring vs baseline). See
   `w4a8_fp8_wmma/TASK6_HANDOFF.md` and the bench in `w4a8_fp8_wmma/bench_vs_triton.py`.
5. **Perf claims** must come from `test/bench_tp2.py` (warmup-generate then timed) on
   a **warm** Triton cache, and state the GPU layout — the published baseline is
   **298 dec / 1887 total tok/s** (heterogeneous TP=2, CU_NUM=56). Quote the config.

## Style

- Match the surrounding file: the Dockerfiles and compose carry **load-bearing
  comments** explaining *why* a flag/pin exists. If you change behavior, update the
  comment in the same diff. Don't drop a "why" comment to save a line.
- Shell scripts: `set -euo pipefail`, keep them runnable both inside and outside a
  container where it makes sense.
- Don't reflexively raise the W4A8 low-batch caps (`max_model_len`/`max_num_batched_tokens`)
  — they prevent a real MoE-apply OOM on 16 GB cards.

## What we'd love help with

- **Upstreaming the `moe_wna16` `tp_size` fix** to vLLM — it's a generic upstream bug
  (`RoutedExperts` exposes TP only via `moe_config.tp_size`), not gfx1201-specific. A
  clean PR there lets us drop the patch here.
- **Real vision support:** route the ViT off Torch-SDPA (the gfx1201 AOTriton
  block-(0,0,0) hazard) through flash_attn, so `--limit-mm-per-prompt` can be lifted.
- **W4A8 perf** in the mid-M regime where it trails Triton; a single-GPU `single`
  profile model that's known-good; FLA-GDN autotune configs that prewarm faster.

## Licensing & credit

This repo's glue is [Apache-2.0](LICENSE). By contributing you agree your changes
are licensed the same. Every upstream component keeps its own license — preserve
[`NOTICE`](NOTICE) and add credit there if your change pulls in new third-party work.
Sign commits off (`git commit -s`, Developer Certificate of Origin) so provenance is
clear.

## Reporting issues

Include: GPU model(s) + CU counts, `ROCR_VISIBLE_DEVICES`/`CU_NUM`, the profile, the
model id, whether the Triton cache was warm, and the relevant worker log lines. For a
suspected hang, attach a `py-spy dump` (`--cap-add SYS_PTRACE` is already set) — it's
the only reliable way to tell a cold compile from an actual wedge (DIARY "Act II").
