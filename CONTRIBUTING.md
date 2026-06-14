# Contributing

Thanks for helping improve the gfx1201 (RDNA4) vLLM stack. This repo is mostly
**packaging** around upstream projects — read [`DIARY.md`](DIARY.md) first; it
explains why almost every odd-looking decision exists, and most debugging walls
already have a documented door.

> **The stack (2026-06): the combined image.** All development targets
> **`Dockerfile.combined`** → `vllm22-w4a8:combined`, built `FROM tcclaviger/vllm22:dev`
> (the collaborator's tuned-attention vLLM 0.22.69) with our W4A8 kernel built in-image from the
> **in-repo `w4a8_fp8_wmma/` source** + a surgical `moe_wna16` patch. Run via `docker-compose.yml`
> (the only compose; `--profile serve` / `--profile single`). The wheel-based from-source
> Dockerfiles have been **retired** — the base image now provides vLLM.

## What lives where

| If you want to change… | Edit it… |
|---|---|
| **the combined image (the stack)** | **`Dockerfile.combined`, `docker-compose.yml`** |
| heterogeneous-TP (64:56 sharding) | `patches/het_tp.py` + `patches/het_tp_vllm.patch` — see `patches/HET_TP_HANDOFF.md` |
| the WNA16-MoE `tp_size` source fix | applied surgically in `Dockerfile.combined` (sed) |
| the W4A8-FP8-WMMA kernel | **in-repo `w4a8_fp8_wmma/`** (the single source of truth; built in-image) |
| the kernels/engine themselves (vllm / aiter / flash-attention) | **not here** — from the base image, see "Upstream" below |

vLLM core, attention, aiter and flash-attention come from the **base image**
(`tcclaviger/vllm22:dev`), not from this repo. A change to vLLM core or attention goes to the
base image / its upstreams; this repo layers the W4A8 kernel and the small source patches on
top. (Historically these shipped as three `patcarter883/*-gfx1201` wheels — see DIARY Act I.)

## The one hard rule: ABI

Everything is pinned to the combined base **`tcclaviger/vllm22:dev`** — gfx1201 +
Python 3.12 + torch 2.10 + ROCm 7.2.1 + vLLM 0.22.69. Native artifacts built against any
other torch/arch will fail to load (`ImportError: undefined symbol _ZN3c10...`).
Consequences for contributors:

- **Never commit a prebuilt `.so`.** The W4A8 kernel is compiled *inside* the image from
  `w4a8_fp8_wmma/`; `.gitignore` already blocks build output.
- If you bump the base image or any pinned version, the in-image build rebuilds the W4A8
  kernel against it automatically — but verify the ABI import check still passes.

## Dev workflow

1. **Fork + branch.** Branch from `main`; keep one logical change per PR.
2. **Validate compose** for any `docker-compose*.yml` / `Dockerfile*` change:
   ```bash
   docker compose config -q
   ```
3. **Build + smoke** on real gfx1201 hardware when you touch the image or runtime:
   ```bash
   docker compose --profile serve up --build      # TP=2 (or --profile single for 1 GPU)
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
6. **Offline scripts that build `vllm.LLM(...)` with TP>1 MUST guard the engine under
   `if __name__ == "__main__":`** — vLLM spawns TP workers, which re-import the module;
   an unguarded top-level `LLM(...)` dies at worker spawn with `RuntimeError: An attempt
   has been made to start a new process before ... bootstrapping` (looks like a model
   bug, isn't). Template: `def main(): llm = LLM(...); ...` + `if __name__ == "__main__":
   main()`. This has bitten us repeatedly. (Example: `patches/het_e2e_check.py`.)

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
