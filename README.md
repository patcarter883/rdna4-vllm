<h1 align="center">vllm-gfx1201</h1>

<p align="center">
  <strong>vLLM on AMD RDNA4 (gfx1201 — RX 9070 XT / RX 9070).<br>
  Qwen3.6-35B-A3B AWQ-INT4 MoE · tensor-parallel · OpenAI-compatible · one <code>docker compose up</code>.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Working-brightgreen" alt="Status" />
  <img src="https://img.shields.io/badge/TP%3D2-298_dec_%2F_1887_total_tok%2Fs-red" alt="Throughput" />
  <img src="https://img.shields.io/badge/GPU-gfx1201_(RDNA4)-ED1C24?logo=amd&logoColor=white" alt="GPU" />
  <img src="https://img.shields.io/badge/ROCm-7.2.1-ED1C24?logo=amd&logoColor=white" alt="ROCm" />
  <img src="https://img.shields.io/badge/vLLM-0.22.69-4B2E83" alt="vLLM" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker" />
  <a href="https://github.com/patcarter883/rdna4-vllm/actions/workflows/build-image.yml"><img src="https://github.com/patcarter883/rdna4-vllm/actions/workflows/build-image.yml/badge.svg" alt="build-image" /></a>
</p>

---

This repo brings up [vLLM](https://github.com/vllm-project/vllm) on **AMD RDNA4** GPUs —
gfx1201 (RX 9070 XT / 9070) — which the stock ROCm vLLM image does not support. It layers
our **W4A8-FP8-WMMA MoE kernel** (built in-image from the in-repo `w4a8_fp8_wmma/` source)
plus a surgical `moe_wna16` source fix onto the collaborator's tuned-attention vLLM 0.22.69
base image (`tcclaviger/vllm22:dev`), which provides the engine + RDNA4 attention. The goal
is: clone, set one env var, `docker compose up`.

> ⚠️ **This is overwhelmingly other people's work.** The serving engine is vLLM;
> the kernels are AMD aiter, Dao-AILab flash-attention, and Composable Kernel; the
> base image is AMD's TheRock. See [NOTICE](NOTICE) for full credits. The only
> original parts here are the gfx1201 enablement, the packaging, and the W4A8 kernel.

---

## Quick start

```bash
git clone https://github.com/patcarter883/rdna4-vllm && cd rdna4-vllm
cp .env.template .env
$EDITOR .env                       # set HF_HOME (+ HF_TOKEN for the download)

# Fetch the model into your HF cache (one-time, needs network + token):
HF_HOME=/your/hf-cache hf download cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit

# Bring it up (both GPUs). First boot is SLOW — see "Cold start". A/B the W4A8
# kernel with USE_W4A8=0 in .env (pure tuned-attention baseline).
docker compose --profile serve up --build

# In another shell, once it's serving:
./scripts/smoke.sh
```

Then hit the OpenAI-compatible API at `http://localhost:8000/v1`.

---

## Requirements

- **Two** discrete RDNA4 cards for the 35B model (it needs ~23 GiB of weights >
  16 GiB/card, so it cannot run on one — see [Profiles](#profiles)). A single card
  works for smaller models via the `single` profile.
- A working ROCm/amdgpu driver stack on the host (`/dev/kfd`, `/dev/dri`). You do
  **not** need ROCm libraries on the host — everything runs in the container.
- Docker with Compose v2, ~60 GiB disk for the base image, and the HuggingFace
  model cache (~24 GiB for the 35B AWQ).

---

## Profiles

One image, two `docker compose --profile` modes (see [docker-compose.yml](docker-compose.yml)):

| Profile | GPUs | Model (default) | Notes |
|---|---|---|---|
| `serve` | 0,1 (`CU_NUM=56`) | Qwen3.6-35B-A3B-AWQ-4bit | TP=2; W4A8 on by default — set `USE_W4A8=0` for the stock **298 dec / 1887 total tok/s** baseline. Low batch is mandatory (below). |
| `single` | one card | Qwen2.5-Coder-7B-AWQ | one 16 GB card, quick smoke / coherence checks (35B won't fit one card) |

```bash
docker compose --profile serve  up --build   # TP=2 (W4A8 on; USE_W4A8=0 for baseline)
docker compose --profile single up --build   # one GPU, small model
```

**Why `CU_NUM=56` for TP=2.** The reference rig is heterogeneous (9070 XT = 64 CU,
9070 = 56 CU). aiter asserts all visible GPUs share one CU count, so tensor-parallel
must be told the lower value. The image deliberately does **not** bake `CU_NUM` —
single-card runs auto-detect correctly, and only the TP profiles set it.

**Why text-only by default.** The compose files pass `--limit-mm-per-prompt
{"image":0,"video":0}`. These are vision-language models, but the ViT path uses
Torch-SDPA, which trips a gfx1201 AOTriton hazard and hangs memory profiling.
Text generation routes through the base image's tuned `triton_attn` backend (the gfx1201
default on the combined image, with the startup autotuner — not the narrow `rocm_attn` fast
path). (Enabling vision means routing the ViT through flash_attn / disabling AOTriton SDPA —
out of scope here.)

---

## gfx1200 (RX 9060 XT / 9060) — ISA-compatible, combined image is gfx1201-only for now

gfx1200 (Navi 44) is the smaller RDNA4 die and shares the exact ISA this stack targets
(FP8 WMMA, wave32, no TDM). **Note (2026-06):** since consolidating onto the combined image,
the W4A8 kernel is built for **gfx1201 only** (`GPU_ARCHS=gfx1201` in `docker-compose.yml`),
and gfx1200 coverage also depends on the base image (`tcclaviger/vllm22:dev`). To target a
9060: build the W4A8 layer fat — `w4a8_fp8_wmma/setup.py` defaults to `gfx1200;gfx1201`, so set
`GPU_ARCHS=gfx1200;gfx1201` for the build — and confirm the base image carries gfx1200 objects.

> ⚠️ **Not hardware-validated** — there's no gfx1200 card in the lab; see
> [issue #2](https://github.com/patcarter883/rdna4-vllm/issues/2). The crossover caches were
> tuned on gfx1201's 64-CU die, so they'll be suboptimal (not broken) on the 32-CU Navi 44; and
> the 35B MoE needs ~23 GB, i.e. two **16 GB** 9060 XTs.

---

## The W4A8-FP8-WMMA MoE kernel

`w4a8_fp8_wmma/` is a custom HIP kernel that expands packed INT4 expert weights to
FP8 e4m3 **in-register** and feeds RDNA4's FP8 WMMA units. It is compiled **inside
the image** against the container's torch (ABI must match) and auto-engages via a
`vllm.general_plugins` entry point in every EngineCore worker — no code changes to
your serving script.

- Built into the image from the in-repo `w4a8_fp8_wmma/` source (the single source of
  truth), compiled for gfx1201 against the base image's torch.
- Toggle at run time without a rebuild: `VLLM_ROCM_USE_W4A8_FP8_WMMA=0` (or `USE_W4A8=0`
  in `.env`) disables it for a pure tuned-attention baseline.
- **Low batch is mandatory** for the `serve` profile with W4A8 on. The MoE apply scratch is
  O(M·top_k) padded-sorted and OOMs the KV cache at the 8192 profiling batch on a
  16 GB card — hence `--max-model-len 2048 --max-num-batched-tokens 2048
  --gpu-memory-utilization 0.92`.
- Bisect knobs: `VLLM_ROCM_W4A8_FP8_WMMA_MOE=0` (fall back to the Marlin baseline),
  `VLLM_ROCM_W4A8_FP8_WMMA_MOE_VERSION=0` (scalar golden kernel, isolates v5 WMMA
  numerics).

Confirm it's engaged: the worker logs show `WNA16 MoE -> grouped FP8-WMMA`.

---

## Cold start & the Triton cache

The first TP=2 boot pays a **15–30 minute** one-time cold compile: the Qwen3.6
linear-attention (FLA-GDN) Triton kernels autotune from scratch (LLVM-AMDGCN
codegen, many configs). This looks identical to a hang — both workers at 100% CPU,
GPU idle — but it terminates. The compose files mount a **persistent Triton cache**
(`./.triton-cache` → `/root/.triton`) so subsequent boots load from it and start in
~1–2 minutes. Don't delete it.

If you need to confirm it's compiling and not wedged: the containers run with
`SYS_PTRACE`, so `docker exec <cid> bash -lc "pip install -q py-spy; py-spy dump
--pid 1 --nonblocking"` should show `make_amdgcn → autotuner → chunk_*`.

**Parallel cold-compile (on by default).** That cold compile is normally *serial* —
one kernel config at a time, one CPU core, GPU idle. The W4A8 plugin installs two
best-effort accelerators that fan the compiles across cores (they only fire on a cold
cache and are no-ops once it's warm; neither changes the autotuned result, only *when*
each kernel is compiled):

- **`VLLM_TRITON_PARALLEL_COMPILE`** (default `1`) — parallel-compiles every
  `@triton.autotune` config in a thread pool before the serial timing pass. This covers
  the GDN/SSD `ssd_*` autotune (~58 configs — the bulk of the cold boot). Set `0` to
  disable. Workers: `VLLM_TRITON_PARALLEL_COMPILE_WORKERS` (default 8).
- **`VLLM_ATTN_AUTOTUNE_PARALLEL`** (default `1`) — process-pool pre-warm of vLLM's
  *custom* attention startup autotuner (which the Triton hook above can't reach). It
  spawns short-lived GPU workers during startup; on a warm cache they cache-hit, adding
  only small spawn overhead. Set `0` to disable. Workers:
  `VLLM_ATTN_AUTOTUNE_PARALLEL_WORKERS` (default 6 — the measured sweet spot).

Measured speedups on gfx1201: ~3.1× (Triton autotune) and ~2.8× (attention autotune).
Both stay out of the way of warm production boots, where autotune is skipped entirely.

---

## Verifying / benchmarking

```bash
./scripts/smoke.sh                       # /v1/models + one chat completion
# Throughput (run inside the image — needs the gfx1201 vLLM/torch):
docker compose --profile serve run --rm --entrypoint python3 \
  serve /workspace/test/bench_tp2.py     # warmup + timed; ~298/1887 on a warm cache
```

`test/bench_tp2.py` does a warmup `generate()` (to JIT-compile decode/MoE kernels)
then times a second one, so the number isn't compile-contaminated.

---

## Building / iterating the W4A8 kernel

There are no wheels to fetch — vLLM, RDNA4 attention, aiter and flash-attention come from the
base image (`tcclaviger/vllm22:dev`), and the W4A8 kernel is compiled **inside the image** from
the in-repo `w4a8_fp8_wmma/` source on every `docker compose build` (cross-compiled for gfx1201;
no GPU needed to build). The surgical `moe_wna16` `tp_size` fix is applied inline in
[`Dockerfile.combined`](Dockerfile.combined).

To iterate the kernel on bare metal (op-level tests + microbench) using the TheRock build venv:
```bash
source /home/pat/code/vllm-rocm714-gfx1250/activate-build-env.sh
cd w4a8_fp8_wmma && python setup.py build_ext --inplace && python test_correctness.py
```
`w4a8_fp8_wmma/` is the single source of truth; edits there flow into the next image build.

---

## A second model: ZAYA1-8B (hybrid CCA + MoE)

The same combined base also serves **Zyphra's ZAYA1-8B** — a hybrid model that alternates
**CCA (Compressed Convolutional Attention)** with MoE layers. It's **baked into the same combined image** (`WITH_ZAYA`, on by default —
additive, a no-op for the 35B path) and runs via the `zaya` profile in the main compose:

```bash
cp .env.template .env                     # set HF_HOME, ZAYA_MODELS_DIR, ZAYA_MODEL_ID
docker compose --profile zaya up --build              # TP=1, one card, backend on :8001
# Optional: Recursive Self-Aggregation (RSA) test-time-compute proxy on :8100 (no GPU):
docker compose --profile zaya --profile rsa up
```

- **CCA as fused HIP kernels.** The eager CCA step is the graph-broken `vllm::cca` op
  (~1.4M tiny ATen launches/decode); three HIP kernels (decode / prefill / mixed) collapse each
  region into one launch. Bit-exact, and **+38% chat-decode / +49% RSA-decode tok/s** vs eager
  measured **with cudagraphs on** — a real production win, because the launch storm is exactly
  what graph capture can't fix (`docs/zaya/cca-kernel-perf.md`). Toggle with `ZAYA_CCA_HIP=0`.
- **Scope:** single card / **TP=1** today (CCA has no tensor-parallel split — per-head RMSNorm +
  grouped-mean state break under column-wise sharding). Experts are FP8, so the W4A8 kernel is off
  for this path (`ZAYA_USE_W4A8=0`). Multi-card (DP + expert-parallel) is the next profile. See
  `zaya/ZAYA_HANDOFF.md`. Build a leaner W4A8-only image with `WITH_ZAYA=0`.

---

## The story

[`DIARY.md`](DIARY.md) is a development diary of how this stack came together — the
RDNA4-vs-gfx1250 kernel reframing, the three sequential model-load blockers (and the
gdb/py-spy detective work behind each), the VRAM wall that forces TP=2, the
W4A8-FP8-WMMA kernel, and the packaging decisions behind this repo. Read it before you
debug something here — most walls already have a door.

---

## Credits & license

See [NOTICE](NOTICE) for the full acknowledgments. This repo's own glue code is
under [Apache-2.0](LICENSE); every upstream component retains its own license.
The `moe_wna16` `tp_size` fix is a generic upstream vLLM bug and is owed back as a PR.
