<h1 align="center">vllm-gfx1201</h1>

<p align="center">
  <strong>vLLM on AMD RDNA4 (gfx1201 — RX 9070 XT / RX 9070).<br>
  Qwen3.6-35B-A3B AWQ-INT4 MoE · tensor-parallel · OpenAI-compatible · one <code>docker compose up</code>.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Working-brightgreen" alt="Status" />
  <img src="https://img.shields.io/badge/TP%3D2-298_dec_%2F_1887_total_tok%2Fs-red" alt="Throughput" />
  <img src="https://img.shields.io/badge/GPU-gfx1201_(RDNA4)-ED1C24?logo=amd&logoColor=white" alt="GPU" />
  <img src="https://img.shields.io/badge/ROCm-TheRock_7.14-ED1C24?logo=amd&logoColor=white" alt="ROCm" />
  <img src="https://img.shields.io/badge/vLLM-0.22.0-4B2E83" alt="vLLM" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker" />
  <a href="https://github.com/patcarter883/rdna4-vllm/actions/workflows/build-image.yml"><img src="https://github.com/patcarter883/rdna4-vllm/actions/workflows/build-image.yml/badge.svg" alt="build-image" /></a>
</p>

---

This repo brings up [vLLM](https://github.com/vllm-project/vllm) on **AMD RDNA4** GPUs —
gfx1201 (RX 9070 XT / 9070) and gfx1200 (RX 9060 XT / 9060) — which the stock ROCm vLLM
image does not support. It ships three prebuilt **fat `gfx1200;gfx1201`** wheels (vLLM,
aiter, flash-attention) that run on either die, the two source fixes needed to load
Qwen3.5/3.6 MoE models, and an optional **W4A8-FP8-WMMA MoE kernel**. The goal is: clone,
set one env var, `docker compose up`.

> ⚠️ **This is overwhelmingly other people's work.** The serving engine is vLLM;
> the kernels are AMD aiter, Dao-AILab flash-attention, and Composable Kernel; the
> base image is AMD's TheRock. See [NOTICE](NOTICE) for full credits. The only
> original parts here are the gfx1201 enablement, the packaging, and the W4A8 kernel.

---

## Quick start

```bash
git clone https://github.com/patcarter883/rdna4-vllm && cd rdna4-vllm
cp .env.template .env
$EDITOR .env                       # set HF_HOME (+ HF_TOKEN for the download), and WHEELS_BASE

# Fetch the model into your HF cache (one-time, needs network + token):
HF_HOME=/your/hf-cache hf download cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit

# Bring it up (both GPUs, stock kernels). First boot is SLOW — see "Cold start".
docker compose --profile tp2-baseline up --build

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

One image, three `docker compose --profile` modes (see [docker-compose.yml](docker-compose.yml)):

| Profile | GPUs | Model (default) | Kernels | Notes |
|---|---|---|---|---|
| `tp2-baseline` | 0,1 (`CU_NUM=56`) | Qwen3.6-35B-A3B-AWQ-4bit | stock vLLM | the validated **298 dec / 1887 total tok/s** result |
| `tp2-w4a8` | 0,1 (`CU_NUM=56`) | Qwen3.6-35B-A3B-AWQ-4bit | W4A8-FP8-WMMA MoE | custom kernel; low batch is mandatory (below) |
| `single` | one card | Mellum2-12B-A2.5B-AWQ-INT4 | stock or W4A8 | 35B won't fit one 16 GB card |

```bash
docker compose --profile tp2-baseline up --build   # proven baseline
docker compose --profile tp2-w4a8     up --build    # custom MoE kernel
docker compose --profile single       up --build    # one GPU, small model
```

**Why `CU_NUM=56` for TP=2.** The reference rig is heterogeneous (9070 XT = 64 CU,
9070 = 56 CU). aiter asserts all visible GPUs share one CU count, so tensor-parallel
must be told the lower value. The image deliberately does **not** bake `CU_NUM` —
single-card runs auto-detect correctly, and only the TP profiles set it.

**Why text-only by default.** The compose files pass `--limit-mm-per-prompt
{"image":0,"video":0}`. These are vision-language models, but the ViT path uses
Torch-SDPA, which trips a gfx1201 AOTriton hazard and hangs memory profiling.
Text generation routes through the working `ROCM_ATTN` backend. (Enabling vision
means routing the ViT through flash_attn / disabling AOTriton SDPA — out of scope here.)

---

## gfx1200 (RX 9060 XT / 9060) — same fast path, no extra steps

gfx1200 (Navi 44) is the smaller RDNA4 die and shares the exact ISA this stack targets
(FP8 WMMA, wave32, no TDM), so it's a first-class target. **The default Release wheels
are fat `gfx1200;gfx1201`** — one set runs on either die — so a 9060 uses the **same
`docker compose up`** as a 9070, nothing special required:

```bash
docker compose --profile single up --build      # or tp2-baseline on two 16 GB 9060 XTs
```

(How: AMD GPU code objects are arch-exact, so the wheels carry *both* arches' objects —
verified: flash-attn 2662 gfx1200 + 2662 gfx1201; the vLLM `_C/_moe_C/_rocm_C` likewise;
aiter JITs per-arch at runtime. The W4A8 kernel is built fat in-image too.)

A from-source build (`docker-compose.fromsource.yml`, `GPU_ARCHS` selectable) is there if
you want to compile it yourself or trim to one die.

> ⚠️ **Built for it, not yet hardware-validated** — there's no gfx1200 card in the lab.
> The binaries provably contain gfx1200 code objects and the identical RDNA4 stack is
> validated on gfx1201, but a 9060 owner should expect to shake out the first real-hardware
> bugs — **please post on [issue #2](https://github.com/patcarter883/rdna4-vllm/issues/2).**
> Notes: the aiter A8W8 tuning configs and the kernel crossover cache were tuned on
> gfx1201's 64-CU die, so they'll be suboptimal (not broken) on the 32-CU Navi 44; and the
> 35B MoE needs ~23 GB, i.e. two **16 GB** 9060 XTs (8 GB cards host only smaller models
> via the `single` profile).

---

## The W4A8-FP8-WMMA MoE kernel

`w4a8_fp8_wmma/` is a custom HIP kernel that expands packed INT4 expert weights to
FP8 e4m3 **in-register** and feeds RDNA4's FP8 WMMA units. It is compiled **inside
the image** against the container's torch (ABI must match) and auto-engages via a
`vllm.general_plugins` entry point in every EngineCore worker — no code changes to
your serving script.

- Built in by default (`WITH_W4A8=1`); set `WITH_W4A8=0` to ship a pure baseline image.
- Toggle at run time without a rebuild: `VLLM_ROCM_USE_W4A8_FP8_WMMA=0` disables it.
- **Low batch is mandatory** for the `tp2-w4a8` profile. The MoE apply scratch is
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

---

## Verifying / benchmarking

```bash
./scripts/smoke.sh                       # /v1/models + one chat completion
# Throughput (run inside the image — needs the gfx1201 vLLM/torch):
docker compose --profile tp2-baseline run --rm --entrypoint python3 \
  tp2-baseline /workspace/test/bench_tp2.py     # warmup + timed; ~298/1887 on a warm cache
```

`test/bench_tp2.py` does a warmup `generate()` (to JIT-compile decode/MoE kernels)
then times a second one, so the number isn't compile-contaminated.

---

## Building the wheels yourself

The default `docker compose up` **downloads** the three gfx1201 wheels from a GitHub
Release — point `WHEELS_BASE` at yours. Two ways to produce them:

1. **From-source Docker build** (fully reproducible, ~2–4h): switch to the
   from-source overlay. It clones the patched trees and compiles in-container.
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.fromsource.yml \
     --profile tp2-baseline up --build
   ```
   You must set the `*_REPO`/`*_REF` build-args to your pushed/forked patched
   trees — see [Dockerfile.fromsource](Dockerfile.fromsource) for the exact commits.

2. **Bare-metal build** (how the published wheels were made): on a gfx1201 host
   with the TheRock build venv, run [`scripts/build-wheels.sh`](scripts/build-wheels.sh),
   then `gh release create v0.22.0-gfx1201 wheels/*.whl`.

The wheels are pinned to **gfx1201 + py3.12 + torch 2.10+rocm7.14**; they will not
load on a different ABI. The base image `rocm/vllm-dev:nightly-therock714` and the
two source fixes ([patches/moe_wna16.py](patches/moe_wna16.py) for the WNA16-MoE
`tp_size` attribute, and the `apache-tvm-ffi==0.1.10` pin for the tilelang MHC path)
are documented inline in the [Dockerfile](Dockerfile).

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
