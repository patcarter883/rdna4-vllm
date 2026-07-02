<h1 align="center">vllm-gfx1201</h1>

<p align="center">
  <strong>vLLM on AMD RDNA4 (gfx1201 — RX 9070 XT / RX 9070).<br>
  Qwen3.6-35B-A3B AWQ-INT4 MoE · tensor-parallel · OpenAI-compatible · one <code>docker compose up</code>.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Working-brightgreen" alt="Status" />
  <img src="https://img.shields.io/badge/TP%3D2_cudagraphs-957_dec_%2F_2571_total_tok%2Fs-red" alt="Throughput" />
  <img src="https://img.shields.io/badge/GPU-gfx1201_(RDNA4)-ED1C24?logo=amd&logoColor=white" alt="GPU" />
  <img src="https://img.shields.io/badge/ROCm-7.2.1-ED1C24?logo=amd&logoColor=white" alt="ROCm" />
  <img src="https://img.shields.io/badge/vLLM-0.22.69-4B2E83" alt="vLLM" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker" />
  <a href="https://github.com/patcarter883/rdna4-vllm/actions/workflows/ci.yml"><img src="https://github.com/patcarter883/rdna4-vllm/actions/workflows/ci.yml/badge.svg" alt="ci" /></a>
</p>

---

This repo brings up [vLLM](https://github.com/vllm-project/vllm) on **AMD RDNA4** GPUs —
gfx1201 (RX 9070 XT / 9070) — which the stock ROCm vLLM image does not support. It layers
our **native HIP compute kernels** (GDN, attention, and the W4A8-FP8-WMMA MoE kernel from the
in-repo `gdn_hip/`, `attn_decode/`, `attn_prefill_paged/`, `w4a8_fp8_wmma/` sources) plus a
surgical `moe_wna16` source fix onto the base image (`tcclaviger/vllm22:dev`), which provides
the vLLM engine, the ROCm/torch runtime, and the upstream RXF W4A8 quant path. The goal is:
clone, set one env var, `docker compose up`.

> ⚠️ **The serving engine and runtime are other people's work.** The vLLM engine, the
> ROCm/torch runtime, and the upstream **RXF** W4A8 quant path come from the base image
> (`tcclaviger/vllm22:dev`, @tcclaviger). The attention / GDN / CCA compute, though, now
> runs through our own **native HIP kernels** (plus the W4A8-FP8-WMMA MoE kernel) — shared
> with the sibling `minisgl-rdna4` project and drawing on SGLang's design lineage. They
> replaced the earlier aiter / flash-attention / Composable-Kernel path, which was a dead
> end here. See [NOTICE](NOTICE) for full credits.

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
| `serve` | 0,1 (`CU_NUM=56`) | Qwen3.6-35B-A3B-AWQ-4bit | TP=2 het-TP, cudagraphs on: **957 dec / 2571 total tok/s** stock, **917 / 2464** with W4A8 (on by default; `USE_W4A8=0` for stock). `max-num-seqs=32`, `--no-async-scheduling` — don't raise `gpu-util`/`max-num-seqs` above the boot-profiled budget (16 GB is tight). |
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
  `VLLM_ROCM_W4A8_FP8_WMMA_MOE_KERNEL=scalar` (scalar golden kernel, isolates WMMA
  numerics). The grouped kernel is named — `wmma` (default) / `scalar` / `gemv`; the old
  numeric `MOE_VERSION` is removed and now hard-errors at boot if set. A-residence
  (formerly v5-vs-v6) is the `VLLM_W4A8_MOE_A_IN_LDS` knob.
- **Kernel names, not version numbers.** The dispatch table reads as descriptive kernels
  (`DenseKernel*` / `mmq_regdirect_*` dense, `wmma`/`scalar`/`gemv` MoE), and the two tiled
  dense kernels are consolidated onto one `TileConfig` policy (`gemm_tiled.h`), now the served
  dense default.
- **mxfp4 (OCP E2M1) rides the same path.** Because E2M1 ⊂ e4m3, mxfp4 weight decode is a
  lossless decode-table swap in the in-register int4→fp8 expansion — no new kernel. vLLM dispatch
  routes mxfp4 dense linears and gpt-oss E2M1 MoE experts to the grouped FP8-WMMA path
  (GPU bit-exact); a capability today, since no lab model ships mxfp4 yet.

Confirm it's engaged: the worker logs show `WNA16 MoE -> grouped FP8-WMMA`.

### RXF (the base image's quant path) and the rotation experiment

The base image (`tcclaviger/vllm22:dev`) ships the collaborator's **RXF — "Rotated eXtra Fast"** —
which does W4A8 as **int8×int8 with an integer (IQ4-NL) codebook** (exact — no fp8 round-trip) plus a
**FWHT-32 activation rotation**. The one surviving lever we explored on top of it is a **wider / learned
/ importance-aware rotation** — a *pre-pass-only* change (the K=32 int8 GEMM, the pack format, and the
per-group scale are untouched). All three stages were built and measured (`paroquant_rotation/`):
**the shipped span-32 Hadamard is the right default.** A wider fixed span, a learned Givens rotation,
and activation-conditioning are all ≤ noise or worse on PPL — on this W4A8 model the bottleneck is the
4-bit *weight*, and the int8 *activation* quant is already near-lossless (int8 ≈ fp16 PPL). The machinery
is kept for a future regime where the activation quant is the bottleneck (true 4-bit / NVFP4). Cheap
gate before any future rotation experiment: `paroquant_rotation/analyze_act_conditioning.py` (one GPU
forward over real activations). See [`DIARY.md`](DIARY.md) Act XVIII for the full closure.

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
# Throughput — one command; ~957/2571 (stock, cudagraphs) on a warm cache:
./scripts/bench.sh                       # stamps git SHA, appends profiling/bench-results/results.jsonl
USE_W4A8=1 ./scripts/bench.sh            # bench the W4A8 kernel path instead
ENFORCE_EAGER=1 ./scripts/bench.sh       # eager (profiling only — not the shipped path)
# equivalently, the raw compose invocation the wrapper runs:
docker compose --profile bench run --rm bench
```

The `bench` profile runs `test/bench_tp2.py` (offline LLM API) with the same TP=2 /
het-TP env as `serve`, warmup-`generate()` then a timed one so the number isn't
compile-contaminated. **`enforce_eager` is profiling-only** — every published number is
the shipped **cudagraphs-on** path. Each run appends a self-describing record (throughput
+ full config + git SHA + timestamp) to `profiling/bench-results/results.jsonl`, so a
number always maps back to a commit.

**Measured** (35B Qwen3.6-A3B-AWQ-4bit, TP=2 het-TP 64:56, 32 prompts × 128 out, gpu-util 0.90):

| path | decode tok/s | total tok/s |
|---|---|---|
| stock, **cudagraphs** | **957.5** | **2571.0** |
| W4A8, cudagraphs | 917.5 | 2463.6 |
| stock, eager (profiling) | 510.8 | 1371.6 |

- These are **offline** batched throughput (`bench_tp2.py`, fixed batch). The **served** HTTP
  endpoint is memory-tighter, so the `serve` profile now ships **`gpu-util=0.90`** (not the
  bench's aggressive setting): at high util the 16 GB cards have ~no headroom, and **heavy
  concurrent *streaming* load OOMs** (a single non-streaming burst of 32 sustained ~900 tok/s gen,
  but `vllm bench serve` streaming at conc 16/32 hit `HIP out of memory` at 0.96). 0.90 trades a
  little KV cache for a server that survives streaming out of the box; raise it via
  `VLLM_GPU_MEMORY_UTIL` (or lower `max-num-seqs`) once you know your load fits.
- **Cudagraphs are the throughput win: +87 %** decode over eager (957 vs 511) on this stack.
- On the 35B, **W4A8 is ~4 % under stock** under graphs — it quantizes only the MoE experts and
  graphs don't help MoE; its dense-path win is on other models. (The graph-compat work's value
  here is that W4A8 *runs* under cudagraphs at all — it was eager-only before.)
- **ZAYA1-8B CCA** (the `zaya` profile): the fused HIP `cca_decode_qk` kernel does **36.1 vs
  23.0 tok/s = +57 %** single-stream decode vs the eager ATen path (`ZAYA_CCA_HIP=0`).
- Warm restart is **~90 s** (persisted Triton/autotune + vLLM compile caches) vs ~22 min cold.
- Note: upstream `vllm-openai-rocm:nightly` can't initialise TP=2 on gfx1201 (RCCL
  `ncclCommInitRank` → `HIP invalid argument`); the base fork's PYNCCL path is what enables it.

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

## Observability & cloud training (landed on `main`)

Two pieces of dev-infrastructure now ship on `main`, both CPU-only and decoupled from the GPU lease:

- **Local metrics capture** (`docker compose --profile monitoring up -d` — **no** GPU lease). vLLM's
  `/metrics` counters live in the serve process and vanish when it stops; a CPU-only Prometheus +
  Grafana stack polls the host serve ports (8000/8001) and retains the series in a named volume
  (30-day retention), so it auto-captures every serve that comes and goes. Start it once and leave it
  up. Prometheus on `:9090`; a baked-in Grafana dashboard (throughput / TTFT-ITL / KV-cache /
  spec-decode panels) on `:3000`. See the metrics-capture section in [`CLAUDE.md`](CLAUDE.md).
- **`cloud-lease` — the cloud sibling of `gpu-lease`** (`scripts/cloud-lease.sh`). Provisions a cloud
  GPU box (Vultr / RunPod, including **ROCm MI300X**), rsyncs the repo over, runs a command, and tears
  the box down on exit — the same lock-is-the-process model as `gpu-lease`, for work that won't fit two
  16 GB consumer cards (a from-scratch or full-fine-tune training run). Hardened for real runs: `--env`
  keeps a secret (`HF_TOKEN`) off the argv/`ps`, `--restart-on-crash` resumes from a box-local
  checkpoint, `--setup-timeout` fails a wedged/dud pod instead of billing forever. This is what drove
  the TiDAR experts run below.

## In flight (feature branches, not on `main` yet)

Four workstreams are live on their own worktrees; each has a fuller write-up in [`DIARY.md`](DIARY.md)
(Acts XIX–XXIII). Summary of where they stand:

- **DFlash speculative decoding** (`feat/dflash-spec`, `feat/zaya-dflash`). The infrastructure works —
  a non-causal `TRITON_ATTN` patch (`patches/dflash_triton_noncausal/`, the `vllm22-w4a8:dflash`
  overlay) lets a diffusion drafter's bidirectional block run on the RDNA4-default attention, boots
  TP=2, coherent and lossless. The lever is the target's **quant format**: INT4 mis-conditions the
  drafter (~0.8% accept), but `Laguna-XS.2-NVFP4` (emulates → bf16) recovers it to ~25.7% (pos-0 ~70%).
  A parallel effort trains our **own** CCA-aware drafter for ZAYA1-8B (sidesteps the format wall).
- **Titans — test-time neural memory** (`feat/titans`, arXiv 2501.00663). The first *training*
  workstream: enwik8 trained from scratch (val BPC 1.83), MQAR recall probe PASS-directional, and a
  5-arm ladder that picked **GDN-2 (decoupled gates)** to provisionally ship while deferring the
  deep-memory kernel. Next: training-path choice + serving the checkpoint in `minisgl-rdna4`.
- **`gdn_hip` — native HIP GDN kernels** (`feat/gdn-hip`). A framework-agnostic `torch.ops` extension
  that replaces the ~15 FLA Triton GDN kernels with AOT-compiled HIP — **the 15–30 min cold GDN
  compile cliff goes away** (compile once, run any shape). Numeric parity ≤ ~1e-7; serves 4B TP1/TP2
  and 35B TP2 coherently. Gated `WITH_GDN_HIP` / `VLLM_GDN_HIP=1` (default off). Open work: a
  WMMA chunked-prefill path (the recurrent one is correct but ~4× slower than a matrix-core form).
- **TiDAR — converting ZAYA1-8B to a diffusion model** (`feat/tidar-convert`, `feat/tidar-serve`,
  arXiv 2511.08923). TiDAR fuses a diffusion drafter and an AR verifier into one forward; since Zyphra
  shipped ZAYA1-8B-Diffusion, the bet is *convert-and-serve* rather than train-from-scratch. The
  conversion half trains the TiDAR objective (a doubled `[clean | all-mask]` sequence — AR-causal plus a
  one-step block-bidirectional diffusion loss) on ZAYA: a four-rung ladder (LoRA → full-FT attention →
  +100× data → **+experts unfrozen**) lifted diffusion accuracy `0.089 → 0.254`, with **unfreezing the
  MoE experts the emphatic biggest lever**. The converged full-FT-all model trained on an **MI300X via
  `cloud-lease`** and is published on HF at `pat883/zaya1-8b-tidar-experts`. The serving half
  (`feat/tidar-serve`) wires the diffusion-draft + AR-verify path into the RDNA4 stack and picks up that
  checkpoint next.

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
