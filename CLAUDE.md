# CLAUDE.md — instructions for all agents working in this repo

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the stack contract (what-lives-where, the ABI
rule) and [`DIARY.md`](DIARY.md) for why. This file is the **operational protocol every agent
must follow when building/testing containers**, so we stop wasting each other's GPU time.

## Branch / worktree protocol (MANDATORY — read FIRST)

**NEVER do feature work in the shared `main` checkout — a dedicated worktree is REQUIRED.**
Multiple agents work different features in this repo at the same time. Two failure modes, both
chaos:
1. Editing on `main` directly → uncommitted changes collide and entangle (e.g. one agent's edits
   layered on another's uncommitted refactor — impossible to split cleanly). Already bit us.
2. `git switch`/`checkout -b` in the *shared* checkout → you change the branch out from under
   every other agent working in that same directory. Switching branches underneath each other is
   just as much chaos as sharing a dirty `main`. **Branch-switching in the shared checkout is NOT
   an acceptable substitute for a worktree.**

- **Before touching code, create your own feature-branch *worktree* (a separate directory)** and
  work entirely inside it — never `git switch` the shared checkout:
  ```
  git worktree add -b feat/<short-topic> ../vllm-gfx1201-<short-topic> main
  cd ../vllm-gfx1201-<short-topic>
  ```
  Each agent/effort gets its own worktree dir, so no one's branch or working tree moves under
  anyone else. This is mandatory, not preferred.
- **One worktree/branch = one concern.** Do not bundle an inherited uncommitted refactor with your
  own change; if you find a working tree already dirty with someone else's WIP, STOP and flag it —
  don't commit it under your change.
- **Commit/push only when asked.** When you do, it is on your feature branch, never `main`.
- Read-only analysis, profiling runs, and container builds are fine from the shared checkout; it's
  *source edits* that must live in your own worktree.

## GPU sharing protocol (MANDATORY — no human booking manager)

The box has **two** discrete gfx1201 compute cards: **GPU 0** (RX 9070 XT, 16 GB) and **GPU 1**
(RX 9070, 16 GB). ROCm device **2** is the Ryzen iGPU (gfx1036) — it drives the display and is
**never** a compute target. Multiple agents share these two cards.

**Do NOT ask a human "can I have the GPU?" and do NOT just check `rocm-smi` and hope. EVERY GPU
workload — `docker compose` serve/bench, a raw `python`/pytorch probe, anything that touches a
card — MUST be launched through `scripts/gpu-lease.sh`.** It is an `flock`-based arbiter: the lock
is held by the launched process and dies with it, so a crash/Ctrl-C auto-frees the cards with no
stale "reserved" state and no janitor. This *is* the booking manager — there is no human in the
loop.

```
# Foreground job (bench, probe, smoke test) on whichever card is free; blocks until one is:
scripts/gpu-lease.sh -n 1 -- docker compose --profile bench run --rm bench
scripts/gpu-lease.sh -n 1 -- bash -c 'source /app/.venv/bin/activate && python my_probe.py'

# Long-lived detached server; blocks until enough cards are free, then returns immediately and
# keeps the lease alive until the CONTAINER stops (not until this command returns):
scripts/gpu-lease.sh -n 2 --detach --name serve35b -- docker compose --profile serve up -d --build
scripts/gpu-lease.sh -n 1 --detach --name zaya     -- docker compose --profile zaya  up -d

# See who holds what (truth = flock state, self-healing):   scripts/gpu-status.sh
```

Rules:
- **`-n` = HOW MANY cards, NOT which card.** It is a count: `-n 1` = one card, `-n 2` = both. TP=2
  serve/bench/zaya → `-n 2`; a single-card model/probe → `-n 1` (the default). **There is no
  "pin a specific GPU" flag and you never need one** — the arbiter auto-assigns the lowest free
  card and injects `ROCR/HIP_VISIBLE_DEVICES` for you. Do NOT pass `-n 2` thinking it selects
  "card #2": that leases BOTH cards and starves every other agent. Want a single card? `-n 1`,
  always — let the arbiter choose which one.
- **`--detach` for any `up -d`** (the holder binds the lease to container lifetime). Foreground jobs
  (`run --rm`, a script that blocks) need no flag — the lease releases when they exit.
- **Let it block (the default).** Waiting *is* the coordination — don't poll `rocm-smi` and don't
  fall back to asking. Use `--nowait` only when you genuinely want to skip if the box is busy, and
  `--timeout S` for a bounded wait.
- gpu-lease injects `ROCR/HIP_VISIBLE_DEVICES`, a unique `COMPOSE_PROJECT_NAME`/`LEASE_NAME`, and a
  per-card host port (8000 + lowest card) — so two leased compose jobs never collide on device,
  container name, or port. The compose services read these with their current values as defaults,
  so a non-leased `docker compose` call is unchanged. **Don't hand-set `HIP_VISIBLE_DEVICES`/ports
  for a leased run — gpu-lease owns them.**
- CPU-only work (builds, static analysis, editing) needs no lease.
- **⚠ POWER SAFETY (MANDATORY for the default `:combined`).** It now defaults the native HIP serve
  path on (`VLLM_GDN_HIP=1`, `VLLM_ATTN_DECODE_HIP=1`); the GDN + attention **WMMA prefill** kernels
  saturate the matrix engine and RDNA4 can't damp the di/dt (a ~500W transient on a 374W cap was
  seen). A power cap **alone is insufficient** — you MUST also set a **negative GPU clock offset**
  (RDNA4 lacks RDNA3.5's power-ramp-speed control). Apply both per `patches/POWER_SAFETY.md` before
  serving, or fall back at runtime: `VLLM_GDN_HIP_RECURRENT_ONLY=1` + `VLLM_ATTN_DECODE_HIP=0`.

## Container testing protocol (MANDATORY)

### 1. Always test against the *latest* image — never a stale tag
- The canonical stack is `vllm22-w4a8:combined`, built from `Dockerfile.combined`
  (`FROM tcclaviger/vllm22:dev`, W4A8 kernel built in-image from `w4a8_fp8_wmma/`).
- **Before a test campaign**, make sure the image reflects current sources: rebuild if the
  base image, `w4a8_fp8_wmma/` csrc, or any applied patch changed since the tag was built.
  A stale image is exactly what bit us before (adapter calling v10/v11 against a v5-only `.so`).
- Build variants with a **distinct tag** so the known-good `:combined` is preserved, and tell
  the others which tag is current. Het-TP example:
  `docker build -f Dockerfile.combined -t vllm22-w4a8:hettp --build-arg WITH_HET_TP=1 \
   --build-context w4a8_src=<latest w4a8 csrc> .`
- Legacy tags (`vllm-gfx1201-w4a8bench:latest`, the retired wheel images) are **not** the
  stack — only use them for an explicit legacy A/B, never as "the image."

### 2. Always mount the *same* Triton cache — one dir per image/toolchain
The 35B is a GDN hybrid: a cold boot pays a ~15-30 min FLA-GDN + attention autotune compile.
A shared, persistent Triton cache makes that **one-time** and is reused across runs, models,
and configs. So **every** container run mounts the repo cache to `/root/.triton`:
```
-v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton   # combined image
```
- **One cache dir PER image/toolchain.** `.triton-cache-combined` for the combined image;
  `.triton-cache-old` for the legacy bench image. The cache is keyed by kernel-source hash +
  shapes, so different **models/shapes on the same image** safely add entries — sharing is the
  whole point (e.g. the het-TP equivalence runs warmed the cache so the het profile server
  skipped the cold GDN compile).
- **ZAYA is in the combined image, not a separate one.** ZAYA1-8B + CCA is folded into
  `Dockerfile.combined` behind `WITH_ZAYA` (default on), so `vllm22-w4a8:combined` itself serves it
  via the `zaya` profile in `docker-compose.yml` — same image, same toolchain, same
  `.triton-cache-combined` (already mounted by that profile). ZAYA support is additive (a no-op for
  the 35B W4A8/het-TP path); build `--build-arg WITH_ZAYA=0` for a leaner W4A8-only image.

#### ⚠ docker-compose defaults to a COLD cache outside the main checkout — set the env (MANDATORY)
`docker-compose.yml` mounts `${VLLM_HOST_TRITON_CACHE:-./.triton-cache-combined}` and
`${HF_HOME:-./.hf-cache}` — **relative paths**. Run compose from any worktree other than the main
checkout (e.g. the `gpu-lease` worktree) and those resolve to **empty/cold** dirs in that worktree
→ a ~15-30 min cold GDN compile *and* an offline `LocalEntryNotFoundError` for the model. This has
bitten us repeatedly. **Before any compose GPU run, export both** so they point at the real warm
artifacts (or an isolated copy — see below):
```
export HF_HOME=/home/pat/.cache/huggingface
export VLLM_HOST_TRITON_CACHE=<warm cache or a copy of it>      # see procedure below
```

#### Use the existing warm cache via an ISOLATED COPY (don't compile cold, don't corrupt production)
The canonical warm cache `/home/pat/code/vllm-gfx1201/.triton-cache-combined` (~170-200 MB, the
GDN/attention autotune) already exists — **reuse it; never pay the cold compile.** For a throwaway
test/experiment image, mount a **copy** so concurrent compiles can't corrupt the shared production
cache (the §3 race caveat). The cache is **root-owned** (written by the container as root), so a
plain `cp` as your user silently copies only the few files it can read — **copy via a root
container**:
```
docker run --rm -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/src:ro \
  -v /home/pat/code/.triton-cache-<tag>:/dst --entrypoint sh vllm22-w4a8:combined \
  -c 'rm -rf /dst/* 2>/dev/null; cp -a /src/. /dst/'
export VLLM_HOST_TRITON_CACHE=/home/pat/code/.triton-cache-<tag>     # warm copy, isolated
```
The Triton cache is keyed by kernel-source hash, so it is reused **regardless of a W4A8 `.so`
change** (that's a separate HIP extension, not a Triton kernel) — a fused/variant image still hits
the warm GDN/attention entries. Verify warmth: `find <dir> -name '*.json' | wc -l` should be ~1800+,
and a boot should reach "Application startup complete" in ~2-5 min, not 30. Put the copy on real
disk under `/home/pat/code` (never tmpfs — it's lost on reboot).

### 3. When NOT to reuse the cache (the caveat — start a fresh dir instead)
Reusing across an **incompatible toolchain** can load a stale/wrong kernel and crash or
silently misbehave. Use a new cache dir (or wipe the per-image one) when:
- The image's **vLLM / Triton / ROCm / torch / GPU-arch** changed (new base image, version
  bump). The combined-vs-old split exists for exactly this reason — keep them separate.
- You suspect **cache corruption**: a kernel crash that disappears after `rm -rf` the cache dir.
- **Concurrent compiles** into one dir can race on writes. Sequential runs sharing a dir is
  safe and normal; for runs that compile in parallel, pre-warm the cache once then run, or give
  each concurrent run its own dir.

### 4. Container-run gotchas (combined image)
- It bakes `HIP_VISIBLE_DEVICES=ROCR_VISIBLE_DEVICES=0,1,2,3`. Override **both together** to
  your device set (e.g. `-e HIP_VISIBLE_DEVICES=0,1 -e ROCR_VISIBLE_DEVICES=0,1`). A mismatch
  disables Triton and crashes model inspection.
- To run a script (not `vllm serve`), use
  `--entrypoint bash IMG -lc 'source /app/.venv/bin/activate && exec python <script>'`
  so the venv PATH is set for Triton's JIT — not `--entrypoint python`.
- Mount HF cache: `-v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1`.
- Shared **2× gfx1201** box: **never ask a human for a GPU window** — acquire cards through
  `scripts/gpu-lease.sh`. See the **GPU sharing protocol** section below (MANDATORY).

Reference runners that already follow all of the above: `patches/run_het_e2e_combined.sh`
(equivalence), `profiling/run_het_profile.sh` (profiling), `profiling/run_compare.sh`.

## Metrics capture: keep the monitoring stack up alongside any serve (MANDATORY)

vLLM exposes a `/metrics` Prometheus endpoint, but those counters live **in the serve
process** — they vanish the moment the container stops, and a removed container also loses
its logs. So if you want to understand how a serve behaved *after the fact* (throughput,
TTFT/ITL, queue depth, KV-cache utilisation, prefix-cache hit rate, spec-decode acceptance),
something must be **scraping and retaining** the series while it runs. We lost an all-day
Laguna run's data exactly this way (the scraping Prometheus was off-box; no local capture).

**Whenever you run a vLLM serve for real inference work, also have the local monitoring stack
running.** It is CPU-only, takes **no** GPU lease, and scrapes the host serve ports
(8000/8001) by polling — so it is fully decoupled from the leased serve project. Start it
once and leave it up; it then auto-captures every serve that comes and goes:

```
docker compose --profile monitoring up -d        # NOT under gpu-lease — CPU only
# Prometheus → http://localhost:9090   Grafana → http://localhost:3000 (anon admin)
```

- TSDB persists in the `vllm-prom-data` named volume with `--storage.tsdb.retention.time`
  (default 30d), so data survives container recreate and outlives the serve — config in
  `monitoring/prometheus.yml`, services in `docker-compose.yml`.
- It scrapes **8000 and 8001**; gpu-lease maps a serve to `8000 + lowest leased card`, and
  the `serve_port` + vLLM's `model_name` labels distinguish two concurrent serves.
- After a run, query Prometheus (`http://localhost:9090`) or Grafana for the analysis — do
  **not** rely on the live `/metrics` endpoint or `docker logs`, which are gone once the
  serve stops.

## Trace analysis: TraceLens (MANDATORY after trace collection)

After any profiling run that produces `*.pt.trace.json.gz` files, **run TraceLens before
drawing conclusions from the raw traces.**  The existing `profiling/analyze_torch_trace.py`
does flat kernel bucketing — TraceLens adds hierarchical Python→GPU linkage, per-kernel
TFLOPS/TB/s roofline, and (for TP≥2) straggler analysis that the flat script cannot.

```bash
# Analyse a trace directory — host-side, no GPU required, do NOT wrap in gpu-lease.sh:
profiling/run_tracelens.sh <trace_dir>

# Override output location:
profiling/run_tracelens.sh --out /some/other/dir <trace_dir>
```

Output goes to `profiling/tracelens/<trace_dir_basename>/`:
- `rank<N>/gpu_timeline.csv` — computation / comm / memcpy / idle breakdown
- `rank<N>/ops_summary_by_category.csv` — time by op class (GEMM, elementwise, attention …)
- `rank<N>/kernel_summary.csv` — per-kernel time + parent CPU op
- `rank<N>/unified_perf_summary.csv` — TFLOPS/s and TB/s per op (roofline)
- `multi/straggler_summary.csv` — per-rank wait time and arrived-last% (TP≥2 only)

**Prerequisites (host-side, one-time):**
```bash
uv tool install "git+https://github.com/AMD-AGI/TraceLens.git"
```
TraceLens is installed as a `uv` tool and is available on PATH after that. It does not
belong in the container — post-processing only.
