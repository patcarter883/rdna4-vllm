# cloud-lease — design (for review, no code yet)

Status: **DRAFT for review.** Worktree `feat/cloud-lease`. Nothing is implemented until this is
signed off.

## 1. Why this exists

`scripts/gpu-lease.sh` arbitrates the **two fixed, shared, already-paid-for** local gfx1201 cards
between agents. Larger training runs (the first real one: TiDAR conversion of ZAYA1-8B — a
production 5k–10k-step run is 1–3 h and wants more than 2×16 GB) need to spill to the cloud. The
trainer is the Zyphra transformers fork — **arch-agnostic pure torch**, so it runs on stock NVIDIA
with no HIP port (unlike the local serving stack, which is permanently gfx1201-local).

This doc designs `cloud-lease` — the cloud sibling of `gpu-lease`. It is **not** an extension of
`gpu-lease.sh`: they share a *UX contract* but zero *mechanism*.

## 2. The lease contract — what carries over, what doesn't

`gpu-lease`'s value is an `flock` whose lock dies with the process, so a crash auto-frees the card
with no janitor. In the cloud almost none of that mechanism transfers, but the **contract** does:

| | local `gpu-lease` | `cloud-lease` |
|---|---|---|
| scarce thing | 2 fixed cards, many agents contend | nothing — you provision what you ask for |
| `-n` semantics | how-many of a fixed set | instance **type** to spin up |
| acquire | `flock -n` (instant) | provider API call + poll-until-SSH-ready (~1–5 min) |
| "release" | close an fd → free | **`destroy` the instance → stop paying** |
| cost of "forgot to release" | another agent waits | **money bleeds** (cf. the stray $28.82 Vultr node) |
| new concerns | — | code/data sync **up**, checkpoint sync **down**, preemption |

**The one property worth being paranoid about: guaranteed teardown.** In the cloud, "lock dies with
the process" becomes "**instance is destroyed on exit / Ctrl-C / crash**" via a `trap`. That trap is
the whole safety story — getting it wrong is how you leak a running H100.

## 3. Architecture: one front-end, pluggable backends

Vultr is **not** a native SkyPilot cloud (SkyPilot covers AWS/GCP/Azure/Lambda/RunPod/Vast.ai/…;
Vultr only via standing up Vultr Kubernetes — too heavy for ad-hoc training). Since Vultr therefore
needs a direct-API backend *regardless*, we drop SkyPilot as the spine and keep one uniform
direct-API shape for every backend. (SkyPilot stays a *possible later* option purely for cross-cloud
cheapest-spot auto-failover — it would cover RunPod/Vast but never Vultr.)

```
cloud-lease (front-end)
   │  parse args, pick backend, own the teardown trap + state record
   ▼
Backend interface  ── provision → wait_ready → sync_up → run → (checkpoint sync_down)* → destroy
   ├── vultr     direct REST (api.vultr.com/v2)        [account exists]
   ├── runpod    direct REST (rest.runpod.io/v1)       [ad-hoc burst]
   └── vastai    stub behind the same interface         [later]
```

### Backend interface (the five verbs every provider implements)

1. `provision(gpu, region, spot) -> instance_id` — create the instance (API call).
2. `wait_ready(instance_id) -> ssh_target` — poll the provider until status=running + SSH answers.
3. `sync_up(ssh_target, workdir, dataset)` — rsync the working dir + dataset to the instance.
4. `run(ssh_target, cmd)` — execute CMD over SSH, stream stdout/stderr back.
5. `destroy(instance_id)` — terminate. **Idempotent** (safe to call twice; the trap may double-fire).

Plus `list_live()` for `cloud-status` (queries the provider API = source of truth).

### Implementation language — Python engine, bash shim (DECIDED)

`gpu-lease.sh` is pure bash with no deps, and that's right for an `flock`. But cloud orchestration is
JSON over HTTP + polling + retries + rsync/SSH across three providers — pure bash + `curl` + `jq`
becomes a maintenance trap. **Proposal:** a thin `scripts/cloud-lease.sh` shim (keeps the family UX)
delegating to `scripts/cloud_lease/` Python (a `Backend` ABC + one module per provider, `requests`
for HTTP, `subprocess` for ssh/rsync). **DECIDED 2026-06-28: Python engine + bash shim.**

## 4. Provider roles & sizing

The pricing pulled 2026-06-28 maps onto a **planned-vs-burst** split:

- **Vultr = standing / planned-run backend.** Account exists; on-demand **A100 80 GB ≈ $1.29–2.40
  /GPU-hr** — fits the **full fine-tune / Track F** target (8B + optimizer states unsharded on one
  80 GB card). ⚠ Vultr's **H100 is contract/prepaid-priced, not freely on-demand** — don't rely on
  Vultr for spot-cheap or on-demand H100.
- **RunPod = ad-hoc burst backend.** Clean REST/GraphQL, Docker-native, **community(spot) +
  on-demand**, fast spin-up — the "both local cards are busy, I want a throwaway box now" role.
  A40/A6000 48 GB ≈ $0.40–0.80/hr is ideal for **LoRA / Track R**.
- **Vast.ai = stub, same interface.** Cheapest floor + SkyPilot-native, but marketplace reliability
  variance → wired later for short interruptible LoRA, not unattended 3 h runs.

GPU shorthand the front-end maps per-provider (`a100-80`, `a6000-48`, `h100-80`, …) so the CLI is
provider-independent and the backend resolves it to that provider's plan/gpuType id.

## 5. Lifecycle in detail (and the teardown guarantee)

```
cloud-lease --provider vultr --gpu a100-80 --region ewr -- python -m zaya.tidar.train ...
```

1. **acquire** — `provision()` → `instance_id`. Immediately write a **state record** (§7) and arm
   `trap 'destroy + rm record' EXIT INT TERM`. From this line on, *every* exit path tears down.
2. **wait_ready** — poll provider status until running, then probe SSH until it answers. Bounded by
   `--provision-timeout` (default ~600 s); on timeout → destroy + fail (never leave a half-booted
   paid node).
3. **sync_up** — rsync `$PWD` (respecting `.gitignore`/an rsync-filter) + `--dataset` path up.
4. **run** — CMD over SSH, output streamed to the caller's terminal. Caller's Ctrl-C → SIGINT →
   trap → destroy.
5. **checkpoint sync_down** — every `--checkpoint-every` (and once more on exit, best-effort) rsync
   the remote checkpoint dir **down** to a local `--checkpoint-dir`. This is what makes a teardown or
   preemption non-catastrophic.
6. **release** — trap fires `destroy(instance_id)` (idempotent) and removes the state record.

Failure-mode table:

| event | result |
|---|---|
| CMD exits 0 | sync_down final checkpoint → destroy → exit 0 |
| CMD crashes / non-zero | sync_down (best-effort) → destroy → propagate exit code |
| Ctrl-C / SIGTERM on the launcher | trap → sync_down → destroy |
| provision/wait_ready timeout | destroy whatever was created → exit 75 |
| spot preemption (RunPod/Vast) | detected as instance vanished → last synced checkpoint is the resume point (see §6) |
| launcher process is `kill -9`'d | trap can't run → orphan node ⇒ caught by `cloud-status` reaper (§7) |

## 6. Hard prerequisite: checkpoint + resume in the trainer

Independent of provider, and the **first work item**. The TiDAR trainer currently saves only at the
end; on any cloud tier a teardown/preemption mid-run then loses everything. Required trainer
contract:

- `save_state(dir, step)` every N steps — model + optimizer + scheduler + step + RNG.
- `--resume <dir>` restores all of the above and continues from `step`.
- atomic checkpoint writes (tmp + rename) so a sync_down/preemption mid-write can't corrupt the
  latest good checkpoint.

`cloud-lease` then: syncs `--checkpoint-dir` **up** on resume, **down** periodically, and a relaunch
(`--resume`) on a fresh instance continues the run. This is the precondition for using any
spot/community pricing at all.

## 7. State, `cloud-status`, and the orphan reaper

There is no kernel-held lock — **the provider API is the source of truth.** Mirroring how
`gpu-status` reconciles flock state:

- On provision, write `~/.config/cloud-lease/state/<provider>-<id>.json`
  (`provider, instance_id, region, gpu, started, cmd, pid`).
- On destroy, remove it.
- `scripts/cloud-status.sh` calls each backend's `list_live()` and **reconciles**:
  - live instance **with** a record, launcher pid alive → healthy.
  - live instance **with** a record but launcher pid dead → **orphan** (the `kill -9` case) → offer
    `--reap` to destroy it.
  - live instance **without** any record → **stray** (e.g. the $28.82 node) → surface loudly.
  - record **without** a live instance → stale → clean the record.

This is the dollar-safety net the local flock gives for free and the cloud doesn't.

## 8. Security / key handling

- API keys read **only** from `~/.config/cloud-lease/<provider>.env` (chmod 600) or env vars
  (`VULTR_API_KEY`, `RUNPOD_API_KEY`, `VAST_API_KEY`). **Never** pasted into a session, never in the
  repo, never on a command line (process list leak).
- ✅ **Resolved 2026-06-28:** the earlier-pasted Vultr key has been rotated. The node behind the
  $28.82 is not a stray — it's an existing video-streaming headend service (legitimate, leave it).
- SSH: an `cloud-lease`-managed keypair uploaded to each provider once; private key chmod 600 under
  `~/.config/cloud-lease/`.

## 9. File layout

```
scripts/cloud-lease.sh          # front-end shim (UX, trap, dispatch)
scripts/cloud-status.sh         # cross-provider live-instance + orphan reconciler
scripts/cloud_lease/
  __init__.py
  backend.py                    # Backend ABC: provision/wait_ready/sync_up/run/destroy/list_live
  vultr.py                      # direct REST (api.vultr.com/v2)
  runpod.py                     # direct REST (rest.runpod.io/v1)
  vastai.py                     # stub behind the interface
  state.py                      # state-record read/write/reconcile
  gpu_map.py                    # a100-80/a6000-48/... -> per-provider plan ids
docs/cloud-lease/design.md      # this doc
~/.config/cloud-lease/          # keys, ssh keypair, state/ (NOT in repo)
```

## 10. CLI sketch

```
cloud-lease --provider {vultr|runpod|vastai} --gpu <shorthand> [--region R] [--spot]
            [--dataset PATH] [--checkpoint-dir DIR] [--checkpoint-every N]
            [--provision-timeout S] [--name LABEL] -- CMD...

cloud-status [--reap]      # list live across providers; reap orphans
```

## 11. Phasing (build order) — status as of 2026-06-28

1. ✅ **Trainer checkpoint/resume** (§6) — DONE by the ZAYA TiDAR agent in `train_tidar_zaya.py`
   (full state: params+AdamW+LR sched+RNG+step; atomic writes; SIGTERM handler; auto-resume;
   device-map-safe). Validated CPU-only by `test_ckpt_resume.py` (resume reproduces step-9 weights
   bit-identically). This is the unblocker — landed.
2. ✅ **Front-end + state + teardown trap + Vultr backend** — BUILT (`scripts/cloud_lease/`,
   `scripts/cloud-lease.sh`). Teardown guarantee validated offline by `scripts/test_cloud_lease.py`
   (destroy fires on normal/non-zero/exception/signal exit; idempotent; label-prefixed) — 10/10.
   ⏳ **Remaining gate:** a live smoke (`provision → nvidia-smi → destroy`) to confirm the Vultr
   wire format — `os_id` for Ubuntu 24.04, and the GPU **plan ids** (currently `None` in
   `gpu_map._VULTR` → provisioning refuses until filled via `cloud-lease list-gpus --provider vultr`).
3. ✅ **`cloud-status` reaper** — BUILT (`scripts/cloud-status.sh`, `status.py`). **Label-filtered**:
   only ever lists/reaps `cloud-lease-`-prefixed instances, so the user's production
   video-streaming headend on the same Vultr account is invisible to it.
4. ✅ **RunPod backend** — BUILT (`runpod.py`, incl. `--spot`). ⏳ same live-smoke gate (port schema +
   `gpuTypeId`s confirm against `list-gpus`).
5. ⏭ **First real TiDAR run** on Vultr A100-80 with checkpoint sync — after the step-2 smoke fills
   the plan ids.
6. ⏭ **Vast.ai backend** — `vastai.py` is a stub behind the interface; fill when a cost-floor LoRA
   run justifies it.

### What's verified vs what needs a live smoke
- **Verified offline (no cloud):** all modules import; bash shims valid; CLI/argparse; the teardown
  guarantee on every exit path; label prefixing; idempotent destroy; record reconcile logic.
- **Needs one live smoke per provider (the only thing local tests can't cover):** exact provider
  wire formats — Vultr `os_id`/plan ids, RunPod runtime port schema/`gpuTypeId`s. These are isolated
  in `gpu_map.py` + clearly-commented constants so the smoke is fill-in-the-blank, not a rewrite.
4. **RunPod backend** (incl. `--spot`/community) — the burst role.
5. **First real TiDAR run** on Vultr A100-80 with checkpoint sync.
6. **Vast.ai backend** — fill the stub when a cost-floor LoRA run justifies it.

## 12. Decisions (resolved 2026-06-28)

1. **Engine** — ✅ Python engine + bash shim (§3).
2. **Multi-GPU** — ✅ single-GPU only for v1; no `--gpus N`. One 80 GB card covers 8B.
3. **Spot resume** — ✅ manual relaunch. On preemption, `cloud-lease` preserves the last synced
   checkpoint and exits; the human relaunches with `--resume`. (Auto-relaunch is the SkyPilot
   feature we'd be re-implementing — out of scope for v1.)
4. **Dataset transfer** — see §13; default chosen there.

## 13. Dataset transfer — options & decision

Two distinct artifacts, handled differently:

- **Model weights (ZAYA1-8B, ~17.7 GB, public on HF):** *always* pulled from HF **on the instance**,
  never pushed from the home box. Datacenter↔HF is fast; home-upload of 17.7 GB over residential
  upstream is not. Not in question — this is fixed.
- **Training dataset:** the real choice below (depends on size, whether it's local-only or also on a
  hub, and how often it's reused across runs).

| option | pros | cons |
|---|---|---|
| **A. rsync from local each run** (over the same SSH the working-dir sync already uses) | simplest; zero extra cloud resource or standing cost; works for a locally-generated/preprocessed corpus; provider-agnostic; one code path reused for workdir+data | bottlenecked by **home upstream** (residential up is slow — GBs = minutes, tens of GB painful); re-uploads every run (no reuse); the paid instance is *running while you upload* (you pay for transfer time); home box must stay up through the transfer |
| **B. provider persistent volume** (Vultr Block Storage / RunPod Network Volume) | upload once, reuse across many runs; fast instance↔volume IO; survives teardown → relaunch re-attaches instantly | **standing $ while allocated** (pay even with no instance); **region/provider-pinned → breaks multi-provider** (a Vultr volume can't attach a RunPod pod → one volume per provider); still need an instance to load it initially; more lifecycle/API surface = more leak risk; RunPod volumes only in some DCs → constrains GPU availability |
| **C. object storage, S3-compatible** (Cloudflare R2 / Vultr Object Storage / S3) | **provider-agnostic** — any instance pulls from one bucket → fits multi-provider; upload once, fast datacenter download each run; cheap at rest (R2 ~$0.015/GB-mo, **R2 zero egress**); durable; decoupled from instance lifecycle | initial upload still home-upstream-bound (local→bucket); per-run download adds startup latency (+ egress cost on non-R2); needs creds + `rclone`/`aws-cli` on the instance; more moving parts |
| **D. pull dataset on the instance from its source** (HF dataset / public URL) | **zero home-upload** (datacenter bandwidth); trivially multi-provider; no standing storage cost; same path as the model-weight pull; reproducible | only works if the dataset is actually hub-hosted/public — **no help for a local-only/just-preprocessed corpus**; re-downloads each run unless cached on a volume; depends on the external host |

**Decision (default, v1):** **Option A (rsync each run)** for the dataset, because the TiDAR
conversion corpus is modest and A is the simplest path with no standing cost and no provider lock-in
— *and* it reuses the working-dir sync the lease already does. **If a dataset is hub-hosted, prefer
D** (skip the home upload entirely; same mechanism as the weight pull). **Escalate to C (Cloudflare
R2)** only once a corpus is reused across enough runs (or large enough) that per-run re-upload
dominates — R2 keeps the multi-provider story intact and has no egress fee. **Avoid B** — its
region/provider pinning directly fights the multi-provider goal and adds standing cost + orphan
surface. The `--dataset` flag accepts a local path (A) or a `hf://` / `s3://` / `r2://` URI (D/C),
so the choice is per-run, not baked in.
