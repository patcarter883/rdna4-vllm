# GPU leasing — how agents share the two cards without a human booking manager

**TL;DR for agents: never ask "can I have the GPU?" Run every GPU workload through
`scripts/gpu-lease.sh`. It blocks until your card(s) are free, runs your job, and frees them when
the job dies. That is the whole protocol.** (Also stated as MANDATORY in `CLAUDE.md`.)

## The hardware

| ROCm dev | Card                | Arch    | VRAM  | Leasable? |
|----------|---------------------|---------|-------|-----------|
| 0        | RX 9070 XT          | gfx1201 | 16 GB | yes       |
| 1        | RX 9070             | gfx1201 | 16 GB | yes       |
| 2        | Ryzen 7800X3D iGPU  | gfx1036 | —     | **no** (drives the display; excluded by construction) |

The combined image bakes `HIP/ROCR_VISIBLE_DEVICES=0,1,2,3`, but only **0 and 1** are real
compute targets. gpu-lease only ever hands out `{0, 1}`.

## Why flock (and why it removes the human)

The lease is an `flock` held *by the launched process*. An advisory lock is released when the last
process holding the open file descriptor closes it — i.e. when your job exits, crashes, or is
Ctrl-C'd. So:

- There is **no registry** of who booked what, therefore **no janitor** and no stale "reserved"
  state to clean up after a crash. A booking database would need one; flock doesn't.
- Waiting on a busy card is just `flock` blocking. **The wait *is* the coordination** — you don't
  poll `rocm-smi`, and you never fall back to pinging a human.

`scripts/gpu-status.sh` reports state by *trying* to grab each lock (truth = flock state); the
`.info` sidecar files are only human-readable labels and are reconciled/cleaned automatically, so
they can never drift the system out of sync.

## Usage

```
scripts/gpu-lease.sh [-n N] [--wait|--nowait|--timeout S] [--detach] [--name LABEL] -- CMD...
```

| flag         | meaning |
|--------------|---------|
| `-n N`       | cards to lease: `1` (default) or `2`. TP=2 serve/bench/zaya → `2`; single model/probe → `1`. |
| `--wait`     | block until N cards are free (**default**). |
| `--nowait`   | fail now (exit 75) if N cards aren't free this instant. |
| `--timeout S`| block up to S seconds, then fail (exit 75). |
| `--detach`   | for `docker compose ... up -d`: launch, return immediately, keep the lease alive in a background holder until the container(s) stop. |
| `--name L`   | label for the lease (used in project/container name and `gpu-status`). |
| `-- CMD...`  | the command to run while holding the lease. |

### What it injects into `CMD`'s environment

| var                                       | example         | purpose |
|-------------------------------------------|-----------------|---------|
| `ROCR_VISIBLE_DEVICES` / `LEASE_ROCR_DEVICES` | `1` or `0,1`    | leased **physical** card ids |
| `HIP_VISIBLE_DEVICES` / `LEASE_HIP_DEVICES`   | `0` or `0,1`    | 0-based re-index of the leased set |
| `COMPOSE_PROJECT_NAME` / `LEASE_NAME`     | `lease-serve35b`| unique per lease → two compose jobs don't collide on project/container name |
| `VLLM_HOST_PORT` / `ZAYA_HOST_PORT`       | `8000` / `8001` | 8000 + lowest leased card (unless you already set it) |

The `docker-compose.yml` GPU services read `${LEASE_ROCR_DEVICES}` / `${LEASE_HIP_DEVICES}` /
`${LEASE_NAME}` with their pre-existing values as defaults — so a **non-leased** `docker compose`
call behaves exactly as before. Under a lease, don't also hand-set devices/ports; gpu-lease owns
them.

## Examples

```bash
# TP=2 35B server: wait for both cards, then detach (lease lives until the container stops).
scripts/gpu-lease.sh -n 2 --detach --name serve35b -- \
  docker compose --profile serve up -d --build

# Second model on the OTHER free card at the same time:
scripts/gpu-lease.sh -n 1 --detach --name zaya -- \
  docker compose --profile zaya up -d

# Foreground single-card bench on whichever card frees up first:
scripts/gpu-lease.sh -n 1 -- docker compose --profile bench run --rm bench

# Raw pytorch probe, but skip if the box is fully busy:
scripts/gpu-lease.sh -n 1 --nowait -- \
  docker run --rm --entrypoint bash vllm22-w4a8:combined -lc \
    'source /app/.venv/bin/activate && exec python /workspace/probe.py'

# Who holds what right now:
scripts/gpu-status.sh

# Release a detached server early:
docker compose -p lease-serve35b down
```

## Design notes / guarantees

- **Deadlock-free.** Acquisition is all-or-nothing in ascending card order (0 before 1). A 2-GPU
  lease only ever waits for a *higher-numbered* card than any it holds, so there is no circular
  wait. A 2-GPU job that can't grab both *releases* card 0 between attempts, so a 1-GPU job is
  never starved behind it.
- **Detached lifetime binding.** `--detach` launches the (fast-returning) `up -d`, resolves the
  project's container ids, then hands the held lock fds to a `setsid` holder that blocks on
  `docker wait` until every container stops. The lock is on the open file description, so it stays
  held by the holder after gpu-lease itself exits, and frees the instant the last container dies.
  (Verified with real (dummy) containers: a **multi-container** project stays BUSY for the whole
  container lifetime and goes FREE after `down`; a launch that creates no container, or whose
  command fails, releases the lease immediately instead of leaking it.)
- **Lock dir is a fixed absolute path** (`/home/pat/code/vllm-gfx1201/.gpu-locks`), not relative —
  so every agent in every git worktree coordinates on the *same* lock files.

## Validated / remaining

- Validated on CPU (no GPU needed): foreground acquisition, queueing, deadlock-free 2-GPU wait,
  `--nowait`/exit-75, `gpu-status` self-heal, compose substitution (leased + unleased), and the
  full detached path with real dummy containers (multi-container lifetime, no-container release,
  failing-launch release).
- Remaining: the first real detached serve through `--profile serve`/`zaya` is the only thing not
  yet exercised on actual hardware. Nothing GPU-specific is in the lease logic, so this is a smoke
  confirmation, not a risk — watch that `gpu-status` shows BUSY for the server's lifetime and FREE
  after `docker compose -p <project> down`.
