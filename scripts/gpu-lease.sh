#!/usr/bin/env bash
# gpu-lease — flock-based GPU arbiter for the shared 2× gfx1201 box.
#
# WHY THIS EXISTS
#   Multiple agents share two discrete gfx1201 cards (GPU 0 = RX 9070 XT, GPU 1 = RX 9070,
#   both 16 GB). The iGPU (gfx1036, ROCm device 2) drives the display and is NEVER leasable.
#   Instead of a human booking GPU windows, every GPU workload acquires its card(s) through an
#   flock here. The lock is held *by a process* and released when that process dies — so a crashed
#   or Ctrl-C'd job auto-frees its GPUs with NO stale "reserved" state to clean up. There is no
#   registry, therefore no janitor: that is the whole point.
#
# USAGE
#   gpu-lease [-n N] [--wait|--nowait|--timeout S] [--detach] [--name LABEL] -- CMD...
#
#     -n N            number of GPUs to lease: 1 (default) or 2.
#     --wait          block until N cards are free (DEFAULT).
#     --nowait        fail immediately (exit 75) if N cards aren't free right now.
#     --timeout S     block up to S seconds, then fail (exit 75).
#     --detach        for `docker compose up -d` style jobs: launch, return to the caller
#                     immediately, and keep the lease alive in a background holder until the
#                     launched container(s) stop. Lock lifetime == container lifetime.
#     --name LABEL    human label for the lease (used in COMPOSE_PROJECT_NAME / container name
#                     and shown by gpu-status). Defaults to a name derived from the leased cards.
#     -- CMD...       the command to run while holding the lease.
#
# WHAT IT INJECTS INTO CMD's ENVIRONMENT
#   LEASE_ROCR_DEVICES / ROCR_VISIBLE_DEVICES  physical leased card ids, e.g. "1" or "0,1"
#   LEASE_HIP_DEVICES  / HIP_VISIBLE_DEVICES   0-based re-index of the same, e.g. "0" or "0,1"
#   COMPOSE_PROJECT_NAME / LEASE_NAME          unique per lease → two compose jobs don't collide
#   VLLM_HOST_PORT / ZAYA_HOST_PORT            8000 + lowest leased card (so card1 → 8001), unless
#                                              already set by the caller.
#   The docker-compose.yml services read ${LEASE_ROCR_DEVICES}/${LEASE_HIP_DEVICES}/${LEASE_NAME}
#   with their current values as defaults, so a NON-leased `docker compose` call is unchanged.
#
# EXAMPLES
#   # TP=2 35B server, blocks until both cards free, then detaches:
#   gpu-lease -n 2 --detach --name serve35b -- docker compose --profile serve up -d --build
#
#   # quick single-card bench in the foreground on whichever card is free:
#   gpu-lease -n 1 -- docker compose --profile bench run --rm bench
#
#   # raw pytorch script on one free card, fail fast if the box is busy:
#   gpu-lease -n 1 --nowait -- python my_probe.py
set -euo pipefail

# --- constants ---------------------------------------------------------------------------------
# Leasable PHYSICAL compute cards. The iGPU (ROCm device 2, gfx1036) is excluded by construction —
# it can never be passed through here.
readonly LEASABLE=(0 1)
# Fixed ABSOLUTE lock dir so every agent in every git worktree coordinates on the SAME files.
# Must not be relative to the caller's cwd/worktree.
readonly LOCKDIR=/home/pat/code/vllm-gfx1201/.gpu-locks
readonly POLL_SECS=2

# --- args --------------------------------------------------------------------------------------
n=1 mode=wait timeout=0 detach=0 name=""
cmd=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n)         n="$2"; shift 2 ;;
    --wait)     mode=wait; shift ;;
    --nowait)   mode=nowait; shift ;;
    --timeout)  mode=timeout; timeout="$2"; shift 2 ;;
    --detach)   detach=1; shift ;;
    --name)     name="$2"; shift 2 ;;
    --)         shift; cmd=("$@"); break ;;
    -h|--help)  sed -n '2,40p' "$0"; exit 0 ;;
    *)          echo "gpu-lease: unknown arg '$1' (did you forget '--' before the command?)" >&2; exit 2 ;;
  esac
done

[[ "$n" =~ ^[12]$ ]]       || { echo "gpu-lease: -n must be 1 or 2 (have ${#LEASABLE[@]} cards)" >&2; exit 2; }
[[ ${#cmd[@]} -gt 0 ]]     || { echo "gpu-lease: no command given (use '-- CMD...')" >&2; exit 2; }
mkdir -p "$LOCKDIR"

# --- acquisition -------------------------------------------------------------------------------
# Atomic try-N: grab the first N free cards in ascending order, nonblocking. If we can't get all
# N, release whatever we got and (per mode) retry / give up. Ascending order + all-or-nothing =
# no circular wait, and a 1-GPU job is never stuck behind a 2-GPU job squatting on one card.
declare -a held_fds=() held_cards=()

release_held() {
  local fd
  for fd in "${held_fds[@]:-}"; do [[ -n "$fd" ]] && exec {fd}>&- 2>/dev/null || true; done
  for c in "${held_cards[@]:-}"; do [[ -n "$c" ]] && rm -f "$LOCKDIR/gpu${c}.info" 2>/dev/null || true; done
  held_fds=() held_cards=()
}

try_acquire() {   # returns 0 if it grabbed exactly N cards
  held_fds=() held_cards=()
  local card fd
  for card in "${LEASABLE[@]}"; do
    [[ ${#held_cards[@]} -ge $n ]] && break
    exec {fd}>"$LOCKDIR/gpu${card}.lock"
    if flock -n -x "$fd"; then
      held_fds+=("$fd"); held_cards+=("$card")
    else
      exec {fd}>&-
    fi
  done
  if [[ ${#held_cards[@]} -lt $n ]]; then release_held; return 1; fi
  return 0
}

start_ts=$SECONDS announced=0
while ! try_acquire; do
  case "$mode" in
    nowait)  echo "gpu-lease: $n GPU(s) not free right now (need ${LEASABLE[*]}; nowait)" >&2; exit 75 ;;
    timeout) (( SECONDS - start_ts >= timeout )) && { echo "gpu-lease: timed out after ${timeout}s waiting for $n GPU(s)" >&2; exit 75; } ;;
  esac
  if [[ $announced -eq 0 ]]; then echo "gpu-lease: waiting for $n free GPU(s)…" >&2; announced=1; fi
  sleep "$POLL_SECS"
done

# --- derive lease env ----------------------------------------------------------------------------
IFS=, ; rocr="${held_cards[*]}"; unset IFS                # physical ids, e.g. "0,1" or "1"
hip="$(seq -s, 0 $((${#held_cards[@]} - 1)))"             # 0-based re-index, e.g. "0,1" or "0"
[[ -z "$name" ]] && name="gpu$(IFS=-; echo "${held_cards[*]}")"   # default label from cards
lowest="${held_cards[0]}"
port=$(( 8000 + lowest ))

export LEASE_ROCR_DEVICES="$rocr"   ROCR_VISIBLE_DEVICES="$rocr"
export LEASE_HIP_DEVICES="$hip"     HIP_VISIBLE_DEVICES="$hip"
export LEASE_NAME="lease-${name}"
export COMPOSE_PROJECT_NAME="lease-${name}"
: "${VLLM_HOST_PORT:=$port}"; export VLLM_HOST_PORT
: "${ZAYA_HOST_PORT:=$port}";  export ZAYA_HOST_PORT

# Best-effort descriptive sidecar (NOT the source of truth — flock is). gpu-status reconciles it.
for c in "${held_cards[@]}"; do
  printf 'pid=%s name=%s since=%s\ncmd=%s\n' "$$" "$name" "$(date '+%F %T')" "${cmd[*]}" \
    > "$LOCKDIR/gpu${c}.info" 2>/dev/null || true
done

echo "gpu-lease: leased card(s) [$rocr] as '$name'  (HIP=$hip, port=$VLLM_HOST_PORT, project=$COMPOSE_PROJECT_NAME)" >&2

# --- run -----------------------------------------------------------------------------------------
if [[ $detach -eq 1 ]]; then
  # Launch the (fast-returning) detached command, then hand the held fds to a background holder
  # that inherits them and outlives this script. The advisory lock is on the open file description,
  # so it stays held until the LAST inheriting process closes it — i.e. until the holder exits when
  # the container stops. This script's own fd copies close on exit; the holder keeps the lease.
  rc=0; "${cmd[@]}" || rc=$?   # `|| rc=$?` so set -e doesn't exit before we can release on failure
  if [[ $rc -ne 0 ]]; then echo "gpu-lease: launch command failed (rc=$rc); releasing lease" >&2; release_held; exit $rc; fi

  # Resolve the project's container ids (compose may take a moment to create them).
  cids=""
  for _ in $(seq 1 30); do
    cids=$(docker compose -p "$COMPOSE_PROJECT_NAME" ps -q 2>/dev/null || true)
    [[ -n "$cids" ]] && break
    sleep 1
  done
  if [[ -z "$cids" ]]; then
    echo "gpu-lease: WARNING — couldn't find a container for project '$COMPOSE_PROJECT_NAME'." >&2
    echo "gpu-lease: was the command a 'docker compose ... up -d'? Releasing lease to avoid a leak." >&2
    release_held; exit 1
  fi

  # Holder: a backgrounded process that INHERITS the locked fds and blocks on `docker wait` until
  # EVERY leased container has stopped, then exits → fds close → lease released. setsid detaches it
  # from the caller's TTY so the agent's shell returns cleanly. `$cids` is passed UNQUOTED so it
  # word-splits into separate positional args ("$@") — `docker wait` then blocks on all of them.
  # (Do NOT interpolate the multi-line id list into a command string: the embedded newline would
  # corrupt a multi-container project's holder and free the lease while the server is still up.)
  infos=""
  for c in "${held_cards[@]}"; do infos+=" $LOCKDIR/gpu${c}.info"; done
  setsid bash -c 'docker wait "$@" >/dev/null 2>&1; rm -f '"$infos" _ $cids >/dev/null 2>&1 &
  disown || true

  echo "gpu-lease: detached — lease held until container(s) stop. Release early with: docker compose -p $COMPOSE_PROJECT_NAME down" >&2
  exit 0
else
  # Foreground: run as a child while we hold the fds; lease releases when this script exits
  # (normally, on Ctrl-C, or on crash — flock guarantees it). Propagate the child's exit code.
  trap release_held EXIT
  "${cmd[@]}"
  exit $?
fi
