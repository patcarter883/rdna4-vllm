#!/usr/bin/env bash
# gpu-status — show which gfx1201 cards are leased right now.
#
# Truth is the flock state, not the .info sidecar: a card is BUSY iff its lock can't be grabbed
# nonblocking. The .info file is only a best-effort label; if it's present for a card that flock
# says is FREE, the holder crashed and we clean the stale label up here. This is what keeps the
# system janitor-free — there is no registry to drift out of sync.
set -euo pipefail
readonly LEASABLE=(0 1)
readonly LOCKDIR=/home/pat/code/vllm-gfx1201/.gpu-locks
mkdir -p "$LOCKDIR"

printf '%-6s %-6s %s\n' GPU STATE HOLDER
for card in "${LEASABLE[@]}"; do
  lock="$LOCKDIR/gpu${card}.lock" info="$LOCKDIR/gpu${card}.info"
  exec {fd}>"$lock"
  if flock -n -x "$fd"; then
    # We grabbed it → it was FREE. Any .info is stale (crashed holder); remove it.
    [[ -f "$info" ]] && rm -f "$info"
    printf '%-6s %-6s %s\n' "$card" FREE -
    exec {fd}>&-
  else
    holder="?"
    [[ -f "$info" ]] && holder="$(tr '\n' ' ' < "$info")"
    printf '%-6s %-6s %s\n' "$card" BUSY "$holder"
    exec {fd}>&-
  fi
done
