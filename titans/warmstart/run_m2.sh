#!/usr/bin/env bash
# Run a titans script inside titans:dev on a leased card. Container is --name'd so it can be stopped
# with `docker kill/stop` (TaskStop on the wrapper orphans the container — see CLAUDE.md / memory).
# Usage (already inside a gpu-lease, env HIP/ROCR injected):
#   scripts/gpu-lease.sh -n 1 --name titans-m2 -- \
#     titans/warmstart/run_m2.sh <cname> [--entry <script-rel-/work>] -- <python args...>
# Default entry = warmstart/m2_adapter.py.
set -euo pipefail
CNAME="${1:-titans-m2}"; shift || true
ENTRY="warmstart/m2_adapter.py"
if [ "${1:-}" = "--entry" ]; then ENTRY="$2"; shift 2; fi
[ "${1:-}" = "--" ] && shift || true
exec docker run --rm --name "$CNAME" \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 16g \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
  -e PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -v /home/pat/code/vllm-gfx1201-titans/titans:/work \
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
  --entrypoint bash titans:dev -lc "source /app/.venv/bin/activate && python -u /work/$ENTRY $*"
