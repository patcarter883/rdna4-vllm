#!/usr/bin/env bash
# Run the 35B-A3B HF probe inside titans:dev across BOTH leased cards (device_map="auto").
# Forwards the lease's HIP/ROCR_VISIBLE_DEVICES pair VERBATIM (for -n 2 these are "0,1") so torch
# sees both cards and can shard the int4 GDN-MoE. Mirrors run_m2.sh but: (1) forwards BOTH cards,
# (2) entry is probe_relrep_35b.py, (3) passes PROBE_SMOKE through for the 1-anchor smoke.
# Usage (already inside `gpu-lease.sh -n 2 --name <cname>`):
#   titans/warmstart/run_probe35b.sh <cname>
set -euo pipefail
CNAME="${1:-titans-probe35b}"
exec docker run --rm --name "$CNAME" \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 16g \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}" \
  -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1}" \
  -e PROBE_SMOKE="${PROBE_SMOKE:-0}" \
  -e PROBE_MODEL="${PROBE_MODEL:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}" \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -v /home/pat/code/vllm-gfx1201-titans/titans:/work \
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
  --entrypoint bash titans:dev -lc \
    "source /app/.venv/bin/activate && python -u /work/warmstart/probe_relrep_35b.py"
