#!/usr/bin/env bash
# Serve the Givens-rotated Qwen3.5-4B-RXF on one leased gfx1201 card, with the
# WORKTREE runtime (rxf.py + rxf_kernels.py = the learned-Givens path) bind-mounted
# over the baked image. Warm Triton cache + HF cache exported (CLAUDE.md §2/§3).
set -euo pipefail
cd /home/pat/code/vllm-gfx1201
PQ=/home/pat/code/vllm-gfx1201-paroquant-rotation/paroquant_rotation
export HF_HOME=/home/pat/.cache/huggingface
export VLLM_HOST_TRITON_CACHE=/home/pat/code/vllm-gfx1201/.triton-cache-combined
export VLLM_SINGLE_MODEL_ID=/root/.cache/huggingface/Qwen3.5-4B-RXF-${1:-givens}
export RXF_ACT_DTYPE="${RXF_ACT_DTYPE:-int8}"
exec scripts/gpu-lease.sh -n 1 --detach --name rxfgiv -- \
  docker compose -f docker-compose.yml -f "$PQ/givens-serve.override.yml" \
    --profile single up -d
