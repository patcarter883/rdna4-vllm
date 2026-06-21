#!/usr/bin/env bash
# Full GPU quant of Qwen3.5-4B to RXF with the LEARNED Givens rotation (stage b,
# uniform importance), W4A8 int8 path — for the runtime serve-validation. One
# leased gfx1201 card; my quantize_rxf.py (with the givens fit) bind-mounted over
# the image's. Output -> HF cache so the serve can pick it up.
set -euo pipefail
cd /home/pat/code/vllm-gfx1201
PQ=/home/pat/code/vllm-gfx1201-paroquant-rotation/paroquant_rotation
QWEN_IN=/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
KIND="${1:-givens}"          # givens | hadamard
NCARDS="${NCARDS:-1}"        # lease this many cards; --gpu-stream uses 1 layer/GPU
OUT="/root/.cache/huggingface/Qwen3.5-4B-RXF-${KIND}"
exec scripts/gpu-lease.sh -n "$NCARDS" --name rxfqg -- \
  docker run --rm --name rxf-quant-qwen-"$KIND" \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
    -v "$PQ/quantize_rxf.py:/app/tools/rxf_quant/quantize_rxf.py:ro" \
    --entrypoint bash tcclaviger/vllm22:dev -lc "
      source /app/.venv/bin/activate && cd /app/tools/rxf_quant
      exec python -u quantize_rxf.py --input $QWEN_IN --output $OUT \
        --rotation-kind $KIND --act-dtype int8 --no-eval \
        --gpu-stream --load-chunks 8 2>&1"
