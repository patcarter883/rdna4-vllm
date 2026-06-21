#!/usr/bin/env bash
# Quant Qwen3.5-4B to RXF with a WIDER fixed Hadamard span (stage a), W4A8 int8.
# The real-activation analysis (analyze_act_conditioning.py) showed a wider span
# cuts the per-token int8 ACTIVATION-quant MSE up to 2.3x at S=256 on real Qwen
# activations (while the LEARNED rotation REGRESSES it) — so the activation-
# conditioning lever is span WIDTH, not learning. This produces the checkpoints
# for the PPL A/B vs the shipped Hadamard-32.
#   ./quant_qwen_hadamard_span.sh 128   ->  Qwen3.5-4B-RXF-hadamard128
set -euo pipefail
cd /home/pat/code/vllm-gfx1201
PQ=/home/pat/code/vllm-gfx1201-paroquant-rotation/paroquant_rotation
QWEN_IN=/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
SPAN="${1:?usage: quant_qwen_hadamard_span.sh <span>}"
NCARDS="${NCARDS:-1}"
OUT="/root/.cache/huggingface/Qwen3.5-4B-RXF-hadamard${SPAN}"
exec scripts/gpu-lease.sh -n "$NCARDS" --name rxfqh -- \
  docker run --rm --name rxf-quant-qwen-h"$SPAN" \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
    -v "$PQ/quantize_rxf.py:/app/tools/rxf_quant/quantize_rxf.py:ro" \
    --entrypoint bash tcclaviger/vllm22:dev -lc "
      source /app/.venv/bin/activate && cd /app/tools/rxf_quant
      exec python -u quantize_rxf.py --input $QWEN_IN --output $OUT \
        --rotation-kind hadamard --rotation-span $SPAN --act-dtype int8 --no-eval \
        --gpu-stream --load-chunks 8 2>&1"
