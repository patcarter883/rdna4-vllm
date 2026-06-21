#!/usr/bin/env bash
# PPL A/B: learned-Givens vs fixed-Hadamard RXF on Qwen3.5-4B, W4A8 (int8) path.
# One leased card; the WORKTREE runtime (rxf.py + rxf_kernels.py, the Givens path)
# bind-mounted over the image so BOTH checkpoints load through the same runtime.
# Each checkpoint is scored in its OWN python process (one engine at a time → no
# two-engine OOM on a 16 GB card). Lower PPL = better; Givens should be <= Hadamard.
set -euo pipefail
cd /home/pat/code/vllm-gfx1201
PQ=/home/pat/code/vllm-gfx1201-paroquant-rotation/paroquant_rotation
ACT="${RXF_ACT_DTYPE:-int8}"
exec scripts/gpu-lease.sh -n 1 --name rxfppl -- \
  docker run --rm --name rxf-ppl-ab \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES -e HF_HUB_OFFLINE=1 \
    -e RXF_ACT_DTYPE="$ACT" \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
    -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
    -v /home/pat/code/vllm-gfx1201-rxf-laguna:/work \
    -v "$PQ/rxf.py:/app/vllm/vllm/model_executor/layers/quantization/rxf.py:ro" \
    -v "$PQ/rxf_kernels.py:/app/vllm/vllm/model_executor/layers/quantization/utils/rxf_kernels.py:ro" \
    --entrypoint bash vllm22-w4a8:combined -lc "
      source /app/.venv/bin/activate
      echo '==================== GIVENS (learned) ===================='
      python /work/eval_rxf_ppl.py --model /root/.cache/huggingface/Qwen3.5-4B-RXF-givens --tp 1
      echo '==================== HADAMARD (fixed) ===================='
      python /work/eval_rxf_ppl.py --model /root/.cache/huggingface/Qwen3.5-4B-RXF-hadamard --tp 1
    "
