#!/usr/bin/env bash
# PPL A/B across Hadamard rotation SPANS on Qwen3.5-4B RXF — does a wider fixed
# Hadamard (which cuts the per-token int8 activation MSE up to 2.3x on real
# activations, see analyze_act_conditioning.py) translate to lower PPL? Answers
# the §4.4 open question on real data. Each checkpoint scored in its OWN process
# (one 16 GB card, one engine at a time). The WORKTREE runtime (wider-span aware)
# is bind-mounted over the combined image. Set RXF_ACT_DTYPE=int8|fp16.
#   CKPTS="hadamard hadamard128 hadamard256" ./eval_ppl_span.sh
set -euo pipefail
cd /home/pat/code/vllm-gfx1201
PQ=/home/pat/code/vllm-gfx1201-paroquant-rotation/paroquant_rotation
ACT="${RXF_ACT_DTYPE:-int8}"
CKPTS="${CKPTS:-hadamard hadamard128 hadamard256}"
SCORE=""
for c in $CKPTS; do
  SCORE+="echo '==================== ${c} (act=${ACT}) ===================='; "
  SCORE+="python /work/eval_rxf_ppl.py --model /root/.cache/huggingface/Qwen3.5-4B-RXF-${c} --tp 1; "
done
exec scripts/gpu-lease.sh -n 1 --name rxfppl -- \
  docker run --rm --name rxf-ppl-span \
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
      $SCORE
    "
