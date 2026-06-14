#!/usr/bin/env bash
# Het-TP COMM-bubble A/B via the OFFLINE torch profiler (profiling/het_profile_check.py).
# Loads the 35B TP=2 twice (even baseline vs het=64,56), profiles a decode wave, and per-rank
# buckets the kineto traces (profiling/analyze_torch_trace.py). Reuses the equivalence runs'
# warm Triton cache, so startup is fast. See CLAUDE.md container-testing protocol.
#
#   bash profiling/run_het_profile_offline.sh
set -uo pipefail

MODEL="${MODEL:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}"
IMG="${HET_IMG:-vllm22-w4a8:hettp}"
P=/home/pat/code/vllm-gfx1201
HF=/home/pat/.cache/huggingface
TRITON=$P/.triton-cache-combined
OUTBASE="${OUTBASE:-$P/profiling/het-profile}"
W4A8="${HET_W4A8:-0}"

run() {  # $1=tag  $2=cu_weights ("" = even)
  local tag="$1" cuw="$2" OUT="$OUTBASE/$1"
  rm -rf "$OUT"; mkdir -p "$OUT"
  echo "=== [$tag] profile run  cuw='${cuw}'  W4A8=$W4A8 ==="
  docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add CAP_SYS_PTRACE \
    --ipc host --shm-size 16g \
    -e HIP_VISIBLE_DEVICES=0,1 -e ROCR_VISIBLE_DEVICES=0,1 -e CU_NUM=56 \
    -e VLLM_ROCM_USE_W4A8_FP8_WMMA=$W4A8 -e VLLM_ROCM_USE_AITER=0 \
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface -e VLLM_LOGGING_LEVEL=WARNING \
    -e HET_MODEL="$MODEL" -e HET_TAG="$tag" -e HET_OUT=/profiles \
    -e HET_TP=2 -e HET_LIMIT_MM="${HET_LIMIT_MM:-1}" \
    -e HET_MAXLEN="${HET_MAXLEN:-2048}" -e HET_GPUUTIL="${HET_GPUUTIL:-0.90}" \
    -e HET_NSEQ="${HET_NSEQ:-16}" -e HET_DECODE_TOKS="${HET_DECODE_TOKS:-128}" \
    ${cuw:+-e VLLM_TP_CU_WEIGHTS="$cuw"} \
    -v "$P/profiling/het_profile_check.py:/tmp/het_profile_check.py:ro" \
    -v "$TRITON:/root/.triton" -v "$HF:/root/.cache/huggingface" -v "$OUT:/profiles" \
    --entrypoint bash "$IMG" -lc 'source /app/.venv/bin/activate && exec python /tmp/het_profile_check.py'
  echo "[$tag] traces:"; ls -la "$OUT"/*.json.gz 2>/dev/null || echo "  (no .json.gz!)"
}

run even ""
run het  "64,56"

echo
echo "############## ANALYSIS (per-rank kernel buckets) ##############"
for tag in even het; do
  echo; echo "================= $tag ================="
  python3 "$P/profiling/analyze_torch_trace.py" "$OUTBASE/$tag"/*.json.gz 2>/dev/null \
    | grep -iE '^===|total device-kernel|all-reduce|collective|MoE expert|dense GEMM|paged att|other ' \
    || echo "(no traces)"
done
