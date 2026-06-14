#!/usr/bin/env bash
# Het-TP COMM-bubble A/B profile: serve the 35B TP=2 twice (even baseline vs het=64,56),
# drive a concurrent decode wave around vLLM's torch profiler, and per-rank bucket the
# kineto traces. The "all-reduce / collective (TP)" bucket reveals the sync bubble: under
# EVEN split the faster (64-CU) card spin-waits at the barrier -> inflated collective time
# on one rank; under HET=64,56 the work is balanced -> the imbalance should shrink and
# aggregate decode tok/s rise.
#
#   bash profiling/run_het_profile.sh
#
# Knobs: HET_IMG (default vllm22-w4a8:hettp), HET_W4A8 (0=stock/warm-cache, matches the
# equivalence runs; the bubble is a TP-balance effect independent of W4A8), DECODE_TOKS.
set -uo pipefail

MODEL="${MODEL:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}"
IMG="${HET_IMG:-vllm22-w4a8:hettp}"
P=/home/pat/code/vllm-gfx1201
HF=/home/pat/.cache/huggingface
TRITON=$P/.triton-cache-combined
DRIVER=$P/profiling/drive_tp2_profile.py
OUTBASE="${OUTBASE:-$P/profiling/het-profile}"
DECODE_TOKS="${DECODE_TOKS:-128}"
W4A8="${HET_W4A8:-0}"
PORT=8000

serve_and_profile() {  # $1=tag  $2=cu_weights ("" = even)
  local tag="$1" cuw="$2" name="hetprof-$1"
  local OUT="$OUTBASE/$tag"
  rm -rf "$OUT"; mkdir -p "$OUT"
  docker rm -f "$name" >/dev/null 2>&1 || true
  echo "=== [$tag] start server  cuw='${cuw}'  W4A8=$W4A8 ==="
  docker run -d --name "$name" --rm \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add CAP_SYS_PTRACE \
    --ipc host --shm-size 16g -p $PORT:8000 \
    -e HIP_VISIBLE_DEVICES=0,1 -e ROCR_VISIBLE_DEVICES=0,1 -e CU_NUM=56 \
    -e VLLM_ROCM_USE_W4A8_FP8_WMMA=$W4A8 -e VLLM_ROCM_USE_AITER=0 \
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface -e VLLM_LOGGING_LEVEL=WARNING \
    -e VLLM_TORCH_PROFILER_DIR=/profiles \
    ${cuw:+-e VLLM_TP_CU_WEIGHTS="$cuw"} \
    -v "$TRITON:/root/.triton" -v "$HF:/root/.cache/huggingface" -v "$OUT:/profiles" \
    "$IMG" \
    "$MODEL" --host 0.0.0.0 --port 8000 --tensor-parallel-size 2 \
    --served-model-name model \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --max-model-len 2048 --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.90 --enforce-eager >/dev/null

  echo "[$tag] waiting for /health (up to ~30 min for cold compile)..."
  local ok=0
  for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then ok=1; echo "[$tag] healthy after ~$((i*10))s"; break; fi
    if ! docker ps -q --filter "name=$name" | grep -q .; then echo "[$tag] CONTAINER DIED during startup"; docker logs "$name" 2>&1 | tail -30; return 1; fi
    sleep 10
  done
  if [ "$ok" != 1 ]; then echo "[$tag] health TIMEOUT"; docker logs "$name" 2>&1 | tail -30; docker rm -f "$name" >/dev/null 2>&1; return 1; fi

  echo "[$tag] driving profile ($DECODE_TOKS decode toks x 16 seqs)..."
  python3 "$DRIVER" "$DECODE_TOKS" 2>&1 | tee "$OUT/driver.log"
  docker stop -t 40 "$name" >/dev/null 2>&1 || true
  echo "[$tag] traces written:"; ls -la "$OUT"/*.json.gz 2>/dev/null || echo "  (no .json.gz — check driver.log)"
}

serve_and_profile even ""    || echo "EVEN profile failed"
serve_and_profile het  "64,56" || echo "HET profile failed"

echo
echo "############## ANALYSIS (per-rank kernel buckets) ##############"
for tag in even het; do
  echo
  echo "================= $tag ================="
  grep -h 'tok/s aggregate' "$OUTBASE/$tag/driver.log" 2>/dev/null
  python3 "$P/profiling/analyze_torch_trace.py" "$OUTBASE/$tag"/*.json.gz 2>/dev/null \
    | grep -iE '===|total device-kernel|all-reduce|collective|MoE expert|dense GEMM|paged att' || echo "(no traces to analyze)"
done
