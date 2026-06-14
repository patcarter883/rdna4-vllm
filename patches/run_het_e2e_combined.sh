#!/usr/bin/env bash
# Heterogeneous-TP greedy-equivalence E2E — runs against the COMBINED image with the
# het-TP patch BAKED IN (built via Dockerfile.combined --build-arg WITH_HET_TP=1). Unlike
# the older patches/run_het_e2e.sh, it does NOT mount source files over site-packages — the
# patch is in the image, so we only toggle VLLM_TP_CU_WEIGHTS between the two runs.
#
#   bash patches/run_het_e2e_combined.sh [MODEL] [TP]
#
# Loads the model twice (even baseline: VLLM_TP_CU_WEIGHTS unset; het: "64,56") and diffs
# the greedy token ids — they MUST match (het sharding is math-preserving).
#
# Knobs (env):
#   HET_IMG       image tag (default vllm22-w4a8:hettp)
#   HET_W4A8      0|1 -> VLLM_ROCM_USE_W4A8_FP8_WMMA (default 0 = stock loaders, isolates
#                 the het sharding from the W4A8 kernel). Set 1 to test the shipped config.
#   HET_LIMIT_MM  1 for multimodal models (Qwen3.6 35B) to skip the ViT dummy forward.
#   HET_MAXLEN    max_model_len (default 2048; W4A8 wants <=2048).
#   HET_MAXTOK    greedy tokens per prompt (default 64).
#   HET_GPUUTIL   gpu_memory_utilization (default 0.85).
#   GPU           single-card index for TP=1 (default 0).
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct-AWQ}"
TP="${2:-2}"
GPU="${GPU:-0}"
IMG="${HET_IMG:-vllm22-w4a8:hettp}"
PATCHES=/home/pat/code/vllm-gfx1201/patches
TRITON=/home/pat/code/vllm-gfx1201/.triton-cache-combined
HF=/home/pat/.cache/huggingface
OUT="${HET_OUT_DIR:-/home/pat/code/vllm-gfx1201/profiling/het-e2e}"
mkdir -p "$OUT" "$TRITON"

if [ "$TP" = "2" ]; then DEVS="0,1"; CUW_HET="64,56"; CU_ENV=(-e CU_NUM=56)
else                     DEVS="$GPU"; CUW_HET="64";    CU_ENV=(); fi

run() {  # $1=tag  $2=cu_weights("" for even)
  local tag="$1" cuw="$2"
  echo "=== run: $tag  model=$MODEL  TP=$TP  devs=$DEVS  W4A8=${HET_W4A8:-0}  VLLM_TP_CU_WEIGHTS='${cuw}' ==="
  # NOTE: the combined image bakes HIP_VISIBLE_DEVICES=ROCR_VISIBLE_DEVICES=0,1,2,3; they
  # MUST be overridden TOGETHER (mismatch -> "Disabling Triton" + a model-inspect subprocess
  # crash). Run via `bash -lc 'source activate && ...'` (the proven run_compare.sh pattern)
  # rather than --entrypoint python, so the venv PATH is set up for Triton's JIT.
  docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add CAP_SYS_PTRACE \
    --ipc host --shm-size 16g \
    -e HIP_VISIBLE_DEVICES="$DEVS" -e ROCR_VISIBLE_DEVICES="$DEVS" "${CU_ENV[@]}" \
    -e VLLM_ROCM_USE_W4A8_FP8_WMMA="${HET_W4A8:-0}" -e VLLM_ROCM_USE_AITER=0 \
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface -e VLLM_LOGGING_LEVEL=WARNING \
    -e HET_MODEL="$MODEL" -e HET_TAG="$tag" -e HET_OUT=/out \
    -e HET_TP="$TP" -e HET_LIMIT_MM="${HET_LIMIT_MM:-0}" \
    -e HET_MAXLEN="${HET_MAXLEN:-2048}" -e HET_MAXTOK="${HET_MAXTOK:-64}" \
    -e HET_GPUUTIL="${HET_GPUUTIL:-0.85}" \
    ${cuw:+-e VLLM_TP_CU_WEIGHTS="$cuw"} \
    -v "$PATCHES/het_e2e_check.py:/tmp/het_e2e_check.py:ro" \
    -v "$TRITON:/root/.triton" -v "$HF:/root/.cache/huggingface" -v "$OUT:/out" \
    --entrypoint bash "$IMG" -lc 'source /app/.venv/bin/activate && exec python /tmp/het_e2e_check.py'
}

run even ""
run het  "$CUW_HET"

echo "=== DIFF (token_ids must be identical) ==="
python3 - "$OUT/het_even.json" "$OUT/het_het.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
ok = True
for x, y in zip(a, b):
    same = x["token_ids"] == y["token_ids"]; ok &= same
    print(("MATCH " if same else "DIFFER"), repr(x["prompt"][:40]))
    if not same:
        # first divergence index
        n = min(len(x["token_ids"]), len(y["token_ids"]))
        d = next((i for i in range(n) if x["token_ids"][i] != y["token_ids"][i]), n)
        print(f"  first diverge @ tok {d}/{n}")
        print("  even:", x["token_ids"][max(0,d-2):d+6])
        print("  het :", y["token_ids"][max(0,d-2):d+6])
        print("  even text:", repr(x["text"][:80]))
        print("  het  text:", repr(y["text"][:80]))
print("\nRESULT:", "PASS - het == even" if ok else "FAIL - het diverged")
sys.exit(0 if ok else 1)
PY
