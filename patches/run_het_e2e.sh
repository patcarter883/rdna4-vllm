#!/usr/bin/env bash
# Heterogeneous-TP greedy-equivalence E2E. Loads a model twice (even baseline vs het)
# and diffs the generated token ids — they MUST match (het is math-preserving).
#
#   bash patches/run_het_e2e.sh [MODEL] [TP]
#
# Defaults to a SMALL, single-card-fittable, no-FLA-GDN-compile model so we can share
# GPUs and iterate fast:
#   dense (default): Qwen/Qwen2.5-Coder-7B-Instruct-AWQ           (AWQ-INT4 g128, 5.2G, cached)
#   moe            : Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4        (GPTQ-INT4 g128, ~8G, Qwen2MoE;
#                    exercises routed_experts het (inter 1408->768/640) AND dense/shared-expert
#                    het (5632->2944/2688). Run with HET_W4A8=1 — its GPTQ qkv qzeros trip the
#                    stock triton_w4a16 assert; the W4A8 plugin handles that layout.)
#   (Mellum2 rejected: MellumForCausalLM is a custom arch, unsupported by upstream vLLM.)
#
# HET_W4A8=0|1 selects VLLM_ROCM_USE_W4A8_FP8_WMMA (default 0 = stock loaders, the path
# the het edits live in). Dense AWQ works at 0; the GPTQ MoE needs 1.
#
# TP=2 (default): even vs "64,56" across GPU0,1 (the real uneven split).
# TP=1: even vs "64" on a single card (het path reduces to identity; fast smoke,
#       leaves the other GPUs free) — set GPU=<idx> to pick the card.
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct-AWQ}"
TP="${2:-2}"
GPU="${GPU:-1}"                       # single-card index for TP=1
IMG="${HET_IMG:-vllm-gfx1201-w4a8bench:latest}"
SP=/opt/python/lib/python3.12/site-packages/vllm
SRC=/home/pat/code/zaya/vllm-therock/vllm
PATCHES=/home/pat/code/vllm-gfx1201/patches
TRITON=/home/pat/code/vllm-gfx1201/profiling/triton-cache
HF=/home/pat/.cache/huggingface
OUT=/home/pat/code/vllm-gfx1201/profiling/het-e2e
LIMIT_MM="${HET_LIMIT_MM:-0}"         # set 1 for multimodal models (Qwen3.6)
mkdir -p "$OUT"

if [ "$TP" = "2" ]; then DEVS="0,1"; CUW_HET="64,56"; CU_ENV=(-e CU_NUM=56)
else                     DEVS="$GPU"; CUW_HET="64";    CU_ENV=(); fi

run() {  # $1=tag  $2=cu_weights("" for even)
  local tag="$1" cuw="$2"
  echo "=== run: $tag  TP=$TP  devs=$DEVS  VLLM_TP_CU_WEIGHTS='${cuw}' ==="
  docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --ipc host --shm-size 16g \
    -e ROCR_VISIBLE_DEVICES="$DEVS" "${CU_ENV[@]}" \
    -e VLLM_ROCM_USE_W4A8_FP8_WMMA="${HET_W4A8:-0}" -e VLLM_ROCM_USE_AITER=0 \
    -e HF_HUB_OFFLINE=1 -e VLLM_LOGGING_LEVEL=WARNING \
    -e HET_MODEL="$MODEL" -e HET_TAG="$tag" -e HET_OUT=/out \
    -e HET_TP="$TP" -e HET_LIMIT_MM="$LIMIT_MM" \
    ${cuw:+-e VLLM_TP_CU_WEIGHTS="$cuw"} \
    -v "$SRC/distributed/het_tp.py:$SP/distributed/het_tp.py:ro" \
    -v "$SRC/model_executor/parameter.py:$SP/model_executor/parameter.py:ro" \
    -v "$SRC/model_executor/layers/linear.py:$SP/model_executor/layers/linear.py:ro" \
    -v "$SRC/model_executor/layers/fused_moe/routed_experts.py:$SP/model_executor/layers/fused_moe/routed_experts.py:ro" \
    -v "$SRC/model_executor/layers/fused_moe/config.py:$SP/model_executor/layers/fused_moe/config.py:ro" \
    -v "$PATCHES/het_e2e_check.py:/tmp/het_e2e_check.py:ro" \
    -v "$TRITON:/root/.triton" -v "$HF:/root/.cache/huggingface" -v "$OUT:/out" \
    --entrypoint python "$IMG" /tmp/het_e2e_check.py
}

run even ""
run het  "$CUW_HET"

echo "=== DIFF (token_ids must be identical) ==="
python - "$OUT/het_even.json" "$OUT/het_het.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
ok = True
for x, y in zip(a, b):
    same = x["token_ids"] == y["token_ids"]; ok &= same
    print(("MATCH" if same else "DIFFER"), repr(x["prompt"][:40]))
    if not same:
        print("  even:", x["token_ids"][:12]); print("  het :", y["token_ids"][:12])
print("\nRESULT:", "PASS — het ≡ even" if ok else "FAIL — het diverged")
sys.exit(0 if ok else 1)
PY
