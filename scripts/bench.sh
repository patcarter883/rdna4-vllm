#!/usr/bin/env bash
# Turnkey TP=2 throughput benchmark — reproduces the published 298 dec / 1887 total tok/s.
#
# Runs the `bench` compose profile (test/bench_tp2.py: warmup generate -> timed generate)
# and stamps the current git SHA into each appended result so a published number maps back
# to a commit. Results are appended to profiling/bench-results/results.jsonl (host-visible).
#
# Usage:
#   ./scripts/bench.sh                  # stock headline baseline (USE_W4A8=0)
#   USE_W4A8=1 ./scripts/bench.sh       # bench the W4A8 kernel path instead
#
# Notes:
#   - Needs a free 2-GPU window (cards are shared) and a WARM Triton cache; a cold boot pays
#     the ~15-30 min FLA-GDN autotune compile once into .triton-cache-combined.
#   - Builds the image if missing: docker compose --profile bench build
set -euo pipefail
cd "$(dirname "$0")/.."

BENCH_GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
  BENCH_GIT_SHA="${BENCH_GIT_SHA}-dirty"
fi
export BENCH_GIT_SHA
export USE_W4A8="${USE_W4A8:-0}"

# Init-OOM guard: the engine-init memory-profiling forward is driven by max_num_batched_tokens
# (stock activation memory), NOT max_model_len (which only sizes KV cache), and the W4A8 MoE
# apply scratch is already bounded by MOE_APPLY_CHUNK in the image. bench_tp2.py decouples the
# two — MAX_NUM_BATCHED defaults to 4096 (chunked prefill) — so the batch is bounded regardless
# of context length and the W4A8 path needs no MAX_MODEL_LEN cap. Override MAX_NUM_BATCHED to
# tune the per-forward batch; set it =MAX_MODEL_LEN to recover the old coupled behaviour.
export MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-4096}"

RESULTS="profiling/bench-results/results.jsonl"
mkdir -p "$(dirname "$RESULTS")"

echo "=== bench: git=${BENCH_GIT_SHA}  USE_W4A8=${USE_W4A8}  (warm Triton cache assumed) ==="
docker compose --profile bench run --rm bench

echo
echo "=== last result (${RESULTS}) ==="
tail -n 1 "$RESULTS" 2>/dev/null || echo "(no result row written — check the run output above)"
