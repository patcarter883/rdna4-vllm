#!/usr/bin/env bash
# Post-`up` health probe: wait for the server, list models, run one completion.
# Usage: ./scripts/smoke.sh [host:port]   (default localhost:8000)
set -euo pipefail
ADDR="${1:-localhost:${VLLM_HOST_PORT:-8000}}"
BASE="http://${ADDR}/v1"

echo "Waiting for ${BASE}/models (first TP=2 boot can take 15-30 min on a cold Triton cache)..."
for i in $(seq 1 360); do
  if curl -sf "${BASE}/models" >/dev/null 2>&1; then break; fi
  sleep 10
done

echo "=== /v1/models ==="
curl -s "${BASE}/models" | python3 -m json.tool
MODEL="$(curl -s "${BASE}/models" | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')"
echo "Serving model: $MODEL"

echo "=== completion ==="
curl -s "${BASE}/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, what is an AMD RDNA4 GPU?\"}],\"max_tokens\":64}" \
  | python3 -m json.tool
echo "OK"
