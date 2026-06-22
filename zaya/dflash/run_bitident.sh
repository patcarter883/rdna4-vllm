#!/usr/bin/env bash
# Boot one ZAYA bit-identity profile under a GPU lease, wait for health, run the
# greedy token-capture client against it, then tear it down (releasing the lease).
#
#   zaya/dflash/run_bitident.sh <profile> <lease-name> <out.json> [EXTRA_ENV_KV ...]
#
# Example (ngram spec run):
#   zaya/dflash/run_bitident.sh zaya-dflash-all bi-spec /tmp/bi/spec.json \
#     'ZAYA_DFLASH_SPEC={"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":3,"prompt_lookup_min":2}'
#
# Must run from the worktree root (where docker-compose.yml + .env + scripts/ live).
set -euo pipefail

PROFILE="${1:?profile}"; LEASE="${2:?lease-name}"; OUT="${3:?out.json}"; shift 3 || true
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Extra KEY=VALUE env (e.g. ZAYA_DFLASH_SPEC=...) exported for the compose boot.
for kv in "$@"; do export "$kv"; done

PROJECT="lease-${LEASE}"
CTR="${PROJECT}-${PROFILE}"

cleanup() {
  echo ">>> tearing down $PROJECT (releases lease)" >&2
  docker compose -p "$PROJECT" down -t 20 >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo ">>> leasing 1 card + booting profile=$PROFILE name=$LEASE" >&2
scripts/gpu-lease.sh -n 1 --detach --name "$LEASE" -- \
  docker compose --profile "$PROFILE" up -d

# Resolve the published host port (8000 + lowest leased card).
PORT=""
for _ in $(seq 1 30); do
  PORT="$(docker port "$CTR" 8000 2>/dev/null | sed -n 's/.*:\([0-9]\+\)$/\1/p' | head -1)"
  [ -n "$PORT" ] && break
  sleep 1
done
[ -n "$PORT" ] || { echo "could not resolve host port for $CTR" >&2; docker ps >&2; exit 4; }
URL="http://localhost:${PORT}"
echo ">>> $PROFILE serving target: $URL (container $CTR)" >&2

# Wait for /health (cold-ish boot even with warm cache; allow generous window).
python3 zaya/dflash/bitident_client.py --url "$URL" --out "$OUT" --label "$PROFILE" --wait 900
rc=$?
echo ">>> client exit $rc; last 25 log lines:" >&2
docker logs --tail 25 "$CTR" 2>&1 | sed 's/^/    /' >&2 || true
exit $rc
