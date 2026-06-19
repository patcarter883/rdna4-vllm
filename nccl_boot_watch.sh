#!/usr/bin/env bash
# Watches a queued boot test: when the server reaches startup-complete, fire ONE inference
# through the container (localhost:8000), record it, then tear the container down (drops the
# gpu-lease). Exits on boot failure too.
#   nccl_boot_watch.sh <container-name-substr> <logfile> <resultfile>
set -uo pipefail
NAME_SUBSTR="${1:?need container name substring}"
LOG="${2:?need logfile}"
RESULT="${3:?need resultfile}"
: > "$RESULT"

find_cid() {  # the run container whose name matches our lease
  docker ps --filter "name=$NAME_SUBSTR" --format '{{.ID}}' 2>/dev/null | head -1
}

echo "[watch] waiting for boot (startup-complete) or failure [name~$NAME_SUBSTR]..." | tee -a "$RESULT"
deadline=$(( $(date +%s) + 3600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if grep -qiE "Application startup complete|Uvicorn running on" "$LOG" 2>/dev/null; then
    echo "[watch] BOOT OK detected" | tee -a "$RESULT"; break
  fi
  if grep -qiE "Traceback|EngineCore failed|Engine core initialization failed|raise RuntimeError|EXITCODE=" "$LOG" 2>/dev/null; then
    echo "[watch] BOOT FAILED — see log" | tee -a "$RESULT"
    grep -iE "error|assert|traceback|nccl|rccl|invalid|exit" "$LOG" 2>/dev/null | tail -25 | tee -a "$RESULT"
    exit 1
  fi
  sleep 10
done

CID=$(find_cid)
[ -z "$CID" ] && { echo "[watch] could not locate boot container" | tee -a "$RESULT"; exit 1; }
echo "[watch] container=$CID — sending one inference" | tee -a "$RESULT"

docker exec "$CID" bash -lc 'source /app/.venv/bin/activate 2>/dev/null; python3 - <<PY
import json,urllib.request
def get(url,data=None):
    req=urllib.request.Request(url, data=(json.dumps(data).encode() if data else None),
                               headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))
mid=get("http://localhost:8000/v1/models")["data"][0]["id"]
out=get("http://localhost:8000/v1/completions",
        {"model":mid,"prompt":"The capital of France is","max_tokens":16,"temperature":0})
print("MODEL_ID:", mid)
print("COMPLETION:", repr(out["choices"][0]["text"]))
print("USAGE:", out.get("usage"))
PY' 2>&1 | tee -a "$RESULT"
rc=${PIPESTATUS[0]}
echo "[watch] inference exit=$rc" | tee -a "$RESULT"

echo "[watch] tearing down container $CID (releases lease)" | tee -a "$RESULT"
docker stop -t 30 "$CID" >/dev/null 2>&1 || true
echo "[watch] done" | tee -a "$RESULT"
exit $rc
