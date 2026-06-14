#!/usr/bin/env bash
# Wait until both dGPUs (GPU0, GPU1) are released, then run the het-TP E2E check.
# Polls VRAM; launches run_het_e2e.sh once both drop below THRESH. 6h safety cap.
set -uo pipefail

THRESH=3000000000        # 3 GB "free enough" per dGPU
INTERVAL=120
MAXWAIT=$((6 * 3600))
elapsed=0

echo "[wait] watching GPU0/GPU1 VRAM (free threshold ${THRESH} B, poll ${INTERVAL}s)"
while true; do
  read -r g0 g1 <<EOF
$(rocm-smi --showmeminfo vram 2>/dev/null | awk '
  /GPU\[0\].*Used Memory/ {g0=$NF}
  /GPU\[1\].*Used Memory/ {g1=$NF}
  END {print g0+0, g1+0}')
EOF
  if [ "${g0:-0}" -lt "$THRESH" ] && [ "${g1:-0}" -lt "$THRESH" ] && [ "${g0:-0}" -gt 0 -o "${g1:-0}" -ge 0 ]; then
    echo "[wait] GPUs free after ${elapsed}s (g0=${g0} g1=${g1}) — launching E2E"
    break
  fi
  if [ "$elapsed" -ge "$MAXWAIT" ]; then
    echo "[wait] TIMEOUT after ${elapsed}s (g0=${g0} g1=${g1}); not launching"
    exit 2
  fi
  sleep "$INTERVAL"; elapsed=$((elapsed + INTERVAL))
done

exec bash /home/pat/code/vllm-gfx1201/patches/run_het_e2e.sh
