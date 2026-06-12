#!/usr/bin/env bash
# Download the three prebuilt gfx1201 wheels from the GitHub Release into ./wheels.
#
# The default Dockerfile fetches these itself at build time, so you normally do
# NOT need this. It exists for: (a) offline/air-gapped builds — pre-stage wheels/
# then build with `--build-arg WHEELS_BASE=file:///wheels` style flows, and
# (b) sanity-checking the Release URLs.
#
# Usage: WHEELS_BASE=https://github.com/<you>/vllm-gfx1201/releases/download/<tag> ./scripts/fetch-wheels.sh
set -euo pipefail

WHEELS_BASE="${WHEELS_BASE:-https://github.com/CHANGEME/vllm-gfx1201/releases/download/v0.22.0-gfx1201}"
VLLM_WHL="${VLLM_WHL:-vllm-0.22.0+rocm714-cp312-cp312-linux_x86_64.whl}"
AITER_WHL="${AITER_WHL:-amd_aiter-0.1.14rc1.dev264+g2e93b80ab.d20260611-cp312-cp312-linux_x86_64.whl}"
FA_WHL="${FA_WHL:-flash_attn-2.8.4-cp312-cp312-linux_x86_64.whl}"

DEST="$(cd "$(dirname "$0")/.." && pwd)/wheels"
mkdir -p "$DEST"
for w in "$VLLM_WHL" "$AITER_WHL" "$FA_WHL"; do
  echo "==> $w"
  curl -fSL --retry 3 -o "$DEST/$w" "$WHEELS_BASE/$w"
done
echo "Wheels in $DEST:"
ls -la "$DEST"
