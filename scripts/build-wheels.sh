#!/usr/bin/env bash
# Build the three gfx1201 wheels FROM SOURCE on bare metal, for publishing to a
# GitHub Release. This is the "how the prebuilt wheels were made" recipe — most
# users never run it (they consume the Release via the default Dockerfile).
#
# Prereqs (see README "Building the wheels yourself"):
#   - An RDNA4 (gfx1201) host with ROCm and the TheRock build venv set up to
#     MATCH the container: torch 2.10.0+rocm7.14, py3.12.13, triton 3.6.0.
#   - The three patched source trees checked out (vllm, aiter, flash-attention).
#   - `source activate-build-env.sh` (sets VIRTUAL_ENV, ROCM_PATH, GPU_ARCHS=gfx1201,
#     CU_NUM=64, MAX_JOBS, and adds ld.lld/amdclang to PATH).
#
# Build sequentially (16 GB RAM/job; 2x16 jobs OOMs). flash-attention is ~2h
# (2669 CK FMHA steps); aiter + vllm are minutes each.
set -euo pipefail

: "${AITER_DIR:?set AITER_DIR to the patched aiter tree}"
: "${FA_DIR:?set FA_DIR to the patched flash-attention tree}"
: "${VLLM_DIR:?set VLLM_DIR to the patched vllm tree}"
OUT="${OUT:-$(pwd)/wheels}"
mkdir -p "$OUT"

echo "GPU_ARCHS=${GPU_ARCHS:-unset}  CU_NUM=${CU_NUM:-unset}  (source activate-build-env.sh first)"

# aiter — PREBUILD_KERNELS=0 is REQUIRED for gfx1201 (PREBUILD=1 hardcodes a
# gfx950 MoE matrix that all fail under flydsl 0.2.0 x ROCm7.14). Kernels JIT at
# runtime instead.
echo "=== aiter ==="
( cd "$AITER_DIR" && GPU_ARCHS=gfx1201 CU_NUM=64 PREBUILD_KERNELS=0 \
    python setup.py bdist_wheel && cp dist/amd_aiter-*.whl "$OUT/" )

# flash-attention — needs its CK submodule populated first.
echo "=== flash-attention (slow, ~2h) ==="
( cd "$FA_DIR" && git submodule update --init csrc/composable_kernel && \
    GPU_ARCHS=gfx1201 python setup.py bdist_wheel && cp dist/flash_attn-*.whl "$OUT/" )

# vLLM — relabel the stale setuptools_scm base tag so version gates behave.
echo "=== vllm ==="
( cd "$VLLM_DIR" && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.22.0 \
    GPU_ARCHS=gfx1201 PYTORCH_ROCM_ARCH=gfx1201 python setup.py bdist_wheel && \
    cp dist/vllm-*.whl "$OUT/" )

echo "Built wheels in $OUT:"
ls -la "$OUT"
echo "Publish: gh release create v0.22.0-gfx1201 $OUT/*.whl --title 'gfx1201 wheels' --notes '...'"
