# vLLM on AMD RDNA4 (gfx1201: RX 9070 XT / RX 9070) — wheel-injection image.
#
# Default ("fast") build: fetches three prebuilt gfx1201 wheels from a GitHub
# Release and installs them over the TheRock base image, then bakes in the two
# source fixes (tilelang apache-tvm-ffi pin + moe_wna16 tp_size) and, optionally,
# compiles the W4A8-FP8-WMMA MoE kernel against the container's torch.
#
# For a fully-from-source build (compile the 3 wheels in-container, ~2-4h) use
# Dockerfile.fromsource instead.
#
# The three wheels are gfx1201 + py3.12 + torch2.10+rocm7.14 specific. They are
# built on bare metal (see scripts/build-wheels.sh) and published as Release
# assets; override WHEELS_BASE / *_WHL to point at your own Release.

ARG BASE_IMAGE=rocm/vllm-dev:nightly-therock714
FROM ${BASE_IMAGE}

ENV PYTORCH_ROCM_ARCH=gfx1201
ENV GPU_ARCHS=gfx1201
ENV AITER_ROCM_ARCH=gfx1201
# Do NOT bake CU_NUM: aiter's get_cu_num() uses CU_NUM if set, else auto-detects
# the live GPU's Compute Unit count via rocminfo. The reference host is
# heterogeneous (RX 9070 XT = 64 CU, RX 9070 = 56 CU), so leaving CU_NUM unset
# lets each card get its correct count. Set it per-profile in docker-compose.yml
# (TP=2 needs the lower of the two, 56). Pin one discrete gfx1201 per process
# with ROCR_VISIBLE_DEVICES (excludes any iGPU) at run time.

# ---------------------------------------------------------------------------
# 1. Fetch + install the three gfx1201 wheels.
# ---------------------------------------------------------------------------
ARG WHEELS_BASE=https://github.com/CHANGEME/vllm-gfx1201/releases/download/v0.22.0-gfx1201
ARG VLLM_WHL=vllm-0.22.0+rocm714-cp312-cp312-linux_x86_64.whl
ARG AITER_WHL=amd_aiter-0.1.14rc1.dev264+g2e93b80ab.d20260611-cp312-cp312-linux_x86_64.whl
ARG FA_WHL=flash_attn-2.8.4-cp312-cp312-linux_x86_64.whl

RUN set -e; \
    mkdir -p /tmp/gfx1201-wheels && cd /tmp/gfx1201-wheels; \
    for w in "$VLLM_WHL" "$AITER_WHL" "$FA_WHL"; do \
      echo "fetching $w"; curl -fSL --retry 3 -O "$WHEELS_BASE/$w"; \
    done; \
    ls -la /tmp/gfx1201-wheels; \
    # Pin the bundled TheRock torch/triton so no transitive dep can change them.
    python3 -c "import torch,triton; open('/tmp/c.txt','w').write(f'torch=={torch.__version__}\ntriton=={triton.__version__}\n')"; \
    cat /tmp/c.txt; \
    # flash_attn: gfx1201 build replaces base's; deps (einops/torch) already present.
    pip install --force-reinstall --no-deps /tmp/gfx1201-wheels/flash_attn-*.whl; \
    # aiter: install WITH deps (constraints protect torch/triton) so its runtime deps
    # are satisfied — critically flydsl==0.2.0 (base ships 0.1.4.2, too old for aiter 0.1.14).
    pip install -c /tmp/c.txt /tmp/gfx1201-wheels/amd_aiter-*.whl; \
    # vLLM 0.22.0 + its ~95 runtime deps; constraints protect torch/triton.
    pip install -c /tmp/c.txt /tmp/gfx1201-wheels/vllm-0.22.0*.whl; \
    # tilelang fix: base ships apache-tvm-ffi 0.1.12, which double-registers the
    # TVM-FFI TypeAttr `__ffi_repr__` (type index 130) against tilelang 0.1.10's
    # bundled libtvm_compiler.so static init -> `import tilelang` aborts with
    # "Rust cannot catch foreign exceptions". tilelang 0.1.10 was built against
    # apache-tvm-ffi 0.1.10; pin to it (still satisfies its
    # `apache-tvm-ffi>=0.1.10,~=0.1.0`). Needed for the Qwen3_5Moe
    # linear-attention (MHC) tilelang kernel path.
    pip install "apache-tvm-ffi==0.1.10"; \
    pip show amd-aiter flash_attn vllm apache-tvm-ffi | grep -iE 'Name|Version'; \
    python3 -c "import torch, flash_attn; print('torch', torch.__version__, '| flash_attn', flash_attn.__version__)"; \
    rm -rf /tmp/gfx1201-wheels /tmp/c.txt

# ---------------------------------------------------------------------------
# 2. moe_wna16 tp_size fix (source patch).
# ---------------------------------------------------------------------------
# The WNA16 MoE weight loader reads layer.tp_size, but the current expert
# container `RoutedExperts` exposes TP only via layer.moe_config.tp_size (the
# legacy FusedMoE.tp_size attribute is gone). Without this, AWQ/GPTQ MoE models
# that fall back to the WNA16 kernel (e.g. Qwen3.5/3.6 MoE) crash at load with
# `AttributeError: 'RoutedExperts' object has no attribute 'tp_size'`. The
# patched file falls back to moe_config.tp_size (no-op for layers that have
# tp_size, so no regression). patches/moe_wna16.py must match the installed
# vLLM version (built from the same tree as the wheel).
COPY patches/moe_wna16.py /tmp/moe_wna16.patched.py
RUN set -e; \
    # Locate the installed file via path glob (NOT `import vllm...`, which prints
    # INFO logs to stdout and would corrupt the captured path).
    dest="$(find /opt/python/lib -path '*vllm/model_executor/layers/quantization/moe_wna16.py' -type f | head -1)"; \
    test -n "$dest" || { echo "moe_wna16.py not found"; exit 1; }; \
    echo "patching $dest"; \
    cp /tmp/moe_wna16.patched.py "$dest"; \
    grep -q 'moe_config.tp_size' "$dest"; \
    python3 -c "import ast; ast.parse(open('$dest').read())"; \
    echo "moe_wna16 tp_size fix applied OK"; \
    rm -f /tmp/moe_wna16.patched.py

# ---------------------------------------------------------------------------
# 3. (Optional) W4A8-FP8-WMMA MoE kernel.
# ---------------------------------------------------------------------------
# Compiled against the container's torch (ABI must match — that's why it's built
# here, not shipped as a .so). The vllm.general_plugins entry point auto-engages
# the dispatcher hooks in every EngineCore worker; no in-script register() call
# is needed. Disable at run time (no rebuild) with VLLM_ROCM_USE_W4A8_FP8_WMMA=0.
# Set WITH_W4A8=0 to skip building it entirely (pure baseline image).
ARG WITH_W4A8=1
COPY w4a8_fp8_wmma/ /opt/w4a8_fp8_wmma/
RUN set -e; \
    if [ "$WITH_W4A8" = "1" ]; then \
      export PATH="$ROCM_PATH/lib/llvm/bin:$PATH"; \
      cd /opt/w4a8_fp8_wmma && GPU_ARCHS=gfx1201 pip install . --no-build-isolation --no-deps; \
      python3 -c "import w4a8_fp8_wmma; print('[w4a8] import OK')"; \
    else \
      echo "WITH_W4A8=0 — skipping W4A8 kernel build"; \
    fi

# aiter (JIT) + vllm GPU verification deferred to run time (needs the GPU).
