# References used building the W4A8-FP8-WMMA kernel

External docs, repos, headers, and local resources consulted — with what each was
useful for. Keep updated as new sources are used.

## RDNA4 WMMA — authoritative

- **AMD GPUOpen — "How to accelerate AI applications ... WMMA on RDNA 4" (3 parts)**
  - Part 1: https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/
  - Part 2: https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/
  - Part 3: https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-3/
  - Gave: A and B are **K-major (column-major)**, D accumulator **M-major**; each
    thread loads **8 contiguous elements**; RDNA4 removes RDNA3 element
    duplication; **double-K** technique = fuse 2 WMMAs for **128-bit loads** (fp8);
    `__builtin_amdgcn_global_load_tr_*` transpose loads. (Part 3 is more about
    in-register transpose than operand loading.)

- **Local RDNA4 ISA XML**: `/home/pat/code/vllm-rocm714-gfx1250/amdgpu_isa_rdna4.xml`
  (~194k lines). Full instruction/format list. Has `V_WMMA_F32_16X16X16_*` and a
  native `V_WMMA_I32_16X16X16_IU4` (int4) op. ISA definition, not a perf model.

## Compiler builtins / headers (in `.venv-therock714/.../_rocm_sdk_devel/`)

- **rocWMMA** `include/rocwmma/internal/wmma_impl.hpp:1821` — the fp8 WMMA builtin
  `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(int2 A, int2 B, float8 C)`
  gated to `AMDGCN_ARCH_ID_GFX1200/1201`. (rocWMMA's `load_matrix_sync` emits a
  mix of `ds_load_b32/b64` → the perf ceiling we replaced with raw loads.)
- **FP8 conversions** `include/hip/amd_detail/amd_hip_fp8.h` — OCP e4m3,
  `__builtin_amdgcn_cvt_pk_fp8_f32` / `cvt_f32_fp8` (E4M3_MAX 448).
- **Transpose-load builtins** (grep the SDK): `ds_read_tr8_b64_v2i32`,
  `global_load_tr8_b64_v2i32` (8-bit), `ds_read_tr16_b64_*` (16-bit), etc.
- **CK transpose-load usage** (how to address them):
  `flash-attention/csrc/composable_kernel/include/ck/utility/amd_transpose_load.hpp`
  and `.../include/ck_tile/core/arch/amd_buffer_addressing_hip.hpp` (~line 3104:
  `ds_read_tr8_b64_v2i32(lds_ptr)` per-thread LDS pointer for fp8/int8).

## Scaffold / prior art

- **hec-ovi/vllm-awq4-qwen** — https://github.com/hec-ovi/vllm-awq4-qwen
  Cloned to `reference/vllm-awq4-qwen/`. gfx1151 (RDNA3.5) **INT8 MMQ** kernel for
  AWQ-INT4 (the structural template). Gave: MMQ tiling, per-group fp32 dequant at
  K-boundaries, `MPLinearKernel` adapter + `_POSSIBLE_KERNELS[ROCM]` registration,
  M-dispatch (their `csrc/awq_mmq_gfx1151/`). **Targets the same `cyankiwi` AWQ4
  g32 model** that is our final goal.

## Performance background (CDNA-oriented — design inspiration, not RDNA4 code)

- **HazyResearch "AMD brr" blog** —
  https://hazyresearch.stanford.edu/blog/2025-11-09-amd-brr
  Gave: no universal LDS swizzle (per-dtype); **wave/warp specialization
  underperforms on wave32** (don't port producer/consumer to RDNA4); 8-wave
  ping-pong; match LDS swizzle to `ds_read` granularity; VGPR/occupancy budgeting.
- **HipKittens** — https://github.com/HazyResearch/HipKittens — CDNA3/4 tile DSL
  (no RDNA4 path yet). Design reference only.
- **Modular `warp_spec_matmul.mojo`** —
  https://github.com/modular/modular/blob/main/max/kernels/src/linalg/matmul/gpu/amd/warp_spec_matmul.mojo
  Producer/consumer + ring-buffer pipelining (CDNA-oriented; the pattern the blog
  warns against for wave32).

## vLLM internals (in the container `kyuz0/vllm-therock-gfx1201`)

- `vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py` — base
  class + `MPLinearLayerConfig` our adapter subclasses.
- `.../mixed_precision/triton_w4a16.py` — `triton_w4a16_gemm(a, b_q [K,N//8],
  scales [K//g,N], qzeros, group_size, zp_bias)` — the baseline we dispatch to.
- `.../linear/__init__.py` — `_POSSIBLE_KERNELS[PlatformEnum.ROCM]` (we prepend).
- `vllm/model_executor/layers/quantization/utils/marlin/marlin_int4_fp8_preprocess.cu`
  (in the vllm source tree) — AWQ zp-fold reference for future AWQ support.

## Models & infra

- **Final-goal model**: `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` (HF) — MoE, AWQ-4bit g32.
- **Test models**: `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` (works end-to-end),
  `Qwen/Qwen2.5-{1.5B,3B}-Instruct-GPTQ-Int4`. `neuralmagic/Qwen2-0.5B-Instruct-
  quantized.w4a16` (has desc_act/g_idx → we correctly decline).
- **Container**: `kyuz0/vllm-therock-gfx1201:latest` (vllm 0.21.1, torch 2.13,
  gfx1201) — build kernel in-container (ABI), HF cache mounted at `/hf`.
- **Build env**: bare-metal host `blue`, `source activate-build-env.sh`
  (`.venv-therock714`, torch 2.10, GPU_ARCHS=gfx1201).
