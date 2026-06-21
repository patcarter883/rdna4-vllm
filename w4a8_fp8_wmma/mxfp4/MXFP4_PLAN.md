# mxfp4 on the W4A8 kernel — plan & coexistence with activation-rotation work

Branch: `feat/mxfp4-w4a8` (worktree `../vllm-gfx1201-mxfp4`).

## Goal

Let mxfp4-quantized models (OCP E2M1 weights + E8M0 group scales — gpt-oss, compressed-tensors
`mxfp4-pack-quantized`) run on this repo's W4A8 fp8-WMMA kernel. Today vLLM **parses and accepts**
mxfp4 on gfx1201 but has **no working RDNA4 GEMM**: native MoE is gfx950/AITER-gated, the dense
linear kernel registry has no ROCm entry, Marlin/FlashInfer are CUDA-only. The only runnable RDNA4
path is bf16-dequant emulation. Our kernel can be the **only real mxfp4 GEMM on RDNA4**.

## Why it's a decode-table swap, not a new kernel (proven, CPU-only)

The W4A8 kernel is a *"decode a 4-bit code → fp8 e4m3, then WMMA fp8×fp8"* engine. mxfp4's E2M1
codebook `{0,±.5,±1,±1.5,±2,±3,±4,±6}` is non-uniform (can't reuse the int4 subtract-zp integer
decode) **but every magnitude is exactly representable in fp8 e4m3**. So mxfp4 = a different
16-entry decode LUT fed through the *same* hardware `f32→e4m3` + the *same* fp8 WMMA core, with the
per-group E8M0 scale folded into the existing fp16 group-scale at load time.

Verified bit-exactly on CPU (`mxfp4/test_mxfp4_decode.py`, run in the image venv, no GPU):
- All 16 E2M1 values round-trip through e4m3 with **zero error** → decode is lossless.
- Our LUT == vLLM's `CompressedTensorsW4A16Mxfp4._FP4_E2M1_LUT`.
- uint8-nibble → (N,K//8) int32 repack round-trips through the kernel's `(word>>4j)&0xF` read.
- Full decode emulation (`code→e4m3→f32 × fp16 group-scale`) == vLLM reference dequant
  (`_FP4_E2M1_LUT × 2^(s-127)`), **bit-exact** for in-fp16-range scales.

## Status (updated 2026-06-21, branch rebased onto de-numbered main)

**DONE — converter + decode (CPU, host-validated):**
- `tile_config.h` / `w4a8_fp8_wmma_kernel.hip` / `moe_kernel.hip`: added the e2m1 decode helpers
  (`e2m1_to_e4m3`, `decode_w4_to_e4m3`, `decode_w4_to_f32`) in all three namespaces — additive,
  never touches `int4_signed_to_e4m3` or the activation path.
- `mxfp4/convert.py`: load-time converter, 2D (`convert_mxfp4_weight`) + 3D MoE
  (`convert_mxfp4_moe`). E2M1 uint8 + E8M0 → kernel-native packed int32 + fp16 group scale,
  `w_zeros=None`. Surfaces fp16 scale-range overflow.
- `mxfp4/test_mxfp4_decode.py`: host bit-exactness, all pass vs vLLM's `_FP4_E2M1_LUT`.

**DONE — kernel threading (compiles clean, dense GPU-validated):**
- Threaded an explicit, defaulted `bool weight_is_e2m1` through the LIVE kernels (pure/cudagraph-safe
  — no hidden device state): dense `reference_scalar`/`prefill_wmma`/`prefill_wmma_ashuffle`/
  `decode_gemv` (+ gated `gemm_tiled_ashuffle`), MoE `scalar`/`wmma` (tiled `moe_gemm_tiled.h`)/
  `gemv` + fused `gemm1_silu` v5/v6 — through every launcher, the 4 bindings ops (schema + forward +
  fake), and the Python wrappers. Full extension compiles.
- `mxfp4/test_mxfp4_kernel_gpu.py`: **GPU bit-exact PASS** — e2m1 output == the validated int4 path
  (`max|diff|=0`) for all dense kernels + decode_gemv at M=4/16/64, and an fp8 oracle confirms the
  non-integer 0.5/1.5 values decode through the live kernels. int4 path unchanged (regression guard).

**REMAINING:**
- MoE GPU equivalence test (`test_mxfp4_kernel_gpu.py` covers dense; MoE kernels compile + share the
  identical threaded helper — add the e2m1==int4 MoE check).
- vLLM dispatch wiring (next §) + image build + serve smoke.

## Coexistence with the activation-rotation work — orthogonal by construction

The rotation effort improves **activation (A-side)** quant. mxfp4 here is entirely **weight
(B-side)**. They touch disjoint code:

| Concern | A-side (rotation agent) | B-side (this work) |
|---|---|---|
| Kernel fns | `compute_act_fp8_and_scales_kernel` (w4a8_fp8_wmma_kernel.hip:126), `compute_block_act_scales` (:197) | weight-decode sites `int4_signed_to_e4m3((nibble)-zp)` → `e2m1_to_e4m3(code)` |
| Data | `act_scales`, the fp8 `x_fp8` activation tensor | `w_packed`, `scales` (group), `w_zeros` |
| Helper | (act quant / rotation) | `tile_config.h::e2m1_to_e4m3` (new), `int4_signed_to_e4m3` (unchanged) |
| Load-time | rotation matrix / online transform | `mxfp4/convert.py` |

Only shared surfaces:
1. The fp8 WMMA core (`WmmaFp8::mma`) — **neither side changes it**.
2. Each `__global__` launcher's argument list / dispatch lines — both increments may add a flag
   here (rotation: an A-side toggle; mxfp4: a B-side decode toggle). These are independent params;
   a merge conflict here is mechanical, not semantic. Keep the mxfp4 decode flag named distinctly
   (e.g. `bool weight_is_e2m1`) so it reads orthogonally to any activation flag.

**Net: the rotation change lands cleanly before/after this with no semantic interaction.** In fact
it *helps* the open accuracy gate below.

## Open accuracy gate (the go/no-go) — partly addressed by rotation

mxfp4 reference models are **W4A16** (bf16 activations). Our kernel forces **A8** (dynamic per-row
fp8 activations), which injects activation-quant error the reference doesn't have. This must pass a
PPL/quality check on a real mxfp4 model vs the bf16-emulation reference before claiming success.
**The activation pre-quant rotation directly reduces this error** (spreads outliers before fp8
quant), so the two efforts compound: rotation widens the margin this gate needs.

## Next increments (need a kernel build + GPU smoke; do under gpu-lease)

2. **Kernel decode variant (dense first).** Thread a `bool weight_is_e2m1` through the dense
   launcher; at each decode site choose `e2m1_to_e4m3(code)` vs `int4_signed_to_e4m3(code-zp)`.
   Non-templated kernels → either a runtime flag arg or a thin `#define`/wrapper per variant.
   Validate eager bit-exactness of one GEMM vs `convert.py` + `dequant_reference`.
3. **vLLM dispatch hook.** Replace the dead RDNA4 path in
   `CompressedTensorsW4A16Mxfp4.process_weights_after_loading` (`on_gfx1x()` branch — currently
   dequant-to-bf16 + `F.linear`) with: run `convert.py`, stash the W4A8 packed tensors, and route
   the layer through `RocmW4A8Fp8WmmaLinearKernel`. Mirror the adapter's existing weight-loading.
4. **fp16 scale range.** E8M0 spans `2^±127`; the kernel stores fp16 group scales (`2^±15`).
   `convert.py` flags overflow. Real trained block scales sit near weight magnitude (small
   exponents) so fp16 is fine in practice — but confirm on the target checkpoint; if it overflows,
   add an fp32 group-scale path or factor a per-tensor global scale.
5. **MoE experts (the real win).** mxfp4 quantizes MoE experts only (like our 35B). Apply the same
   decode swap + E8M0→scale conversion to the grouped-MoE kernels (`moe_kernel.hip`,
   `moe_gemm_*.h`) and wire `GptOssMxfp4MoEMethod` / `Mxfp4MoEMethod`. This is where most of the
   model weight (and speedup) lives.

## How to re-run the CPU test
```
docker run --rm --entrypoint bash \
  -v /home/pat/code/vllm-gfx1201-mxfp4/w4a8_fp8_wmma:/work vllm22-w4a8:combined \
  -lc 'source /app/.venv/bin/activate && cd /work && python -m mxfp4.test_mxfp4_decode'
```
