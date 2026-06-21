"""mxfp4 (OCP E2M1 + E8M0) -> W4A8-FP8-WMMA kernel-native weight format.

The W4A8 kernel is fundamentally a "decode a 4-bit code into fp8 e4m3, then WMMA fp8xfp8" engine
(see tile_config.h::e2m1_to_e4m3). mxfp4 is therefore NOT a new compute path -- it is a different
4-bit *decode table* plus a power-of-two (E8M0) group scale. This module does the load-time format
conversion so the existing kernel container ((N, K//8) packed nibbles + (N, K//group) fp16 scales)
carries mxfp4 data:

  * E2M1 weight nibbles are repacked into the kernel's (N, K//8) int32 container UNCHANGED -- the
    *codes* are stored verbatim; the kernel's e2m1_to_e4m3 LUT (not the int4 subtract-zp path) turns
    each code into the right fp8 byte. So this is purely a re-packing, no value remap.
  * E8M0 per-group scales (uint8, value = 2^(s-127)) are converted to the kernel's existing fp16
    per-group scale array. The kernel epilogue (out_acc * wscale * a_scale) is unchanged.
  * No zero-points: mxfp4 is symmetric, so w_zeros = None (kernel uses implicit-symmetric decode).

Input format (compressed-tensors "mxfp4-pack-quantized", a.k.a. CompressedTensorsW4A4Mxfp4):
  weight_packed : uint8  (N, K//2)   -- 2 E2M1 nibbles/byte, low nibble = lower K index
  weight_scale  : uint8  (N, K//32)  -- E8M0, one shared exponent per 32-element block

Output (kernel-native, matches vllm_adapter.py weight contract):
  w_packed : int32  (N, K//8)        -- 8 E2M1 codes/word, code j at bits [4j, 4j+3]
  scales   : fp16   (N, K//32)       -- 2^(s-127), per group
  group_size = 32 (the OCP MX block size)

CPU-only; no GPU and no kernel build required.
"""
from __future__ import annotations

import torch

# OCP E2M1 codebook, indexed by the raw 4-bit code (bit3=sign, bits2-1=exp, bit0=mantissa).
# MUST match vLLM's _FP4_E2M1_LUT and tile_config.h::e2m1_to_e4m3's LUT.
FP4_E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

OCP_MX_BLOCK_SIZE = 32  # E8M0 shares one exponent per 32-element block
E8M0_BIAS = 127


def unpack_e2m1_nibbles(weight_packed: torch.Tensor) -> torch.Tensor:
    """(N, K//2) uint8 -> (N, K) uint8 codes. Low nibble = lower (even) K index."""
    assert weight_packed.dtype == torch.uint8, weight_packed.dtype
    assert weight_packed.ndim == 2, weight_packed.shape
    n, k_half = weight_packed.shape
    codes = torch.empty((n, k_half * 2), dtype=torch.uint8)
    codes[:, 0::2] = weight_packed & 0x0F
    codes[:, 1::2] = (weight_packed >> 4) & 0x0F
    return codes


def pack_codes_to_int32(codes: torch.Tensor) -> torch.Tensor:
    """(N, K) uint8 codes -> (N, K//8) int32, code j at bits [4j, 4j+3] (low nibble first).

    Matches the kernel's read: ((word >> (j*4)) & 0xF) for j in 0..7.
    """
    n, k = codes.shape
    assert k % 8 == 0, f"K={k} must be a multiple of 8 for int32 packing"
    c = codes.to(torch.int64) & 0xF                      # widen so shifts don't overflow
    words = torch.zeros((n, k // 8), dtype=torch.int64)
    for j in range(8):
        words |= c[:, j::8] << (4 * j)
    # Wrapping cast to int32 preserves the 32-bit pattern (incl. a set bit 31 -> negative). The
    # kernel does ((word >> 4*j) & 0xF), and the trailing &0xF recovers the right nibble regardless
    # of the word's sign / arithmetic-shift fill, so a "negative" int32 here is harmless.
    return (words & 0xFFFFFFFF).to(torch.int32)


def e8m0_to_fp16_scales(weight_scale: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """(N, K//32) uint8 E8M0 -> (N, K//32) fp16 scales = 2^(s-127).

    Returns (scales_fp16, info). E8M0 spans 2^-127..2^128, which OVERFLOWS fp16's ~2^15 range; the
    kernel stores fp16 group scales, so we surface any out-of-range exponents rather than silently
    saturating. Real trained mxfp4 block scales sit near the weight magnitude (small exponents), so
    this is informational for typical checkpoints, but a robust path may need fp32 group scales.
    """
    assert weight_scale.dtype == torch.uint8, weight_scale.dtype
    exp = weight_scale.to(torch.int32) - E8M0_BIAS        # true exponent
    # 255 is the E8M0 NaN code; flag if present.
    nan_count = int((weight_scale == 0xFF).sum())
    fp16_max_exp, fp16_min_norm_exp = 15, -14
    over = int((exp > fp16_max_exp).sum())
    under = int(((exp < fp16_min_norm_exp) & (weight_scale != 0)).sum())
    scales = torch.pow(torch.tensor(2.0, dtype=torch.float32), exp.float()).to(torch.float16)
    info = {
        "exp_min": int(exp.min()), "exp_max": int(exp.max()),
        "fp16_overflow_groups": over, "fp16_subnormal_groups": under,
        "e8m0_nan_groups": nan_count,
        "fp16_range_ok": over == 0 and nan_count == 0,
    }
    return scales, info


def convert_mxfp4_weight(weight_packed: torch.Tensor,
                         weight_scale: torch.Tensor) -> dict:
    """Full conversion for one mxfp4 linear/expert weight matrix.

    weight_packed (N, K//2) uint8, weight_scale (N, K//32) uint8 ->
    {w_packed (N,K//8) int32, scales (N,K//32) fp16, w_zeros None, group_size 32, scale_info}.
    """
    n, k_half = weight_packed.shape
    k = k_half * 2
    ns, k_groups = weight_scale.shape
    assert ns == n, f"row mismatch: weight {n} vs scale {ns}"
    assert k_groups == k // OCP_MX_BLOCK_SIZE, (
        f"scale groups {k_groups} != K//{OCP_MX_BLOCK_SIZE} = {k // OCP_MX_BLOCK_SIZE}")

    codes = unpack_e2m1_nibbles(weight_packed)
    w_packed = pack_codes_to_int32(codes)
    scales, scale_info = e8m0_to_fp16_scales(weight_scale)
    return {
        "w_packed": w_packed,          # (N, K//8) int32, E2M1 codes (decode via e2m1_to_e4m3)
        "scales": scales,              # (N, K//32) fp16
        "w_zeros": None,               # symmetric
        "group_size": OCP_MX_BLOCK_SIZE,
        "scale_info": scale_info,
        "shape": (n, k),
    }


def convert_mxfp4_moe(weight_packed: torch.Tensor,
                      weight_scale: torch.Tensor) -> dict:
    """Stacked per-expert MoE variant of convert_mxfp4_weight.

    weight_packed (E, N, K//2) uint8, weight_scale (E, N, K//32) uint8 ->
    {w_packed (E,N,K//8) int32, scales (E,N,K//32) fp16, w_zeros None, group_size 32}.
    The kernel's MoE ops take exactly this 3D (E,N,*) layout. Flattens E into the row dim,
    reuses the 2D path, reshapes back (the repack/scale math is per-row, E-independent).
    """
    assert weight_packed.ndim == 3 and weight_scale.ndim == 3, (
        weight_packed.shape, weight_scale.shape)
    e, n, k_half = weight_packed.shape
    k = k_half * 2
    flat = convert_mxfp4_weight(weight_packed.reshape(e * n, k_half),
                                weight_scale.reshape(e * n, k // OCP_MX_BLOCK_SIZE))
    return {
        "w_packed": flat["w_packed"].reshape(e, n, k // 8).contiguous(),
        "scales": flat["scales"].reshape(e, n, k // OCP_MX_BLOCK_SIZE).contiguous(),
        "w_zeros": None,
        "group_size": OCP_MX_BLOCK_SIZE,
        "scale_info": flat["scale_info"],
        "shape": (e, n, k),
    }


def dequant_reference(weight_packed: torch.Tensor,
                      weight_scale: torch.Tensor) -> torch.Tensor:
    """Reference dequant matching vLLM's _FP4_E2M1_LUT * 2^(s-127). (N, K) float32.

    This is the GOLDEN value the kernel decode path must reproduce. Used by the host bit-exactness
    test; not on the kernel hot path.
    """
    codes = unpack_e2m1_nibbles(weight_packed).to(torch.int64)
    lut = torch.tensor(FP4_E2M1_LUT, dtype=torch.float32)
    w = lut[codes]                                              # (N, K)
    exp = weight_scale.to(torch.int32) - E8M0_BIAS
    scale = torch.pow(torch.tensor(2.0), exp.float())          # (N, K//32)
    scale = scale.repeat_interleave(OCP_MX_BLOCK_SIZE, dim=-1)  # (N, K)
    return w * scale
