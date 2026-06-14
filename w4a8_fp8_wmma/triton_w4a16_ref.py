"""Standalone host copy of vLLM's triton_w4a16_gemm (gfx1201 config path).

Extracted verbatim from
  vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py
so the dense W4A8 kernel can be benchmarked against the *real production Triton
W4A16 baseline* on the bare-metal gfx1201 host, where the full `vllm` package is
not importable (build venv lacks vllm's runtime deps). Only torch + triton.

The block-size selection is hard-coded to the on_gfx1x()==True branch (gfx1201
matches "gfx12"), i.e. the RDNA3.5-tuned configs the production path actually
uses on this card. Keep this in sync if vllm's triton_w4a16 block table changes.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _triton_w4a16_gemm_kernel(
    a_ptr, b_ptr, scales_ptr, zeros_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    group_size,
    HAS_ZP: tl.constexpr, ZP_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_bn = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    shifts_row = tl.arange(0, 8) * 4
    shifts_1d_2d = tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 8, 8))
    shifts_1d = tl.reshape(shifts_1d_2d, (BLOCK_N,))
    shifts = tl.broadcast_to(shifts_1d[None, :], (BLOCK_K, BLOCK_N))
    offs_sn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        mask_b = mask_k[:, None] & (offs_bn[None, :] < N // 8)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = (b >> shifts) & 0xF
        g_idx = (k_start * BLOCK_K) // group_size
        scale_offset = g_idx * N + offs_sn
        scale_mask = offs_sn < N
        scales = tl.load(scales_ptr + scale_offset, mask=scale_mask, other=1.0)
        scales = tl.broadcast_to(scales[None, :], (BLOCK_K, BLOCK_N))
        if HAS_ZP:
            zero_offset = g_idx * (N // 8) + offs_bn
            zero_mask = offs_bn < N // 8
            z_packed = tl.load(zeros_ptr + zero_offset, mask=zero_mask, other=0)
            z = tl.interleave(z_packed, z_packed)
            z = tl.interleave(z, z)
            z = tl.interleave(z, z)
            z = (z >> shifts_1d) & 0xF
            z = tl.broadcast_to(z[None, :], (BLOCK_K, BLOCK_N))
        else:
            z = tl.full((BLOCK_K, BLOCK_N), ZP_BIAS, dtype=tl.int32)
        b_fp = (b - z).to(a.dtype) * scales
        accumulator += tl.dot(a, b_fp, out_dtype=tl.float32)
    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def triton_w4a16_gemm(a, b_q, scales, qzeros, group_size, zp_bias=8):
    """a [M,K] fp16; b_q [K,N//8] int32; scales [K//G,N]; qzeros [K//G,N//8] or None."""
    assert a.is_contiguous() and b_q.is_contiguous() and scales.is_contiguous()
    M, K = a.shape
    N = b_q.shape[1] * 8
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    has_zp = qzeros is not None
    zeros_ptr = qzeros if has_zp else b_q
    # gfx1201 -> on_gfx1x()==True branch (RDNA3.5-tuned).
    if M <= 32:
        BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
    elif M <= 64:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 32, 64
    if group_size < BLOCK_K:
        BLOCK_K = group_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _triton_w4a16_gemm_kernel[grid](
        a, b_q, scales, zeros_ptr, c, M, N, K,
        a.stride(0), a.stride(1), b_q.stride(0), b_q.stride(1),
        c.stride(0), c.stride(1),
        group_size=group_size, HAS_ZP=has_zp, ZP_BIAS=zp_bias,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c
