"""Hardware test for the AWQ decode-path fix (small-M Triton fallback).

The adapter routes small M (decode) to vllm's triton_w4a16_gemm. The fix builds
the Triton-format zeros as a transpose of our op's (N//8, K//group) zeros into
(K//group, N//8). This test inlines the EXACT triton_w4a16 kernel from this
vllm checkout and verifies that feeding it `our_zeros.t()` yields the AWQ
dequant (q - zp) * scale, matching an fp16 reference.

This isolates the fix without needing a full vLLM/model load: it exercises the
real Triton kernel + the real zero layout (our op's (N//8,K//G) packing,
validated separately by test_correctness.py's asym cases).
"""
import sys

import torch
import triton
import triton.language as tl


# ---- EXACT copy of triton_w4a16_gemm_kernel from this checkout's vllm
#      (vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py) ----
@triton.jit
def triton_w4a16_gemm_kernel(
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
    M, K = a.shape
    N = b_q.shape[1] * 8
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    has_zp = qzeros is not None
    zeros_ptr = qzeros if has_zp else b_q
    # gfx1201 is gfx12 (RDNA4); use the RDNA3.5/gfx1x small-M block sizes path.
    if M <= 32:
        BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
    elif M <= 64:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 32, 64
    if group_size < BLOCK_K:
        BLOCK_K = group_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    triton_w4a16_gemm_kernel[grid](
        a, b_q, scales, zeros_ptr, c, M, N, K,
        a.stride(0), a.stride(1), b_q.stride(0), b_q.stride(1),
        c.stride(0), c.stride(1),
        group_size=group_size, HAS_ZP=has_zp, ZP_BIAS=zp_bias,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c


def pack_kn(w_int4):  # (N,K) int -> our op's (N, K//8) [unused for triton] ...
    pass


def build_tri_bq(w_int4):
    """(N,K) uint4 -> triton b_q [K, N//8] (N packed, nibble n%8)."""
    N, K = w_int4.shape
    w_kn = w_int4.t().contiguous().to(torch.int32)  # (K, N)
    N8 = N // 8
    bq = torch.zeros((K, N8), dtype=torch.int32, device=w_int4.device)
    for j in range(8):
        bq |= (w_kn[:, j::8] & 0xF) << (j * 4)
    return bq


def pack_our_zeros(zeros):
    """(N, G) uint4 -> our op layout (N//8, G), nibble n%8 (== test_correctness.pack_zeros)."""
    N, G = zeros.shape
    z = zeros.to(torch.int32)
    packed = torch.zeros((N // 8, G), dtype=torch.int32, device=z.device)
    for i in range(8):
        packed |= (z[i::8, :] & 0xF) << (i * 4)
    return packed


def run(M, N, K, group_size):
    dev = torch.device("cuda")
    G = K // group_size
    x = torch.randn(M, K, dtype=torch.float16, device=dev)
    w_int4 = torch.randint(0, 16, (N, K), dtype=torch.int8, device=dev)
    scales = torch.randn(N, G, dtype=torch.float16, device=dev).abs() * 0.02 + 0.001
    zeros = torch.randint(0, 16, (N, G), dtype=torch.int8, device=dev)

    # Our op's zero layout (N//8, G), then the adapter's transpose -> (G, N//8).
    our_zeros = pack_our_zeros(zeros)            # (N//8, G)
    tri_zp = our_zeros.t().contiguous()          # (G, N//8)  <-- the fix
    tri_bq = build_tri_bq(w_int4)                # (K, N//8)
    tri_s = scales.t().contiguous()              # (G, N)

    out = triton_w4a16_gemm(x, tri_bq, tri_s, qzeros=tri_zp,
                            group_size=group_size, zp_bias=0)

    # fp16 reference: (q - zp) * scale, then matmul.
    zp_full = zeros.to(torch.float32).repeat_interleave(group_size, dim=1)  # (N,K)
    w_deq = (w_int4.to(torch.float32) - zp_full) * \
        scales.to(torch.float32).repeat_interleave(group_size, dim=1)        # (N,K)
    ref = (x.float() @ w_deq.t()).to(torch.float16)

    diff = (out.float() - ref.float()).abs()
    rel = diff.mean().item() / (ref.float().abs().mean().item() + 1e-9)
    ok = rel < 0.02
    print(f"M={M} N={N} K={K} g={group_size}: mean={diff.mean():.5f} "
          f"max={diff.max():.5f} |ref|mean={ref.abs().float().mean():.4f} "
          f"rel={rel:.5f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("FAIL: no CUDA/HIP device"); sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=== AWQ decode-path (Triton fallback) with adapter's transposed zeros ===")
    shapes = [
        (1, 4096, 4096, 128),   # decode (M=1), g128
        (1, 1024, 1024, 32),    # decode, g32
        (8, 2048, 2048, 128),   # small prefill
        (32, 512, 1024, 32),    # small-M boundary
    ]
    res = [run(M, N, K, g) for M, N, K, g in shapes]
    if all(res):
        print(f"ALL PASSED ({len(res)})"); sys.exit(0)
    print(f"FAIL: {sum(1 for r in res if not r)}/{len(res)}"); sys.exit(1)


if __name__ == "__main__":
    main()
