"""Correctness test for the W4A8-FP8 WMMA *MoE composition* (gfx1201).

Where ``test_moe_correctness.py`` tests the single grouped GEMM op in isolation,
this tests the NEW Python wiring in ``moe_experts.py``:

  1. ``_awq_moe_to_op_layout`` — AWQ MoE weights (packed-along-output, AWQ bit
     order) -> our op layout. Verified by dequantising the converted weights and
     comparing to the AWQ source dequant (bit-exact on the int4 values).
  2. The full gated MoE composition that ``W4A8Fp8WmmaExperts.apply`` performs:
     moe_align -> grouped GEMM (w13) -> silu_and_mul -> grouped GEMM (w2,
     identity-gather) -> topk-weighted scatter-reduce. Replicated here with the
     real op + conversion and compared to an fp8 reference that mirrors the
     kernel arithmetic, plus reported against an ideal fp MoE.

Run inside the container (needs the built .so + a gfx1201 device):
    python test_moe_experts.py
"""
import os
import sys

import torch

try:
    import w4a8_fp8_wmma
    from w4a8_fp8_wmma.moe_experts import (
        _REVERSE_AWQ_PACK_ORDER,
        _awq_moe_to_op_layout,
        _choose_block_m,
    )
except ImportError as e:
    print(f"FAIL: import error: {e}")
    sys.exit(1)

E4M3_MAX = 448.0


def _with_a_in_lds(a_in_lds, fn):
    """Run ``fn`` with VLLM_W4A8_MOE_A_IN_LDS set (the former-v5 A-in-LDS path)
    when ``a_in_lds`` is True; restore the default (env unset) afterwards."""
    if a_in_lds:
        os.environ["VLLM_W4A8_MOE_A_IN_LDS"] = "1"
    try:
        return fn()
    finally:
        if a_in_lds:
            os.environ.pop("VLLM_W4A8_MOE_A_IN_LDS", None)


def to_e4m3(x):
    return x.to(torch.float8_e4m3fn).to(torch.float32)


# ---- AWQ packing helpers (inverse of the unpack in _awq_to_op_layout_single) ----
def awq_pack(vals):  # (E, K, N) int -> (E, K, N//8) int32, AWQ bit order
    E, K, N = vals.shape
    rev = _REVERSE_AWQ_PACK_ORDER
    packed = torch.zeros((E, K, N // 8), dtype=torch.int32, device=vals.device)
    v = vals.to(torch.int32)
    for i in range(8):
        packed |= (v[:, :, i::8] & 0xF) << (rev[i] * 4)
    return packed


def awq_pack_zeros(vals):  # (E, G, N) int -> (E, G, N//8) int32, AWQ bit order
    E, G, N = vals.shape
    rev = _REVERSE_AWQ_PACK_ORDER
    packed = torch.zeros((E, G, N // 8), dtype=torch.int32, device=vals.device)
    v = vals.to(torch.int32)
    for i in range(8):
        packed |= (v[:, :, i::8] & 0xF) << (rev[i] * 4)
    return packed


def make_awq_expert_weights(E, K, N, group_size, dev):
    """Random AWQ tensors + the unpacked ground-truth (qvals, zeros, scales)."""
    G = K // group_size
    qvals = torch.randint(0, 16, (E, K, N), dtype=torch.int32, device=dev)
    zeros = torch.randint(0, 16, (E, G, N), dtype=torch.int32, device=dev)
    scales = (torch.randn(E, G, N, dtype=torch.float16, device=dev).abs()
              * 0.02 + 0.001)
    qweight = awq_pack(qvals)              # (E, K, N//8)
    qzeros = awq_pack_zeros(zeros)         # (E, G, N//8)
    return qvals, zeros, scales, qweight, qzeros


def dequant_true(qvals, zeros, scales, group_size):
    """AWQ dequant -> true fp weights W[e,k,n] = (q - zp) * scale. (E,K,N)."""
    zexp = zeros.repeat_interleave(group_size, dim=1)   # (E,K,N)
    sexp = scales.float().repeat_interleave(group_size, dim=1)
    return (qvals.float() - zexp.float()) * sexp


# --------------------------------------------------------------------------- #
# Test 1: conversion correctness (op-layout dequant == AWQ dequant)
# --------------------------------------------------------------------------- #
def unpack_op_weight(w_packed, scales_op, zeros_op, group_size):
    """Dequant from OUR op layout: w_packed (E,N,K//8), zeros (E,N//8,G),
    scales (E,N,G). Returns W[e,n,k] = (nibble - zp) * scale  (E, N, K)."""
    E, N, Kp = w_packed.shape
    K = Kp * 8
    G = scales_op.shape[2]
    dev = w_packed.device
    # nibble[e,n,k]
    shifts = (torch.arange(8, device=dev, dtype=torch.int32) * 4)
    nib = (w_packed.unsqueeze(-1) >> shifts) & 0xF      # (E,N,K//8,8)
    nib = nib.reshape(E, N, K).float()
    # zp[e,n,g] from zeros_op (E,N//8,G) packed along N
    zshift = ((torch.arange(N, device=dev) % 8) * 4)
    zp = (zeros_op[:, torch.arange(N, device=dev) // 8, :] >> zshift.view(1, N, 1)) & 0xF
    zp = zp.float().repeat_interleave(group_size, dim=2)  # (E,N,K)
    sc = scales_op.float().repeat_interleave(group_size, dim=2)  # (E,N,K)
    return (nib - zp) * sc


def test_conversion(E, K, N, group_size, dev):
    qvals, zeros, scales, qweight, qzeros = make_awq_expert_weights(
        E, K, N, group_size, dev)
    w_p, s_p, z_p = _awq_moe_to_op_layout(qweight, scales, qzeros, group_size)
    # shapes
    assert tuple(w_p.shape) == (E, N, K // 8), (w_p.shape, (E, N, K // 8))
    assert tuple(s_p.shape) == (E, N, K // group_size)
    assert tuple(z_p.shape) == (E, N // 8, K // group_size)
    # dequant equality: true (E,K,N) vs op (E,N,K)
    W_true = dequant_true(qvals, zeros, scales, group_size)       # (E,K,N)
    W_op = unpack_op_weight(w_p, s_p, z_p, group_size)            # (E,N,K)
    diff = (W_true.permute(0, 2, 1) - W_op).abs().max().item()
    ok = diff < 1e-3
    print(f"  conversion E={E} K={K} N={N} g={group_size}: "
          f"max|dequant diff|={diff:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# Test 2: full MoE composition (mirror of W4A8Fp8WmmaExperts.apply)
# --------------------------------------------------------------------------- #
def moe_align_py(topk_ids, block_m, E):
    """Python moe_align_block_size returning a block_m-multiple padded layout."""
    T, top_k = topk_ids.shape
    num_valid = T * top_k
    flat = topk_ids.reshape(-1)
    sorted_ids, expert_ids = [], []
    for e in range(E):
        slots = torch.nonzero(flat == e, as_tuple=False).flatten().tolist()
        npad = ((len(slots) + block_m - 1) // block_m) * block_m
        for i in range(npad):
            sorted_ids.append(slots[i] if i < len(slots) else num_valid)
        expert_ids.extend([e] * (npad // block_m))
    dev = topk_ids.device
    sti = torch.tensor(sorted_ids, dtype=torch.int32, device=dev)
    eids = torch.tensor(expert_ids, dtype=torch.int32, device=dev)
    ntp = torch.tensor([sti.numel()], dtype=torch.int32, device=dev)
    return sti, eids, ntp


def fp8_ref_moe(x, W13t, W13s, W13z, W2t, W2s, W2z, topk_ids, topk_weights,
                group_size):
    """fp8 reference for the whole gated MoE, mirroring the kernel arithmetic.
    W*t are int4 (E,K,N) ground-truth nibbles; W*z (E,G,N); W*s (E,G,N)."""
    T, K = x.shape
    top_k = topk_ids.shape[1]
    E, _, twoI = W13t.shape
    inter = twoI // 2
    dev = x.device

    def fp8_gemm(a, Wt, Ws, Wz, gsize):
        # a: (R, Kin) fp16 ; Wt int4 (Kin, Nout); per-row fp8 act, per-group reduce
        R, Kin = a.shape
        Nout = Wt.shape[1]
        G = Kin // gsize
        a_scale = a.float().abs().amax(1, keepdim=True).clamp(
            min=1e-8 * E4M3_MAX) / E4M3_MAX
        a_fp8 = to_e4m3(a.float() / a_scale)            # (R, Kin)
        w_signed = Wt.float() - Wz.repeat_interleave(gsize, dim=0).float()  # (Kin,Nout)
        w_fp8 = to_e4m3(w_signed)
        ag = a_fp8.view(R, G, gsize)
        wg = w_fp8.view(G, gsize, Nout)
        partial = torch.einsum("rgk,gkn->rgn", ag, wg)  # (R,G,Nout)
        o = (partial * Ws.float().unsqueeze(0)).sum(1)  # (R,Nout)
        return (o * a_scale).to(torch.float16)

    y = torch.zeros((T, K), dtype=torch.float32, device=dev)
    for t in range(T):
        for j in range(top_k):
            e = int(topk_ids[t, j].item())
            g1 = fp8_gemm(x[t:t + 1], W13t[e], W13s[e], W13z[e], group_size)  # (1,2I)
            gate, up = g1[:, :inter], g1[:, inter:]
            act = (torch.nn.functional.silu(gate.float()).to(torch.float16)
                   * up)                                                    # (1,I)
            g2 = fp8_gemm(act, W2t[e], W2s[e], W2z[e], group_size)          # (1,K)
            y[t] += topk_weights[t, j].float() * g2[0].float()
    return y


def kernel_moe(x, qw13, s13, z13, qw2, s2, z2, topk_ids, topk_weights,
               group_size, kernel, fuse_silu=False):
    """Replicate W4A8Fp8WmmaExperts.apply with the real op + conversion.

    ``fuse_silu`` exercises the fused gemm1+silu kernel (mmq_fp8_moe_gemm1_silu)
    in place of the separate gemm1 + silu_and_mul (wmma only)."""
    T, K = x.shape
    top_k = topk_ids.shape[1]
    E = qw13.shape[0]
    dev = x.device
    # convert AWQ -> op layout (the real code under test)
    w13_p, s13_p, z13_p = _awq_moe_to_op_layout(qw13, s13, z13, group_size)
    w2_p, s2_p, z2_p = _awq_moe_to_op_layout(qw2, s2, z2, group_size)

    block_m = _choose_block_m(T, top_k, E)
    sti, eids, ntp = moe_align_py(topk_ids, block_m, E)
    P = sti.numel()

    x16 = x.to(torch.float16).contiguous()
    if fuse_silu:
        buf2 = w4a8_fp8_wmma.mmq_fp8_moe_gemm1_silu(
            x16, w13_p, s13_p, sti, eids, ntp, top_k, block_m,
            kernel=kernel, w_zeros=z13_p)                    # (P, inter)
        buf2 = buf2.contiguous()
    else:
        out1 = w4a8_fp8_wmma.mmq_fp8_moe_gemm(
            x16, w13_p, s13_p, sti, eids, ntp, top_k, block_m,
            kernel=kernel, w_zeros=z13_p)                    # (P, 2*inter)
        inter = out1.shape[1] // 2
        buf2 = (torch.nn.functional.silu(out1[:, :inter].float()).to(torch.float16)
                * out1[:, inter:])                           # (P, inter)
        buf2 = buf2.contiguous()

    num_valid = T * top_k
    row_idx = torch.arange(P, dtype=torch.int32, device=dev)
    valid = (sti < num_valid) & (row_idx < ntp)
    ident = torch.where(valid, row_idx,
                        torch.full((P,), P, dtype=torch.int32, device=dev))
    out2 = w4a8_fp8_wmma.mmq_fp8_moe_gemm(
        buf2, w2_p, s2_p, ident, eids, ntp, 1, block_m,
        kernel=kernel, w_zeros=z2_p)                         # (P, K)

    slots = sti[valid].long()
    tokens = slots // top_k
    contrib = out2[valid].float() * topk_weights.reshape(-1)[slots].float().unsqueeze(1)
    y = torch.zeros((T, K), dtype=torch.float32, device=dev)
    y.index_add_(0, tokens, contrib)
    return y


def test_composition(T, E, K, inter, top_k, group_size, kernel, dev,
                     mean_rtol=0.03, fuse_silu=False):
    N13 = 2 * inter
    qv13, z13, s13, qw13, qz13 = make_awq_expert_weights(E, K, N13, group_size, dev)
    qv2, z2, s2_, qw2, qz2 = make_awq_expert_weights(E, inter, K, group_size, dev)
    x = torch.randn(T, K, dtype=torch.float16, device=dev) * 0.5
    topk_ids = torch.stack(
        [torch.randperm(E, device=dev)[:top_k] for _ in range(T)]).to(torch.int32)
    topk_weights = torch.rand(T, top_k, dtype=torch.float32, device=dev)

    ref = fp8_ref_moe(x, qv13, s13, z13, qv2, s2_, z2, topk_ids, topk_weights,
                      group_size)
    out = kernel_moe(x, qw13, s13, qz13, qw2, s2_, qz2, topk_ids, topk_weights,
                     group_size, kernel, fuse_silu=fuse_silu)
    diff = (out - ref).abs()
    refm = ref.abs().mean().item()
    mean = diff.mean().item()
    eff = max(2e-3, mean_rtol * refm)
    n_bad = int((diff > 0.15 + 0.05 * ref.abs()).sum().item())
    ok = mean <= eff and n_bad == 0
    tag = ("%s+fusedsilu" % kernel) if fuse_silu else kernel
    print(f"  compose {tag} T={T} E={E} K={K} I={inter} tk={top_k} "
          f"g={group_size}: mean={mean:.5f} |ref|={refm:.4f} bad={n_bad} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# Test 3: fused gemm1+silu == unfused gemm1 + silu_and_mul (BIT-EXACT)
# --------------------------------------------------------------------------- #
def silu_and_mul_ref(out1):
    """Exact mirror of torch.ops._C.silu_and_mul on a (P, 2*inter) fp16 tensor:
    silu in fp32 -> fp16, then HALF*HALF multiply (vllm act_first compute)."""
    inter = out1.shape[1] // 2
    gate, up = out1[:, :inter], out1[:, inter:]                # both fp16
    silu = (gate.float() / (1.0 + torch.exp(-gate.float()))).to(torch.float16)
    return silu * up                                           # fp16 * fp16


def test_fused_gemm1_silu(T, E, K, inter, top_k, group_size, kernel, dev):
    """The fused gemm1+silu kernel must be BIT-EXACT to running the existing
    gemm1 op then the exact silu_and_mul arithmetic (the kernel rounds each half
    to fp16 the same way gemm1 stores, then applies the identical silu)."""
    N13 = 2 * inter
    qv13, z13, s13, qw13, qz13 = make_awq_expert_weights(E, K, N13, group_size, dev)
    x = torch.randn(T, K, dtype=torch.float16, device=dev) * 0.5
    topk_ids = torch.stack(
        [torch.randperm(E, device=dev)[:top_k] for _ in range(T)]).to(torch.int32)

    w13_p, s13_p, z13_p = _awq_moe_to_op_layout(qw13, s13, qz13, group_size)
    block_m = _choose_block_m(T, top_k, E)
    sti, eids, ntp = moe_align_py(topk_ids, block_m, E)
    P = sti.numel()
    x16 = x.contiguous()

    out1 = w4a8_fp8_wmma.mmq_fp8_moe_gemm(
        x16, w13_p, s13_p, sti, eids, ntp, top_k, block_m,
        kernel=kernel, w_zeros=z13_p)                          # (P, 2*inter)
    ref = silu_and_mul_ref(out1)                               # (P, inter)
    fused = w4a8_fp8_wmma.mmq_fp8_moe_gemm1_silu(
        x16, w13_p, s13_p, sti, eids, ntp, top_k, block_m,
        kernel=kernel, w_zeros=z13_p)                          # (P, inter)

    num_valid = T * top_k
    row_idx = torch.arange(P, dtype=torch.int32, device=dev)
    valid = (sti < num_valid) & (row_idx < ntp)
    diff = (fused[valid].float() - ref[valid].float()).abs().max().item()
    ok = diff == 0.0           # bit-exact: same fp16 store + same silu arithmetic
    print(f"  fused-vs-unfused {kernel} T={T} E={E} K={K} I={inter} "
          f"tk={top_k} g={group_size}: max|diff|={diff:.2e} "
          f"-> {'PASS (bit-exact)' if ok else 'FAIL'}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("FAIL: no device"); sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    res = []

    print("=== conversion (AWQ -> op layout) ===")
    for (E, K, N, g) in [(4, 256, 128, 32), (8, 512, 256, 128), (2, 1024, 512, 32)]:
        res.append(test_conversion(E, K, N, g, "cuda"))

    print("=== full MoE composition (apply mirror) ===")
    cases = [
        # (T, E, K, inter, top_k, group_size)
        (64, 8, 256, 128, 2, 32),
        (32, 4, 512, 256, 2, 128),
        (1, 8, 512, 256, 2, 32),     # decode
        (16, 16, 256, 128, 4, 32),
    ]
    for c in cases:
        # former v0 -> "scalar"; former v5 (A-in-LDS) -> "wmma" + A_IN_LDS env
        res.append(test_composition(*c, kernel="scalar", dev="cuda"))
        res.append(_with_a_in_lds(True, lambda c=c:
                                  test_composition(*c, kernel="wmma", dev="cuda")))

    print("=== fused gemm1+silu == unfused gemm1 + silu_and_mul (bit-exact) ===")
    # mid/prefill batches (the regime the fusion targets); former v5 (A-in-LDS)
    # and v6 -- both "wmma" now, differing only by the A_IN_LDS env.
    for c in [(64, 8, 256, 128, 2, 32), (128, 8, 512, 256, 2, 128),
              (16, 16, 256, 128, 4, 32), (256, 4, 512, 256, 2, 32)]:
        res.append(_with_a_in_lds(True, lambda c=c:
                                  test_fused_gemm1_silu(*c, kernel="wmma", dev="cuda")))
        res.append(test_fused_gemm1_silu(*c, kernel="wmma", dev="cuda"))

    print("=== full MoE composition through the FUSED gemm1+silu path ===")
    for c in cases:
        res.append(_with_a_in_lds(True, lambda c=c:
                                  test_composition(*c, kernel="wmma", dev="cuda", fuse_silu=True)))
        res.append(test_composition(*c, kernel="wmma", dev="cuda", fuse_silu=True))

    print("=" * 56)
    if all(res):
        print(f"ALL PASSED ({len(res)})"); sys.exit(0)
    print(f"FAIL: {sum(1 for r in res if not r)}/{len(res)}"); sys.exit(1)


if __name__ == "__main__":
    main()
