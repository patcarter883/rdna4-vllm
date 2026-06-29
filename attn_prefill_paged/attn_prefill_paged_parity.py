"""Numeric parity for the native paged/chunked-prefill kernel (GPU).

Builds a batch of sequences each with (prefix_len, q_len): the prefix sits in a SHUFFLED paged cache,
the q_len NEW tokens are the packed query. Checks torch.ops.attn_prefill_paged.flash_prefill_paged vs
an fp32 SDPA reference where new query i (global pos prefix_len+i) attends causally over keys
0..prefix_len+i (the cached prefix ⧺ the new tokens). The kernel is fp32-internal / bf16-out, so we
judge with cos-sim (primary) + max|Δ| vs a bf16-rounded reference (same bar as attn_hip).

Run inside vllm22-w4a8:combined under a 1-card lease (see attn_decode_parity.py for the docker line).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import op  # noqa: F401  loads the .so + registers torch.ops.attn_prefill_paged.*

DEV = "cuda"
torch.manual_seed(0)
COS_MIN = 0.9995
# A faithful fp32-internal/bf16-out kernel differs from the bf16-rounded fp32 reference by at most a
# couple of bf16 ULP (reduction-order differences straddling a rounding boundary). bf16 ULP is
# RELATIVE (~2^-7 of the magnitude), so the bar is rtol≈2 ULP + a small atol — NOT a fixed abs delta
# (which mis-flags peaked rows where |out|~1-4). cos-sim (magnitude-robust) is the primary gate.
ULP_RTOL = 0.016   # ~2 bf16 ULP
ULP_ATOL = 0.004


def check(name, specs, Hq, Hk, D, sw=0, block_size=16) -> bool:
    """specs: list of (prefix_len, q_len) per sequence."""
    scale = D ** -0.5
    S = len(specs)
    ctx = [p + q for p, q in specs]
    qlens = [q for _, q in specs]
    total_q = sum(qlens)
    maxctx = max(ctx)
    rep = Hq // Hk

    kd = torch.randn(S, maxctx, Hk, D, device=DEV, dtype=torch.bfloat16)
    vd = torch.randn(S, maxctx, Hk, D, device=DEV, dtype=torch.bfloat16)
    q = torch.randn(total_q, Hq, D, device=DEV, dtype=torch.bfloat16)

    bps = (maxctx + block_size - 1) // block_size
    num_blocks = S * bps + 3
    perm = torch.randperm(num_blocks, device=DEV).int()
    k_cache = torch.zeros(num_blocks, block_size, Hk, D, device=DEV, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros(S, bps, device=DEV, dtype=torch.int32)
    for b in range(S):
        for lb in range(bps):
            phys = int(perm[b * bps + lb].item())
            block_table[b, lb] = phys
            for off in range(block_size):
                j = lb * block_size + off
                if j < ctx[b]:
                    k_cache[phys, off] = kd[b, j]
                    v_cache[phys, off] = vd[b, j]
    cu = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0).tolist()), device=DEV, dtype=torch.int32)
    ctxt = torch.tensor(ctx, device=DEV, dtype=torch.int32)

    got = torch.ops.attn_prefill_paged.flash_prefill_paged(
        q, k_cache, v_cache, block_table, cu, ctxt, scale, 1, sw, max(qlens), 0).float()

    ref = torch.empty(total_q, Hq, D, device=DEV)
    for b in range(S):
        p, ql = specs[b]
        cl = ctx[b]
        off = int(cu[b].item())
        kbe = kd[b, :cl].float().repeat_interleave(rep, dim=1)   # [cl, Hq, D]
        vbe = vd[b, :cl].float().repeat_interleave(rep, dim=1)
        qi = q[off:off + ql].float()                            # [ql, Hq, D]
        scores = torch.einsum("qhd,khd->qhk", qi, kbe) * scale  # [ql, Hq, cl]
        qpos = p + torch.arange(ql, device=DEV)
        kpos = torch.arange(cl, device=DEV)
        mask = kpos[None, :] > qpos[:, None]                    # causal (prefix-offset)
        if sw > 0:
            mask = mask | ((qpos[:, None] - kpos[None, :]) >= sw)
        scores = scores.masked_fill(mask[:, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        ref[off:off + ql] = torch.einsum("qhk,khd->qhd", attn, vbe)

    ref_b = ref.bfloat16().float()
    d = (got - ref_b).abs().max().item()
    # within-2-ULP violation (0 = every element within the bf16 rounding bound)
    viol = ((got - ref_b).abs() - (ULP_ATOL + ULP_RTOL * ref_b.abs())).clamp(min=0).max().item()
    cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    ok = (cos >= COS_MIN) and (viol <= 1e-6)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} cos={cos:.6f}  max|Δ|bf16={d:.3e}  ulp_viol={viol:.2e}")
    return ok


def check_fp8(name, specs, Hq, Hk, D, k_descale=1.0, v_descale=1.0, sw=0, block_size=16) -> bool:
    """fp8 (e4m3) paged KV + per-tensor descale. The reference dequantizes the SAME fp8 bytes
    (* descale) — fp8 quant error is in both kernel and ref, so the residual is only bf16 output
    rounding. Exercises the descale fold (k into the score, v into the output)."""
    scale = D ** -0.5
    S = len(specs)
    ctx = [p + q for p, q in specs]
    qlens = [q for _, q in specs]
    total_q = sum(qlens)
    maxctx = max(ctx)
    rep = Hq // Hk

    kd = torch.randn(S, maxctx, Hk, D, device=DEV).to(torch.float8_e4m3fn)
    vd = torch.randn(S, maxctx, Hk, D, device=DEV).to(torch.float8_e4m3fn)
    q = torch.randn(total_q, Hq, D, device=DEV, dtype=torch.bfloat16)

    bps = (maxctx + block_size - 1) // block_size
    num_blocks = S * bps + 3
    perm = torch.randperm(num_blocks, device=DEV).int()
    k_cache = torch.zeros(num_blocks, block_size, Hk, D, device=DEV, dtype=torch.float8_e4m3fn)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros(S, bps, device=DEV, dtype=torch.int32)
    for b in range(S):
        for lb in range(bps):
            phys = int(perm[b * bps + lb].item())
            block_table[b, lb] = phys
            for off in range(block_size):
                j = lb * block_size + off
                if j < ctx[b]:
                    k_cache[phys, off] = kd[b, j]
                    v_cache[phys, off] = vd[b, j]
    cu = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0).tolist()), device=DEV, dtype=torch.int32)
    ctxt = torch.tensor(ctx, device=DEV, dtype=torch.int32)

    got = torch.ops.attn_prefill_paged.flash_prefill_paged_fp8(
        q, k_cache, v_cache, block_table, cu, ctxt, scale, k_descale, v_descale, 1, sw, max(qlens), 0).float()

    kdq = kd.float() * k_descale
    vdq = vd.float() * v_descale
    ref = torch.empty(total_q, Hq, D, device=DEV)
    for b in range(S):
        p, ql = specs[b]
        cl = ctx[b]
        off = int(cu[b].item())
        kbe = kdq[b, :cl].repeat_interleave(rep, dim=1)
        vbe = vdq[b, :cl].repeat_interleave(rep, dim=1)
        qi = q[off:off + ql].float()
        scores = torch.einsum("qhd,khd->qhk", qi, kbe) * scale
        qpos = p + torch.arange(ql, device=DEV)
        kpos = torch.arange(cl, device=DEV)
        mask = kpos[None, :] > qpos[:, None]
        if sw > 0:
            mask = mask | ((qpos[:, None] - kpos[None, :]) >= sw)
        scores = scores.masked_fill(mask[:, None, :], float("-inf"))
        ref[off:off + ql] = torch.einsum("qhk,khd->qhd", F.softmax(scores, dim=-1), vbe)

    ref_b = ref.bfloat16().float()
    viol = ((got - ref_b).abs() - (ULP_ATOL + ULP_RTOL * ref_b.abs())).clamp(min=0).max().item()
    cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    ok = (cos >= COS_MIN) and (viol <= 1e-6)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} cos={cos:.6f}  ulp_viol={viol:.2e}")
    return ok


def main() -> None:
    print("=== attn_prefill_paged parity (vs fp32 SDPA over prefix⧺new, prefix-offset causal) ===")
    ok = True
    ok &= check("cold prefix0 q64 D128 Hq16/Hk2", [(0, 64)], 16, 2, 128)        # == dense prefill
    ok &= check("extend prefix100 q32 D128     ", [(100, 32)], 16, 2, 128)
    ok &= check("batch mixed [(0,16)(50,40)(128,8)]", [(0, 16), (50, 40), (128, 8)], 16, 2, 128)
    ok &= check("ragged prefix37 q27 (ctx64)    ", [(37, 27)], 16, 2, 128)
    ok &= check("SWA=64 prefix200 q32 D128      ", [(200, 32)], 16, 2, 128, sw=64)
    ok &= check("MHA prefix64 q48 D128 Hq8/Hk8  ", [(64, 48)], 8, 8, 128)
    ok &= check("D64 prefix30 q30 Hq8/Hk1       ", [(30, 30)], 8, 1, 64)
    ok &= check("long prefix1000 q64 bs32 D128  ", [(1000, 64)], 16, 2, 128, block_size=32)
    print("--- head_dim 256 (BR=BC=16 tile) ---")
    ok &= check("D256 cold prefix0 q48 Hq16/Hk2 ", [(0, 48)], 16, 2, 256)
    ok &= check("D256 extend prefix100 q32      ", [(100, 32)], 16, 2, 256)
    ok &= check("D256 batch [(0,16)(50,40)(128,8)]", [(0, 16), (50, 40), (128, 8)], 16, 2, 256)
    ok &= check("D256 ragged prefix37 q27       ", [(37, 27)], 16, 2, 256)
    ok &= check("D256 SWA=64 prefix200 q32      ", [(200, 32)], 16, 2, 256, sw=64)
    ok &= check("D256 MHA prefix64 q48 Hq8/Hk8  ", [(64, 48)], 8, 8, 256)
    print("--- fp8-KV (e4m3) + per-tensor descale fold ---")
    ok &= check_fp8("fp8 cold prefix0 q64 D128      ", [(0, 64)], 16, 2, 128)
    ok &= check_fp8("fp8 extend prefix100 q32 D128  ", [(100, 32)], 16, 2, 128)
    ok &= check_fp8("fp8 batch [(0,16)(50,40)(128,8)]", [(0, 16), (50, 40), (128, 8)], 16, 2, 128)
    ok &= check_fp8("fp8 descale k2.0/v0.5 ext200   ", [(200, 32)], 16, 2, 128, k_descale=2.0, v_descale=0.5)
    ok &= check_fp8("fp8 SWA=64 prefix200 q32       ", [(200, 32)], 16, 2, 128, sw=64)
    ok &= check_fp8("fp8 D256 extend prefix100 q32  ", [(100, 32)], 16, 2, 256)
    ok &= check_fp8("fp8 D256 descale k2.0/v0.5     ", [(200, 32)], 16, 2, 256, k_descale=2.0, v_descale=0.5)
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
