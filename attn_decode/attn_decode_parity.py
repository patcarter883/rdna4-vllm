"""Numeric parity for the native attn_decode flash-decode kernel (GPU).

Checks torch.ops.attn_decode.flash_decode against a pure-torch fp32 reference (single-query SDPA
over the cached KV, with GQA + optional sliding window). The kernel accumulates scores and the
output in fp32, so a faithful kernel matches to a few e-3 in bf16; a real bug (bad cross-warp
combine, wrong strided d-indexing, GQA head map, or mask) shows as a large max|Δ|.

Run inside the combined ROCm image UNDER a 1-card lease (executes HIP kernels):
    scripts/gpu-lease.sh -n 1 -- bash -c 'docker run --rm \
      -v <repo>:/engine -e HIP_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0 \
      -v <repo>/.triton-cache-combined:/root/.triton \
      --entrypoint bash vllm22-w4a8:combined -lc \
      "source /app/.venv/bin/activate && cd /engine/attn_decode && \
       GPU_ARCHS=gfx1201 python setup.py build_ext --inplace && python attn_decode_parity.py"'
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import op  # noqa: F401  loads the .so + registers torch.ops.attn_decode.* (run from inside the pkg dir)

DEV = "cuda"
torch.manual_seed(0)


def ref_decode(q, k, v, scale, sliding_window):
    """q:[B,Hq,D]  k/v:[B,S,Hk,D] -> [B,Hq,D]. fp32 reference; GQA via head repeat."""
    B, Hq, D = q.shape
    S, Hk = k.shape[1], k.shape[2]
    rep = Hq // Hk
    qf = q.float().unsqueeze(2)                                   # [B,Hq,1,D]
    kf = k.float().repeat_interleave(rep, dim=2).permute(0, 2, 1, 3)  # [B,Hq,S,D]
    vf = v.float().repeat_interleave(rep, dim=2).permute(0, 2, 1, 3)
    attn = torch.matmul(qf, kf.transpose(-1, -2)) * scale         # [B,Hq,1,S]
    if sliding_window > 0:
        j = torch.arange(S, device=q.device)
        masked = (S - 1 - j) >= sliding_window                    # older than window
        attn = attn.masked_fill(masked[None, None, None, :], float("-inf"))
    out = torch.matmul(F.softmax(attn, dim=-1), vf)               # [B,Hq,1,D]
    return out.squeeze(2).contiguous()


def check(name, B, Hq, Hk, S, D, sw=0, tol=4e-3) -> bool:
    scale = D ** -0.5
    q = torch.randn(B, Hq, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(B, S, Hk, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(B, S, Hk, D, device=DEV, dtype=torch.bfloat16)
    got = torch.ops.attn_decode.flash_decode(q, k, v, scale, sw).float()
    ref = ref_decode(q, k, v, scale, sw)
    d = (got - ref).abs().max().item()
    scl = ref.abs().mean().item()
    ok = d <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} max|Δ|={d:.3e}  (ref |·|~{scl:.3e})")
    return ok


def check_paged(name, B, Hq, Hk, D, ctx_lens, block_size=16, sw=0, tol=4e-3) -> bool:
    """Scatter dense K/V into a SHUFFLED paged cache + block table, check it matches the
    dense fp32 reference. Shuffled physical blocks exercise the block-table indirection."""
    scale = D ** -0.5
    S = max(ctx_lens)
    q = torch.randn(B, Hq, D, device=DEV, dtype=torch.bfloat16)
    # dense KV (the ground truth), per-seq valid prefix = ctx_lens[b]
    kd = torch.randn(B, S, Hk, D, device=DEV, dtype=torch.bfloat16)
    vd = torch.randn(B, S, Hk, D, device=DEV, dtype=torch.bfloat16)
    blocks_per_seq = (S + block_size - 1) // block_size
    num_blocks = B * blocks_per_seq + 3                       # +slack
    # shuffled physical block ids so logical != physical (real indirection test)
    perm = torch.randperm(num_blocks, device=DEV).int()
    k_cache = torch.zeros(num_blocks, block_size, Hk, D, device=DEV, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros(B, blocks_per_seq, device=DEV, dtype=torch.int32)
    ctx = torch.tensor(ctx_lens, device=DEV, dtype=torch.int32)
    for b in range(B):
        for lb in range(blocks_per_seq):
            phys = int(perm[b * blocks_per_seq + lb].item())
            block_table[b, lb] = phys
            for off in range(block_size):
                j = lb * block_size + off
                if j < S:
                    k_cache[phys, off] = kd[b, j]
                    v_cache[phys, off] = vd[b, j]
    got = torch.ops.attn_decode.flash_decode_paged(q, k_cache, v_cache, block_table, ctx, scale, sw).float()
    # reference: mask keys >= ctx_len[b]
    ref = torch.empty(B, Hq, D, device=DEV)
    for b in range(B):
        cl = ctx_lens[b]
        ref[b] = ref_decode(q[b:b+1], kd[b:b+1, :cl], vd[b:b+1, :cl], scale, sw)[0]
    d = (got - ref).abs().max().item()
    ok = d <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} max|Δ|={d:.3e}")
    return ok


def check_paged_fp8(name, B, Hq, Hk, D, ctx_lens, k_descale=1.0, v_descale=1.0,
                    block_size=16, sw=0, tol=8e-3) -> bool:
    """fp8 (e4m3) paged KV. The reference dequantizes the SAME fp8 bytes (* descale) so the only
    error vs the kernel is bf16 output rounding — fp8 quant error is in both. Exercises the
    per-tensor descale fold (k into the score, v into the output)."""
    scale = D ** -0.5
    S = max(ctx_lens)
    q = torch.randn(B, Hq, D, device=DEV, dtype=torch.bfloat16)
    kd = torch.randn(B, S, Hk, D, device=DEV).to(torch.float8_e4m3fn)   # dense fp8 ground truth
    vd = torch.randn(B, S, Hk, D, device=DEV).to(torch.float8_e4m3fn)
    blocks_per_seq = (S + block_size - 1) // block_size
    num_blocks = B * blocks_per_seq + 3
    perm = torch.randperm(num_blocks, device=DEV).int()
    k_cache = torch.zeros(num_blocks, block_size, Hk, D, device=DEV, dtype=torch.float8_e4m3fn)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros(B, blocks_per_seq, device=DEV, dtype=torch.int32)
    ctx = torch.tensor(ctx_lens, device=DEV, dtype=torch.int32)
    for b in range(B):
        for lb in range(blocks_per_seq):
            phys = int(perm[b * blocks_per_seq + lb].item())
            block_table[b, lb] = phys
            for off in range(block_size):
                j = lb * block_size + off
                if j < S:
                    k_cache[phys, off] = kd[b, j]
                    v_cache[phys, off] = vd[b, j]
    got = torch.ops.attn_decode.flash_decode_paged_fp8(
        q, k_cache, v_cache, block_table, ctx, scale, k_descale, v_descale, sw).float()
    # reference: dequantize the same fp8 bytes * descale, fp32 SDPA over the valid prefix
    k_deq = kd.float() * k_descale
    v_deq = vd.float() * v_descale
    ref = torch.empty(B, Hq, D, device=DEV)
    for b in range(B):
        cl = ctx_lens[b]
        # k_deq/v_deq are fp32 (exact fp8 value * descale) — match what the kernel reads; do NOT
        # bf16-round them (ref_decode .float()s internally; q stays bf16 as the kernel sees it).
        ref[b] = ref_decode(q[b:b+1], k_deq[b:b+1, :cl], v_deq[b:b+1, :cl], scale, sw)[0]
    d = (got - ref).abs().max().item()
    ok = d <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} max|Δ|={d:.3e}")
    return ok


def main() -> None:
    print("=== attn_decode flash_decode parity (vs fp32 single-query SDPA) ===")
    ok = True
    # Qwen3.5/3.6-ish attention geometry (head_dim 128, GQA), varying KV length.
    ok &= check("D128 S128  B1  Hq16/Hk2", 1, 16, 2, 128, 128)
    ok &= check("D128 S2048 B1  Hq16/Hk2", 1, 16, 2, 2048, 128)   # long KV (BW path)
    ok &= check("D128 S1000 B4  Hq16/Hk2", 4, 16, 2, 1000, 128)   # batched, non-pow2 S
    ok &= check("D128 S37   B1  Hq16/Hk2", 1, 16, 2, 37, 128)     # S < NWARPS*something
    ok &= check("D128 S512  B1  Hq8/Hk8 ", 1, 8, 8, 512, 128, )   # MHA (no GQA)
    ok &= check("SWA=256 D128 S2048 Hq16/Hk2", 1, 16, 2, 2048, 128, sw=256)
    ok &= check("D64  S256  B2  Hq8/Hk1 ", 2, 8, 1, 256, 64)
    ok &= check("D256 S300  B1  Hq8/Hk2 ", 1, 8, 2, 300, 256)
    print("--- paged (block-table indirection, shuffled physical blocks) ---")
    ok &= check_paged("paged D128 ctx[128] Hq16/Hk2", 1, 16, 2, 128, [128])
    ok &= check_paged("paged D128 ctx[100,250,37] mixed", 3, 16, 2, 128, [100, 250, 37])
    ok &= check_paged("paged D128 ctx[1000] bs32", 1, 16, 2, 128, [1000], block_size=32)
    ok &= check_paged("paged SWA=64 ctx[500]", 1, 16, 2, 128, [500], sw=64)
    print("--- paged fp8-KV (e4m3) + per-tensor descale fold ---")
    ok &= check_paged_fp8("fp8 D128 ctx[256] descale=1", 1, 16, 2, 128, [256])
    ok &= check_paged_fp8("fp8 D128 ctx[100,250,37] mix", 3, 16, 2, 128, [100, 250, 37])
    ok &= check_paged_fp8("fp8 descale k2.0/v0.5 ctx[300]", 1, 16, 2, 128, [300],
                          k_descale=2.0, v_descale=0.5)
    ok &= check_paged_fp8("fp8 SWA=64 ctx[500]", 1, 16, 2, 128, [500], sw=64)
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
