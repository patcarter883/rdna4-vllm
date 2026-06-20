"""Correctness test for the grouped (MoE) W4A8-FP8 GEMM op (gfx1201).

Builds a random top-k routing, the moe_align_block_size routing tensors
(sorted_token_ids / expert_ids / num_tokens_post_padded), stacked per-expert
int4 weights (symmetric uint4b8 and asymmetric AWQ), and checks the grouped
kernels by NAME ("scalar" golden, "wmma" tiled WMMA, "gemv" decode GEMV) against
an fp8 reference that mirrors the kernel's arithmetic (per-token fp8 act,
per-expert fp8 weights, per-group fp32 reduce). The former v5-vs-v6 A-residence
split is now the env knob VLLM_W4A8_MOE_A_IN_LDS (read per-dispatch in C++), so
"wmma" is exercised with it both unset (A-shuffle, former v6) and =1 (A-in-LDS,
former v5) in one process.
"""
import os
import sys

import torch

try:
    import w4a8_fp8_wmma
except ImportError as e:
    print(f"FAIL: import error: {e}")
    sys.exit(1)

E4M3_MAX = 448.0


def pack_uint4(w):  # (E,N,K) int8 -> (E,N,K//8) int32
    E, N, K = w.shape
    w = w.to(torch.int32)
    packed = torch.zeros((E, N, K // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= (w[:, :, i::8] & 0xF) << (i * 4)
    return packed


def pack_zeros(z):  # (E,N,G) int8 -> (E,N//8,G) int32 (N-packed, nibble n%8)
    E, N, G = z.shape
    z = z.to(torch.int32)
    packed = torch.zeros((E, N // 8, G), dtype=torch.int32, device=z.device)
    for i in range(8):
        packed |= (z[:, i::8, :] & 0xF) << (i * 4)
    return packed


def to_e4m3(x):
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def moe_align(topk_ids, block_m, E):
    """Python moe_align_block_size: sort token-slots by expert, pad each expert's
    run to a multiple of block_m with a sentinel (= num_valid_tokens)."""
    T, top_k = topk_ids.shape
    num_valid = T * top_k
    flat = topk_ids.reshape(-1)  # expert per slot, slot id = t*top_k + j
    sorted_ids, expert_ids = [], []
    for e in range(E):
        slots = torch.nonzero(flat == e, as_tuple=False).flatten().tolist()
        n = len(slots)
        npad = ((n + block_m - 1) // block_m) * block_m
        for i in range(npad):
            sorted_ids.append(slots[i] if i < n else num_valid)
        expert_ids.extend([e] * (npad // block_m))
    dev = topk_ids.device
    sti = torch.tensor(sorted_ids, dtype=torch.int32, device=dev)
    eids = torch.tensor(expert_ids, dtype=torch.int32, device=dev)
    ntp = torch.tensor([sti.numel()], dtype=torch.int32, device=dev)
    return sti, eids, ntp, num_valid


def reference(x, w_int4, scales, zeros, sti, eids, ntp, top_k, block_m, group_size):
    """fp8 reference in the padded sorted layout. zeros=None for symmetric."""
    T, K = x.shape
    E, N, _ = w_int4.shape
    G = K // group_size
    dev = x.device
    P = sti.numel()
    num_valid = T * top_k

    a_scale = x.float().abs().amax(1, keepdim=True).clamp(min=1e-8 * E4M3_MAX) / E4M3_MAX
    x_fp8 = to_e4m3(x.float() / a_scale)  # (T,K)

    if zeros is None:
        zp = torch.full((E, N, G), 8, dtype=torch.int32, device=dev)
    else:
        zp = zeros.to(torch.int32)
    w_signed = w_int4.to(torch.float32) - zp.repeat_interleave(group_size, dim=2)
    w_fp8 = to_e4m3(w_signed)  # (E,N,K) exact

    out = torch.zeros((P, N), dtype=torch.float16, device=dev)
    for r in range(P):
        offs = int(sti[r].item())
        if offs >= num_valid:
            continue
        t = offs // top_k
        e = int(eids[r // block_m].item())  # expert for this block
        xg = x_fp8[t].view(G, group_size)
        wg = w_fp8[e].view(N, G, group_size)
        partial = torch.einsum("gk,ngk->ng", xg, wg)  # (N,G)
        o = (partial * scales[e].float()).sum(1)  # (N,)
        out[r] = (o * a_scale[t]).to(torch.float16)
    return out


def run(T, E, N, K, top_k, block_m, group_size, kernel, asym, mean_rtol=0.02,
        label=None):
    dev = torch.device("cuda")
    G = K // group_size
    x = torch.randn(T, K, dtype=torch.float16, device=dev)
    w_int4 = torch.randint(0, 16, (E, N, K), dtype=torch.int8, device=dev)
    w_packed = pack_uint4(w_int4)
    scales = torch.randn(E, N, G, dtype=torch.float16, device=dev).abs() * 0.02 + 0.001
    # random top-k routing
    topk_ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(T)]
                           ).to(torch.int32)
    sti, eids, ntp, _ = moe_align(topk_ids, block_m, E)

    if asym:
        zeros = torch.randint(0, 16, (E, N, G), dtype=torch.int8, device=dev)
        zp_packed = pack_zeros(zeros)
    else:
        zeros, zp_packed = None, None

    ref = reference(x, w_int4, scales, zeros, sti, eids, ntp, top_k, block_m, group_size)
    out = w4a8_fp8_wmma.mmq_fp8_moe_gemm(
        x, w_packed, scales, sti, eids, ntp, top_k, block_m,
        kernel=kernel, w_zeros=zp_packed)

    # compare only valid (non-padding) rows
    num_valid = T * top_k
    valid = (sti < num_valid)
    diff = (out[valid].float() - ref[valid].float()).abs()
    refm = ref[valid].float().abs().mean().item()
    mean = diff.mean().item()
    eff = max(2e-3, mean_rtol * refm)
    n_bad = int((diff > 0.15 + 0.03 * ref[valid].float().abs()).sum().item())
    ok = mean <= eff and n_bad == 0
    tag = "ASYM" if asym else "SYM "
    name = label or kernel
    print(f"  {name:14s} {tag} T={T} E={E} N={N} K={K} tk={top_k} bm={block_m} "
          f"g={group_size}: mean={mean:.5f} |ref|={refm:.4f} bad={n_bad} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("FAIL: no device"); sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    cases = [
        # (T, E, N, K, top_k, block_m, group_size)
        (64, 8, 128, 256, 2, 16, 32),
        (32, 4, 256, 512, 2, 32, 128),
        (128, 8, 512, 1024, 2, 64, 128),
        (16, 16, 256, 256, 4, 16, 32),
        (1, 8, 512, 1024, 2, 16, 128),   # single-token decode
    ]
    res = []
    print("=== grouped scalar (golden) ===")
    for c in cases:
        res.append(run(*c, kernel="scalar", asym=False))
        res.append(run(*c, kernel="scalar", asym=True))
    print("=== grouped wmma (A-shuffle / B-only LDS; default, former v6) ===")
    os.environ.pop("VLLM_W4A8_MOE_A_IN_LDS", None)
    for c in cases:
        res.append(run(*c, kernel="wmma", asym=False))
        res.append(run(*c, kernel="wmma", asym=True))
    print("=== grouped wmma + VLLM_W4A8_MOE_A_IN_LDS=1 (A-in-LDS, former v5) ===")
    os.environ["VLLM_W4A8_MOE_A_IN_LDS"] = "1"
    for c in cases:
        res.append(run(*c, kernel="wmma", asym=False, label="wmma[A_IN_LDS]"))
        res.append(run(*c, kernel="wmma", asym=True, label="wmma[A_IN_LDS]"))
    os.environ.pop("VLLM_W4A8_MOE_A_IN_LDS", None)
    print("=== grouped gemv (decode GEMV) ===")
    for c in cases:
        res.append(run(*c, kernel="gemv", asym=False))
        res.append(run(*c, kernel="gemv", asym=True))
    print("=" * 50)
    if all(res):
        print(f"ALL PASSED ({len(res)})"); sys.exit(0)
    print(f"FAIL: {sum(1 for r in res if not r)}/{len(res)}"); sys.exit(1)


if __name__ == "__main__":
    main()
