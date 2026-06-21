"""GPU bit-exactness for the e2m1 GROUPED-MoE kernels (gfx1201).

Same rigorous approach as the dense test: for integer-valued E2M1 codes, the symmetric int4 path
(code = v+8, zp=8 -> v) produces the same fp8 weight byte, so the e2m1 MoE kernels must give
BIT-IDENTICAL output to the validated int4 MoE kernels — using the kernel's own fp8 act-quant for
both, so no external oracle. Covers the grouped GEMM (wmma tiled / gemv / scalar) + the fused
gemm1_silu + the scatter epilogue, across the routed/padded layout from moe_align_block_size.

Run under a single-card GPU lease (build in-container — see test_mxfp4_kernel_gpu.py header).
"""
import torch

import w4a8_fp8_wmma as w4a8
from mxfp4.convert import FP4_E2M1_LUT, pack_codes_to_int32
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_ok = True
INT_CODES = [c for c in range(16) if float(FP4_E2M1_LUT[c]).is_integer()]


def check(name, cond, detail=""):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}{(' -- ' + detail) if detail else ''}")


def _make_experts(E, N, K, group_size, gen):
    """Return (w_e2m1 int32, w_int4 int32, scales fp16) for E experts, integer-valued codes."""
    idx = torch.randint(0, len(INT_CODES), (E * N, K), generator=gen)
    codes_e2m1 = torch.tensor(INT_CODES, dtype=torch.int64)[idx]
    vals = torch.tensor(FP4_E2M1_LUT, dtype=torch.float32)[codes_e2m1]
    codes_int4 = (vals + 8).to(torch.int64)
    w_e2m1 = pack_codes_to_int32(codes_e2m1.to(torch.uint8)).reshape(E, N, K // 8).contiguous()
    w_int4 = pack_codes_to_int32(codes_int4.to(torch.uint8)).reshape(E, N, K // 8).contiguous()
    scales = (torch.rand(E, N, K // group_size, generator=gen).to(torch.float16) * 0.5 + 0.25)
    return w_e2m1.cuda(), w_int4.cuda(), scales.cuda()


def _routing(T, E, top_k, block_m, gen):
    topk_ids = torch.stack([torch.randperm(E, generator=gen)[:top_k] for _ in range(T)]).int()
    sorted_ids, expert_ids, ntp = moe_align_block_size(
        topk_ids.cuda(), block_m, E, None, pad_sorted_ids=True, ignore_invalid_experts=True)
    return topk_ids.cuda(), sorted_ids, expert_ids, ntp


def _valid(sorted_ids, T, top_k):
    """Padded-sorted rows the kernels actually write (others left uninitialised, masked by ntp)."""
    return sorted_ids < (T * top_k)


def run_gemm(kernel, T, E, N, K, top_k=2, group_size=32, block_m=16, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w_e2m1, w_int4, scales = _make_experts(E, N, K, group_size, g)
    _, sorted_ids, expert_ids, ntp = _routing(T, E, top_k, block_m, g)
    x = (torch.randn(T, K, generator=g) * 0.5).to(torch.float16).cuda()
    o_e = w4a8.mmq_fp8_moe_gemm(x, w_e2m1, scales, sorted_ids, expert_ids, ntp,
                                top_k, block_m, kernel=kernel, weight_is_e2m1=True)
    o_i = w4a8.mmq_fp8_moe_gemm(x, w_int4, scales, sorted_ids, expert_ids, ntp,
                                top_k, block_m, kernel=kernel, weight_is_e2m1=False)
    m = _valid(sorted_ids, T, top_k)
    check(f"moe_gemm e2m1==int4 {kernel:<7} T={T:<3} E={E} N={N} K={K} bm={block_m}",
          torch.equal(o_e[m], o_i[m]),
          f"max|diff|={(o_e[m].float()-o_i[m].float()).abs().max().item():.3e}")


def run_gemm1_silu(T, E, inter, K, top_k=2, block_m=16, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w_e2m1, w_int4, scales = _make_experts(E, 2 * inter, K, 32, g)   # w13 = [gate|up]
    _, sorted_ids, expert_ids, ntp = _routing(T, E, top_k, block_m, g)
    x = (torch.randn(T, K, generator=g) * 0.5).to(torch.float16).cuda()
    o_e = w4a8.mmq_fp8_moe_gemm1_silu(x, w_e2m1, scales, sorted_ids, expert_ids, ntp,
                                      top_k, block_m, kernel="wmma", weight_is_e2m1=True)
    o_i = w4a8.mmq_fp8_moe_gemm1_silu(x, w_int4, scales, sorted_ids, expert_ids, ntp,
                                      top_k, block_m, kernel="wmma", weight_is_e2m1=False)
    m = _valid(sorted_ids, T, top_k)
    check(f"gemm1_silu e2m1==int4 wmma  T={T:<3} E={E} inter={inter} K={K}",
          torch.equal(o_e[m], o_i[m]),
          f"max|diff|={(o_e[m].float()-o_i[m].float()).abs().max().item():.3e}")


def run_scatter(T, E, N, K, top_k=2, block_m=8, seed=2):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w_e2m1, w_int4, scales = _make_experts(E, N, K, 32, g)
    topk_ids, sorted_ids, expert_ids, ntp = _routing(T, E, top_k, block_m, g)
    P = sorted_ids.size(0)
    # scatter's `x` is the (P, K) padded-sorted post-activation buffer, NOT (T, K).
    x = (torch.randn(P, K, generator=g) * 0.5).to(torch.float16).cuda()
    tw = (torch.rand(T * top_k, generator=g).to(torch.float32)).cuda()
    out_e = torch.zeros(T, N, dtype=torch.float32, device="cuda")
    out_i = torch.zeros(T, N, dtype=torch.float32, device="cuda")
    w4a8.mmq_fp8_moe_gemm_scatter(x, w_e2m1, scales, sorted_ids, expert_ids, ntp, tw,
                                  out_e, top_k, block_m, kernel="gemv", weight_is_e2m1=True)
    w4a8.mmq_fp8_moe_gemm_scatter(x, w_int4, scales, sorted_ids, expert_ids, ntp, tw,
                                  out_i, top_k, block_m, kernel="gemv", weight_is_e2m1=False)
    check(f"gemm_scatter e2m1==int4 gemv T={T:<3} E={E} N={N} K={K}",
          torch.equal(out_e, out_i), f"max|diff|={(out_e-out_i).abs().max().item():.3e}")


if __name__ == "__main__":
    print(f"torch {torch.__version__}\n")
    print("A. grouped GEMM (wmma / gemv / scalar)")
    for kern in ("wmma", "gemv", "scalar"):
        run_gemm(kern, T=8, E=8, N=128, K=512, block_m=(8 if kern == "gemv" else 16))
        run_gemm(kern, T=64, E=8, N=128, K=512, block_m=(8 if kern == "gemv" else 16))
    print("B. fused gemm1+silu")
    run_gemm1_silu(T=8, E=8, inter=128, K=512)
    run_gemm1_silu(T=64, E=8, inter=128, K=512)
    print("C. gemm2 scatter epilogue")
    run_scatter(T=8, E=8, N=512, K=128)
    print("\n" + ("ALL MoE GPU CHECKS PASSED" if _ok else "SOME MoE GPU CHECKS FAILED"))
    raise SystemExit(0 if _ok else 1)
