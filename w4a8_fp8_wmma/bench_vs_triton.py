"""Head-to-head GEMM benchmark: our W4A8-FP8-WMMA op vs vLLM's default
TritonW4A16 (int4->fp16) kernel, on identical logical 4-bit weights.

Target (the goal): our op TFLOP/s >= Triton's across prefill (large M) and decode
(M=1) shapes. Run inside kyuz0/vllm-therock-gfx1201 with the GPU mounted.
"""
import sys
import time
import torch

import w4a8_fp8_wmma  # noqa: F401  loads torch.ops.w4a8_fp8_wmma
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    triton_w4a16_gemm,
)


def build_weights(N, K, G, dev):
    w_int4 = torch.randint(0, 16, (N, K), dtype=torch.int32, device=dev)
    scale = torch.rand(N, K // G, dtype=torch.float16, device=dev) * 0.02 + 0.001
    # our op: w_packed (N, K//8) packed along K; scales (N, K//G)
    our_wp = torch.zeros(N, K // 8, dtype=torch.int32, device=dev)
    for j in range(8):
        our_wp |= (w_int4[:, j::8] & 0xF) << (j * 4)
    our_sc = scale
    # triton: b_q (K, N//8) packed along N; scales (K//G, N)
    wt = w_int4.t().contiguous()  # (K, N)
    tri_bq = torch.zeros(K, N // 8, dtype=torch.int32, device=dev)
    for j in range(8):
        tri_bq |= (wt[:, j::8] & 0xF) << (j * 4)
    tri_sc = scale.t().contiguous()  # (K//G, N)
    return our_wp, our_sc, tri_bq, tri_sc


def bench(fn, it=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it


def main():
    dev = "cuda"
    G = 128
    # (M, K, N): decode + prefill, model-ish shapes.
    shapes = [
        (512, 4096, 4096),
        (1024, 4096, 4096),
        (1536, 4096, 4096),
        (2048, 4096, 4096),
        (3072, 4096, 4096),
        (4096, 4096, 4096),
    ]
    print(f"{'M':>5} {'K':>6} {'N':>6} | {'ours TFLOP/s':>13} {'triton TFLOP/s':>15} "
          f"{'ratio':>6} | {'maxdiff':>8}")
    empty = torch.empty(0, dtype=torch.int32, device=dev)
    for (M, K, N) in shapes:
        x = torch.randn(M, K, dtype=torch.float16, device=dev)
        owp, osc, tbq, tsc = build_weights(N, K, G, dev)
        flops = 2 * M * N * K

        def run_ours():
            return torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm(x, owp, osc, empty, 5)

        def run_tri():
            return triton_w4a16_gemm(a=x, b_q=tbq, scales=tsc, qzeros=None,
                                     group_size=G, zp_bias=8)

        o = run_ours(); t = run_tri()
        maxdiff = (o.float() - t.float()).abs().max().item()
        t_o = bench(run_ours)
        t_t = bench(run_tri)
        to_tf = flops / t_o / 1e12
        tt_tf = flops / t_t / 1e12
        print(f"{M:>5} {K:>6} {N:>6} | {to_tf:>13.2f} {tt_tf:>15.2f} "
              f"{to_tf / tt_tf:>6.2f} | {maxdiff:>8.3f}")


if __name__ == "__main__":
    main()
