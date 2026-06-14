"""Prefill-regime compute throughput (TFLOP/s). At large M the GEMM is
compute-bound; fp16 hipBLASLt is the bar our 4-bit path must beat. Same shapes
and paths as bench_decode_bw.py for a direct comparison.
    HIP_VISIBLE_DEVICES=0 python /tmp/bench_prefill.py
"""
import time
import torch
import w4a8_fp8_wmma  # noqa: F401
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    triton_w4a16_gemm,
)

op = torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm


def pack_our(w):
    N, K = w.shape
    wp = torch.zeros(N, K // 8, dtype=torch.int32, device=w.device)
    for j in range(8):
        wp |= (w[:, j::8].to(torch.int32) & 0xF) << (j * 4)
    return wp


def pack_tri(w):
    N, K = w.shape
    wt = w.t().contiguous()
    bq = torch.zeros(K, N // 8, dtype=torch.int32, device=w.device)
    for j in range(8):
        bq |= (wt[:, j::8].to(torch.int32) & 0xF) << (j * 4)
    return bq


def bench(fn, it=50, warmup=15):
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
    print(f"device: {torch.cuda.get_device_name(0)}\n")
    shapes = [(4096, 4096), (4096, 11008), (8192, 16384)]
    empty = torch.empty(0, dtype=torch.int32, device=dev)
    for (K, N) in shapes:
        w = torch.randint(0, 16, (N, K), dtype=torch.int32, device=dev)
        sc = torch.rand(N, K // G, dtype=torch.float16, device=dev) * 0.02 + 0.001
        owp = pack_our(w)
        tbq = pack_tri(w)
        tsc = sc.t().contiguous()
        wf16 = (w.t().to(torch.float16) - 8) * 0.01
        print(f"=== K={K} N={N} (TFLOP/s; fp16 = the bar) ===")
        print(f"{'M':>5} | {'v5':>7} {'v7':>7} {'tri':>7} {'fp16':>7} | "
              f"{'v7/fp16':>7} {'v7/tri':>6}")
        for M in [256, 512, 1024, 2048, 4096]:
            x = torch.randn(M, K, dtype=torch.float16, device=dev) * 0.5
            flops = 2 * M * N * K
            f5 = lambda: op(x, owp, sc, empty, 5)
            f7 = lambda: op(x, owp, sc, empty, 7)
            ft = lambda: triton_w4a16_gemm(a=x, b_q=tbq, scales=tsc, qzeros=None,
                                           group_size=G, zp_bias=8)
            ff = lambda: torch.mm(x, wf16)
            t5, t7, tt, tf = bench(f5), bench(f7), bench(ft), bench(ff)
            tf5, tf7 = flops / t5 / 1e12, flops / t7 / 1e12
            tft, tff = flops / tt / 1e12, flops / tf / 1e12
            print(f"{M:>5} | {tf5:>7.1f} {tf7:>7.1f} {tft:>7.1f} {tff:>7.1f} | "
                  f"{tf7/tff:>6.2f}x {tf7/tft:>5.2f}x")
        print()


if __name__ == "__main__":
    main()
