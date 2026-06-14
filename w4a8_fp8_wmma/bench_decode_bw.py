"""Decode-regime memory-bandwidth test. At small M a 4-bit GEMM is weight-traffic
bound; the question is whether our W4A8-FP8 kernel SATURATES VRAM bandwidth (and
how it compares to Triton W4A16 and fp16 torch.mm). Reports effective
weight-read GB/s = (weight_bytes) / time, against the GPU's sustainable peak.

    HIP_VISIBLE_DEVICES=0 python /tmp/bench_decode_bw.py
"""
import time
import torch
import w4a8_fp8_wmma  # noqa: F401
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    triton_w4a16_gemm,
)

op = torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm


def pack_our(w):  # (N,K) -> (N,K//8)
    N, K = w.shape
    wp = torch.zeros(N, K // 8, dtype=torch.int32, device=w.device)
    for j in range(8):
        wp |= (w[:, j::8].to(torch.int32) & 0xF) << (j * 4)
    return wp


def pack_tri(w):  # (N,K) -> b_q (K, N//8)
    N, K = w.shape
    wt = w.t().contiguous()
    bq = torch.zeros(K, N // 8, dtype=torch.int32, device=w.device)
    for j in range(8):
        bq |= (wt[:, j::8].to(torch.int32) & 0xF) << (j * 4)
    return bq


def bench(fn, it=100, warmup=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it


def peak_bw(dev):
    # sustainable read BW proxy: sum() reads the whole buffer.
    n = 512 * 1024 * 1024 // 2  # 512 MB of fp16
    a = torch.randn(n, dtype=torch.float16, device=dev)
    t = bench(lambda: a.sum(), it=50, warmup=20)
    return (n * 2) / t / 1e9  # GB/s (read)


def main():
    dev = "cuda"
    G = 128
    print(f"device: {torch.cuda.get_device_name(0)}")
    pk = peak_bw(dev)
    print(f"sustainable read BW (sum of 512MB): {pk:.0f} GB/s\n")

    # last shape's 4-bit weights (67MB) and fp16 (268MB) both exceed the 64MB
    # RDNA4 MALL/Infinity Cache, so it reports TRUE VRAM-bound bandwidth.
    shapes = [(4096, 4096), (4096, 11008), (8192, 16384)]
    empty = torch.empty(0, dtype=torch.int32, device=dev)
    for (K, N) in shapes:
        w = torch.randint(0, 16, (N, K), dtype=torch.int32, device=dev)
        sc = torch.rand(N, K // G, dtype=torch.float16, device=dev) * 0.02 + 0.001
        owp = pack_our(w)
        tbq = pack_tri(w)
        tsc = sc.t().contiguous()
        wf16 = (w.t().to(torch.float16) - 8) * 0.01  # (K,N) for torch.mm
        wbytes4 = N * K * 0.5      # 4-bit weight traffic
        wbytes16 = N * K * 2.0     # fp16 weight traffic
        print(f"=== K={K} N={N} | 4bit weights {wbytes4/1e6:.1f}MB, "
              f"fp16 {wbytes16/1e6:.1f}MB ===")
        print(f"{'M':>4} | {'v5 us':>7} {'GB/s':>6} | {'v7 us':>7} {'GB/s':>6} | "
              f"{'tri us':>7} {'GB/s':>6} | {'fp16 us':>7} {'GB/s':>6} | "
              f"{'v7/fp16':>7}")
        print(f"{'M':>4} | {'v7 us':>7} {'GB/s':>6} | {'v10 us':>7} {'GB/s':>6} | "
              f"{'tri us':>7} {'GB/s':>6} | {'fp16 us':>7} {'GB/s':>6}")
        for M in [1, 2, 4, 8, 16, 32]:
            x = torch.randn(M, K, dtype=torch.float16, device=dev) * 0.5
            f7 = lambda: op(x, owp, sc, empty, 7)
            f10 = lambda: op(x, owp, sc, empty, 10)
            ft = lambda: triton_w4a16_gemm(a=x, b_q=tbq, scales=tsc, qzeros=None,
                                           group_size=G, zp_bias=8)
            ff = lambda: torch.mm(x, wf16)
            t7, t10, tt, tf = bench(f7), bench(f10), bench(ft), bench(ff)
            bw7, bw10 = wbytes4 / t7 / 1e9, wbytes4 / t10 / 1e9
            bwt, bwf = wbytes4 / tt / 1e9, wbytes16 / tf / 1e9
            print(f"{M:>4} | {t7*1e6:>7.1f} {bw7:>6.0f} | {t10*1e6:>7.1f} {bw10:>6.0f} "
                  f"| {tt*1e6:>7.1f} {bwt:>6.0f} | {tf*1e6:>7.1f} {bwf:>6.0f}")
        print()


if __name__ == "__main__":
    main()
