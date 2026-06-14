"""v8 (double-buffered) correctness vs v5 + prefill TFLOP/s vs v7(256x128) & fp16.
CFG via env VLLM_W4A8_V8_CFG (default 96x128).
    HIP_VISIBLE_DEVICES=0 VLLM_W4A8_V8_CFG=96x128 python /tmp/bench_v8.py
"""
import os, time, torch
import w4a8_fp8_wmma  # noqa
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    triton_w4a16_gemm)
op = torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm


def pack(w):
    N, K = w.shape
    wp = torch.zeros(N, K // 8, dtype=torch.int32, device=w.device)
    for j in range(8):
        wp |= (w[:, j::8].to(torch.int32) & 0xF) << (j * 4)
    return wp


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
    dev = "cuda"; G = 128
    cfg = os.environ.get("VLLM_W4A8_V8_CFG", "96x128")
    print(f"device {torch.cuda.get_device_name(0)} | V8_CFG={cfg} G={G}")
    shapes = [(4096, 4096), (4096, 11008), (8192, 16384)]
    empty = torch.empty(0, dtype=torch.int32, device=dev)
    for (K, N) in shapes:
        w = torch.randint(0, 16, (N, K), dtype=torch.int32, device=dev)
        sc = torch.rand(N, K // G, dtype=torch.float16, device=dev) * 0.02 + 0.001
        wp = pack(w); wf16 = (w.t().to(torch.float16) - 8) * 0.01
        print(f"=== K={K} N={N} (TFLOP/s) ===")
        print(f"{'M':>5} | {'v7':>7} {'v8':>7} {'fp16':>7} | {'v8/v7':>6} "
              f"{'v8/fp16':>7} | {'maxdiff':>8}")
        for M in [256, 512, 1024, 2048, 4096]:
            x = torch.randn(M, K, dtype=torch.float16, device=dev) * 0.5
            flops = 2 * M * N * K
            f7 = lambda: op(x, wp, sc, empty, 7)   # v7 256x128 (default)
            f8 = lambda: op(x, wp, sc, empty, 8)
            ff = lambda: torch.mm(x, wf16)
            d = (f8().float() - f7().float()).abs().max().item()
            t7, t8, tf = bench(f7), bench(f8), bench(ff)
            tf7, tf8, tff = flops/t7/1e12, flops/t8/1e12, flops/tf/1e12
            print(f"{M:>5} | {tf7:>7.1f} {tf8:>7.1f} {tff:>7.1f} | {tf8/tf7:>6.2f} "
                  f"{tf8/tff:>6.2f}x | {d:>8.1e}")
        print()


if __name__ == "__main__":
    main()
