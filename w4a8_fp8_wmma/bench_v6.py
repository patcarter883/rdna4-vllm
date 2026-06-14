"""v6 (K-extension) vs v5 TFLOP/s on dense shapes. Pad swept via env per run:
    HIP_VISIBLE_DEVICES=1 VLLM_W4A8_V6_PAD=16 python /tmp/bench_v6.py
"""
import os
import time
import torch
import w4a8_fp8_wmma  # noqa: F401

op = torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm


def pack_w(w_int4):
    N, K = w_int4.shape
    wp = torch.zeros(N, K // 8, dtype=torch.int32, device=w_int4.device)
    for j in range(8):
        wp |= (w_int4[:, j::8].to(torch.int32) & 0xF) << (j * 4)
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
    dev = "cuda"
    pad = os.environ.get("VLLM_W4A8_V6_PAD", "16")
    print(f"device: {torch.cuda.get_device_name(0)}  |  V6_PAD={pad}")
    G = int(os.environ.get("BENCH_G", "128"))
    shapes = [
        (512, 4096, 4096), (1024, 4096, 4096), (1536, 4096, 4096),
        (2048, 4096, 4096), (3072, 4096, 4096), (4096, 4096, 4096),
        (2048, 4096, 11008), (4096, 4096, 14336),
    ]
    empty = torch.empty(0, dtype=torch.int32, device=dev)
    print(f"G={G}")
    print(f"{'M':>5} {'K':>6} {'N':>6} | {'v5 TF/s':>8} {'v6 TF/s':>8} {'v6/v5':>6}")
    for (M, K, N) in shapes:
        x = torch.randn(M, K, dtype=torch.float16, device=dev) * 0.5
        w_int4 = torch.randint(0, 16, (N, K), dtype=torch.int32, device=dev)
        scale = torch.rand(N, K // G, dtype=torch.float16, device=dev) * 0.02 + 0.001
        wp = pack_w(w_int4)
        flops = 2 * M * N * K
        f5 = lambda: op(x, wp, scale, empty, 5)
        f6 = lambda: op(x, wp, scale, empty, 6)
        # sanity once
        d = (f6().float() - f5().float()).abs().max().item()
        t5, t6 = bench(f5), bench(f6)
        tf5, tf6 = flops / t5 / 1e12, flops / t6 / 1e12
        print(f"{M:>5} {K:>6} {N:>6} | {tf5:>8.1f} {tf6:>8.1f} {tf6/tf5:>6.2f}"
              f"   (maxdiff {d:.1e})")


if __name__ == "__main__":
    main()
