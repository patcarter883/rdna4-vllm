"""v7 perf breakdown via VLLM_W4A8_V7_DIAG bits (set per-call here through the
env BEFORE import won't work — diag is read at launch, so we set os.environ and
it's picked up each call). Reports time per diag mode to localize the bottleneck.
    HIP_VISIBLE_DEVICES=0 python /tmp/bench_v7_diag.py
"""
import os, time, torch
import w4a8_fp8_wmma  # noqa
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
    dev = "cuda"
    G = 128
    print(f"device {torch.cuda.get_device_name(0)} G={G}")
    modes = [(0, "full"), (1, "no-WMMA"), (8, "no-LDSread"),
             (9, "no-WMMA+LDSrd"), (2, "no-global"), (6, "no-mem"),
             (14, "no-mem+LDSrd"), (15, "all-off")]
    for (K, N, M) in [(4096, 4096, 4096), (8192, 16384, 4096)]:
        w = torch.randint(0, 16, (N, K), dtype=torch.int32, device=dev)
        sc = torch.rand(N, K // G, dtype=torch.float16, device=dev) * 0.02 + 0.001
        wp = pack(w); empty = torch.empty(0, dtype=torch.int32, device=dev)
        x = torch.randn(M, K, dtype=torch.float16, device=dev) * 0.5
        flops = 2 * M * N * K
        print(f"\n=== K={K} N={N} M={M} (flops={flops/1e9:.0f} GF) ===")
        base = None
        for (d, name) in modes:
            os.environ["VLLM_W4A8_V7_DIAG"] = str(d)
            t = bench(lambda: op(x, wp, sc, empty, 7))
            tf = flops / t / 1e12
            if d == 0:
                base = t
            frac = t / base * 100
            tag = f"{tf:.1f} TF/s" if d == 0 else f"{frac:.0f}% of full"
            print(f"  diag={d:>1} {name:>14}: {t*1e6:>8.1f} us   {tag}")
    os.environ["VLLM_W4A8_V7_DIAG"] = "0"


if __name__ == "__main__":
    main()
