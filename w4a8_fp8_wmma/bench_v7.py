"""v7 (templated tile) vs v5. CFG via env VLLM_W4A8_V7_CFG=BMxBN.
    HIP_VISIBLE_DEVICES=1 VLLM_W4A8_V7_CFG=128x128 python /tmp/bench_v7.py
"""
import os, time, torch
import w4a8_fp8_wmma  # noqa: F401
op = torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm


def pack_w(w):
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
    cfg = os.environ.get("VLLM_W4A8_V7_CFG", "256x64")
    G = int(os.environ.get("BENCH_G", "128"))
    print(f"device {torch.cuda.get_device_name(0)} | CFG={cfg} G={G}")
    shapes = [(512,4096,4096),(1024,4096,4096),(2048,4096,4096),
              (4096,4096,4096),(2048,4096,11008),(4096,4096,14336)]
    empty = torch.empty(0, dtype=torch.int32, device=dev)
    print(f"{'M':>5} {'K':>6} {'N':>6} | {'v5':>7} {'v7':>7} {'v7/v5':>6}")
    for (M,K,N) in shapes:
        x = torch.randn(M,K,dtype=torch.float16,device=dev)*0.5
        w = torch.randint(0,16,(N,K),dtype=torch.int32,device=dev)
        sc = torch.rand(N,K//G,dtype=torch.float16,device=dev)*0.02+0.001
        wp = pack_w(w); flops = 2*M*N*K
        f5 = lambda: op(x,wp,sc,empty,5)
        f7 = lambda: op(x,wp,sc,empty,7)
        d = (f7().float()-f5().float()).abs().max().item()
        t5,t7 = bench(f5),bench(f7)
        tf5,tf7 = flops/t5/1e12, flops/t7/1e12
        print(f"{M:>5} {K:>6} {N:>6} | {tf5:>7.1f} {tf7:>7.1f} {tf7/tf5:>6.2f}  (md {d:.0e})")


if __name__ == "__main__":
    main()
