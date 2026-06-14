"""v6 (b128 double-K) crossover bench: v6 vs v5/v10/Triton across the full M sweep.

v6 issues two back-to-back WMMAs per b128 (16-fp8) LDS read -- half v5's LDS load
instructions per WMMA. This bench finds the mid-M band where v6 beats the kernel the
served dispatch would otherwise pick (v10 for gs in {32,128}, else v5) AND beats
stock Triton, so VLLM_ROCM_W4A8_V6_{MIN,MAX}_M can be set to that band.

    HIP_VISIBLE_DEVICES=0 python bench_v6.py            # full M sweep, all cands
    HIP_VISIBLE_DEVICES=0 BENCH_G=32 python bench_v6.py # group_size 32
    HIP_VISIBLE_DEVICES=0 VLLM_W4A8_V6_PAD=16 python bench_v6.py  # sweep v6 LDS pad

Reads ours-vs-Triton from the extracted byte-faithful triton_w4a16_ref (== vLLM's
gfx1201 production W4A16). Bit-exactness of v6 vs v5 is asserted per shape (maxdiff
0 -- identical operands); see test_v6_correctness.py for the full correctness gate.
"""
import os
import time
import torch
import w4a8_fp8_wmma  # noqa: F401
from triton_w4a16_ref import triton_w4a16_gemm

op = torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm


def pack_ours(w):  # (N,K) int4 -> (N, K//8) int32, low nibble first along K
    N, K = w.shape
    wp = torch.zeros(N, K // 8, dtype=torch.int32, device=w.device)
    for j in range(8):
        wp |= (w[:, j::8].to(torch.int32) & 0xF) << (j * 4)
    return wp


def pack_triton(w):  # (N,K) int4 -> b_q (K, N//8) int32, 8 N-vals per int32
    N, K = w.shape
    wt = w.t().contiguous()  # (K, N)
    bq = torch.zeros(K, N // 8, dtype=torch.int32, device=w.device)
    for j in range(8):
        bq |= (wt[:, j::8].to(torch.int32) & 0xF) << (j * 4)
    return bq


def bench(fn, it=50, warmup=20):
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
    G = int(os.environ.get("BENCH_G", "128"))
    pad = os.environ.get("VLLM_W4A8_V6_PAD", "16")
    torch.manual_seed(0)
    # The mid-M band is the whole point; sweep through it densely.
    Ms = [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    shapes = [(4096, 4096), (4096, 11008), (11008, 4096), (8192, 16384)]
    empty = torch.empty(0, dtype=torch.int32, device=dev)
    v10_ok = G in (32, 128)
    print(f"device {torch.cuda.get_device_name(0)} | G={G} V6_PAD={pad} "
          f"(v10_ok={v10_ok})")
    for (K, N) in shapes:
        w = torch.randint(0, 16, (N, K), dtype=torch.int32, device=dev)
        sc = torch.rand(N, K // G, dtype=torch.float16, device=dev) * 0.02 + 0.001
        wp = pack_ours(w)
        bq = pack_triton(w)
        tsc = sc.t().contiguous()  # (K//G, N)
        # bit-exactness spot check v6 vs v5 (must be 0 -- identical operands)
        x0 = torch.randn(256, K, dtype=torch.float16, device=dev) * 0.5
        bex = (op(x0, wp, sc, empty, 6).float()
               - op(x0, wp, sc, empty, 5).float()).abs().max().item()
        print(f"\n=== K={K} N={N}  (us; v6-v5 maxdiff={bex:.1e}) ===")
        print(f"{'M':>5} | {'triton':>8} {'v10':>8} {'v5':>8} {'v6':>8} | "
              f"{'v6/v5':>6} {'v6/v10':>7} {'v6/tri':>7} {'best':>7}")
        for M in Ms:
            x = torch.randn(M, K, dtype=torch.float16, device=dev) * 0.5
            tt = bench(lambda: triton_w4a16_gemm(x, bq, tsc, None, G, 8)) * 1e6
            t5 = bench(lambda: op(x, wp, sc, empty, 5)) * 1e6
            t6 = bench(lambda: op(x, wp, sc, empty, 6)) * 1e6
            cand = {"triton": tt, "v5": t5, "v6": t6}
            if v10_ok:
                t10 = bench(lambda: op(x, wp, sc, empty, 10)) * 1e6
                cand["v10"] = t10
            else:
                t10 = float("nan")
            best = min(cand, key=cand.get)
            v10s = f"{t10:8.1f}" if v10_ok else f"{'-':>8}"
            r10 = f"{t6 / t10:7.2f}" if v10_ok else f"{'-':>7}"
            print(f"{M:>5} | {tt:8.1f} {v10s} {t5:8.1f} {t6:8.1f} | "
                  f"{t6 / t5:6.2f} {r10} {t6 / tt:7.2f} {best:>7}")


if __name__ == "__main__":
    main()
