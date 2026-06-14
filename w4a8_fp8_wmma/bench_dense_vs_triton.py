"""Dense W4A8 vs production Triton-W4A16, full M sweep, on gfx1201 host.

Compares our op (v5/v10/v11) against the extracted Triton-W4A16 reference
(triton_w4a16_ref, == vLLM's gfx1201 production path) across decode->prefill M.
Symmetric uint4b8 (implicit zp=8) so both paths use the same dequant.

  HIP_VISIBLE_DEVICES=0 python bench_dense_vs_triton.py
"""
import os, time, torch
import w4a8_fp8_wmma  # noqa: F401  registers torch.ops.w4a8_fp8_wmma
from triton_w4a16_ref import triton_w4a16_gemm

op = torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm


def pack_ours(w):  # w (N,K) int4 -> (N, K//8) int32, low nibble first along K
    N, K = w.shape
    wp = torch.zeros(N, K // 8, dtype=torch.int32, device=w.device)
    for j in range(8):
        wp |= (w[:, j::8].to(torch.int32) & 0xF) << (j * 4)
    return wp


def pack_triton(w):  # w (N,K) int4 -> b_q (K, N//8) int32, 8 N-vals per int32
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
    torch.manual_seed(0)
    shapes = [(4096, 4096), (4096, 11008), (8192, 16384)]
    Ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    empty = torch.empty(0, dtype=torch.int32, device=dev)
    print(f"device {torch.cuda.get_device_name(0)} | G={G}")
    for (K, N) in shapes:
        w = torch.randint(0, 16, (N, K), dtype=torch.int32, device=dev)
        sc = torch.rand(N, K // G, dtype=torch.float16, device=dev) * 0.02 + 0.001
        wp = pack_ours(w)
        bq = pack_triton(w)
        tsc = sc.t().contiguous()  # (K//G, N)
        v10_ok = G in (32, 128)
        # one correctness spot-check (M=64) ours-v5 vs triton (fp8-quant tol)
        x0 = torch.randn(64, K, dtype=torch.float16, device=dev) * 0.5
        a = op(x0, wp, sc, empty, 5).float()
        b = triton_w4a16_gemm(x0, bq, tsc, None, G, 8).float()
        rel = ((a - b).abs().mean() / b.abs().mean().clamp_min(1e-6)).item()
        print(f"\n=== K={K} N={N}  (us; best-ours/triton; v5~triton rel={rel:.2e}) ===")
        print(f"{'M':>5} | {'triton':>8} {'v11':>8} {'v10':>8} {'v6':>8} "
              f"{'v5':>8} | {'best':>8} {'best/tri':>9} {'winner':>7}")
        for M in Ms:
            x = torch.randn(M, K, dtype=torch.float16, device=dev) * 0.5
            tt = bench(lambda: triton_w4a16_gemm(x, bq, tsc, None, G, 8)) * 1e6
            cand = {}
            if K % 1024 == 0 and G % 32 == 0 and M <= 16:
                cand["v11"] = bench(lambda: op(x, wp, sc, empty, 11)) * 1e6
            if v10_ok:
                cand["v10"] = bench(lambda: op(x, wp, sc, empty, 10)) * 1e6
            # v6 (b128 double-K) -- the mid-M lever; needs G % 32 == 0. Bit-exact
            # vs v5, so include it as a candidate and let the winner column show
            # where it beats v5/v10/Triton (-> sets VLLM_ROCM_W4A8_V6_{MIN,MAX}_M).
            if G % 32 == 0:
                cand["v6"] = bench(lambda: op(x, wp, sc, empty, 6)) * 1e6
            cand["v5"] = bench(lambda: op(x, wp, sc, empty, 5)) * 1e6
            bestk = min(cand, key=cand.get)
            best = cand[bestk]
            g = lambda k: f"{cand[k]:8.1f}" if k in cand else f"{'-':>8}"
            print(f"{M:>5} | {tt:8.1f} {g('v11')} {g('v10')} {g('v6')} "
                  f"{g('v5')} | {best:8.1f} {best/tt:9.2f} {bestk:>7}")


if __name__ == "__main__":
    main()
