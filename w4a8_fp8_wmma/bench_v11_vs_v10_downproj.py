"""Payoff check: decode_gemv (v11) vs prefill_wmma_ashuffle (v10) for the down_proj
decode shape that the K%512 fix unlocks. N=5120, K=8704, gs=32, asym (AWQ), M=1/2/4.
Reports per-call us and effective weight-read GB/s (weights = N*K/2 bytes int4 +
scales/zeros). Kernels referenced by their descriptive names (kernel_names.h)."""
import torch, time
import w4a8_fp8_wmma  # noqa


def pack_uint4(w):
    N, K = w.shape; w = w.to(torch.int32)
    p = torch.zeros((N, K // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        p |= (w[:, i::8] & 0xF) << (i * 4)
    return p


def pack_zeros(z):
    N, G = z.shape; z = z.to(torch.int32)
    p = torch.zeros((N // 8, G), dtype=torch.int32, device=z.device)
    for j in range(8):
        p |= (z[j::8, :] & 0xF) << (j * 4)
    return p


def bench(kernel, x, wp, sc, zp, iters=200):
    f = lambda: w4a8_fp8_wmma.mmq_fp8_gemm(x, wp, sc, kernel=kernel, w_zeros=zp)
    for _ in range(20):
        f()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        f()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e6  # us/call


def main():
    dev = "cuda"
    N, K, G = 5120, 8704, 32
    torch.manual_seed(0)
    w = torch.randint(0, 16, (N, K), dtype=torch.int8, device=dev)
    wp = pack_uint4(w)
    sc = torch.rand(N, K // G, dtype=torch.float16, device=dev) * 0.02 + 0.001
    z = torch.randint(0, 16, (N, K // G), dtype=torch.int8, device=dev)
    zp = pack_zeros(z)
    wbytes = N * K // 2 + N * (K // G) * 2 + (N // 8) * (K // G) * 4
    print(f"down_proj N={N} K={K} gs={G} asym | weight bytes ~= {wbytes/1e6:.1f} MB")
    print(f"{'M':>3} {'v10 us':>9} {'v11 us':>9} {'speedup':>8} {'v11 GB/s':>9}")
    for M in (1, 2, 4):
        x = torch.randn(M, K, dtype=torch.float16, device=dev) * 0.5
        t10 = bench("prefill_wmma_ashuffle", x, wp, sc, zp)
        t11 = bench("decode_gemv", x, wp, sc, zp)
        gbps = wbytes / (t11 * 1e-6) / 1e9
        print(f"{M:>3} {t10:>9.1f} {t11:>9.1f} {t10/t11:>7.2f}x {gbps:>8.0f}")


if __name__ == "__main__":
    main()
