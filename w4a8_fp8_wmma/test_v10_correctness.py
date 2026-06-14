"""v10 correctness vs v0 (golden) and v5, over G in {32,128}, sym + asym/AWQ,
M-tails, DB on/off.  HIP_VISIBLE_DEVICES=0 python /tmp/test_v10_correctness.py
"""
import os, torch
import w4a8_fp8_wmma  # noqa
op = torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm


def pack_w(w):
    N, K = w.shape
    wp = torch.zeros(N, K // 8, dtype=torch.int32, device=w.device)
    for j in range(8):
        wp |= (w[:, j::8].to(torch.int32) & 0xF) << (j * 4)
    return wp


def pack_zeros(z):
    N, G = z.shape
    zp = torch.zeros(N // 8, G, dtype=torch.int32, device=z.device)
    for j in range(8):
        zp |= (z[j::8, :].to(torch.int32) & 0xF) << (j * 4)
    return zp


def run(M, K, N, G, asym, dev="cuda"):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.float16, device=dev) * 0.5
    w = torch.randint(0, 16, (N, K), dtype=torch.int32, device=dev)
    sc = torch.rand(N, K // G, dtype=torch.float16, device=dev) * 0.02 + 0.001
    wp = pack_w(w)
    zeros = (pack_zeros(torch.randint(0, 16, (N, K // G), dtype=torch.int32, device=dev))
             if asym else torch.empty(0, dtype=torch.int32, device=dev))
    o0 = op(x, wp, sc, zeros, 0).float()
    o5 = op(x, wp, sc, zeros, 5).float()
    o10 = op(x, wp, sc, zeros, 10).float()

    def rel(a, b):
        return ((a - b).abs().mean() / b.abs().mean().clamp_min(1e-6)).item()
    r5, r0 = rel(o10, o5), rel(o10, o0)
    print(f"M={M:>5} K={K:>5} N={N:>5} G={G:>3} {'asym' if asym else 'sym '} | "
          f"v10-v5 rel={r5:.1e} | v10-v0 rel={r0:.1e}")
    return r5, r0


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_name(0), "DB=", os.environ.get("VLLM_W4A8_V10_DB", "1"))
    w5 = w0 = 0.0
    for c in [(256,4096,4096,128,False),(512,4096,4096,32,True),
              (1024,2304,2304,32,True),(2048,4096,11008,128,False),
              (37,4096,4096,128,False),(333,2304,2304,32,True),
              (96,4096,4096,32,False),(2048,8192,16384,128,True)]:
        a, b = run(*c); w5 = max(w5, a); w0 = max(w0, b)
    print(f"\nWORST: v10-v5 rel={w5:.1e}  v10-v0 rel={w0:.1e}")
    print("RESULT:", "PASS" if (w5 < 1e-3 and w0 < 0.06) else "FAIL")
