"""v7 correctness vs v0 (golden) and v5, over model-relevant cases
(G=32/64/128, sym + asym/AWQ, M-tails). CFG via env (default 256x128).
    HIP_VISIBLE_DEVICES=1 VLLM_W4A8_V7_CFG=256x128 python /tmp/test_v7_correctness.py
"""
import torch
import w4a8_fp8_wmma  # noqa: F401
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
    o7 = op(x, wp, sc, zeros, 7).float()

    def rel(a, b):
        return ((a - b).abs().mean() / b.abs().mean().clamp_min(1e-6)).item()
    r75, r70 = rel(o7, o5), rel(o7, o0)
    md = (o7 - o5).abs().max().item()
    print(f"M={M:>5} K={K:>5} N={N:>5} G={G:>3} {'asym' if asym else 'sym '} | "
          f"v7-v5 max={md:.2e} rel={r75:.1e} | v7-v0 rel={r70:.1e}")
    return r75, r70


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_name(0))
    w75 = w70 = 0.0
    for c in [(256,4096,4096,128,False),(512,4096,4096,32,True),
              (1024,2304,2304,32,True),(2048,4096,11008,128,False),
              (37,4096,4096,128,False),(4096,4096,4096,64,True),
              (333,2304,2304,32,True),(96,4096,4096,32,False)]:
        a, b = run(*c); w75 = max(w75, a); w70 = max(w70, b)
    print(f"\nWORST: v7-v5 rel={w75:.1e}  v7-v0 rel={w70:.1e}")
    print("RESULT:", "PASS" if (w75 < 1e-3 and w70 < 0.06) else "FAIL")
