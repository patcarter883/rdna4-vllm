"""v6 (K-extension) correctness vs v0 (scalar golden) and v5 (raw b64 WMMA).

v6 loads byte-identical operands to v5 via b128 dual-subtile reads, so it must
match v5 EXACTLY (the loaded bytes are bit-identical; only the LDS load width /
WMMA issue order differ -- WMMA accumulation order is unchanged) and v0 to
fp8-quant tolerance. Run on GPU1:
    HIP_VISIBLE_DEVICES=1 python /tmp/test_v6_correctness.py

Expected tolerance (the served-dispatch gate VLLM_ROCM_W4A8_V6_{MIN,MAX}_M relies
on this): v6-vs-v5 relmean < 1e-3 (effectively 0 -- identical operands & MMA
order; any nonzero is fp16-store rounding noise), v6-vs-v0 relmean < 0.06 (the
shared fp8 activation-quant tolerance, same bound v5 meets). Covers every
group_size the gate can engage (gs % 32 == 0, <= 128: 32/64/96/128), sym + asym.
"""
import torch
import w4a8_fp8_wmma  # noqa: F401  loads torch.ops.w4a8_fp8_wmma

op = torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm


def pack_w(w_int4):  # (N,K) int in [0,16) -> (N,K//8) int32
    N, K = w_int4.shape
    wp = torch.zeros(N, K // 8, dtype=torch.int32, device=w_int4.device)
    for j in range(8):
        wp |= (w_int4[:, j::8].to(torch.int32) & 0xF) << (j * 4)
    return wp


def pack_zeros(z_int4):  # (N,G) -> (N//8,G) int32 packed along N
    N, G = z_int4.shape
    zp = torch.zeros(N // 8, G, dtype=torch.int32, device=z_int4.device)
    for j in range(8):
        zp |= (z_int4[j::8, :].to(torch.int32) & 0xF) << (j * 4)
    return zp


def run_case(M, K, N, G, asym, dev="cuda"):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.float16, device=dev) * 0.5
    w_int4 = torch.randint(0, 16, (N, K), dtype=torch.int32, device=dev)
    scale = (torch.rand(N, K // G, dtype=torch.float16, device=dev) * 0.02 + 0.001)
    wp = pack_w(w_int4)
    if asym:
        z_int4 = torch.randint(0, 16, (N, K // G), dtype=torch.int32, device=dev)
        zeros = pack_zeros(z_int4)
    else:
        zeros = torch.empty(0, dtype=torch.int32, device=dev)

    o0 = op(x, wp, scale, zeros, 0).float()
    o5 = op(x, wp, scale, zeros, 5).float()
    o6 = op(x, wp, scale, zeros, 6).float()

    def stats(a, b):
        d = (a - b).abs()
        denom = b.abs().mean().clamp_min(1e-6)
        return d.max().item(), (d.mean() / denom).item()

    md65, rm65 = stats(o6, o5)
    md60, rm60 = stats(o6, o0)
    tag = "asym" if asym else "sym "
    print(f"M={M:>5} K={K:>5} N={N:>5} G={G:>3} {tag} | "
          f"v6-v5 max={md65:.4e} relmean={rm65:.2e} | "
          f"v6-v0 max={md60:.4e} relmean={rm60:.2e}")
    return rm65, rm60


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_name(0))
    worst65 = worst60 = 0.0
    cases = [
        (256, 4096, 4096, 128, False),
        (256, 4096, 4096, 128, True),
        (512, 4096, 4096, 32, False),
        (512, 4096, 4096, 32, True),
        (1024, 2304, 2304, 32, True),   # Qwen3.6-ish hidden
        (2048, 4096, 11008, 128, False),
        (37, 4096, 4096, 128, False),   # M tail
        (4096, 4096, 4096, 64, True),   # G=64
        (768, 4608, 4096, 96, False),   # G=96 (gate also engages here)
        (768, 4608, 4096, 96, True),
    ]
    for (M, K, N, G, asym) in cases:
        r65, r60 = run_case(M, K, N, G, asym)
        worst65 = max(worst65, r65)
        worst60 = max(worst60, r60)
    print(f"\nWORST relmean: v6-v5={worst65:.2e}  v6-v0={worst60:.2e}")
    ok = worst65 < 1e-3 and worst60 < 0.06
    print("RESULT:", "PASS" if ok else "FAIL")
