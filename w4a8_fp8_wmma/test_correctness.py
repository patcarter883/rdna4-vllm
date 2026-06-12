"""Correctness test for the W4A8-FP8 MMQ HIP custom op (gfx1201).

v0 (scalar fp8 reference) is checked against a CPU/torch model of the SAME fp8
arithmetic: int4 weights expanded to e4m3, activations quantized to e4m3 per row,
f32 accumulate, per-group weight scale, per-row activation scale.

The point of this test is to confirm two things at once:
  1. The data path / packing / scales plumbing is correct.
  2. int4 -> fp8 e4m3 is lossless for symmetric weights (the research claim):
     with NO activation quant (x already representable), v0 should match an
     exact int4 dequant matmul to within fp16 rounding.
"""
import sys

import torch

try:
    import w4a8_fp8_wmma
except ImportError as e:
    print(f"FAIL: import error: {e}")
    print("Did you run `python setup.py build_ext --inplace` first?")
    sys.exit(1)

E4M3_MAX = 448.0


def pack_uint4_weights(w_int4: torch.Tensor) -> torch.Tensor:
    assert w_int4.dtype == torch.int8
    assert ((w_int4 >= 0) & (w_int4 <= 15)).all()
    N, K = w_int4.shape
    assert K % 8 == 0
    w_int4 = w_int4.to(torch.int32)
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=w_int4.device)
    for i in range(8):
        packed |= (w_int4[:, i::8] & 0xF) << (i * 4)
    return packed


def pack_zeros(zeros_int4: torch.Tensor) -> torch.Tensor:
    """Pack per-(col, group) zero points into the op's (N//8, K//group) layout.

    zeros_int4: (N, num_groups) int in [0,15]. 8 consecutive output columns are
    packed per int32 (nibble = n % 8, standard order) — mirrors how the kernel
    reads w_zeros_packed[(n//8)*num_groups + g] >> ((n%8)*4).
    """
    assert zeros_int4.dtype == torch.int8
    assert ((zeros_int4 >= 0) & (zeros_int4 <= 15)).all()
    N, G = zeros_int4.shape
    assert N % 8 == 0
    z = zeros_int4.to(torch.int32)
    packed = torch.zeros((N // 8, G), dtype=torch.int32, device=z.device)
    for i in range(8):
        packed |= (z[i::8, :] & 0xF) << (i * 4)
    return packed


def to_e4m3_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Round a float tensor through torch's e4m3 and back (matches the HW cvt)."""
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def reference_fp8_model(x, w_packed, scales, group_size=32):
    """CPU/torch model of v0's exact fp8 arithmetic."""
    N, K_packed = w_packed.shape
    K = K_packed * 8
    w_unpacked = torch.zeros((N, K), dtype=torch.int32, device=w_packed.device)
    for i in range(8):
        w_unpacked[:, i::8] = (w_packed >> (i * 4)) & 0xF
    # symmetric uint4b8: signed = nibble - 8, in [-8,7], exact in e4m3
    w_signed = (w_unpacked - 8).to(torch.float32)
    w_fp8 = to_e4m3_roundtrip(w_signed)  # exact for [-8,7]

    # per-row activation scale = max_abs / 448, quantize to e4m3
    max_abs = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8 * E4M3_MAX)
    a_scale = max_abs / E4M3_MAX
    x_fp8 = to_e4m3_roundtrip(x.float() / a_scale)

    # Accumulate per K-group, matching the kernel's reduction ORDER so the test
    # isolates real logic bugs from fp32 non-associativity (a single big matmul
    # rounds differently than 32-wide group sums summed across groups).
    M, K2 = x_fp8.shape
    N = w_fp8.shape[0]
    num_groups = K // group_size
    xg = x_fp8.view(M, num_groups, group_size)
    wg = w_fp8.view(N, num_groups, group_size)
    # per-group partial sums: (M, N, num_groups)
    partial = torch.einsum("mgk,ngk->mng", xg, wg)
    out = (partial * scales.to(torch.float32).unsqueeze(0)).sum(dim=-1)
    return (out * a_scale).to(torch.float16)


def reference_fp8_model_asym(x, w_packed, scales, zeros, group_size=32):
    """CPU/torch model of the kernel's ASYMMETRIC (AWQ) fp8 arithmetic.

    Same as reference_fp8_model but the per-group zero point is explicit
    (zeros: (N, num_groups) int in [0,15]) instead of the implicit 8. The signed
    weight nibble-zp lies in [-15,15], which is still exact in e4m3, so the only
    precision loss is on the activation side.
    """
    N, K_packed = w_packed.shape
    K = K_packed * 8
    w_unpacked = torch.zeros((N, K), dtype=torch.int32, device=w_packed.device)
    for i in range(8):
        w_unpacked[:, i::8] = (w_packed >> (i * 4)) & 0xF
    num_groups = K // group_size
    zp_full = zeros.to(torch.int32).repeat_interleave(group_size, dim=1)  # (N, K)
    w_signed = (w_unpacked - zp_full).to(torch.float32)
    w_fp8 = to_e4m3_roundtrip(w_signed)  # exact for [-15,15]

    max_abs = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8 * E4M3_MAX)
    a_scale = max_abs / E4M3_MAX
    x_fp8 = to_e4m3_roundtrip(x.float() / a_scale)

    M = x_fp8.shape[0]
    xg = x_fp8.view(M, num_groups, group_size)
    wg = w_fp8.view(N, num_groups, group_size)
    partial = torch.einsum("mgk,ngk->mng", xg, wg)
    out = (partial * scales.to(torch.float32).unsqueeze(0)).sum(dim=-1)
    return (out * a_scale).to(torch.float16)


def run_one_asym(M, N, K, version=0, group_size=32, atol=0.15, rtol=0.03,
                 mean_tol=2e-3, mean_rtol=0.01, x_scale=1.0):
    """Asymmetric (AWQ) variant of run_one: random per-(col,group) zero points,
    packed into the op's (N//8, K//group) layout and passed as w_zeros.

    Pass criterion is magnitude-aware: the mean abs error must be within
    max(mean_tol, mean_rtol * mean|ref|). AWQ weights span [-15,15] (vs [-8,7]
    symmetric), so with deep K the outputs are larger and the legitimate fp8
    activation noise scales with them; a fixed absolute mean_tol calibrated for
    small outputs would false-fail. mean_rtol=1% still catches real logic bugs
    (a dropped/wrong zero point shifts the mean by O(|ref|))."""
    print(f"\n=== v{version} ASYM: M={M}, N={N}, K={K}, g={group_size} ===")
    device = torch.device("cuda")
    x = torch.randn(M, K, dtype=torch.float16, device=device) * x_scale
    w_int4 = torch.randint(0, 16, (N, K), dtype=torch.int8, device=device)
    w_packed = pack_uint4_weights(w_int4)
    num_groups = K // group_size
    scales = torch.randn(N, num_groups, dtype=torch.float16,
                         device=device).abs() * 0.01 + 0.001
    zeros = torch.randint(0, 16, (N, num_groups), dtype=torch.int8, device=device)
    zeros_packed = pack_zeros(zeros)

    out_ref = reference_fp8_model_asym(x, w_packed, scales, zeros, group_size)
    out_ours = w4a8_fp8_wmma.mmq_fp8_gemm(
        x, w_packed, scales, version=version, w_zeros=zeros_packed)

    diff = (out_ours.float() - out_ref.float()).abs()
    ref = out_ref.float().abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    mean_ref = ref.mean().item()
    eff_mean_tol = max(mean_tol, mean_rtol * mean_ref)
    n_bad = int((diff > atol + rtol * ref).sum().item())
    print(f"  diff: max_abs={max_abs:.6f}  mean_abs={mean_abs:.6f}  "
          f"|ref|mean={mean_ref:.4f}  out_of_tol={n_bad}/{diff.numel()}  "
          f"(eff_mean_tol={eff_mean_tol:.6f})")
    ok = (mean_abs <= eff_mean_tol) and (n_bad == 0)
    print("  PASS" if ok
          else f"  FAIL (mean_abs>{eff_mean_tol:.6f} or {n_bad} elems out of tol)")
    return ok


def run_one(M, N, K, version=0, group_size=32, atol=0.15, rtol=0.03, mean_tol=2e-3, x_scale=1.0):
    """Pass criteria reflect the fp8 precision regime:
      - mean_abs is the global correctness signal (a real bug shows a systematic
        offset and blows this up); held to mean_tol.
      - per-element atol has a floor (0.15) because near-zero outputs, after K-deep
        sign-cancelling fp32 sums, are dominated by fp8 rounding noise where the
        HW cvt and torch's fp8 cast disagree by ~1 ULP. rtol covers larger outputs.
    """
    print(f"\n=== v{version}: M={M}, N={N}, K={K} ===")
    device = torch.device("cuda")
    x = torch.randn(M, K, dtype=torch.float16, device=device) * x_scale
    w_int4 = torch.randint(0, 16, (N, K), dtype=torch.int8, device=device)
    w_packed = pack_uint4_weights(w_int4)
    scales = torch.randn(N, K // group_size, dtype=torch.float16, device=device).abs() * 0.01 + 0.001

    out_ref = reference_fp8_model(x, w_packed, scales, group_size)
    out_ours = w4a8_fp8_wmma.mmq_fp8_gemm(x, w_packed, scales, version=version)

    diff = (out_ours.float() - out_ref.float()).abs()
    ref = out_ref.float().abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    n_bad = int((diff > atol + rtol * ref).sum().item())
    print(f"  diff: max_abs={max_abs:.6f}  mean_abs={mean_abs:.6f}  "
          f"out_of_tol={n_bad}/{diff.numel()}  (mean_tol={mean_tol})")
    ok = (mean_abs <= mean_tol) and (n_bad == 0)
    print("  PASS" if ok else f"  FAIL (mean_abs>{mean_tol} or {n_bad} elems out of tol)")
    return ok


def main():
    if not torch.cuda.is_available():
        print("FAIL: no CUDA/HIP device")
        sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # (M, N, K, group_size). v1 now handles M/N tails and group_size in {32,64,128}.
    v0_shapes = [(64, 256, 256, 32), (128, 1024, 1024, 128), (100, 80, 256, 32)]
    v1_shapes = [
        (16, 16, 32, 32),       # exact tile, g32
        (64, 64, 256, 128),     # g128
        (128, 256, 1024, 128),  # g128 larger
        (256, 4096, 4096, 128), # model-ish, g128
        (1, 4096, 4096, 128),   # decode shape (M=1 tail)
        (100, 80, 256, 32),     # M and N tails
        (17, 48, 256, 32),      # odd M tail
    ]

    # Asymmetric (AWQ) shapes: explicit per-group zero points. v0 is the golden
    # scalar path; v5 is the optimized raw-builtin kernel used at large M.
    asym_shapes = [
        (64, 256, 256, 32),     # g32
        (128, 1024, 1024, 128), # g128
        (256, 512, 1024, 32),   # v5 tile-ish, g32
        (1, 4096, 4096, 128),   # decode shape (M=1 tail), g128
        (100, 80, 256, 32),     # M and N tails
    ]

    print("\n" + "=" * 60 + "\nv0 (scalar fp8 reference)\n" + "=" * 60)
    v0 = [run_one(M, N, K, version=0, group_size=g) for M, N, K, g in v0_shapes]
    print("\n" + "=" * 60 + "\nv1 (rocWMMA fp8 16x16x16, tails + g128)\n" + "=" * 60)
    v1 = [run_one(M, N, K, version=1, group_size=g) for M, N, K, g in v1_shapes]
    print("\n" + "=" * 60 + "\nv0 ASYM (AWQ zero points, golden)\n" + "=" * 60)
    a0 = [run_one_asym(M, N, K, version=0, group_size=g)
          for M, N, K, g in asym_shapes]
    print("\n" + "=" * 60 + "\nv5 ASYM (AWQ zero points, optimized)\n" + "=" * 60)
    a5 = [run_one_asym(M, N, K, version=5, group_size=g)
          for M, N, K, g in asym_shapes]

    print("\n" + "=" * 60)
    results = v0 + v1 + a0 + a5
    if all(results):
        print(f"ALL PASSED ({len(v0)} v0 + {len(v1)} v1 + {len(a0)} v0-asym "
              f"+ {len(a5)} v5-asym)")
        sys.exit(0)
    print(f"FAIL: v0={sum(1 for r in v0 if not r)}/{len(v0)}, "
          f"v1={sum(1 for r in v1 if not r)}/{len(v1)}, "
          f"v0_asym={sum(1 for r in a0 if not r)}/{len(a0)}, "
          f"v5_asym={sum(1 for r in a5 if not r)}/{len(a5)}")
    sys.exit(1)


if __name__ == "__main__":
    main()
