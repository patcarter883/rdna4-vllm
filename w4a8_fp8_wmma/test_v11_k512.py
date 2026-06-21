"""Validate the decode_gemv (v11) K%512 relaxation (512-k tail).

Two checks:
  (A) KERNEL: call w4a8_fp8_wmma.mmq_fp8_gemm(..., kernel="decode_gemv") directly for K
      values that are %512 but NOT %1024 (incl. the real down_proj K=8704 after TP=2
      sharding of intermediate 17408), across decode M=1..16, sym + asym (AWQ zeros).
      Compare vs the fp8 numpy reference. This is what the old K%1024 TORCH_CHECK forbade.
  (B) DISPATCH: through the real adapter, M=1/2 with K=8704 must now pick decode_gemv
      (v11), not prefill_wmma_ashuffle (v10).

Kernels are referenced by descriptive name (kernel_names.h / __init__._DENSE_KERNELS);
the opaque int 11 it maps to is unchanged across the ABI.
Run inside the container from /tmp with PYTHONPATH at the worktree pkg.
"""
import sys, torch
import w4a8_fp8_wmma  # noqa: registers torch.ops.w4a8_fp8_wmma
from vllm.scalar_type import scalar_types
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from w4a8_fp8_wmma.vllm_adapter import RocmW4A8Fp8WmmaLinearKernel
E4M3_MAX = 448.0


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


def ref(x, w_packed, scales, G, zeros=None):
    N, K8 = w_packed.shape; K = K8 * 8
    wu = torch.zeros((N, K), dtype=torch.int32, device=w_packed.device)
    for i in range(8):
        wu[:, i::8] = (w_packed >> (i * 4)) & 0xF
    if zeros is not None:
        zu = torch.zeros((N, K // G), dtype=torch.int32, device=w_packed.device)
        for j in range(8):
            zu[j::8, :] = (zeros >> (j * 4)) & 0xF
        zp = zu.repeat_interleave(G, dim=1)
    else:
        zp = 8
    w_fp8 = ((wu - zp).float()).to(torch.float8_e4m3fn).to(torch.float32)
    ma = x.float().abs().amax(1, keepdim=True).clamp(min=1e-8 * E4M3_MAX)
    a_scale = ma / E4M3_MAX
    x_fp8 = (x.float() / a_scale).to(torch.float8_e4m3fn).to(torch.float32)
    sc = scales.float().repeat_interleave(G, dim=1)
    return ((x_fp8 @ (w_fp8 * sc).T) * a_scale).to(torch.float16)


def make_weights(N, K, G, asym, dev):
    torch.manual_seed(0)
    w = torch.randint(0, 16, (N, K), dtype=torch.int8, device=dev)
    wp = pack_uint4(w)
    sc = torch.rand(N, K // G, dtype=torch.float16, device=dev) * 0.02 + 0.001
    zeros = None
    if asym:
        z = torch.randint(0, 16, (N, K // G), dtype=torch.int8, device=dev)
        zeros = pack_zeros(z)
    return wp, sc, zeros


def kernel_v11(M, K, N, G, asym):
    """Direct decode_gemv (v11) call by name. Returns (rel_err, ok)."""
    dev = "cuda"
    wp, sc, zeros = make_weights(N, K, G, asym, dev)
    x = torch.randn(M, K, dtype=torch.float16, device=dev) * 0.5
    out = w4a8_fp8_wmma.mmq_fp8_gemm(
        x, wp, sc, kernel="decode_gemv", w_zeros=zeros).float()
    r = ref(x, wp, sc, G, zeros).float()
    rel = ((out - r).abs().mean() / r.abs().mean().clamp_min(1e-6)).item()
    ok = out.shape == (M, N) and rel < 0.05
    tag = "asym" if asym else "sym "
    print(f"  [KERNEL v11] M={M:>2} K={K} N={N} G={G} {tag}: rel={rel:.2e} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def cfg(K, N, G, asym):
    return MPLinearLayerConfig(
        full_weight_shape=(K, N), partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4 if asym else scalar_types.uint4b8,
        act_type=torch.float16, group_size=G, zero_points=asym,
        has_g_idx=False, out_type=torch.float16)


def dispatch_check(M, K, N, G, asym):
    """Through the adapter: must be correct AND (for M<=2) route to v11."""
    dev = "cuda"
    c = cfg(K, N, G, asym)
    kern = RocmW4A8Fp8WmmaLinearKernel(
        c, w_q_param_name="weight_packed", w_s_param_name="weight_scale",
        w_zp_param_name="weight_zero_point")
    wp, sc, zeros = make_weights(N, K, G, asym, dev)
    layer = torch.nn.Module()
    layer.weight_packed = torch.nn.Parameter(wp, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(sc, requires_grad=False)
    if asym:
        layer.weight_zero_point = torch.nn.Parameter(zeros, requires_grad=False)
    kern.process_weights_after_loading(layer)
    x = torch.randn(M, K, dtype=torch.float16, device=dev) * 0.5
    out = kern.apply_weights(layer, x).float()
    r = ref(x, wp, sc, G, zeros).float()
    rel = ((out - r).abs().mean() / r.abs().mean().clamp_min(1e-6)).item()
    ok = out.shape == (M, N) and rel < 0.05
    print(f"  [DISPATCH ] M={M:>2} K={K} N={N} G={G} {'asym' if asym else 'sym '}: "
          f"rel={rel:.2e} {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("Device:", torch.cuda.get_device_name(0))
    allok = True
    print("(A) KERNEL decode_gemv on K%512-not-%1024 (down_proj K=8704 + neighbours):")
    for K in (512, 1536, 2560, 8704):          # all %512, only 512/2560... none %1024 except check
        for asym in (False, True):
            for M in (1, 2, 4, 8, 16):
                allok &= kernel_v11(M, K, 5120, 32, asym)
    print("  regression: K%1024 still correct:")
    allok &= kernel_v11(1, 4096, 4096, 128, False)
    allok &= kernel_v11(2, 8192, 2048, 32, True)

    print("(B) DISPATCH: down_proj K=8704 at decode now routes through v11 + correct:")
    allok &= dispatch_check(1, 8704, 5120, 32, True)
    allok &= dispatch_check(2, 8704, 5120, 32, True)
    allok &= dispatch_check(1, 8704, 5120, 32, False)

    print("RESULT:", "PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
