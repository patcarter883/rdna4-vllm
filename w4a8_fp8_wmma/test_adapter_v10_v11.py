"""Adapter dispatch test: decode (M small) -> v11, prefill (M large) -> v10,
through the real RocmW4A8Fp8WmmaLinearKernel.apply_weights path. Sym + AWQ.
Run from /tmp (not the source pkg dir) inside the container.
"""
import sys, torch
import w4a8_fp8_wmma  # noqa
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


def cfg(K, N, G, asym):
    return MPLinearLayerConfig(
        full_weight_shape=(K, N), partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4 if asym else scalar_types.uint4b8,
        act_type=torch.float16, group_size=G, zero_points=asym,
        has_g_idx=False, out_type=torch.float16)


def run(M, K, N, G, asym, expect):
    dev = "cuda"
    torch.manual_seed(0)
    c = cfg(K, N, G, asym)
    kern = RocmW4A8Fp8WmmaLinearKernel(
        c, w_q_param_name="weight_packed", w_s_param_name="weight_scale",
        w_zp_param_name="weight_zero_point")
    w = torch.randint(0, 16, (N, K), dtype=torch.int8, device=dev)
    wp = pack_uint4(w)
    sc = torch.rand(N, K // G, dtype=torch.float16, device=dev) * 0.02 + 0.001
    layer = torch.nn.Module()
    layer.weight_packed = torch.nn.Parameter(wp, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(sc, requires_grad=False)
    zeros = None
    if asym:
        z = torch.randint(0, 16, (N, K // G), dtype=torch.int8, device=dev)
        zeros = pack_zeros(z)
        layer.weight_zero_point = torch.nn.Parameter(zeros, requires_grad=False)
    kern.process_weights_after_loading(layer)
    x = torch.randn(M, K, dtype=torch.float16, device=dev) * 0.5
    out = kern.apply_weights(layer, x).float()
    r = ref(x, wp, sc, G, zeros).float()
    rel = ((out - r).abs().mean() / r.abs().mean().clamp_min(1e-6)).item()
    ok = out.shape == (M, N) and rel < 0.05
    print(f"  M={M:>4} K={K} N={N} G={G} {'asym' if asym else 'sym '} "
          f"({expect}): rel={rel:.2e} {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("Device:", torch.cuda.get_device_name(0))
    allok = True
    # decode -> v11 (K%1024==0, M<=4)
    allok &= run(1, 4096, 4096, 128, False, "v11")
    allok &= run(1, 4096, 4096, 32, True, "v11")
    allok &= run(2, 8192, 2048, 128, False, "v11")
    # prefill -> v10 (M>=256)
    allok &= run(512, 4096, 4096, 128, False, "v10")
    allok &= run(512, 4096, 2048, 32, True, "v10")
    # mid-M -> Triton fallback (4 < M < 256)
    allok &= run(64, 4096, 4096, 128, False, "triton")
    print("RESULT:", "PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
