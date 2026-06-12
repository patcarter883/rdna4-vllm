"""In-container vLLM integration test for the W4A8-FP8 WMMA kernel (gfx1201).

Runs three levels, each strictly more "real" than the last:
  A. Dispatcher selection: vLLM's choose/can_implement picks our kernel for a
     uint4b8 W4A16 g128 config.
  B. Layer apply: drive our adapter through vLLM's MPLinearKernel interface
     (process_weights_after_loading + apply_weights) on random packed weights and
     compare to an fp8 dequant reference.
Run inside kyuz0/vllm-therock-gfx1201 with the GPU mounted.
"""
import sys
import torch

import w4a8_fp8_wmma  # loads torch.ops.w4a8_fp8_wmma
from w4a8_fp8_wmma.register import register

from vllm.scalar_type import scalar_types
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)

E4M3_MAX = 448.0


def pack_uint4(w_int4):  # (N,K) int8 in [0,15] -> (N,K//8) int32
    N, K = w_int4.shape
    w = w_int4.to(torch.int32)
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= (w[:, i::8] & 0xF) << (i * 4)
    return packed


def fp8_ref(x, w_packed, scales, group_size):
    N, K8 = w_packed.shape
    K = K8 * 8
    wu = torch.zeros((N, K), dtype=torch.int32, device=w_packed.device)
    for i in range(8):
        wu[:, i::8] = (w_packed >> (i * 4)) & 0xF
    w_fp8 = ((wu - 8).float()).to(torch.float8_e4m3fn).to(torch.float32)
    max_abs = x.float().abs().amax(1, keepdim=True).clamp(min=1e-8 * E4M3_MAX)
    a_scale = max_abs / E4M3_MAX
    x_fp8 = (x.float() / a_scale).to(torch.float8_e4m3fn).to(torch.float32)
    sc = scales.float().repeat_interleave(group_size, dim=1)
    return ((x_fp8 @ (w_fp8 * sc).T) * a_scale).to(torch.float16)


def make_cfg(K, N, group_size):
    return MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=group_size,
        zero_points=False,
        has_g_idx=False,
        out_type=torch.float16,
    )


def test_dispatch():
    print("\n=== A. dispatcher selection ===")
    register()
    from w4a8_fp8_wmma.vllm_adapter import RocmW4A8Fp8WmmaLinearKernel
    ok, reason = RocmW4A8Fp8WmmaLinearKernel.can_implement(make_cfg(512, 256, 128))
    print(f"  can_implement(uint4b8, g128) -> {ok} ({reason})")
    # confirm it's at the front of the ROCm kernel list
    from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
    from vllm.platforms import PlatformEnum
    names = [k.__name__ for k in _POSSIBLE_KERNELS[PlatformEnum.ROCM]]
    print("  ROCm kernel order:", names)
    front = names[0] == "RocmW4A8Fp8WmmaLinearKernel"
    return ok and front


def test_layer_apply():
    print("\n=== B. layer apply via vLLM MPLinearKernel interface ===")
    from w4a8_fp8_wmma.vllm_adapter import RocmW4A8Fp8WmmaLinearKernel
    dev = "cuda"
    K, N, G, M = 512, 256, 128, 64
    cfg = make_cfg(K, N, G)
    kern = RocmW4A8Fp8WmmaLinearKernel(
        cfg, w_q_param_name="weight_packed", w_s_param_name="weight_scale")

    w_int4 = torch.randint(0, 16, (N, K), dtype=torch.int8, device=dev)
    w_packed = pack_uint4(w_int4)
    scales = torch.rand(N, K // G, dtype=torch.float16, device=dev) * 0.02 + 0.001

    layer = torch.nn.Module()
    layer.weight_packed = torch.nn.Parameter(w_packed, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    kern.process_weights_after_loading(layer)

    x = torch.randn(M, K, dtype=torch.float16, device=dev)
    out = kern.apply_weights(layer, x)
    ref = fp8_ref(x, w_packed, scales, G)

    diff = (out.float() - ref.float()).abs()
    mean_abs = diff.mean().item()
    n_bad = int((diff > 0.15 + 0.03 * ref.float().abs()).sum().item())
    print(f"  out {tuple(out.shape)} mean_abs={mean_abs:.6f} out_of_tol={n_bad}/{diff.numel()}")
    return out.shape == (M, N) and mean_abs < 2e-3 and n_bad == 0


def main():
    if not torch.cuda.is_available():
        print("FAIL: no GPU"); sys.exit(1)
    print("Device:", torch.cuda.get_device_name(0))
    a = test_dispatch()
    b = test_layer_apply()
    print("\n" + "=" * 50)
    print(f"A dispatcher: {'PASS' if a else 'FAIL'}")
    print(f"B layer apply: {'PASS' if b else 'FAIL'}")
    sys.exit(0 if (a and b) else 1)


if __name__ == "__main__":
    main()
