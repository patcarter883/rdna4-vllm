"""vLLM mxfp4 (OCP E2M1) dense linear kernel for gfx1201, backed by the W4A8 FP8-WMMA op.

vLLM ships NO working RDNA4 mxfp4 GEMM: `CompressedTensorsW4A4Mxfp4.__init__` calls
`init_mxfp4_linear_kernel()`, whose `_POSSIBLE_MXFP4_KERNELS` has no ROCm entry -> it raises and an
mxfp4 model can't even construct on gfx1201. We close that by registering THIS kernel into
`_POSSIBLE_MXFP4_KERNELS[ROCM]` (see register.py::register_mxfp4_dense), so the *unmodified*
W4A4Mxfp4 scheme picks it up: it converts the E2M1/E8M0 weights to the W4A8 kernel's native layout
at load and routes apply() through `torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm(..., weight_is_e2m1=True)`.

Weights handed to us by the scheme (after its process_weights renames weight_packed->weight):
  layer.weight        uint8  (N, K//2)   2 E2M1 nibbles/byte, low nibble = lower K
  layer.weight_scale  uint8  (N, K//32)  E8M0 per-32-block exponent
We replace them with the kernel-native packed int32 + fp16 group scale (mxfp4_convert).

NOTE on accuracy: mxfp4 reference models are W4A16 (bf16 activations); this path adds dynamic
per-row fp8 activation quant (A8). That is an accuracy change vs the reference and must pass a PPL
gate before being trusted (the activation-rotation work narrows this). Weight decode itself is
bit-exact (see mxfp4/test_mxfp4_kernel_gpu.py).
"""
import torch

from vllm.model_executor.kernels.linear.mxfp4.base import (
    MxFp4LinearKernel, MxFp4LinearLayerConfig,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def _pick_dense_kernel(M: int, K: int, group_size: int = 32) -> str:
    """Mirror the int4 adapter's intent: GEMV for decode, A-shuffle WMMA for prefill.

    mxfp4 group is always 32, so prefill_wmma_ashuffle (needs group 32|128) handles any M, and
    decode_gemv (needs M<=16, K%512==0, group%32==0) handles the decode regime.
    """
    if M <= 16 and K % 512 == 0 and group_size % 32 == 0:
        return "decode_gemv"
    return "prefill_wmma_ashuffle"


class RocmW4A8MxFp4LinearKernel(MxFp4LinearKernel):
    """E2M1 dense linear via the W4A8 FP8-WMMA op (gfx1201/RDNA4)."""

    @classmethod
    def is_supported(cls, compute_capability=None):
        from vllm.platforms import current_platform
        if not current_platform.is_rocm():
            return False, "ROCm only"
        try:
            from vllm.platforms.rocm import on_gfx1x
            if not on_gfx1x():
                return False, "requires gfx1x (RDNA4)"
        except Exception as e:  # pragma: no cover - defensive
            return False, f"cannot determine gfx arch: {e}"
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig):
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from mxfp4.convert import convert_mxfp4_weight
        dev = layer.weight.device
        # Convert on CPU (the converter uses host int ops); move the small results back.
        conv = convert_mxfp4_weight(layer.weight.data.cpu(), layer.weight_scale.data.cpu())
        info = conv["scale_info"]
        if not info["fp16_range_ok"]:
            logger.warning(
                "[w4a8 mxfp4] E8M0 group scales exceed the fp16 store on %s "
                "(exp range %d..%d, %d overflow / %d e8m0-NaN groups); results may be wrong — "
                "an fp32 group-scale path is needed for this checkpoint.",
                getattr(layer, "prefix", "<linear>"), info["exp_min"], info["exp_max"],
                info["fp16_overflow_groups"], info["e8m0_nan_groups"])
        layer._w4a8_mxfp4_wpacked = torch.nn.Parameter(
            conv["w_packed"].to(dev), requires_grad=False)
        layer._w4a8_mxfp4_scales = torch.nn.Parameter(
            conv["scales"].to(dev), requires_grad=False)
        # Release the source E2M1/E8M0 tensors.
        del layer.weight
        if hasattr(layer, "weight_scale"):
            del layer.weight_scale

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: torch.Tensor | None = None) -> torch.Tensor:
        import w4a8_fp8_wmma as w4a8
        out_dtype = x.dtype
        x2 = x.reshape(-1, x.shape[-1]).to(torch.float16)
        kern = _pick_dense_kernel(x2.shape[0], x2.shape[1])
        out = w4a8.mmq_fp8_gemm(x2, layer._w4a8_mxfp4_wpacked, layer._w4a8_mxfp4_scales,
                                kernel=kern, weight_is_e2m1=True)
        out = out.to(out_dtype).reshape(*x.shape[:-1], out.shape[-1])
        if bias is not None:
            out = out + bias
        return out
