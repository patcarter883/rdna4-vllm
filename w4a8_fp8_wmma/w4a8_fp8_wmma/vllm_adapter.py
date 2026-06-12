"""vLLM MPLinearKernel adapter for the W4A8-FP8 WMMA HIP op (gfx1201).

Routes 4-bit (compressed-tensors / GPTQ-style uint4b8, optionally AWQ uint4 with
zero points) linear layers through torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm, which
expands int4 weights to fp8 e4m3 in-register and runs RDNA4's FP8 WMMA units.

Tensor contract (compressed_tensors_wNa16-style, verified against the gfx1151 MMQ
reference):
  w_q   weight_packed   [N, K//8]        int32  (8 uint4b8 per int32, low nibble first)
  w_s   weight_scale    [N, K//group]    fp16/bf16
  w_zp  zero_points      packed [N//8, K//group] int32, or None for symmetric
  g_idx                  not supported (no activation reordering)

register() in register.py inserts this at the front of _POSSIBLE_KERNELS[ROCM].
"""
import json
import os

import torch

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

_NEVER = 1 << 30  # sentinel crossover meaning "always use Triton"
_CROSSOVER_TABLE: dict | None = None


def _load_crossover_table() -> dict:
    """Load the AOT crossover cache (O(1) per-shape Triton<->FP8 thresholds).
    Path: $VLLM_ROCM_W4A8_FP8_WMMA_CACHE or crossover_cache.json next to this
    file. Keys are "N,K,group" -> crossover M (or null = never)."""
    global _CROSSOVER_TABLE
    if _CROSSOVER_TABLE is not None:
        return _CROSSOVER_TABLE
    path = os.environ.get(
        "VLLM_ROCM_W4A8_FP8_WMMA_CACHE",
        os.path.join(os.path.dirname(__file__), "crossover_cache.json"))
    table = {}
    try:
        with open(path) as f:
            table = json.load(f)
    except (OSError, ValueError):
        table = {}
    _CROSSOVER_TABLE = table
    return table


def _crossover_for(N: int, K: int, group: int) -> int:
    """O(1) crossover lookup. Env override forces a fixed M for all shapes.
    Unknown shapes -> _NEVER (always Triton), so the pathway stays >= Triton."""
    env = os.environ.get("VLLM_ROCM_W4A8_FP8_WMMA_MIN_M")
    if env:
        return int(env)
    v = _load_crossover_table().get(f"{N},{K},{group}")
    if v is None:
        return _NEVER
    return int(v)

SUPPORTED_QUANT_TYPES = [
    scalar_types.uint4b8,  # symmetric, implicit zero point = 8
    scalar_types.uint4,    # asymmetric, explicit per-group zero points (AWQ)
]


def _on_gfx12x() -> bool:
    try:
        from vllm.platforms.rocm import on_gfx12x
        return on_gfx12x()
    except Exception:
        return False


class RocmW4A8Fp8WmmaLinearKernel(MPLinearKernel):
    """FP8 WMMA (16x16x16) kernel for 4-bit weights on RDNA4 / gfx1201."""

    SUPPORTED_QUANT_TYPES = SUPPORTED_QUANT_TYPES

    @classmethod
    def get_min_capability(cls) -> int:
        return 0  # gated by on_gfx12x() in can_implement instead

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        ok, reason = cls._can_implement_inner(c)
        import logging
        logging.getLogger(__name__).warning(
            "[w4a8_fp8_wmma] can_implement -> %s (%s) | wt=%s act=%s g=%s zp=%s "
            "gidx=%s part=%s",
            ok, reason, c.weight_type, c.act_type, c.group_size,
            c.zero_points, c.has_g_idx, c.partition_weight_shape,
        )
        return ok, reason

    @classmethod
    def _can_implement_inner(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "ROCm only"
        if not _on_gfx12x():
            return False, "requires gfx12x (RDNA4)"
        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return False, f"unsupported weight_type {c.weight_type}"
        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, f"act_type must be fp16/bf16, got {c.act_type}"
        if c.has_g_idx:
            return False, "act reordering (g_idx) not supported"
        gs = c.group_size
        K = c.partition_weight_shape[0]
        if gs == -1:
            return False, "per-channel (group_size=-1) not supported"
        if gs % 16 != 0 or gs > 128:
            return False, f"group_size must be multiple of 16 and <=128, got {gs}"
        if K % gs != 0:
            return False, f"K={K} not divisible by group_size={gs}"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Normalize weights to the op's layout: w_q (N, K//8) int32, scales
        (N, K//group) fp16.

        Two source layouts are handled:
          - compressed-tensors: already (N, K//8) / (N, K//group).
          - AutoGPTQ: qweight (K//8, N) packed along K, scale (K//group, N).
            Repacked here (transpose + re-pack) so the op sees its native layout.
        """
        c = self.config
        K, N = c.partition_weight_shape  # (in, out)
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)
        wq = w_q.data
        ws = w_s.data

        if tuple(wq.shape) == (K // 8, N):
            # ---- AutoGPTQ layout -> our (N, K//8) ----
            dev = wq.device
            shifts = (torch.arange(8, device=dev, dtype=torch.int32) * 4).view(1, 8, 1)
            # (K//8, N) -> (K//8, 8, N) -> (K, N): nibble j of group is k = k8*8+j
            unpacked = ((wq.unsqueeze(1) >> shifts) & 0xF).reshape(K, N)
            w_kn = unpacked.t().contiguous()  # (N, K)
            repacked = torch.zeros((N, K // 8), dtype=torch.int32, device=dev)
            for j in range(8):
                repacked |= (w_kn[:, j::8] & 0xF) << (j * 4)
            self._transform_param(layer, self.w_q_name, lambda _p: repacked)
            ws = ws.t().contiguous()  # (K//group, N) -> (N, K//group)
        elif tuple(wq.shape) != (N, K // 8):
            raise RuntimeError(
                f"unexpected w_q shape {tuple(wq.shape)}; expected (N,K//8)="
                f"{(N, K // 8)} or GPTQ (K//8,N)={(K // 8, N)}")
        elif not wq.is_contiguous():
            self._transform_param(layer, self.w_q_name, lambda p: p.contiguous())

        w_s_fp16 = (
            ws.to(torch.float16).contiguous()
            if ws.dtype != torch.float16 else ws.contiguous()
        )
        layer._w4a8_fp8_w_s = w_s_fp16
        if w_zp is not None and not w_zp.is_contiguous():
            self._transform_param(layer, self.w_zp_name, lambda p: p.contiguous())

        # ---- Triton W4A16 fallback weights (used for small M where Triton wins).
        # triton_w4a16_gemm wants b_q [K, N//8], scales [K//group, N]. Build from
        # our (N, K//8) / (N, K//group) layout.
        wq_now = getattr(layer, self.w_q_name).data  # (N, K//8)
        dev = wq_now.device
        shifts = (torch.arange(8, device=dev, dtype=torch.int32) * 4).view(1, 8)
        unpacked = ((wq_now.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF).reshape(N, K)
        w_kn = unpacked.t().contiguous()  # (K, N)
        N8 = N // 8
        tri_bq = torch.zeros((K, N8), dtype=torch.int32, device=dev)
        for j in range(8):
            tri_bq |= (w_kn[:, j::8] & 0xF) << (j * 4)
        layer._w4a8_tri_bq = tri_bq.contiguous()           # (K, N//8)
        layer._w4a8_tri_s = w_s_fp16.t().contiguous()      # (K//group, N)

        # AWQ asymmetric zeros for the Triton fallback (decode path). Our op's
        # zero layout is (N//8, K//group) N-packed, standard nibble order;
        # triton_w4a16_gemm wants (K//group, N//8) with the same N-packing and
        # nibble order, i.e. a plain transpose. Symmetric (uint4b8) keeps None so
        # Triton uses the implicit zp_bias=8. Without this, decode (small M)
        # silently dropped the per-group zero points -> wrong AWQ outputs.
        if c.zero_points and w_zp is not None:
            zp_now = getattr(layer, self.w_zp_name).data  # (N//8, K//group)
            layer._w4a8_tri_zp = zp_now.t().contiguous()  # (K//group, N//8)
        else:
            layer._w4a8_tri_zp = None

        # Per-layer calibration: the ours-vs-Triton crossover depends on (N,K),
        # not just M (larger N lowers it, larger K raises it; some shapes never
        # cross). Measure THIS layer's crossover once so apply_weights engages the
        # FP8 kernel only where it is actually faster -> pathway is always >=
        # Triton. Cached per (N, K, group) since shapes repeat across layers.
        gs = c.group_size if c.group_size != -1 else K
        # O(1) lookup of the precomputed crossover (AOT Profile & Cache). NO
        # benchmarking at load time. Unknown shapes default to "never" (always
        # Triton) so the pathway is >= Triton even when untuned.
        layer._w4a8_min_m = _crossover_for(N, K, gs)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        N = c.partition_weight_shape[1]
        K = c.partition_weight_shape[0]
        out_shape = x.shape[:-1] + (N,)

        x_2d = x.reshape(-1, x.shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        M = x_2d.size(0)
        orig_dtype = x_2d.dtype

        if M < getattr(layer, "_w4a8_min_m", 1 << 30):
            # Small-M (decode / small prefill): Triton W4A16 is faster.
            from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
                triton_w4a16_gemm,
            )
            gs = c.group_size if c.group_size != -1 else K
            # AWQ: pass the real per-group zeros (HAS_ZP path); zp_bias unused.
            # Symmetric uint4b8: qzeros=None, Triton uses the implicit zp_bias=8.
            tri_zp = getattr(layer, "_w4a8_tri_zp", None)
            zp_bias = c.weight_type.bias if c.weight_type.has_bias() else 0
            out = triton_w4a16_gemm(
                a=x_2d, b_q=layer._w4a8_tri_bq, scales=layer._w4a8_tri_s,
                qzeros=tri_zp, group_size=gs, zp_bias=zp_bias)
        else:
            # Large-M (compute-bound prefill): our FP8-WMMA kernel wins.
            w_q, _w_s_native, w_zp, _ = self._get_weight_params(layer)
            w_s = layer._w4a8_fp8_w_s
            x16 = x_2d if x_2d.dtype == torch.float16 else x_2d.to(torch.float16)
            if c.zero_points and w_zp is not None:
                zp_in = w_zp
            else:
                zp_in = torch.empty(0, dtype=torch.int32, device=x.device)
            out = torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm(x16, w_q, w_s, zp_in, 5)
            if orig_dtype != torch.float16:
                out = out.to(orig_dtype)

        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        if bias is not None:
            out = out + bias
        return out.reshape(out_shape)
