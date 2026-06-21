# SPDX-License-Identifier: Apache-2.0
#
# RXF (Rotated eXtra Fast): IQ4-NL 4-bit weights (two per byte) + a plain fp16
# scale per group of 32. Group = 32 = 2 chained WMMA K-steps, so the runtime can
# run a wide int8 blocked dot (W4A8) on the fast cores; a W4A16 fp16 path ships
# too as the correctness-safe default.
#
# Checkpoint layout (per quantized module):
#   weight_packed   uint8   [N, K//2]   two 4-bit NL indices packed per byte
#   weight_scale    float16 [N, K//32]  one fp16 scale per group of 32
#
# No exponent, no mantissa, no super-block scalar. 4.5 bits per weight.
# Dequant: NL_TABLE[nibble] * weight_scale
#
# act_dtype (config weights.act_dtype, env RXF_ACT_DTYPE overrides):
#   "fp16" (default) -> W4A16 dequant matmul, fp16 cores, no act-quant error.
#   "int8"           -> W4A8 per-token int8 activation x int8 code, fast cores.
# The checkpoint is identical either way; only the served path changes.
#
# Protected fp16 experts (optional): config.json quantization_config may carry
#   fp16_experts: {"<layer_idx>": [expert ids...]}
# Those experts are kept in the model dtype. Per protected expert the checkpoint
# stores dense tensors named <proj>.weight_fp16 — pre-rotated along K when
# rotation=hadamard32. A static per-expert format-tag + compact-slot table is
# built once at load; the MoE kernel resolves each expert tile's format, so the
# launch topology is one kernel with fixed tables and CUDA graphs stay intact.

import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
import re

from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)

logger = init_logger(__name__)

RXF_FORMAT = "rxf-pack-quantized"
GROUP = 32

_ROTATION_RE = re.compile(r"^(hadamard|givens)(\d+)$")


def _parse_rotation(rotation: str | None) -> tuple[str | None, int]:
    """Validate a rotation config tag and return (kind, span).

    Accepts None (un-rotated -> (None, GROUP)), 'hadamard{S}' (the fixed
    in-kernel FWHT), and 'givens{S}' (a learned model-wide R applied externally).
    S is a power of two and a multiple of GROUP (=32), so every size-32 scale
    group lands inside one rotated block and the cancellation/K=32 GEMM stay
    span-agnostic. Raises on any other value (the kernel would be silently wrong
    against a checkpoint it can't reproduce)."""
    if rotation is None:
        return None, GROUP
    m = _ROTATION_RE.match(rotation)
    if m is None:
        raise ValueError(
            f"rxf unknown rotation {rotation!r}; expected 'hadamard<S>', "
            f"'givens<S>' (S a power of two, multiple of {GROUP}) or null")
    kind, span = m.group(1), int(m.group(2))
    if span < GROUP or (span & (span - 1)) != 0 or span % GROUP != 0:
        raise ValueError(
            f"rxf rotation span must be a power of two and a multiple of "
            f"{GROUP}, got {span} (from {rotation!r})")
    return kind, span


def _parse_rotation_span(rotation: str | None) -> int:
    """Back-compat: just the span S of a rotation tag (see _parse_rotation)."""
    return _parse_rotation(rotation)[1]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class RXFConfig(QuantizationConfig):
    def __init__(self, group_size, ignore, full_config, codebook=None,
                 fp8_targets=None, fp8_block=None, rotation=None,
                 fp16_experts=None, act_dtype="fp16", rotation_matrix=None):
        super().__init__()
        self.group_size = group_size
        self.ignore = ignore
        self.full_config = full_config
        # Protected fp16 experts: {layer_idx: sorted [expert ids]}. Decided at
        # quant time, static for the life of the model. Empty for fully
        # quantized checkpoints.
        self.fp16_experts = {
            int(k): sorted(int(e) for e in v)
            for k, v in (fp16_experts or {}).items()
        }
        # Optional custom 16-point dequant grid from config.json. None -> IQ4-NL.
        self.codebook = codebook
        # Optional per-span Hadamard rotation ("hadamard{S}" — e.g. "hadamard32",
        # "hadamard512" — or None). MANDATORY gate: weights are stored
        # pre-rotated over span S, so the kernel MUST apply the matching
        # activation rotation (same span) or the matmul is silently wrong. A
        # plain (un-rotated) checkpoint leaves this None and the kernel stays
        # identity. rotation_span carries S (32 by default / when un-rotated).
        self.rotation = rotation
        self.rotation_kind, self.rotation_span = _parse_rotation(rotation)
        # The model-wide learned Givens R [S,S] for rotation=givens{S} (a nested
        # list in config.json), None otherwise. Built into a device tensor at
        # process_weights_after_loading and applied externally before the GEMM.
        self.rotation_matrix = rotation_matrix
        if self.rotation_kind == "givens" and rotation_matrix is None:
            raise ValueError(
                "rxf rotation=givens requires weights.rotation_matrix in "
                "config.json (the learned R); none found.")
        # Served activation path: "fp16" (W4A16, safe default) or "int8" (W4A8
        # fast path). env RXF_ACT_DTYPE wins over the config value.
        self.act_dtype = act_dtype
        # Optional block-wise FP8 group (e.g. attention on Step-3.7). Empty for
        # plain rxf checkpoints, so models without it are unaffected.
        self.fp8_targets = list(fp8_targets or [])
        self.fp8_block = list(fp8_block) if fp8_block else None

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "rxf"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def _is_rxf_cfg(cls, hf) -> bool:
        if not hf:
            return False
        if hf.get("format") == RXF_FORMAT:
            return True
        if hf.get("quant_method") == "rxf":
            return True
        for g in (hf.get("config_groups") or {}).values():
            if isinstance(g, dict) and g.get("format") == RXF_FORMAT:
                return True
        return False

    @classmethod
    def override_quantization_method(
        cls, hf, user, hf_config=None
    ) -> QuantizationMethods | None:
        return cls.get_name() if cls._is_rxf_cfg(hf) else None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RXFConfig":
        weights = {}
        for g in (config.get("config_groups") or {}).values():
            if isinstance(g, dict) and isinstance(g.get("weights"), dict):
                weights = g["weights"]
                break
        codebook = weights.get("codebook") or config.get("codebook")
        if codebook is not None:
            codebook = [float(v) for v in codebook]
            if len(codebook) != 16:
                raise ValueError(
                    f"rxf codebook must have 16 entries, got {len(codebook)}")
        # Per-span Hadamard rotation gate (mandatory if the checkpoint was
        # quantized with --rotate; absent/None on plain checkpoints). Accepts
        # 'hadamard{S}' for any power-of-two span S that is a multiple of 32 (a
        # wider RXF checkpoint), validated by _parse_rotation_span; rejects
        # anything else.
        rotation = weights.get("rotation") or config.get("rotation")
        rot_kind, _ = _parse_rotation(rotation)   # validate (raises on unknown)
        # Learned model-wide R for rotation=givens{S} (a nested S×S list).
        rotation_matrix = (weights.get("rotation_matrix")
                           or config.get("rotation_matrix"))
        # Activation path: env override wins, then config, then safe default.
        act_dtype = (os.environ.get("RXF_ACT_DTYPE")
                     or weights.get("act_dtype")
                     or config.get("act_dtype") or "fp16")
        if act_dtype not in ("fp16", "int8"):
            raise ValueError(
                f"rxf act_dtype {act_dtype!r}; expected 'fp16' or 'int8'")
        raw_ignore = list(config.get("ignore", []))
        ignore = []
        for entry in raw_ignore:
            ignore.append(entry)
            if entry.startswith("model.language_model."):
                ignore.append(entry[len("model.language_model."):])
            if entry.startswith("model."):
                ignore.append(entry[len("model."):])
        # Optional block-wise FP8 group: e.g. attention quantized to fp8 while
        # MoE/MLP stay rxf. Backward compatible -- absent on plain rxf
        # checkpoints, leaving fp8_targets empty.
        fp8_targets, fp8_block = [], None
        for g in (config.get("config_groups") or {}).values():
            if not isinstance(g, dict):
                continue
            w = g.get("weights") or {}
            if (w.get("type") == "float" and w.get("num_bits") == 8
                    and w.get("strategy") == "block"):
                fp8_targets = list(g.get("targets") or [])
                fp8_block = list(w.get("block_structure") or [128, 128])
                break
        return cls(
            group_size=int(weights.get("group_size",
                                       config.get("group_size", GROUP))),
            ignore=ignore,
            full_config=config,
            codebook=codebook,
            fp8_targets=fp8_targets,
            fp8_block=fp8_block,
            rotation=rotation,
            fp16_experts=config.get("fp16_experts"),
            act_dtype=act_dtype,
            rotation_matrix=rotation_matrix,
        )

    def _is_ignored(self, prefix: str) -> bool:
        for entry in self.ignore:
            if entry.startswith("re:"):
                if re.search(entry[3:], prefix):
                    return True
            elif entry in prefix:
                return True
        return False

    def givens_R(self, device) -> "torch.Tensor | None":
        """Build (and cache per device) the model-wide learned Givens R [S,S] as
        an fp32 tensor, or None when the checkpoint is not Givens-rotated. fp32 to
        keep R·Rᵀ=I tight for cancellation precision; tiny (S² floats)."""
        if self.rotation_kind != "givens":
            return None
        cache = self.__dict__.setdefault("_R_cache", {})
        key = str(device)
        if key not in cache:
            R = torch.tensor(self.rotation_matrix, dtype=torch.float32,
                             device=device)
            S = self.rotation_span
            if R.shape != (S, S):
                raise ValueError(
                    f"rxf rotation_matrix is {tuple(R.shape)}, expected "
                    f"({S},{S}) for rotation=givens{S}")
            ortho = (R @ R.t()
                     - torch.eye(S, device=device)).abs().max().item()
            if ortho > 1e-3:
                raise ValueError(
                    f"rxf learned R is not orthonormal (R·Rᵀ−I max={ortho:.2e}); "
                    "the rotation would not cancel.")
            cache[key] = R
        return cache[key]

    def _matches_fp8(self, prefix: str) -> bool:
        for t in self.fp8_targets:
            if t.startswith("re:"):
                if re.search(t[3:], prefix):
                    return True
            elif t in prefix:
                return True
        return False

    def get_quant_method(self, layer, prefix) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            # Genuinely-dense GDN (linear_attn) sub-modules that are NEVER a
            # quantizable GEMM (depthwise conv1d, the tiny fused in_proj_ba) are
            # checked BEFORE the config so a checkpoint that quantizes the OTHER
            # linear_attn projections can't accidentally pull them in.
            if "linear_attn.conv1d" in prefix or "linear_attn.in_proj_ba" in prefix:
                return UnquantizedLinearMethod()
            if self._is_ignored(prefix):
                return UnquantizedLinearMethod()
            # Block-wise FP8 layers (e.g. attention) when the checkpoint declares
            # an fp8 group. No-op for plain rxf checkpoints (fp8_targets empty).
            if self.fp8_targets and self._matches_fp8(prefix):
                return Fp8BlockLinearMethod(self.fp8_block or [128, 128])
            if "visual" in prefix or "vision" in prefix:
                return UnquantizedLinearMethod()
            if "mlp.gate." in prefix and "shared_expert" not in prefix:
                return UnquantizedLinearMethod()
            if "shared_expert_gate" in prefix:
                return UnquantizedLinearMethod()
            if "mtp" in prefix:
                return UnquantizedLinearMethod()
            return RXFLinearMethod(self)
        if isinstance(layer, FusedMoE):
            if self._is_ignored(prefix):
                return None
            if "mtp" in prefix:
                return None
            # Protected fp16 experts for THIS layer, keyed by the decoder
            # layer index parsed from the module prefix.
            protected: list[int] = []
            if self.fp16_experts:
                m = re.search(r"\.layers\.(\d+)\.", f".{prefix}.")
                if m:
                    protected = self.fp16_experts.get(int(m.group(1)), [])
            return RXFFusedMoEMethod(self, layer.moe_config, protected)
        return None


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------
class RXFLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: RXFConfig):
        self.quant_config = quant_config
        self.group_size = quant_config.group_size

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra):
        del input_size, output_size
        wl = extra.get("weight_loader")
        N = sum(output_partition_sizes)
        K = input_size_per_partition
        gs = self.group_size
        if K % gs != 0:
            raise ValueError(
                f"in_features ({K}) must be a multiple of {gs}")
        layer.logical_widths = output_partition_sizes

        weight = ModelWeightParameter(
            data=torch.empty(N, K // 2, dtype=torch.uint8),
            input_dim=1, output_dim=0, weight_loader=wl,
        )
        layer.register_parameter("weight_packed", weight)

        # One plain fp16 scale per group of 32 along K (sign carries polarity).
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(N, K // gs, dtype=torch.float16),
            input_dim=1, output_dim=0, weight_loader=wl,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer) -> None:
        from vllm.model_executor.layers.quantization.utils.rxf_kernels import (
            set_act_dtype,
            set_codebook,
            set_rotation,
        )
        set_codebook(self.quant_config.codebook)
        # Single config-level gates -> both linear and MoE read the same
        # self.quant_config values, so the two paths can never diverge. For
        # rotation=givens{S} set_rotation turns the IN-KERNEL FWHT off; the
        # learned R is applied externally in apply() (built/cached here).
        set_rotation(self.quant_config.rotation)
        set_act_dtype(self.quant_config.act_dtype)
        layer.weight_packed = torch.nn.Parameter(
            layer.weight_packed.data.contiguous(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data.contiguous(), requires_grad=False)
        layer.rxf_givens_R = self.quant_config.givens_R(
            layer.weight_packed.device)

    def apply(self, layer, x, bias=None):
        from vllm.model_executor.layers.quantization.utils.rxf_kernels import (
            invoke_rxf_givens_rotate,
            invoke_rxf_linear_kernel,
        )
        # Learned Givens rotation (rotation=givens{S}): rotate the activation by
        # the model-wide R before the GEMM. The weights were stored R-rotated
        # offline, so (R x)·(R w) = x·w cancels; the in-kernel FWHT is off.
        R = getattr(layer, "rxf_givens_R", None)
        if R is not None:
            x = invoke_rxf_givens_rotate(x, R)
        return invoke_rxf_linear_kernel(
            x,
            layer.weight_packed,
            layer.weight_scale,
            bias,
        )


# ---------------------------------------------------------------------------
# Block-wise FP8 linear (e.g. attention), DeepSeek-style
# ---------------------------------------------------------------------------
class Fp8BlockLinearMethod(LinearMethodBase):
    """Block-wise FP8 (e4m3) linear, used for layers a checkpoint marks with an
    fp8 block group (Step-3.7 attention: fused qkv_proj + o_proj).

    Weights stay FP8 in VRAM:
        weight           float8_e4m3fn  [N, K]
        weight_scale_inv float32        [N//blk0, K//blk1]
    Activations are quantized per-token-group at runtime and the GEMM runs via
    vLLM's block-scaled fp8 kernel (same path as DeepSeek-V3). Delegates weight
    setup / GEMM to the stock fp8 linear kernel so the math is shared, not
    re-implemented.
    """

    def __init__(self, block_size):
        from vllm.config import get_current_vllm_config
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            GroupShape,
            create_fp8_quant_key,
        )
        self.weight_block_size = [int(block_size[0]), int(block_size[1])]
        self.out_dtype = torch.get_default_dtype()
        self.input_dtype = get_current_vllm_config().model_config.dtype
        self.act_q_group_shape = GroupShape(1, self.weight_block_size[0])
        self.weight_quant_key = create_fp8_quant_key(
            static=True, group_shape=GroupShape(*self.weight_block_size))
        self.activation_quant_key = create_fp8_quant_key(
            static=False, group_shape=self.act_q_group_shape)

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra):
        from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            create_fp8_scale_parameter,
            create_fp8_weight_parameter,
            validate_fp8_block_shape,
        )
        from vllm.model_executor.parameter import BlockQuantScaleParameter

        wl = extra.get("weight_loader")
        out = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.orig_dtype = params_dtype
        layer.weight_block_size = self.weight_block_size
        validate_fp8_block_shape(
            layer, input_size, output_size, input_size_per_partition,
            output_partition_sizes, self.weight_block_size)

        weight = create_fp8_weight_parameter(
            out, input_size_per_partition, wl)
        layer.register_parameter("weight", weight)

        # name 'weight_scale_inv' matches the checkpoint (DeepSeek convention)
        scale = create_fp8_scale_parameter(
            BlockQuantScaleParameter, output_partition_sizes,
            input_size_per_partition, self.weight_block_size, wl)
        layer.register_parameter("weight_scale_inv", scale)

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name="Fp8BlockLinearMethod",
        )

    def process_weights_after_loading(self, layer) -> None:
        layer.input_scale = None
        self.fp8_linear.process_weights_after_loading(layer)

    def apply(self, layer, x, bias=None):
        return self.fp8_linear.apply_weights(layer, x, bias)


# ---------------------------------------------------------------------------
# Fused MoE
# ---------------------------------------------------------------------------
class RXFFusedMoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: "RXFConfig", moe,
                 fp16_expert_ids: list[int] | None = None):
        super().__init__(moe)
        self.quant_config = quant_config
        self.group_size = quant_config.group_size
        # Global expert ids kept in fp16 for this layer (sorted; slot i in the
        # compact fp16 region = i-th id). Assumes no expert parallelism, which
        # the rxf experts backend already requires.
        self.fp16_expert_ids = list(fp16_expert_ids or [])

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype,
                       **extra):
        from vllm.model_executor.layers.fused_moe.layer import (
            FusedMoeWeightScaleSupported,
        )
        from vllm.model_executor.utils import set_weight_attrs

        layer.num_experts = num_experts
        I = intermediate_size_per_partition
        H = hidden_size
        n13 = 2 if self.moe.is_act_and_mul else 1
        gs = self.group_size

        assert "weight_loader" in extra

        def reg(name, *shape, dtype, qm=None):
            p = torch.nn.Parameter(torch.empty(*shape, dtype=dtype),
                                   requires_grad=False)
            layer.register_parameter(name, p)
            ea = dict(extra)
            if qm is not None:
                ea["quant_method"] = qm
            set_weight_attrs(p, ea)

        reg("w13_weight_packed", num_experts, n13 * I, H // 2,
            dtype=torch.uint8)
        reg("w2_weight_packed", num_experts, H, I // 2,
            dtype=torch.uint8)

        # One plain fp16 scale per group of 32 along K (sign carries polarity).
        reg("w13_weight_scale", num_experts, n13 * I, H // gs,
            dtype=torch.float16,
            qm=FusedMoeWeightScaleSupported.GROUP.value)
        reg("w2_weight_scale", num_experts, H, I // gs,
            dtype=torch.float16,
            qm=FusedMoeWeightScaleSupported.GROUP.value)

        # Protected fp16 experts: compact dense regions, one slot per protected
        # expert. Checkpoint tensors are named <proj>.weight_fp16 and exist only
        # for protected experts, so they need a loader that maps the global
        # expert id to its compact slot and applies the same w1/w3-stacking +
        # TP-narrow the stock FusedMoE loader would.
        n_fp16 = len(self.fp16_expert_ids)
        if n_fp16 > 0:
            slot_of = {e: i for i, e in enumerate(self.fp16_expert_ids)}
            tp_rank = self.moe.tp_rank

            def fp16_loader(param, loaded_weight, weight_name, shard_id,
                            expert_id, return_success=False):
                slot = slot_of.get(expert_id)
                if slot is None:
                    raise ValueError(
                        f"{weight_name}: expert {expert_id} carries a "
                        f"weight_fp16 tensor but is not in fp16_experts")
                expert_data = param.data[slot]
                if shard_id == "w2":
                    # [H, I]: narrow the contraction dim to this TP shard.
                    shard = expert_data.shape[1]
                    expert_data.copy_(
                        loaded_weight.narrow(1, shard * tp_rank, shard))
                elif shard_id in ("w1", "w3"):
                    # [n13*I, H]: w1 stacks above w3; narrow I to this shard.
                    shard = expert_data.shape[0] // n13
                    start = 0 if shard_id == "w1" else shard
                    expert_data.narrow(0, start, shard).copy_(
                        loaded_weight.narrow(0, shard * tp_rank, shard))
                else:
                    raise ValueError(f"unknown shard_id {shard_id!r}")
                return True if return_success else None

            for name, shape in (("w13_weight_fp16", (n_fp16, n13 * I, H)),
                                ("w2_weight_fp16", (n_fp16, H, I))):
                p = torch.nn.Parameter(
                    torch.empty(*shape, dtype=params_dtype),
                    requires_grad=False)
                layer.register_parameter(name, p)
                ea = dict(extra)
                ea["weight_loader"] = fp16_loader
                set_weight_attrs(p, ea)

    def process_weights_after_loading(self, layer) -> None:
        from vllm.model_executor.layers.fused_moe.experts.rxf_moe import (
            RXFExpertsMonolithic,
        )
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEQuantConfig,
        )
        from vllm.model_executor.layers.quantization.utils.rxf_kernels import (
            set_act_dtype,
            set_codebook,
            set_givens_rotation,
            set_rotation,
        )
        set_codebook(self.quant_config.codebook)
        set_rotation(self.quant_config.rotation)
        set_act_dtype(self.quant_config.act_dtype)
        # Learned model-wide Givens R (rotation=givens{S}): the in-kernel FWHT is
        # off; invoke_rxf_moe_kernel applies this R to each GEMM's activation.
        set_givens_rotation(
            self.quant_config.givens_R(layer.w13_weight_packed.device))

        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data.contiguous(), requires_grad=False)
        del layer.w13_weight_packed
        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data.contiguous(), requires_grad=False)
        del layer.w2_weight_packed

        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.data.contiguous(), requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.data.contiguous(), requires_grad=False)

        # Protected fp16 experts: build the static format-tag + compact-slot
        # tables once, here, before any CUDA graph capture. Read-only from this
        # point on — the graph bakes in their addresses, the kernel reads them.
        if self.fp16_expert_ids:
            E = layer.w13_weight.shape[0]
            dev = layer.w13_weight.device
            tags = torch.zeros(E, dtype=torch.uint8)
            slots = torch.full((E,), -1, dtype=torch.int32)
            for i, e in enumerate(self.fp16_expert_ids):
                tags[e] = 1
                slots[e] = i
            layer.rxf_format_tag = tags.to(dev)
            layer.rxf_fp16_slot = slots.to(dev)
            layer.w13_weight_fp16 = torch.nn.Parameter(
                layer.w13_weight_fp16.data.contiguous(), requires_grad=False)
            layer.w2_weight_fp16 = torch.nn.Parameter(
                layer.w2_weight_fp16.data.contiguous(), requires_grad=False)
        else:
            layer.rxf_format_tag = None
            layer.rxf_fp16_slot = None

        quant_config = FusedMoEQuantConfig.make(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )
        self._experts = RXFExpertsMonolithic(
            moe_config=self.moe,
            quant_config=quant_config,
            format_tag=layer.rxf_format_tag,
            fp16_slot=layer.rxf_fp16_slot,
            w13_fp16=(layer.w13_weight_fp16
                      if self.fp16_expert_ids else None),
            w2_fp16=(layer.w2_weight_fp16
                     if self.fp16_expert_ids else None),
        )
        self._experts.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(self, layer):
        return None

    def maybe_make_prepare_finalize(self, routing_tables=None):
        raise ValueError(
            f"{self.__class__.__name__} uses the modular kernel "
            "initialization logic. This function should not be called.")

    @property
    def is_monolithic(self) -> bool:
        return True

    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        assert self._experts is not None
        return self._experts.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            router_logits=router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            a1q_scale=None,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
            scoring_func=layer.scoring_func,
            renormalize=layer.renormalize,
            # Monolithic kernels handle routing internally and bypass
            # FusedMoE.select_experts, so a model's custom router would be
            # silently ignored. Forward it so models that pre-pack routing into
            # router_logits (ZAYA MoD/EDA) select the right experts. (Folded in
            # from main 53cf7de; no-op for Laguna/Qwen which set no custom fn.)
            custom_routing_function=getattr(
                layer, "custom_routing_function", None),
        )

    def apply(self, layer, x, topk_weights, topk_ids,
              shared_experts, shared_experts_input):
        import triton.language as tl
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )
        from vllm.model_executor.layers.quantization.utils.rxf_kernels import (
            _RXF_MOE_DEFAULT_CONFIG,
            get_rxf_configs,
            invoke_rxf_moe_kernel,
            _pick_config,
        )

        M, K = x.shape
        w1 = layer.w13_weight
        w2 = layer.w2_weight
        _, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        ct = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16

        K_w1 = w1.shape[2] * 2
        K_w2 = w2.shape[2] * 2
        cfg_w13 = _pick_config(M, get_rxf_configs(N, K_w1, "rxf_moe"),
                               _RXF_MOE_DEFAULT_CONFIG)
        cfg_w2 = _pick_config(M, get_rxf_configs(K, K_w2, "rxf_moe"),
                              _RXF_MOE_DEFAULT_CONFIG)
        bm_w13 = cfg_w13["BLOCK_SIZE_M"]
        bm_w2 = cfg_w2["BLOCK_SIZE_M"]

        # Independent block-m alignment per GEMM: each kernel gets the layout
        # built for its own tuned BLOCK_SIZE_M instead of both forced to max().
        sorted_w13, expert_w13, pad_w13 = moe_align_block_size(
            topk_ids, bm_w13, layer.global_num_experts, layer.expert_map)
        if bm_w2 == bm_w13:
            sorted_w2, expert_w2, pad_w2 = sorted_w13, expert_w13, pad_w13
        else:
            sorted_w2, expert_w2, pad_w2 = moe_align_block_size(
                topk_ids, bm_w2, layer.global_num_experts, layer.expert_map)

        ic1 = torch.empty((M, top_k, N), device=x.device, dtype=x.dtype)
        ic2 = torch.empty((M * top_k, N // 2), device=x.device, dtype=x.dtype)
        ic3 = torch.empty((M, top_k, K), device=x.device, dtype=x.dtype)

        ft = layer.rxf_format_tag
        sl = layer.rxf_fp16_slot
        w13_fp16 = layer.w13_weight_fp16 if ft is not None else None
        w2_fp16 = layer.w2_weight_fp16 if ft is not None else None

        invoke_rxf_moe_kernel(
            x, w1, ic1, layer.w13_weight_scale,
            ft, sl, w13_fp16,
            topk_weights if layer.apply_router_weight_on_input else None,
            sorted_w13, expert_w13, pad_w13,
            layer.apply_router_weight_on_input, top_k, cfg_w13, ct)

        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        if layer.activation == MoEActivation.SILU:
            torch.ops._C.silu_and_mul(ic2, ic1.view(-1, N))
        else:
            torch.ops._C.gelu_and_mul(ic2, ic1.view(-1, N))

        invoke_rxf_moe_kernel(
            ic2, w2, ic3, layer.w2_weight_scale,
            ft, sl, w2_fp16,
            topk_weights if not layer.apply_router_weight_on_input else None,
            sorted_w2, expert_w2, pad_w2,
            not layer.apply_router_weight_on_input, 1, cfg_w2, ct)

        output = torch.empty_like(x)
        torch.ops._moe_C.moe_sum(ic3, output)
        return output
