# SPDX-License-Identifier: Apache-2.0
"""
RXF monolithic fused-MoE experts.

Standalone — no nvfp4 inheritance. IQ4-NL packed weights + a plain fp16 scale
per group of 32 (W4A16 default / W4A8 int8 opt-in). Nothing else.
"""
import torch
import triton.language as tl

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk,
)
from vllm.model_executor.layers.quantization.utils.rxf_kernels import (
    _RXF_MOE_DEFAULT_CONFIG,
    get_rxf_configs,
    invoke_rxf_moe_kernel,
    _pick_config,
)
from vllm.platforms import current_platform


class RXFExpertsMonolithic(mk.FusedMoEExpertsMonolithic):
    """RXF monolithic experts: IQ4-NL + plain fp16 group-32 scale."""

    def __init__(self, moe_config: FusedMoEConfig,
                 quant_config: FusedMoEQuantConfig,
                 format_tag: torch.Tensor | None = None,
                 fp16_slot: torch.Tensor | None = None,
                 w13_fp16: torch.Tensor | None = None,
                 w2_fp16: torch.Tensor | None = None):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        self.topk = moe_config.experts_per_token
        # Protected fp16 experts (all four set, or all None): static per-expert
        # format tag [E] uint8, compact slot map [E] int32 (-1 for quantized),
        # and the compact fp16 weight regions. Written once at load — contents
        # never change per step, so passing them through to every launch is
        # CUDA-graph-safe.
        self.format_tag = format_tag
        self.fp16_slot = fp16_slot
        self.w13_fp16 = w13_fp16
        self.w2_fp16 = w2_fp16

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return

    @staticmethod
    def _supports_current_device() -> bool:
        if not current_platform.is_rocm():
            return False
        from vllm.platforms.rocm import on_gfx1x
        return on_gfx1x()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.GELU)

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        return hidden_dim % 32 == 0

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return (not moe_parallel_config.use_all2all_kernels
                and not moe_parallel_config.enable_eplb)

    @staticmethod
    def _supports_routing_method(routing_method_type, weight_key, activation_key) -> bool:
        return routing_method_type in (
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
            RoutingMethodType.Unspecified,
        )

    @staticmethod
    def _supports_router_logits_dtype(router_logits_dtype, routing_method) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
        scoring_func: str = "softmax",
        renormalize: bool = True,
        custom_routing_function=None,
    ) -> torch.Tensor:
        assert a1q_scale is None
        assert num_expert_group is None and topk_group is None

        if custom_routing_function is not None:
            # Models with a custom router (e.g. ZAYA's MoD/EDA top-1) pre-compute
            # routing and pack (weights | ids) into `router_logits`; they cannot
            # be re-derived with fused_topk. Honor the function here (the
            # monolithic path otherwise bypasses FusedMoE.select_experts and
            # would feed the packed tensor straight into fused_topk → garbage
            # expert selection). Mirrors CustomRoutingRouter._compute_routing.
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.topk,
                renormalize=renormalize,
            )
            topk_weights = topk_weights.to(torch.float32)
            topk_ids = topk_ids.to(torch.int32)
        elif e_score_correction_bias is not None:
            from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
                fused_topk_bias,
            )
            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                e_score_correction_bias=e_score_correction_bias,
                topk=self.topk,
                renormalize=renormalize,
                scoring_func=scoring_func,
            )
        else:
            topk_weights, topk_ids, _ = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.topk,
                renormalize=renormalize,
            )
        if routed_scaling_factor is not None and routed_scaling_factor != 1.0:
            topk_weights = topk_weights * routed_scaling_factor

        M, K = hidden_states.shape
        _, two_I, _ = w1.shape
        N = two_I
        top_k_num = self.topk
        ct = (tl.bfloat16 if hidden_states.dtype == torch.bfloat16
              else tl.float16)

        K_w1 = w1.shape[2] * 2
        K_w2 = w2.shape[2] * 2
        tuned_w13 = get_rxf_configs(N, K_w1, dtype_tag="rxf_moe")
        tuned_w2 = get_rxf_configs(K, K_w2, dtype_tag="rxf_moe")
        cfg_w13 = _pick_config(M, tuned_w13, _RXF_MOE_DEFAULT_CONFIG)
        cfg_w2 = _pick_config(M, tuned_w2, _RXF_MOE_DEFAULT_CONFIG)

        bm_w13 = cfg_w13["BLOCK_SIZE_M"]
        bm_w2 = cfg_w2["BLOCK_SIZE_M"]

        ic1 = torch.empty((M, top_k_num, N), device=hidden_states.device,
                          dtype=hidden_states.dtype)
        ic2 = torch.empty((M * top_k_num, N // 2),
                          device=hidden_states.device,
                          dtype=hidden_states.dtype)
        ic3 = torch.empty((M, top_k_num, K), device=hidden_states.device,
                          dtype=hidden_states.dtype)

        # Independent block-m alignment per GEMM: gate-up and down each get the
        # token->block layout built for their own tuned BLOCK_SIZE_M. Routing is
        # identical between the two GEMMs, so a second moe_align_block_size on
        # the same topk_ids is valid. Avoids forcing the smaller-tile kernel
        # (usually down-proj at decode) up to the larger tile via max(), which
        # inflated padded-row WMMA work and ran it off its tuned launch params.
        sorted_w13, expert_w13, pad_w13 = moe_align_block_size(
            topk_ids, bm_w13,
            global_num_experts, expert_map,
        )
        if bm_w2 == bm_w13:
            sorted_w2, expert_w2, pad_w2 = sorted_w13, expert_w13, pad_w13
        else:
            sorted_w2, expert_w2, pad_w2 = moe_align_block_size(
                topk_ids, bm_w2,
                global_num_experts, expert_map,
            )

        invoke_rxf_moe_kernel(
            hidden_states, w1, ic1,
            self.quant_config.w1_scale,
            self.format_tag, self.fp16_slot, self.w13_fp16,
            topk_weights if apply_router_weight_on_input else None,
            sorted_w13, expert_w13, pad_w13,
            apply_router_weight_on_input, top_k_num,
            cfg_w13, ct,
        )

        apply_moe_activation(activation, ic2, ic1.view(-1, N))

        invoke_rxf_moe_kernel(
            ic2, w2, ic3,
            self.quant_config.w2_scale,
            self.format_tag, self.fp16_slot, self.w2_fp16,
            topk_weights if not apply_router_weight_on_input else None,
            sorted_w2, expert_w2, pad_w2,
            not apply_router_weight_on_input, 1,
            cfg_w2, ct,
        )

        output = torch.empty_like(hidden_states)
        torch.ops._moe_C.moe_sum(ic3, output)
        return output
