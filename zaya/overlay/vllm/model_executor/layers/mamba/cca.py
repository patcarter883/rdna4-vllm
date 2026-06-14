# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.cca_attn import CCAAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


# Optional fused HIP conv+state kernel (cca_hip/). The graph-broken vllm::cca op
# runs its decode forward eagerly; a rocprof attributed a big share of ~1.4M tiny
# pointwise launches to it. This kernel collapses the decode conv + state
# roll/scatter into one launch. Opt in with ZAYA_CCA_HIP=1; falls back to the
# eager path if the extension isn't built.
_CCA_HIP = os.environ.get("ZAYA_CCA_HIP", "0") == "1"
_cca_hip = None
_cca_hip_tried = False


def _get_cca_hip():
    """Return the cca_hip op module (with cca_decode_qk / conv_state_decode), or
    None if ZAYA_CCA_HIP is off or the extension isn't built."""
    global _cca_hip, _cca_hip_tried
    if _cca_hip_tried or not _CCA_HIP:
        return _cca_hip
    _cca_hip_tried = True
    try:
        from vllm.model_executor.layers.mamba.cca_hip import cca_op

        _cca_hip = cca_op
    except Exception as e:  # pragma: no cover - depends on a built .so
        import logging

        logging.getLogger(__name__).warning(
            "ZAYA_CCA_HIP=1 but the cca_hip extension is unavailable (%s); "
            "using the eager CCA decode path.",
            e,
        )
        _cca_hip = None
    return _cca_hip


@CustomOp.register("cca")
class CCA(MambaBase, CustomOp):
    def __init__(
        self,
        config,
        cca_num_k_heads: int = 2,
        cca_num_q_heads: int = 8,
        hidden_size: int | None = None,
        head_dim: int = 128,
        cca_time0: int = 2,
        cca_time1: int = 2,
        layer_number: int = 0,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.layer_number = layer_number
        self.prefix = prefix

        # Use the model's true hidden size unless explicitly overridden.
        # (In Megatron this is the lane's hidden_size_in.)
        self.hidden_size = int(hidden_size or config.hidden_size)

        self.cca_time0 = cca_time0
        self.cca_time1 = cca_time1
        self.padding0 = cca_time0 - 1
        self.padding1 = cca_time1 - 1
        self.total_padding = self.padding0 + self.padding1

        self.num_k_heads = int(cca_num_k_heads)
        self.num_q_heads = int(cca_num_q_heads)

        # Geometry
        self.head_dim = int(head_dim)
        self.latent_k_dim = self.num_k_heads * self.head_dim
        self.latent_q_dim = self.num_q_heads * self.head_dim
        self.sqrt_head_dim = np.sqrt(self.head_dim)
        self.gqa_groups = self.num_q_heads // self.num_k_heads
        assert self.num_q_heads % self.num_k_heads == 0, (
            "q_heads must be a multiple of k_heads"
        )
        assert (self.latent_k_dim + self.latent_q_dim) == (
            self.num_k_heads + self.num_q_heads
        ) * self.head_dim

        # Projections
        self.linear_q = ReplicatedLinear(
            self.hidden_size,
            self.latent_q_dim,
            bias=self.config.attention_bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_q",
        )
        self.linear_k = ReplicatedLinear(
            self.hidden_size,
            self.latent_k_dim,
            bias=self.config.attention_bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_k",
        )
        self.val_proj1 = ReplicatedLinear(
            self.hidden_size,
            self.latent_k_dim // 2,
            bias=self.config.attention_bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.val_proj1",
        )
        self.val_proj2 = ReplicatedLinear(
            self.hidden_size,
            self.latent_k_dim // 2,
            bias=self.config.attention_bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.val_proj2",
        )

        # Depthwise + grouped conv along sequence (exactly like Megatron)
        in_out_ch = self.latent_k_dim + self.latent_q_dim
        self.in_out_ch = in_out_ch
        self.conv_qk = nn.Sequential(
            nn.Conv1d(
                in_channels=in_out_ch,
                out_channels=in_out_ch,
                kernel_size=self.cca_time0,
                groups=in_out_ch,
                padding=0,
                stride=1,
            ),
            nn.Conv1d(
                in_channels=in_out_ch,
                out_channels=in_out_ch,
                kernel_size=self.cca_time1,
                groups=(self.num_k_heads + self.num_q_heads),
                padding=0,
                stride=1,
            ),
        )

        # Per-k head temperature (Megatron: shape [num_k_heads])
        self.temp = nn.Parameter(torch.zeros(self.num_k_heads))

        # Speculative decoding: the verify step feeds (1 + num_spec) tokens per
        # sequence in the decode region. The conv-state block is widened by
        # num_spec columns to hold the per-spec-position rollback window (see
        # get_cca_conv_copy_spec); prev_hs uses the (1 + num_spec) block slots.
        self.num_spec = get_current_vllm_config().num_speculative_tokens

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        return

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        torch.ops.vllm.cca(
            hidden_states,
            output,
            self.prefix,
        )

    def _rms_normalize_qk(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Equivalent to RMSNorm with unit weights and eps=1e-12/head_dim.
        # Normalize one tensor at a time in fp32 to reduce peak memory versus
        # the custom rms_norm op, which materializes an additional fp32 output.
        eps = 1e-12
        sqrt_head_dim = float(self.sqrt_head_dim)

        query_fp32 = query.to(torch.float32)
        q_norm = torch.linalg.vector_norm(query_fp32, ord=2, dim=-1, keepdim=True)
        query_fp32.mul_(torch.rsqrt(q_norm * q_norm + eps))
        query_fp32.mul_(sqrt_head_dim)
        query.copy_(query_fp32)

        key_fp32 = key.to(torch.float32)
        k_norm = torch.linalg.vector_norm(key_fp32, ord=2, dim=-1, keepdim=True)
        key_fp32.mul_(torch.rsqrt(k_norm * k_norm + eps))
        key_fp32.mul_(sqrt_head_dim)
        temp = self.temp.to(torch.float32).view(1, 1, self.num_k_heads, 1)
        if self.config.clamp_temp:
            temp = torch.exp(torch.clamp(temp, 1e-7, 2.0))
        key_fp32.mul_(temp)
        key.copy_(key_fp32)
        return query, key

    def _add_grouped_qk_means_inplace(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_pre: torch.Tensor,
        key_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_k_heads = key_base.shape[2]
        key_base_fp32 = key_base.float()
        query_pre_grouped = query_pre.view(
            *query_pre.shape[:2], num_k_heads, self.gqa_groups, query_pre.shape[-1]
        )
        query_out_grouped = query.view_as(query_pre_grouped)
        query_out_grouped.add_(query_pre_grouped, alpha=0.5)
        query_out_grouped.add_(key_base_fp32.unsqueeze(-2), alpha=0.5)

        query_pre_mean = torch.mean(query_pre_grouped, dim=-2, dtype=torch.float32)
        key.add_(query_pre_mean, alpha=0.5)
        key.add_(key_base_fp32, alpha=0.5)
        return query, key

    def _conv_qk_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Manual conv_qk for decode-sized inputs.

        Decode uses tiny sequence windows (currently total_padding + 1), so the
        generic conv path can spend a disproportionate amount of time on layout
        transforms and kernel setup. This manual implementation preserves the
        two-stage depthwise+grouped conv math while operating directly on the
        compact decode tensor.

        Input:  [N, C, S]
        Output: [N, C, S_out]
        """
        # Stage 1: depthwise conv over sequence.
        w0 = self.conv_qk[0].weight.squeeze(1)  # [C, K0]
        b0 = self.conv_qk[0].bias  # [C] or None

        x = x.to(w0.dtype)
        k0 = w0.shape[1]
        x_windows = x.unfold(-1, k0, 1)  # [N, C, L_mid, K0]
        mid = (x_windows * w0[:, None, :]).sum(dim=-1)  # [N, C, L_mid]
        if b0 is not None:
            mid = mid + b0[None, :, None]

        # Stage 2: grouped conv over the depthwise output.
        w1 = self.conv_qk[1].weight  # [C, D, K1]
        b1 = self.conv_qk[1].bias  # [C] or None
        g = self.num_k_heads + self.num_q_heads
        d = self.head_dim
        k1 = w1.shape[2]
        mid_windows = mid.view(mid.shape[0], g, d, mid.shape[-1]).unfold(-1, k1, 1)
        w1_grouped = w1.view(g, d, d, k1)
        out = torch.einsum("godk,sgdtk->sgot", w1_grouped, mid_windows)
        if b1 is not None:
            out = out + b1.view(1, g, d, 1)
        return out.reshape(x.shape[0], g * d, out.shape[-1])

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        forward_context = get_forward_context()

        attn_metadata: AttentionMetadata = forward_context.attn_metadata
        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)
            attn_metadata = attn_metadata[self.prefix]
            assert isinstance(attn_metadata, CCAAttentionMetadata)
            conv_states = self.kv_cache[0]
            prev_hs = self.kv_cache[1]
            state_indices_tensor_p = attn_metadata.state_indices_tensor_p
            # In 'all' mamba_cache_mode the prefill state-index tensor is the full
            # 2D block table [num_prefills, max_blocks]; this non-'all' CCA prefill
            # path expects a single base slot per sequence, so collapse to column 0.
            # No-op in align/none mode, where state_indices_tensor_p is already 1D.
            if (
                state_indices_tensor_p is not None
                and state_indices_tensor_p.dim() > 1
            ):
                state_indices_tensor_p = state_indices_tensor_p[:, 0]
            state_indices_tensor_d = attn_metadata.state_indices_tensor_d
            # Under spec decode the decode region carries (1 + num_spec) tokens
            # per sequence and state_indices_tensor_d is 2D
            # [num_decode_seqs, 1 + num_spec] (one block slot per spec position).
            # Keep the 2D form for the verify path; the single-token decode path
            # uses the base (column-0) slot.
            state_indices_tensor_d_2d = state_indices_tensor_d
            if state_indices_tensor_d is not None and state_indices_tensor_d.dim() > 1:
                state_indices_tensor_d = state_indices_tensor_d[:, 0]
            has_initial_states_p = attn_metadata.has_initial_states_p
            query_start_loc_p = attn_metadata.query_start_loc_p
            query_start_loc_d = attn_metadata.query_start_loc_d
            num_decode_seqs = attn_metadata.num_decodes
            # Spec-decode cross-step rollback pointers (populated under cudagraph;
            # None in eager). block_idx_last_computed_token = the slot holding the
            # committed state from the previous step; block_idx_last_scheduled_token
            # = the slot to fill last this step. Mirrors mamba2 'none'-mode.
            blk_computed_d = getattr(
                attn_metadata, "block_idx_last_computed_token", None
            )
            blk_scheduled_d = getattr(
                attn_metadata, "block_idx_last_scheduled_token", None
            )
            # 'all' mamba_cache_mode rollback metadata (None in align/none). When
            # present, _decode_verify_spec uses the full-block-table layout for
            # bit-lossless partial-acceptance rollback (DRAFT — GPU-unvalidated).
            blk_scheduled_prev_d = getattr(
                attn_metadata, "block_idx_last_scheduled_token_prev_step", None
            )
            num_accepted_d = getattr(attn_metadata, "num_accepted_tokens", None)

        if attn_metadata is None:
            # V1 profile run
            hs = hidden_states.unsqueeze(0).transpose(0, 1).contiguous()
            hs_d = F.pad(hs[:-1], pad=(0, 0, 0, 0, 1, 0))  # [S, B, H]
            q = self.linear_q(hs)  # [S, B, latent_q_dim]
            k = self.linear_k(hs)  # [S, B, latent_k_dim]
            qk_packed0 = torch.cat([q, k], dim=-1)  # [S, B, latent_q + latent_k]
            del q
            del k

            # Pre-mean tensors in head form (for "qk_mean_{q,k}" calc)
            query_pre = qk_packed0[..., : self.latent_q_dim].view(
                *qk_packed0.shape[:2], self.num_q_heads, self.head_dim
            )  # [S, B, qh, dh]

            key_base = qk_packed0[..., self.latent_q_dim :].view(
                *qk_packed0.shape[:2], self.num_k_heads, self.head_dim
            )  # [S, B, kh, dh]

            qk_packed1 = qk_packed0.permute(1, 2, 0)  # [B, E, S]
            qk_packed2 = F.pad(qk_packed1, (self.total_padding, 0))
            qk_packed3 = self.conv_qk(qk_packed2).permute(2, 0, 1)  # [S, B, E]

            # Build queries/keys from conv output + means
            query = (
                qk_packed3[..., : self.latent_q_dim]
                .view(*qk_packed3.shape[:2], self.num_q_heads, self.head_dim)
                .float()
            )

            key = (
                qk_packed3[..., self.latent_q_dim :]
                .view(*qk_packed3.shape[:2], self.num_k_heads, self.head_dim)
                .float()
            )
            query, key = self._add_grouped_qk_means_inplace(
                query, key, query_pre, key_base
            )
            del query_pre
            del key_base
            del qk_packed0
            del qk_packed3

            # Values from the two time streams
            v1 = self.val_proj1(hs)  # [S, B, latent_k_dim/2]
            v2 = self.val_proj2(hs_d)  # [S, B, latent_k_dim/2]
            value = (
                torch.cat([v1, v2], dim=-1)
                .contiguous()
                .view(*hs.shape[:2], self.num_k_heads, self.head_dim)
            )  # [S, B, kh, dh]

            query, key = self._rms_normalize_qk(query.contiguous(), key.contiguous())

            return hs

        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_decodes + num_prefill_tokens

        # Pure-decode HIP fast path: one fused kernel does conv + grouped-means
        # + per-head RMS-norm + state update, producing normalized q|k directly
        # (qk_full); the shared eager means/normalize/qk-output below is skipped.
        # Only for pure decode (no prefill) with an fp32 conv-state cache.
        cca_hip = _get_cca_hip()
        use_full_hip = (
            cca_hip is not None
            and has_decode
            and not has_prefill
            and self.num_spec == 0
            and self.kv_cache[0].dtype == torch.float32
        )
        qk_full = None

        num_input_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states[:num_actual_tokens]

        # Batch size is effectively 1 in this path, so insert the singleton
        # dimension directly instead of transposing and materializing a copy.
        hs = hidden_states.unsqueeze(1)  # [S, 1, H]
        batch_size = hs.shape[1]

        q = self.linear_q(hs)  # [S, B, latent_q_dim]
        k = self.linear_k(hs)  # [S, B, latent_k_dim]
        qk_packed0 = torch.cat([q, k], dim=-1)  # [S, B, latent_q + latent_k]
        del q
        del k

        # Pre-mean tensors in head form (for "qk_mean_{q,k}" calc)
        query_pre = qk_packed0[..., : self.latent_q_dim].view(
            *qk_packed0.shape[:2], self.num_q_heads, self.head_dim
        )  # [S, B, qh, dh]

        key_base = qk_packed0[..., self.latent_q_dim :].view(
            *qk_packed0.shape[:2], self.num_k_heads, self.head_dim
        )  # [S, B, kh, dh]

        # NOTE: V1 puts decode before prefill
        # Separate prefill and decode by splitting varlen input
        # Split along token dimension
        qk_packed0_d, qk_packed0_p = torch.split(
            qk_packed0[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )
        hs_d, hs_p = torch.split(
            hs[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )

        qk_packed3 = torch.empty(
            (num_actual_tokens, batch_size, self.in_out_ch),
            device=hs.device,
            dtype=hs.dtype,
        )
        hs2 = torch.empty(
            (num_actual_tokens, batch_size, self.hidden_size),
            device=hs.device,
            dtype=hs.dtype,
        )
        decode_is_pad: torch.Tensor | None = None
        if has_prefill:
            assert state_indices_tensor_p is not None
            assert has_initial_states_p is not None
            assert query_start_loc_p is not None
            # Prefill: run the causal conv over all requests in one batched
            # call instead of a per-request Python loop (which also forced a
            # GPU->CPU sync per request via the query_start_loc_p slicing).
            # Each request occupies a contiguous segment
            # [cached state (total_padding cols) | its tokens] in a flat conv
            # input, so every valid output window stays inside its own segment
            # and requests cannot contaminate each other.
            prefill_slice = slice(num_decodes, num_decodes + num_prefill_tokens)
            tp_pad = self.total_padding
            device = hs.device

            req_idx = torch.arange(num_prefills, device=device)
            seq_lens_p = query_start_loc_p[1:] - query_start_loc_p[:-1]
            token_req = torch.repeat_interleave(
                req_idx, seq_lens_p, output_size=num_prefill_tokens
            )
            token_flat = torch.arange(num_prefill_tokens, device=device)
            # Token t of request i sits at qsl[i] + t + (i + 1) * tp_pad; its
            # conv output window starts tp_pad columns earlier.
            token_pos = token_flat + (token_req + 1) * tp_pad
            out_pos = token_flat + token_req * tp_pad
            seg_starts = query_start_loc_p[:-1] + req_idx * tp_pad
            pad_offsets = torch.arange(tp_pad, device=device)

            flat_len = num_prefill_tokens + num_prefills * tp_pad
            flat_in = qk_packed0_p.new_empty((self.in_out_ch, flat_len))
            flat_in[:, token_pos] = qk_packed0_p[:, 0, :].t()

            init_states = conv_states[state_indices_tensor_p].to(flat_in.dtype)
            init_states = torch.where(
                has_initial_states_p.view(-1, 1, 1),
                init_states,
                init_states.new_zeros(()),
            )
            state_pos = (seg_starts.unsqueeze(1) + pad_offsets).reshape(-1)
            flat_in[:, state_pos] = init_states.permute(1, 0, 2).reshape(
                self.in_out_ch, -1
            )

            flat_out = self.conv_qk(flat_in.unsqueeze(0))[0]
            qk_packed3[prefill_slice] = flat_out[:, out_pos].t().unsqueeze(1)

            # New conv state: the last total_padding input columns of each
            # segment (naturally falls back onto the old state when a chunk
            # is shorter than the conv window).
            new_state_pos = (
                (query_start_loc_p[1:] + req_idx * tp_pad).unsqueeze(1) + pad_offsets
            ).reshape(-1)
            new_states = (
                flat_in[:, new_state_pos]
                .reshape(self.in_out_ch, num_prefills, tp_pad)
                .permute(1, 0, 2)
            )
            conv_states[state_indices_tensor_p] = new_states.to(
                device=conv_states.device, dtype=conv_states.dtype
            )

            # hs2 = previous-token hidden states: shift by one within each
            # request; the first token takes the cached last hidden state of
            # the previous chunk (or zero for a fresh request).
            hs2_prefill = hs2[prefill_slice]
            hs2_prefill[1:] = hs_p[:-1]
            init_hs = prev_hs[state_indices_tensor_p].to(hs.dtype)
            init_hs = torch.where(
                has_initial_states_p.view(-1, 1),
                init_hs,
                init_hs.new_zeros(()),
            )
            hs2_prefill[query_start_loc_p[:-1]] = init_hs.unsqueeze(1)

            prev_hs[state_indices_tensor_p] = hs_p[query_start_loc_p[1:] - 1, 0, :].to(
                device=prev_hs.device, dtype=prev_hs.dtype
            )

        if has_decode and self.num_spec > 0:
            # Speculative decode verify step: (1 + num_spec) tokens per sequence.
            # Route through the same flat causal-conv machinery as prefill (each
            # sequence = a short segment seeded with its current conv state), and
            # write the per-spec-position rollback state that the align-mode
            # postprocess (get_cca_conv_copy_spec / get_temporal_copy_spec) reads.
            self._decode_verify_spec(
                qk_packed0_d=qk_packed0_d,
                hs_d=hs_d,
                qk_packed3=qk_packed3,
                hs2=hs2,
                conv_states=conv_states,
                prev_hs=prev_hs,
                state_indices_2d=state_indices_tensor_d_2d,
                query_start_loc_d=query_start_loc_d,
                num_decode_tokens=num_decodes,
                num_decode_seqs=num_decode_seqs,
                blk_computed=blk_computed_d,
                blk_scheduled=blk_scheduled_d,
                blk_scheduled_prev=blk_scheduled_prev_d,
                num_accepted=num_accepted_d,
            )
        elif has_decode:
            assert state_indices_tensor_d is not None
            # Generation
            # In generation B and S are actually the same in meaning
            # That's why we don't need to transpose qk_packed0
            # qk_packed0_d [S, 1, H]
            decode_is_pad = state_indices_tensor_d == PAD_SLOT_ID
            # block_id=0 reserved
            # Zvllm/vllm/v1/core/block_pool.py
            safe_decode_indices = torch.where(
                decode_is_pad,
                torch.zeros_like(state_indices_tensor_d),
                state_indices_tensor_d,
            )
            hs_d = torch.where(
                decode_is_pad.view(-1, 1, 1),
                hs_d.new_zeros(()),
                hs_d,
            )

            if use_full_hip:
                # One fused kernel: conv + grouped-means + per-head RMS-norm +
                # state roll. Produces normalized q|k (qk_full); the shared
                # eager means/normalize/qk-output below is skipped for decode.
                w0f, b0f, w1f, b1f = self._conv_weights_fp32()
                qk_new = qk_packed0_d[:, 0, :].to(torch.float32).contiguous()  # [S, C]
                qk_full = cca_hip.cca_decode_qk(
                    qk_new,
                    conv_states,
                    safe_decode_indices.to(torch.int64),
                    decode_is_pad,
                    w0f,
                    b0f,
                    w1f,
                    b1f,
                    self._temp_eff(),
                    self.num_q_heads,
                    self.gqa_groups,
                    self.latent_q_dim,
                    float(self.sqrt_head_dim),
                )  # [S, C] fp32 normalized q|k; conv_states updated in place
            else:
                qk_packed0_d = torch.where(
                    decode_is_pad.view(-1, 1, 1),
                    qk_packed0_d.new_zeros(()),
                    qk_packed0_d,
                )
                qk_packed0_cached = conv_states[
                    safe_decode_indices
                ]  # [S, H, total_padding]
                qk_packed0_cached = torch.where(
                    decode_is_pad.view(-1, 1, 1),
                    qk_packed0_cached.new_zeros(()),
                    qk_packed0_cached,
                )
                qk_packed0_cached_for_compute = qk_packed0_cached
                decode_qk_dtype = qk_packed0_d.dtype
                if qk_packed0_cached_for_compute.dtype != decode_qk_dtype:
                    qk_packed0_cached_for_compute = qk_packed0_cached_for_compute.to(
                        decode_qk_dtype
                    )
                qk_packed0_cat = torch.cat(
                    [qk_packed0_cached_for_compute, qk_packed0_d.transpose(1, 2)],
                    dim=-1,
                )  # [S, H, total_padding + 1]
                qk_packed3_d = self._conv_qk_decode(qk_packed0_cat).transpose(
                    1, 2
                )  # [S, 1, E]
                qk_packed3[:num_decodes] = qk_packed3_d

                new_qk_packed0_cache = qk_packed0_cached.roll(shifts=-1, dims=-1)
                new_qk_packed0_cache[..., -1] = qk_packed0_d[:, 0, :].to(
                    new_qk_packed0_cache.dtype
                )
                new_qk_packed0_cache = torch.where(
                    decode_is_pad.view(-1, 1, 1),
                    new_qk_packed0_cache.new_zeros(()),
                    new_qk_packed0_cache,
                )
                conv_states[safe_decode_indices] = new_qk_packed0_cache.to(
                    device=conv_states.device, dtype=conv_states.dtype
                )

            hs2_decode = prev_hs[safe_decode_indices].unsqueeze(1)  # [S, 1, H]
            hs2_decode = torch.where(
                decode_is_pad.view(-1, 1, 1),
                hs2_decode.new_zeros(()),
                hs2_decode,
            )
            if hs2_decode.dtype != hs.dtype:
                hs2_decode = hs2_decode.to(hs.dtype)
            hs2[:num_decodes] = hs2_decode
            new_prev_hs = hs_d[:, 0, :].to(prev_hs.dtype)
            new_prev_hs = torch.where(
                decode_is_pad.view(-1, 1),
                new_prev_hs.new_zeros(()),
                new_prev_hs,
            )
            prev_hs[safe_decode_indices] = new_prev_hs.to(
                device=prev_hs.device, dtype=prev_hs.dtype
            )

        del qk_packed0_d
        del qk_packed0_p
        del hs_d
        del hs_p

        # Values from the two time streams
        v1 = self.val_proj1(hs)  # [S, B, latent_k_dim/2]
        v2 = self.val_proj2(hs2)
        value = torch.cat([v1, v2], dim=-1).contiguous()
        value = value.view(
            num_actual_tokens, batch_size, self.num_k_heads, self.head_dim
        )  # [S, B, kh, dh]
        del hs2

        q_end = self.latent_q_dim
        k_end = q_end + self.latent_k_dim
        value = value.reshape(num_actual_tokens, self.latent_k_dim)

        if use_full_hip:
            # qk_full [num_actual_tokens, latent_q+latent_k] is the normalized
            # q|k from the fused kernel (means + RMS-norm already applied).
            output[:num_actual_tokens, :k_end] = qk_full.to(output.dtype)
            output[:num_actual_tokens, k_end:] = value.to(output.dtype)
            del qk_packed0
            del qk_packed3
            del query_pre
            del key_base
        else:
            # Build queries/keys from conv output + means
            query = (
                qk_packed3[..., : self.latent_q_dim]
                .view(num_actual_tokens, batch_size, self.num_q_heads, self.head_dim)
                .float()
            )
            key = (
                qk_packed3[..., self.latent_q_dim :]
                .view(num_actual_tokens, batch_size, self.num_k_heads, self.head_dim)
                .float()
            )
            query, key = self._add_grouped_qk_means_inplace(
                query, key, query_pre, key_base
            )
            del query_pre
            del key_base
            del qk_packed0
            del qk_packed3

            query, key = self._rms_normalize_qk(
                query.contiguous(), key.contiguous()
            )
            query = query.reshape(num_actual_tokens, self.latent_q_dim)
            key = key.reshape(num_actual_tokens, self.latent_k_dim)
            output[:num_actual_tokens, :q_end] = query
            output[:num_actual_tokens, q_end:k_end] = key
            output[:num_actual_tokens, k_end:] = value

        if decode_is_pad is not None:
            decode_output = output[:num_decodes]
            output[:num_decodes] = torch.where(
                decode_is_pad.view(-1, 1),
                decode_output.new_zeros(()),
                decode_output,
            )

    def _decode_verify_spec(
        self,
        *,
        qk_packed0_d: torch.Tensor,
        hs_d: torch.Tensor,
        qk_packed3: torch.Tensor,
        hs2: torch.Tensor,
        conv_states: torch.Tensor,
        prev_hs: torch.Tensor,
        state_indices_2d: torch.Tensor,
        query_start_loc_d: torch.Tensor,
        num_decode_tokens: int,
        num_decode_seqs: int,
        blk_computed: torch.Tensor | None = None,
        blk_scheduled: torch.Tensor | None = None,
        blk_scheduled_prev: torch.Tensor | None = None,
        num_accepted: torch.Tensor | None = None,
    ) -> None:
        """Conv + state update for the spec-decode verify step.

        Each decode sequence carries up to ``1 + num_spec`` candidate tokens.
        We run them through the same flat causal-conv path as prefill (each
        sequence is a segment ``[current conv state | candidate tokens]``),
        write the conv outputs into ``qk_packed3``/the previous-hidden inputs
        into ``hs2``, and persist the per-spec-position rollback state that the
        align-mode postprocess reads via ``get_cca_conv_copy_spec`` (conv) and
        ``get_temporal_copy_spec`` (prev_hs).
        """
        device = hs_d.device
        tp_pad = self.total_padding
        td = num_decode_tokens
        s = num_decode_seqs
        w = tp_pad + self.num_spec  # widened conv-state block width
        p = 1 + self.num_spec  # max candidate tokens per sequence

        # query_start_loc_d is None on steps the builder treats as plain
        # single-token decodes (no accepted-token layout): one token per seq.
        if query_start_loc_d is None:
            qsl = torch.arange(s + 1, device=device, dtype=torch.int32)
        else:
            qsl = query_start_loc_d[: s + 1]
        seq_lens = qsl[1:] - qsl[:-1]  # [s]
        base_block = state_indices_2d[:, 0]  # [s]
        decode_is_pad = base_block == PAD_SLOT_ID
        safe_base = torch.where(
            decode_is_pad, torch.zeros_like(base_block), base_block
        )

        # 'all' mamba_cache_mode delivers the full-block-table rollback layout
        # (block_idx_last_scheduled_token{,_prev_step} + num_accepted_tokens).
        # ⚠ DRAFT — UNVALIDATED ON GPU. Gated so the committed align/none spec
        # path (coherent, ~1.76x) is byte-unchanged when these fields are None.
        # See docs/zaya/cca-all-mode-spec-plan.md.
        all_avail = (
            blk_scheduled is not None
            and blk_scheduled_prev is not None
            and num_accepted is not None
        )
        max_col = state_indices_2d.shape[1] - 1

        # Cross-step rollback: read the committed conv/temporal state from the
        # slot ending at the last accepted token, and write verify position j to
        # its own slot. The previous step's acceptance advances the pointer, so
        # the next step reads the state ending at the last accepted token.
        row = torch.arange(s, device=device)
        if all_avail:
            # 'all' mode: the previous verify step wrote its 1+num_spec outputs to
            # FULL block-table columns [prev .. prev+num_spec]; the committed start
            # is column prev + (num_accepted-1). Mirrors mamba2 selective_state_
            # update's init_token_idx = num_accepted-1. Do NOT clamp to p-1 — that
            # is the compact-layout assumption that corrupts the block column here.
            na = num_accepted[:s].clamp(min=1).to(torch.long)
            read_col = (blk_scheduled_prev[:s].to(torch.long) + (na - 1)).clamp(
                min=0, max=max_col
            )
        elif blk_computed is not None:
            read_col = blk_computed[:s].clamp(min=0, max=p - 1).to(torch.long)
        else:
            read_col = torch.zeros(s, device=device, dtype=torch.long)
        init_block = state_indices_2d[row, read_col]  # [s]
        init_is_pad = init_block == PAD_SLOT_ID
        safe_init = torch.where(
            init_is_pad, torch.zeros_like(init_block), init_block
        )

        # Per-sequence conv input: [current conv state | candidate tokens],
        # padded to a uniform width. Run the SAME conv as the non-spec decode
        # path (_conv_qk_decode) so the verify logits match it bit-for-bit;
        # self.conv_qk (cuDNN/MIOpen) diverges from it enough to flip tokens.
        # Use a fixed width of p = 1 + num_spec (the metadata's max query len)
        # rather than seq_lens.max().item(): avoids a CPU sync so the path stays
        # cudagraph-capturable. Shorter sequences are masked out.
        pos = torch.arange(p, device=device)
        valid = pos.unsqueeze(0) < seq_lens.unsqueeze(1)  # [s, p]
        tok_glob = (qsl[:-1].unsqueeze(1) + pos.unsqueeze(0)).clamp(max=td - 1)

        init_states = conv_states[safe_init].to(qk_packed0_d.dtype)  # [s,C,tp_pad]
        init_states = torch.where(
            decode_is_pad.view(-1, 1, 1), init_states.new_zeros(()), init_states
        )
        toks = (
            qk_packed0_d[tok_glob.reshape(-1), 0, :]
            .reshape(s, p, self.in_out_ch)
            .permute(0, 2, 1)
        )  # [s, C, p]
        toks = torch.where(valid.unsqueeze(1), toks, toks.new_zeros(()))
        buf = torch.cat([init_states, toks], dim=-1)  # [s, C, tp_pad + p]

        conv_out = self._conv_qk_decode(buf)  # [s, C, p]
        conv_flat = conv_out.permute(0, 2, 1).reshape(s * p, self.in_out_ch)
        # Cudagraph-safe scatter into the decode region (no boolean-mask indexing,
        # whose output size is data-dependent and illegal during graph capture):
        # token (i, j) -> global index qsl[i] + j; invalid positions dump to row td.
        target = qsl[:-1].unsqueeze(1) + pos.unsqueeze(0)  # [s, p]
        target = torch.where(valid, target, torch.full_like(target, td))
        scratch = qk_packed3.new_zeros((td + 1, 1, self.in_out_ch))
        scratch[target.reshape(-1)] = conv_flat.unsqueeze(1)
        qk_packed3[:td] = scratch[:td]

        # hs2 = previous-token hidden state; shift within each segment and seed
        # the segment start with the incoming cached prev_hs. Read the OLD
        # prev_hs before the per-position write below.
        incoming = prev_hs[safe_init].unsqueeze(1).to(hs_d.dtype)
        incoming = torch.where(
            decode_is_pad.view(-1, 1, 1), incoming.new_zeros(()), incoming
        )
        hs2_dec = hs2[:td]
        hs2_dec[1:] = hs_d[:-1]
        hs2_dec[qsl[:-1]] = incoming

        # Per-spec-position rollback state. For each verify position j of each
        # sequence we write, into the block slot state_indices_2d[i, j], the
        # state ending at token j: the tp_pad-wide conv window (conv_states) and
        # the token's hidden (prev_hs). The align postprocess then selects slot
        # (num_accepted - 1) for the next step via get_cca_conv_copy_spec /
        # get_temporal_copy_spec. Invalid/pad positions rewrite their slot's
        # existing value (no-op).
        pos = torch.arange(p, device=device)
        pos_valid = pos.unsqueeze(0) < seq_lens.unsqueeze(1)
        # Write verify position j to its rollback slot so the next step reads the
        # state ending at the last accepted token. Fall back to column j when no
        # pointer is delivered (plain single-token / no rollback metadata).
        if all_avail:
            # 'all' mode (DRAFT): verify position j -> full block-table column
            # (blk_scheduled + j); next step reads (blk_scheduled + num_accepted-1).
            write_col = (
                blk_scheduled[:s].unsqueeze(1).to(torch.long) + pos.unsqueeze(0)
            ).clamp(min=0, max=max_col)  # [s, p]
        elif blk_computed is not None:
            write_col = (read_col.unsqueeze(1) + 1 + pos.unsqueeze(0)).clamp(
                min=0, max=p - 1
            )  # [s, p]
        else:
            write_col = pos.unsqueeze(0).expand(s, p)
        write_slot = state_indices_2d.gather(1, write_col)  # [s, p] block ids
        write_ok = (
            pos_valid
            & (~decode_is_pad).unsqueeze(1)
            & (write_slot != PAD_SLOT_ID)
        )
        slot_safe = torch.where(
            write_ok, write_slot, torch.zeros_like(write_slot)
        )

        # conv window ending at position j = buf cols [j+1 : j+1+tp_pad] (the
        # tp_pad pre-conv q|k columns up to and including token j).
        win = torch.arange(tp_pad, device=device)
        bcols = ((pos + 1).unsqueeze(-1) + win).clamp(max=buf.shape[-1] - 1)
        conv_win = (
            buf[:, :, bcols.reshape(-1)]
            .reshape(s, self.in_out_ch, p, tp_pad)
            .permute(0, 2, 1, 3)
        )  # [s, p, in_out_ch, tp_pad]
        old_c = conv_states[slot_safe.reshape(-1)]
        new_c = torch.where(
            write_ok.reshape(-1, 1, 1),
            conv_win.reshape(-1, self.in_out_ch, tp_pad).to(old_c.dtype),
            old_c,
        )
        conv_states[slot_safe.reshape(-1)] = new_c

        # prev_hs: slot j gets verify token j's hidden state.
        tok_global = (qsl[:-1].unsqueeze(1) + pos.unsqueeze(0)).clamp(max=td - 1)
        hidden = prev_hs.shape[-1]
        hs_pos = hs_d[tok_global.reshape(-1), 0, :].reshape(s, p, hidden)
        hs_write = torch.where(
            write_ok.unsqueeze(-1), hs_pos.to(prev_hs.dtype), prev_hs[slot_safe]
        )
        prev_hs[slot_safe.reshape(-1)] = hs_write.reshape(-1, hidden)

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.cca_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.cca_state_shape(
            tp_world_size=get_tensor_model_parallel_world_size(),
            conv_kernel_size=self.total_padding,
            num_k_heads=self.num_k_heads,
            num_q_heads=self.num_q_heads,
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
            num_spec=self.num_spec,
        )

    def _temp_eff(self):
        # Cache the effective key temperature for the HIP op: exp(clamp(temp))
        # if clamp_temp else temp (matches _rms_normalize_qk).
        if getattr(self, "_temp_eff_cache", None) is None:
            t = self.temp.to(torch.float32)
            if self.config.clamp_temp:
                t = torch.exp(torch.clamp(t, 1e-7, 2.0))
            self._temp_eff_cache = t.contiguous()
        return self._temp_eff_cache

    def _conv_weights_fp32(self):
        # Cache fp32 copies of the conv weights for the HIP op (model is bf16;
        # the kernel computes in fp32). Weights are constant in inference.
        if getattr(self, "_conv_w_f32", None) is None:
            w0 = self.conv_qk[0].weight.squeeze(1).float().contiguous()  # [C, K0]
            b0 = self.conv_qk[0].bias.float().contiguous()  # [C]
            b1 = self.conv_qk[1].bias.float().contiguous()  # [C]
            # The grouped-conv weight [C=H*d_out, d_in, K1] is pre-transposed to
            # [H, d_in, d_out, K1] for cca_decode_qk: with that layout the head's
            # threads (output channel = j) read consecutive addresses per di, so
            # the grouped conv's hot loop coalesces (~5.6x faster kernel vs the
            # original [C, d_in, K1] layout, which strided w1 by d*K1 across
            # threads). Done once — conv weights are constant in inference.
            w1 = self.conv_qk[1].weight.float()  # [C, d_in, K1]
            # ZAYA_CCA_W1T=0 keeps the original [C, d_in, K1] layout for an A/B
            # against the pre-transpose kernel; default (transpose) is the fast
            # coalesced path and must match the built cca_kernel.hip.
            if os.environ.get("ZAYA_CCA_W1T", "1") != "0":
                c_out, d_in, k1 = w1.shape
                num_heads = c_out // d_in  # groups == heads, d_out == d_in == dim
                w1 = (
                    w1.view(num_heads, d_in, d_in, k1)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )  # [H, d_in, d_out, K1]
            else:
                w1 = w1.contiguous()
            self._conv_w_f32 = (w0, b0, w1, b1)
        return self._conv_w_f32

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.CCA


def cca(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_cuda(hidden_states=hidden_states, output=output)


def cca_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="cca",
    op_func=cca,
    mutates_args=["output"],
    fake_impl=cca_fake,
)
