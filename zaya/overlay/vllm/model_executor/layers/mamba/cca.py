# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
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
# Fused PREFILL kernel (cca_prefill_qk): collapses the flat-buffer
# scatter/conv/gather + eager grouped-means + fp32 RMS-norm + new-conv-state into
# one launch (2.4-3x faster than the eager flat-conv path, validated bit-exact).
# Also enables the mixed prefill+decode HIP path (both kernels in one forward).
# Separate opt-in from decode (ZAYA_CCA_HIP gates the decode kernel). Default ON:
# the GPU coherence + perf gate passed (coherent; +37-49% e2e vs eager; all three
# paths bit-exact — see docs/zaya/cca-kernel-perf.md). Set =0 to force eager prefill.
_CCA_HIP_PREFILL = os.environ.get("ZAYA_CCA_HIP_PREFILL", "1") == "1"
_cca_hip = None
_cca_hip_tried = False

# Debug: tally which fused HIP path each forward takes — decode-only, prefill-only
# (pure-prefill batch), or mixed (prefill+decode in one batch) — so a serving run
# can confirm all three CCA kernels actually fire. Gated; default off => the guard
# short-circuits before any bookkeeping, zero hot-path overhead.
_CCA_DEBUG_PATHS = os.environ.get("ZAYA_CCA_DEBUG_PATHS", "0") == "1"
_cca_path_counts = {"decode": 0, "prefill": 0, "mixed": 0}


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
            # path expects a single base slot per sequence, so collapse to column 0
            # for READS (initial states). The end-of-prefill WRITE, however, must go
            # to the block_idx_last_scheduled_token column so the first verify step's
            # seed read (blk_scheduled_prev + num_accepted-1) finds it — see the
            # prefill write below. Keep the 2D form to compute that write column.
            # No-op in align/none mode, where state_indices_tensor_p is already 1D.
            state_indices_tensor_p_2d = state_indices_tensor_p
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

        # Fused HIP fast paths: one kernel per region does conv + grouped-means +
        # per-head RMS-norm + state update, producing normalized q|k directly
        # (qk_full); the shared eager means/normalize/qk-output below is skipped.
        # Requires an fp32 conv-state cache and non-spec decode. The decode kernel
        # (cca_decode_qk) is gated by ZAYA_CCA_HIP (implicit in cca_hip being
        # non-None); the prefill kernel (cca_prefill_qk) additionally by
        # ZAYA_CCA_HIP_PREFILL.
        cca_hip = _get_cca_hip()
        hip_base = (
            cca_hip is not None
            and self.num_spec == 0
            and self.kv_cache[0].dtype == torch.float32
        )
        use_decode_hip = hip_base and has_decode
        use_prefill_hip = hip_base and has_prefill and _CCA_HIP_PREFILL
        # The unified qk_full output path bakes means+RMS-norm into the kernel and
        # is all-or-nothing across the batch: take it only when EVERY present
        # region is HIP-covered. Pure decode needs only ZAYA_CCA_HIP; pure prefill
        # (and therefore a MIXED prefill+decode batch) also needs
        # ZAYA_CCA_HIP_PREFILL. In a mixed batch both kernels run: decode and
        # prefill write disjoint conv-state slots and the prefill kernel reads a
        # pre-gathered init_states copy (never live conv_states), so the two
        # launches don't race; their qk outputs are concatenated decode-first at
        # the output stage. When a present region is not HIP-covered the whole
        # batch falls back to the eager means/normalize path (unchanged).
        use_hip_qk = (
            (has_decode or has_prefill)
            and (use_decode_hip or not has_decode)
            and (use_prefill_hip or not has_prefill)
        )
        # Decode/prefill HIP kernels emit their own region's normalized q|k; kept
        # separate so a mixed batch can concatenate them (decode rows first).
        qk_full_d = None
        qk_full_p = None

        if _CCA_DEBUG_PATHS and use_hip_qk:
            _path = (
                "mixed"
                if (has_decode and has_prefill)
                else ("decode" if has_decode else "prefill")
            )
            _cca_path_counts[_path] += 1
            if _cca_path_counts[_path] == 1 or sum(_cca_path_counts.values()) % 1000 == 0:
                logging.getLogger(__name__).info(
                    "ZAYA CCA HIP path counts: %s", _cca_path_counts
                )

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
            # 'all' mamba_cache_mode: write the end-of-prefill conv/temporal state
            # to each request's block_idx_last_scheduled_token block, NOT the
            # collapsed column-0 slot. The first spec-verify step seeds from
            # state_indices_2d[row, blk_scheduled_prev + (num_accepted-1)], and
            # blk_scheduled_prev for that step is the prefill's last scheduled
            # block — so the prefill must deposit its state there or the first
            # verify reads an empty/wrong slot (token-1 divergence). In align/none
            # mode state_indices_tensor_p is 1D and this falls back to it (no-op).
            prefill_write_slots = state_indices_tensor_p
            if (
                self.num_spec > 0
                and state_indices_tensor_p_2d is not None
                and state_indices_tensor_p_2d.dim() > 1
                and blk_scheduled_d is not None
            ):
                # num_spec>0: ALL decode (incl. single-token) goes through the
                # verify path, which reads its seed from the block_idx column (never
                # column 0). So the prefill end-state must live at that column.
                # num_spec==0 keeps column 0 (single-token decode reads [:,0]) —
                # which is why the align/all num_spec=0 paths are already bit-clean.
                blk_sched_p = blk_scheduled_d[
                    num_decodes : num_decodes + num_prefills
                ].to(torch.long)
                prefill_write_slots = state_indices_tensor_p_2d[req_idx, blk_sched_p]
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

            if use_hip_qk:
                # Fused prefill kernel: same flat-segment causal-conv math as the
                # eager path below, but conv + grouped-means + per-head RMS-norm +
                # new-conv-state in ONE launch, emitting normalized q|k (qk_full_p)
                # directly. The kernel reads the masked cached state (init_states)
                # and writes conv_states in place (only each request's last token);
                # the eager flat-buffer scatter/conv/gather/means/norm is skipped.
                w0f, b0f, w1f, b1f = self._conv_weights_fp32()
                qk_new_p = qk_packed0_p[:, 0, :].to(torch.float32).contiguous()
                init_states = conv_states[state_indices_tensor_p].to(torch.float32)
                init_states = torch.where(
                    has_initial_states_p.view(-1, 1, 1),
                    init_states,
                    init_states.new_zeros(()),
                ).contiguous()
                seg_pos = (
                    token_flat - query_start_loc_p[:-1][token_req]
                ).to(torch.int32)
                slot_p = state_indices_tensor_p[token_req].to(torch.int64)
                is_last_p = token_flat == (query_start_loc_p[1:] - 1)[token_req]
                qk_full_p = cca_hip.cca_prefill_qk(
                    qk_new_p,
                    conv_states,
                    init_states,
                    seg_pos,
                    token_req.to(torch.int32),
                    slot_p,
                    is_last_p,
                    w0f,
                    b0f,
                    w1f,
                    b1f,
                    self._temp_eff(),
                    self.num_q_heads,
                    self.gqa_groups,
                    self.latent_q_dim,
                    float(self.sqrt_head_dim),
                )  # [P, latent_q+latent_k] normalized q|k; conv_states updated
            else:
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
                    (query_start_loc_p[1:] + req_idx * tp_pad).unsqueeze(1)
                    + pad_offsets
                ).reshape(-1)
                new_states = (
                    flat_in[:, new_state_pos]
                    .reshape(self.in_out_ch, num_prefills, tp_pad)
                    .permute(1, 0, 2)
                )
                conv_states[prefill_write_slots] = new_states.to(
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

            prev_hs[prefill_write_slots] = hs_p[query_start_loc_p[1:] - 1, 0, :].to(
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

            if use_hip_qk:
                # One fused kernel: conv + grouped-means + per-head RMS-norm +
                # state roll. Produces normalized q|k (qk_full_d); the shared
                # eager means/normalize/qk-output below is skipped for decode.
                w0f, b0f, w1f, b1f = self._conv_weights_fp32()
                qk_new = qk_packed0_d[:, 0, :].to(torch.float32).contiguous()  # [S, C]
                qk_full_d = cca_hip.cca_decode_qk(
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

        if use_hip_qk:
            # Normalized q|k (means + RMS-norm baked in) straight from the fused
            # kernel(s). V1 lays the batch out decode-first, so a mixed batch
            # concatenates the decode-region output (rows [0:num_decodes], from
            # cca_decode_qk) ahead of the prefill-region output (rows
            # [num_decodes:num_actual_tokens], from cca_prefill_qk). Pure batches
            # have only one of the two.
            if qk_full_d is not None and qk_full_p is not None:
                qk_full = torch.cat([qk_full_d, qk_full_p], dim=0)
            else:
                qk_full = qk_full_d if qk_full_d is not None else qk_full_p
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
        # TiDAR fold (serving step 4): this same (1 + num_spec) candidate-window
        # conv + per-spec-position rollback IS the TiDAR evict-on-reject path.
        # num_spec maps to the TiDAR block_len: the verify processes the whole
        # [current state | block_len candidate tokens] block, writes the conv
        # window + prev_hs ENDING at each candidate j to slot
        # state_indices_2d[i, write_col[i, j]], and the next step reads slot
        # (num_accepted - 1) = the accepted-prefix end. Appending the rejected
        # tail then reading the accepted column IS the evict — no truncation.
        # PROVEN equivalent to a from-scratch recompute of the accepted prefix on
        # the REAL ZAYA conv/prev_hs (zaya/tidar/cca_evict_gate.py, max|Δ|=0.0 for
        # k_accept 0..block_len) because conv_qk is causal (left-pad only). The
        # TiDAR structured attention mask rides the SEPARATE standard-attention
        # backend via the active-mask carrier (tidar_attn_metadata.py) +
        # CCAAttentionMetadata.tidar_mask — NOT this conv producer, keeping verify
        # ISOLATED from the mask-replica scratch (the §7.5 fusion-contamination
        # finding). num_spec==0 ⇒ none of this runs (the elif-decode branch) ⇒
        # the non-TiDAR path is byte-identical.

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
        # Per-step diagnostic (M5/task-A): log the first N verify calls so we can
        # see whether all_avail is False only on step 1 (no previous step) or
        # throughout (runner not threading rollback fields). Logs num_accepted /
        # seq_lens values too, to tell a within-block conv bug from a cross-step
        # rollback bug. Gated; default off => zero hot-path cost. Set
        # ZAYA_SPEC_DEBUG_N to change the count (default 40); only layer 0 logs.
        if os.environ.get("ZAYA_SPEC_DEBUG", "0") == "1":
            cls = type(self)
            # Latch the first CCA layer instance to reach this point as the sole
            # logger (avoids 40 layers x N steps of spam).
            if getattr(cls, "_spec_dbg_owner", None) is None:
                cls._spec_dbg_owner = id(self)
            if cls._spec_dbg_owner == id(self):
                _dbgN = int(os.environ.get("ZAYA_SPEC_DEBUG_N", "40"))
                _c = getattr(self, "_spec_dbg_count", 0)
                if _c < _dbgN:
                    self._spec_dbg_count = _c + 1
                    _na = (
                        num_accepted[:num_decode_seqs].tolist()
                        if num_accepted is not None
                        else None
                    )
                    logging.getLogger(__name__).info(
                        "ZAYA CCA verify[%d]: all_avail=%s seqs=%s seq_lens=%s "
                        "num_accepted=%s blk_sched=%s blk_sched_prev=%s",
                        _c,
                        all_avail,
                        num_decode_seqs,
                        seq_lens[:num_decode_seqs].tolist(),
                        _na,
                        (blk_scheduled[:num_decode_seqs].tolist()
                         if blk_scheduled is not None else None),
                        (blk_scheduled_prev[:num_decode_seqs].tolist()
                         if blk_scheduled_prev is not None else None),
                    )
        max_col = state_indices_2d.shape[1] - 1

        # Cross-step rollback: read the committed conv/temporal state from the
        # slot ending at the last accepted token, and write verify position j to
        # its own slot. The previous step's acceptance advances the pointer, so
        # the next step reads the state ending at the last accepted token.
        # 'all' cache mode is active whenever the block_idx pointers are threaded.
        # Single-token decode steps (ngram proposed nothing -> seq_len 1) arrive
        # with num_accepted/blk_scheduled_prev None (vLLM treats them as non-spec
        # decodes) but blk_scheduled/blk_computed present.
        all_mode = blk_scheduled is not None and blk_computed is not None
        row = torch.arange(s, device=device)
        if all_avail:
            # 'all' mode verify: the previous verify step wrote its 1+num_spec
            # outputs to FULL block-table columns [prev .. prev+num_spec]; the
            # committed start is column prev + (num_accepted-1). Mirrors mamba2
            # selective_state_update's init_token_idx = num_accepted-1. Do NOT
            # clamp to p-1 — that compact-layout assumption corrupts the column.
            na = num_accepted[:s].clamp(min=1).to(torch.long)
            read_col = (blk_scheduled_prev[:s].to(torch.long) + (na - 1)).clamp(
                min=0, max=max_col
            )
        elif all_mode:
            # 'all' mode single-token decode: read the committed block, mamba2's
            # num_spec==0 path (mamba_mixer2.py:978-984). Full block table, so
            # clamp to max_col (NOT p-1, which aliased into the spec window and
            # left consecutive single-token steps reading a stale slot).
            read_col = blk_computed[:s].clamp(min=0, max=max_col).to(torch.long)
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
            # 'all' mode verify: verify position j -> full block-table column
            # (blk_scheduled + j); next step reads (blk_scheduled + num_accepted-1).
            write_col = (
                blk_scheduled[:s].unsqueeze(1).to(torch.long) + pos.unsqueeze(0)
            ).clamp(min=0, max=max_col)  # [s, p]
        elif all_mode:
            # 'all' mode single-token decode: write the committed state to the
            # scheduled block (mamba2 num_spec==0 output = gather(blk_scheduled)).
            # In-place within a block, new block at a boundary. Only pos 0 is valid
            # (seq_len 1) via write_ok, so broadcasting blk_scheduled is correct.
            # The old read_col+1 wrote one column ahead, so the next step's
            # read (blk_computed) saw a stale slot -> conv-state staleness ->
            # divergence after a couple of tokens.
            write_col = (
                blk_scheduled[:s].unsqueeze(1).to(torch.long).clamp(min=0, max=max_col)
            ).expand(s, p)  # [s, p]
        elif blk_computed is not None:
            write_col = (read_col.unsqueeze(1) + 1 + pos.unsqueeze(0)).clamp(
                min=0, max=p - 1
            )  # [s, p]
        else:
            write_col = pos.unsqueeze(0).expand(s, p)
        write_slot = state_indices_2d.gather(1, write_col)  # [s, p] block ids
        # Comprehensive addressing trace (M5 fix): for the owner layer's first N
        # verify steps dump read/write columns + the block-table row so the
        # cross-step slot handoff can be read off directly. Gated; default off.
        if os.environ.get("ZAYA_SPEC_DEBUG", "0") == "1" and getattr(
            type(self), "_spec_dbg_owner", None
        ) == id(self) and getattr(self, "_spec_dbg_count2", 0) < int(
            os.environ.get("ZAYA_SPEC_DEBUG_N", "40")
        ):
            self._spec_dbg_count2 = getattr(self, "_spec_dbg_count2", 0) + 1
            logging.getLogger(__name__).info(
                "ZAYA CCA addr[%d]: all_avail=%s seq_lens=%s num_acc=%s "
                "blk_sched=%s blk_sched_prev=%s blk_computed=%s read_col=%s "
                "write_col0=%s row0=%s",
                self._spec_dbg_count2 - 1,
                all_avail,
                seq_lens[:s].tolist(),
                num_accepted[:s].tolist() if num_accepted is not None else None,
                blk_scheduled[:s].tolist() if blk_scheduled is not None else None,
                blk_scheduled_prev[:s].tolist()
                if blk_scheduled_prev is not None else None,
                blk_computed[:s].tolist() if blk_computed is not None else None,
                read_col[:s].tolist(),
                write_col[:s, 0].tolist(),
                state_indices_2d[0, :12].tolist(),
            )
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
