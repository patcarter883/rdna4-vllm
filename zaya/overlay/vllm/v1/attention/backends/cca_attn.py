# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)

if TYPE_CHECKING:
    # The TiDAR per-step structured mask (zaya/tidar/tidar_attn_metadata.py). Imported only for
    # typing — the serving overlay lives outside the vLLM package and is optional at runtime.
    from tidar_attn_metadata import TiDARMaskMeta


class CCAAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "CCA_ATTN"

    @staticmethod
    def get_builder_cls() -> type["CCAAttentionMetadataBuilder"]:
        return CCAAttentionMetadataBuilder


@dataclass
class CCAAttentionMetadata(BaseMambaAttentionMetadata):
    # TiDAR serving (§4.1): the structured causal+block-bidirectional mask for THIS step's
    # ``[prefix | S | R_0..R_{B-1}]`` query block. Built by tidar_attn_metadata.build_tidar_mask_meta
    # and attached here so it travels alongside the CCA conv-state metadata; the standard self.attn
    # backend reads it via the active-mask carrier (it is NOT applied inside CCA — CCA is only a QKV
    # producer, §1). Default None ⇒ a plain (non-TiDAR) decode: zero behavioural change.
    tidar_mask: "TiDARMaskMeta | None" = None


class CCAAttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[CCAAttentionMetadata]
):
    metadata_cls = CCAAttentionMetadata
    supports_update_block_table: bool = False
