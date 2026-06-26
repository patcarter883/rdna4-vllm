"""Python entry for the native PAGED/chunked-prefill flash-attention HIP op (gfx1201).

torch.ops.attn_prefill_paged.flash_prefill_paged: extend/chunked prefill — Q = new tokens (packed
varlen via cu_seqlens_q), K/V from the paged cache (block_table), causal with prefix offset
(query global pos = prefix_len + local). Makes prefill Triton-free for chunked-prefill + prefix-cache
serving (dense attn_hip only covers cold prefill).
    q:[total_q, Hq, D]  k/v_cache:[num_blocks, bs, Hk, D]  block_table:[S, max_blocks]
    cu_seqlens_q:[S+1] int32  context_lens:[S] int32  -> [total_q, Hq, D]
"""
import glob
import os

import torch

_so = glob.glob(os.path.join(os.path.dirname(__file__), "attn_prefill_paged_C*.so"))
if not _so:
    raise ImportError(
        "attn_prefill_paged_C not built; run `GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`"
    )
torch.ops.load_library(_so[0])


@torch.library.register_fake("attn_prefill_paged::flash_prefill_paged")
def _fake(q, k_cache, v_cache, block_table, cu_seqlens_q, context_lens, scale, causal,
          sliding_window, max_seqlen_q, kv_block_stride=0):
    return torch.empty_like(q)


@torch.library.register_fake("attn_prefill_paged::flash_prefill_paged_fp8")
def _fake_fp8(q, k_cache, v_cache, block_table, cu_seqlens_q, context_lens, scale, k_descale,
              v_descale, causal, sliding_window, max_seqlen_q, kv_block_stride=0):
    return torch.empty_like(q)


flash_prefill_paged = torch.ops.attn_prefill_paged.flash_prefill_paged
flash_prefill_paged_fp8 = torch.ops.attn_prefill_paged.flash_prefill_paged_fp8

__all__ = ["flash_prefill_paged", "flash_prefill_paged_fp8"]
