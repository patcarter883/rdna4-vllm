"""vLLM routing shim for the native attn_decode HIP kernel (gfx1201).

`maybe_hip_decode(impl, query, kv_cache, attn_metadata, output, output_scale)` is called from the
patched TritonAttentionImpl.forward, right before `unified_attention(...)`. It returns True (and
writes `output`) iff it routed a **pure-decode bf16** batch through
`torch.ops.attn_decode.flash_decode_paged`; otherwise False → the caller falls through to the
Triton path. Conservative: any unsupported feature (quantized KV, alibi, sinks, softcap, fp8 output,
unsupported head_dim, or a non-decode/mixed batch) bails to Triton.

vLLM KV layout: kv_cache is [num_blocks, 2, block_size, num_kv_heads, head_size]; K = [:,0],
V = [:,1] are non-contiguous views with block stride 2*block_size*num_kv_heads*head_size — passed to
the kernel via kv_block_stride (the generalization validated in attn_decode_parity.py check_paged_vllm).
"""
from __future__ import annotations

import torch

# torch.ops.attn_decode.* are registered when the package's op.py is imported (the patch does
# `from ...attn_decode import op` before importing this module), so no import needed here.

_SUPPORTED_HEAD_DIM = (64, 128, 256)


def _is_quantized(kv_cache_dtype) -> bool:
    s = str(kv_cache_dtype)
    return ("fp8" in s) or ("int8" in s) or ("uint8" in s) or ("e4m3" in s) or ("e5m2" in s)


def maybe_hip_decode(impl, query, kv_cache, attn_metadata, output, output_scale) -> bool:
    md = attn_metadata
    # --- gate: only the plain bf16 causal decode path is supported in v0 ---
    if output_scale is not None:
        return False
    if getattr(impl, "alibi_slopes", None) is not None:
        return False
    if getattr(impl, "sinks", None) is not None:
        return False
    if getattr(impl, "logits_soft_cap", 0):
        return False
    if _is_quantized(getattr(impl, "kv_cache_dtype", "auto")):
        return False
    if getattr(impl, "_is_per_token_head_quant", False):
        return False
    if impl.head_size not in _SUPPORTED_HEAD_DIM:
        return False
    if query.dtype != torch.bfloat16 or kv_cache.dtype != torch.bfloat16:
        return False
    if getattr(md, "max_query_len", None) != 1:        # pure decode only (one query token per seq)
        return False
    if kv_cache.dim() != 5 or kv_cache.shape[1] != 2:  # [num_blocks, 2, block_size, kv_heads, hd]
        return False

    n = md.num_actual_tokens
    _, _, block_size, kv_heads, hd = kv_cache.shape
    if hd != impl.head_size:                            # e.g. scale_pad on a quant cache
        return False

    k_cache, v_cache = kv_cache.unbind(1)              # non-contiguous views
    kv_block_stride = 2 * block_size * kv_heads * hd
    q = query[:n].reshape(n, impl.num_heads, impl.head_size).contiguous()

    seqlens = md.seq_lens
    if seqlens.dtype != torch.int32:
        seqlens = seqlens.to(torch.int32)
    bt = md.block_table
    if bt.dtype != torch.int32:
        bt = bt.to(torch.int32)

    sw = impl.sliding_window[0]                         # vLLM stores (sliding_window-1, .)
    sw_hip = (sw + 1) if sw >= 0 else 0                 # attend most-recent sw_hip keys; 0 = unlimited

    out = torch.ops.attn_decode.flash_decode_paged(
        q, k_cache, v_cache, bt, seqlens, float(impl.scale), int(sw_hip), int(kv_block_stride))
    output[:n].reshape(n, impl.num_heads, impl.head_size).copy_(out)
    return True
