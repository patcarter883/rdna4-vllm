"""Python entry point for the native flash-DECODE attention HIP op (gfx1201).

Loads the AOT-compiled attn_decode_C extension and registers a fake/meta impl so torch.compile
treats the op as opaque (no graph-break). Framework-agnostic: minisgl's decode path and vLLM's
backend can both call torch.ops.attn_decode.flash_decode after swapping their Triton decode attn.

v0: bf16 dense (non-paged) decode, GQA, optional sliding window.
    q:[B, num_q_heads, head_dim]  k/v:[B, Skv, num_kv_heads, head_dim]  -> [B, num_q_heads, head_dim]
"""
import glob
import os

import torch

_so = glob.glob(os.path.join(os.path.dirname(__file__), "attn_decode_C*.so"))
if not _so:
    raise ImportError(
        "attn_decode_C extension not built; run `GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`"
    )
torch.ops.load_library(_so[0])


@torch.library.register_fake("attn_decode::flash_decode")
def _flash_decode_fake(q, k, v, scale, sliding_window):
    return torch.empty_like(q)


@torch.library.register_fake("attn_decode::flash_decode_paged")
def _flash_decode_paged_fake(q, k_cache, v_cache, block_table, context_lens, scale, sliding_window):
    return torch.empty_like(q)


@torch.library.register_fake("attn_decode::flash_decode_paged_fp8")
def _flash_decode_paged_fp8_fake(q, k_cache, v_cache, block_table, context_lens, scale, k_descale,
                                 v_descale, sliding_window):
    return torch.empty_like(q)


flash_decode = torch.ops.attn_decode.flash_decode
flash_decode_paged = torch.ops.attn_decode.flash_decode_paged
flash_decode_paged_fp8 = torch.ops.attn_decode.flash_decode_paged_fp8

__all__ = ["flash_decode", "flash_decode_paged", "flash_decode_paged_fp8"]
