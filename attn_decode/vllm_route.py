"""vLLM routing shim for the native HIP attention kernels (gfx1201).

`maybe_hip_attention(impl, layer, query, kv_cache, attn_metadata, output, output_scale)` is called
from the patched TritonAttentionImpl.forward, right before `unified_attention(...)`. It returns True
(and writes `output`) iff it routed the batch through a native HIP kernel; else False → the caller
falls through to Triton. Dispatch on max_query_len × KV dtype:
  decode  (mql==1)  bf16 -> attn_decode.flash_decode_paged          fp8 -> flash_decode_paged_fp8
  prefill (mql>1)   bf16 -> attn_prefill_paged.flash_prefill_paged  fp8 -> flash_prefill_paged_fp8
attn_prefill_paged is varlen (handles any per-seq q_len, incl. q_len=1 decode seqs mixed into a
prefill batch). Conservative: alibi/sinks/softcap/fp8-output/per-token-head-quant/unsupported
head_dim/non-bf16-Q all bail to Triton.

KV layout: kv_cache [num_blocks, 2, block_size, kv_heads, head_size]; K=[:,0], V=[:,1] are
non-contiguous views with block stride 2*block_size*kv_heads*head_size -> kv_block_stride. fp8 KV is
OCP e4m3 (per-tensor descale from layer._k_scale / layer._v_scale, the FP8_PER_TENSOR path).
"""
from __future__ import annotations

import torch

_DECODE_HEAD_DIM = (64, 128, 256)
_PREFILL_HEAD_DIM = (64, 128)


def _is_fp8_kv(impl, kv_cache) -> bool:
    s = str(getattr(impl, "kv_cache_dtype", "auto"))
    if any(t in s for t in ("fp8", "e4m3", "e5m2")):
        return True
    return "float8" in str(kv_cache.dtype)


def _scalar(t) -> float:
    return float(t.reshape(-1)[0].item())


_DBG_SEEN = set()


def _dbg(msg: str) -> None:
    if msg not in _DBG_SEEN:
        _DBG_SEEN.add(msg)
        print("[attn_hip route]", msg, flush=True)


def maybe_hip_attention(impl, layer, query, kv_cache, attn_metadata, output, output_scale) -> bool:
    md = attn_metadata
    if output_scale is not None:
        return False
    if getattr(impl, "alibi_slopes", None) is not None or getattr(impl, "sinks", None) is not None:
        return False
    if getattr(impl, "logits_soft_cap", 0):
        return False
    if getattr(impl, "_is_per_token_head_quant", False):     # per-token-head descale unsupported
        _dbg("BAIL: per-token-head quant")
        return False
    # Q is bf16 normally; under FP8_PER_TENSOR vLLM quantizes Q to fp8 too (dequant'd below).
    if query.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        _dbg(f"BAIL: q_dtype {query.dtype}")
        return False
    if kv_cache.dim() != 5 or kv_cache.shape[1] != 2:
        _dbg(f"BAIL: kv_shape {tuple(kv_cache.shape)}")
        return False
    if impl.head_size not in _DECODE_HEAD_DIM:
        return False

    mql = getattr(md, "max_query_len", None)
    if mql is None:
        return False

    is_fp8 = _is_fp8_kv(impl, kv_cache)
    if not is_fp8 and kv_cache.dtype != torch.bfloat16:
        return False

    n = md.num_actual_tokens
    _, _, block_size, kv_heads, hd = kv_cache.shape
    if hd != impl.head_size:                                 # e.g. scale_pad on a quant cache
        _dbg(f"BAIL: hd {hd} != head_size {impl.head_size} (fp8={is_fp8})")
        return False
    kv_block_stride = 2 * block_size * kv_heads * hd

    k_cache, v_cache = kv_cache.unbind(1)                    # non-contiguous views
    if is_fp8:
        fp8_dtype = getattr(impl, "fp8_dtype", torch.float8_e4m3fn)
        if k_cache.dtype != fp8_dtype:
            k_cache = k_cache.view(fp8_dtype)
            v_cache = v_cache.view(fp8_dtype)
        try:
            k_descale = _scalar(layer._k_scale)
            v_descale = _scalar(layer._v_scale)
        except Exception as e:
            _dbg(f"BAIL: fp8 descale extract failed: {e!r}")
            return False

    if query.dtype == torch.bfloat16:
        q = query[:n].reshape(n, impl.num_heads, impl.head_size).contiguous()
    else:
        # fp8 Q (FP8_PER_TENSOR): dequant with q_descale -> bf16 (e4m3 ⊂ bf16, lossless). The kernels
        # take bf16 Q + fp8 K/V; q_descale into Q + k_descale into the score reconstructs q_real·k_real.
        if not is_fp8:
            return False
        try:
            q_descale = _scalar(layer._q_scale)
        except Exception as e:
            _dbg(f"BAIL: fp8 q_descale extract failed: {e!r}")
            return False
        q = (query[:n].to(torch.float32) * q_descale).to(torch.bfloat16).reshape(
            n, impl.num_heads, impl.head_size).contiguous()
    seqlens = md.seq_lens
    if seqlens.dtype != torch.int32:
        seqlens = seqlens.to(torch.int32)
    bt = md.block_table
    if bt.dtype != torch.int32:
        bt = bt.to(torch.int32)
    sw = impl.sliding_window[0]
    sw_hip = (sw + 1) if sw >= 0 else 0
    scale = float(impl.scale)

    if mql == 1:
        if is_fp8:
            _dbg("ROUTED decode fp8")
            out = torch.ops.attn_decode.flash_decode_paged_fp8(
                q, k_cache, v_cache, bt, seqlens, scale, k_descale, v_descale, int(sw_hip),
                int(kv_block_stride))
        else:
            _dbg("ROUTED decode bf16")
            out = torch.ops.attn_decode.flash_decode_paged(
                q, k_cache, v_cache, bt, seqlens, scale, int(sw_hip), int(kv_block_stride))
    else:
        if impl.head_size not in _PREFILL_HEAD_DIM:
            _dbg(f"BAIL: prefill head_size {impl.head_size} not in 64/128")
            return False
        cu = md.query_start_loc
        if cu.dtype != torch.int32:
            cu = cu.to(torch.int32)
        if is_fp8:
            _dbg("ROUTED prefill fp8")
            out = torch.ops.attn_prefill_paged.flash_prefill_paged_fp8(
                q, k_cache, v_cache, bt, cu, seqlens, scale, k_descale, v_descale, 1, int(sw_hip),
                int(mql), int(kv_block_stride))
        else:
            _dbg("ROUTED prefill bf16")
            out = torch.ops.attn_prefill_paged.flash_prefill_paged(
                q, k_cache, v_cache, bt, cu, seqlens, scale, 1, int(sw_hip), int(mql),
                int(kv_block_stride))

    output[:n].reshape(n, impl.num_heads, impl.head_size).copy_(out)
    return True


maybe_hip_decode = maybe_hip_attention  # back-compat alias
