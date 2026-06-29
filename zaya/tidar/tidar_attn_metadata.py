"""TiDAR attention-mask metadata + backend plumbing (serving step 3).

The kernel-level mask hooks are already built and GPU-validated:
  * Route A  — attn_hip ``flash_prefill(..., mask_bias=)`` (square [L,L] bias) — gpu_validate Part C.
  * Route B  — triton_attn ``unified_attention(qq_bias=)`` gate (paged) — gpu_validate Part D.
Both consume tensors emitted verbatim by ``tidar_mask.py``.

What was MISSING (and this module supplies) is the connective tissue between the per-step serving
layout and the *standard* ``self.attn`` call in ``ZayaAttention.forward`` (zaya.py:216 —
``attn_output = self.attn(q, k, v)``). On RDNA4 that ``Attention`` dispatches to the triton_attn
backend, whose ``forward`` calls ``unified_attention`` WITHOUT a ``qq_bias`` (only ``tree_attn.py``
passes one today). So a per-step TiDAR mask has no path into the real attention unless something:

  1. builds the mask for the current ``[prefix | S | R_0..R_{B-1}]`` block  (``build_tidar_mask_meta``),
  2. carries it where the backend can reach it without re-plumbing the runner  (the module-level
     *active-mask carrier* — ``set_active_tidar_mask`` / ``get_active_tidar_mask``), and
  3. makes the backend read it and pass ``qq_bias`` into ``unified_attention``  (``wrap_unified_attention``
     / ``install_tidar_attn_hook``).

This mirrors how other per-step state is threaded on this stack: a module-level static object the
captured graph / backend reads, updated in place each step (the §31g static-address pattern). It is
deliberately decoupled from vLLM's frozen ``ForwardContext`` so it needs no runner patch.

CUDAGRAPH NOTE: for FULL-capture (step 6 / §5) the carrier's ``qq_bias`` must live at a *fixed
address* and be updated in place; ``update_active_tidar_mask_`` does that (copies into the existing
buffer instead of rebinding). The eager builder allocates fresh each call — fine for correctness /
this step's validation; the capture step swaps to the in-place updater.

Single source of truth for the mask math stays ``tidar_mask.py`` — this module only *places* and
*routes* those tensors; it never redefines the predicate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tidar_mask import (
    MaskDescriptor,
    additive_bias,
    build_allow_matrix,
    square_additive_bias,
)


@dataclass
class TiDARMaskMeta:
    """Per-step TiDAR mask, both backend forms, built once from the layout.

    descriptor   — the layout ints (single source of truth, ``tidar_mask.MaskDescriptor``).
    qq_bias      — [q_len, q_len] 0/-inf over the query-query (new-token) key columns. This is the
                   slice the triton_attn Route-B gate consumes: prefix keys stay strictly causal in
                   the kernel (correct for TiDAR — the whole prefix precedes the block), and qq_bias
                   alone defines allow/deny among the new tokens.
    square_bias  — [L, L] 0/-inf over the FULL window (L = prefix_len + q_len), for the contiguous
                   self-attention attn_hip Route-A kernel (``mask_bias``). None unless requested.
    block_len    — B; q_len = B*(1+B).
    prefix_len   — committed/cached KV length this step.
    """

    descriptor: MaskDescriptor
    qq_bias: torch.Tensor
    square_bias: torch.Tensor | None
    block_len: int
    prefix_len: int

    @property
    def q_len(self) -> int:
        return self.descriptor.q_len

    @property
    def kv_len(self) -> int:
        return self.descriptor.kv_len


def build_tidar_mask_meta(
    prefix_len: int,
    block_len: int,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
    replica_offset: int = 0,
    sampling_causal: bool = True,
    want_square: bool = False,
) -> TiDARMaskMeta:
    """Build the per-step TiDAR mask for ``[prefix | S | R_0..R_{B-1}]``.

    Shapes are a deterministic function of (prefix_len, block_len) only → cudagraph-static for a
    fixed block_len. ``replica_offset`` / ``sampling_causal`` are the two §7.1 flags pinned by the
    passing β=1 coherence gate (defaults match it); exposed so the gate / a future re-pin can vary
    them without editing call sites.

    The qq_bias is the new-token×new-token slice ``additive_bias(d)[:, prefix_len:]`` — exactly the
    tensor gpu_validate Part D drives the Route-B kernel with, so the wired path reuses the validated
    construction verbatim.
    """
    d = MaskDescriptor(
        prefix_len=prefix_len,
        block_len=block_len,
        replica_offset=replica_offset,
        sampling_causal=sampling_causal,
    )
    full_bias = additive_bias(d, dtype=dtype, device=device)  # [q_len, kv_len]
    qq_bias = full_bias[:, prefix_len : d.kv_len].contiguous()  # [q_len, q_len]
    square_bias = (
        square_additive_bias(d, dtype=dtype, device=device) if want_square else None
    )
    return TiDARMaskMeta(
        descriptor=d,
        qq_bias=qq_bias,
        square_bias=square_bias,
        block_len=block_len,
        prefix_len=prefix_len,
    )


def tidar_allow_matrix(meta: TiDARMaskMeta, device=None) -> torch.Tensor:
    """Boolean [q_len, kv_len] ground-truth allow matrix for this step's mask (validation/debug)."""
    return build_allow_matrix(meta.descriptor, device=device)


# ----------------------------------------------------------------------------------------------
# Active-mask carrier — the per-step bridge from the metadata builder to the standard attention
# backend. Set before the model forward, cleared after; the backend hook reads it. A plain decode
# step (no TiDAR block) leaves it None → the hook is a no-op and the kernel path is byte-identical
# to stock. Single-sequence for now (matches the validation + the β=1 coherence gate); batched-decode
# (one qq_bias per sequence) is a step-4 concern once the loop is on the real cca.py KV path.
# ----------------------------------------------------------------------------------------------
_ACTIVE_TIDAR_MASK: TiDARMaskMeta | None = None


def set_active_tidar_mask(meta: TiDARMaskMeta | None) -> None:
    """Install (or clear, with None) the mask the backend hook injects this forward."""
    global _ACTIVE_TIDAR_MASK
    _ACTIVE_TIDAR_MASK = meta


def get_active_tidar_mask() -> TiDARMaskMeta | None:
    return _ACTIVE_TIDAR_MASK


def clear_active_tidar_mask() -> None:
    set_active_tidar_mask(None)


def update_active_tidar_mask_(meta: TiDARMaskMeta) -> None:
    """Cudagraph-static update: copy ``meta``'s qq_bias INTO the already-active buffer (fixed
    address) instead of rebinding it, so a captured graph that closed over the active mask sees the
    new step's values. Falls back to a plain set when nothing is active yet (first step / warmup).
    The descriptor/layout must be identical (same block_len, prefix_len) — capture sizes are fixed.
    """
    global _ACTIVE_TIDAR_MASK
    cur = _ACTIVE_TIDAR_MASK
    if (
        cur is None
        or cur.qq_bias.shape != meta.qq_bias.shape
        or cur.qq_bias.device != meta.qq_bias.device
    ):
        set_active_tidar_mask(meta)
        return
    cur.qq_bias.copy_(meta.qq_bias)
    if cur.square_bias is not None and meta.square_bias is not None:
        cur.square_bias.copy_(meta.square_bias)


# ----------------------------------------------------------------------------------------------
# Backend hook — wrap the standard triton_attn backend's ``unified_attention`` so it injects the
# active TiDAR qq_bias. Kept as a wrapper (not a fork of the ~900-line backend): triton_attn.py
# binds ``unified_attention`` in its own module namespace at import, so rebinding that name routes
# the backend's single call through us. Null-safe: no active mask, or a caller that already supplied
# qq_bias (tree_attn), passes straight through unchanged.
# ----------------------------------------------------------------------------------------------
def wrap_unified_attention(unified_attention):
    """Return a drop-in for ``unified_attention`` that injects the active TiDAR qq_bias.

    When a TiDAR mask is active AND the caller did not already pass ``qq_bias``, we forward the
    active ``qq_bias``; otherwise we call through untouched (byte-identical). This is the exact hook
    gpu_validate Part E exercises and that ``install_tidar_attn_hook`` installs onto the real
    backend for serving.
    """

    def _wrapped(*args, **kwargs):
        if kwargs.get("qq_bias", None) is None:
            meta = get_active_tidar_mask()
            if meta is not None:
                kwargs["qq_bias"] = meta.qq_bias
        return unified_attention(*args, **kwargs)

    _wrapped.__wrapped__ = unified_attention
    _wrapped.__tidar_hook__ = True
    return _wrapped


def install_tidar_attn_hook() -> bool:
    """Monkeypatch the live triton_attn backend so the standard ``self.attn`` path honors the active
    TiDAR mask. Idempotent; returns True if installed (or already installed), False if the backend
    isn't importable (e.g. CPU-only env). For serving (step 4); the correctness of the wrap itself is
    gated by gpu_validate Part E, which exercises ``wrap_unified_attention`` directly.
    """
    try:
        import vllm.v1.attention.backends.triton_attn as T
    except Exception:
        return False
    if getattr(T.unified_attention, "__tidar_hook__", False):
        return True
    T.unified_attention = wrap_unified_attention(T.unified_attention)
    return True
