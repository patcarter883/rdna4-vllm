"""TiDAR proposer / model-runner integration (serving step-3 carrier+hook + step-4 evict wiring).

This is the CONNECTIVE module that makes the already-validated TiDAR pieces drive a real per-step
decode loop against the live converted checkpoint. It implements *no* new mask math and *no* new
rollback math — it only orchestrates the existing, separately-pinned primitives:

  * mask carrier + backend hook ........ ``tidar_attn_metadata.py``  (step 3, gpu_validate Part D/E)
  * the TiDAR mask predicate ............ ``tidar_mask.py``           (single source of truth, 10/10)
  * β-rejection accept length .......... ``tidar_loop.beta_verify``  (step 2/3, β=1 lossless)
  * conv/prev_hs evict-on-reject ....... ``cca.py:_decode_verify_spec`` (step 4 fold; read accepted
                                          column == evict, PROVEN max|Δ|=0 by cca_evict_gate.py)

Design contract (why this is additive / null-safe):

  * The hook (``install_tidar_attn_hook``) rebinds ``triton_attn.unified_attention`` to a wrapper
    that injects the *active* TiDAR ``qq_bias`` ONLY when one is set on the module-level carrier AND
    the caller didn't already pass ``qq_bias``. With the carrier cleared (or TiDAR serving off) the
    wrapper calls through byte-identically — a plain decode step is the stock path (gpu_validate
    Part E ``check_wired_regression_nomask``: max|Δ|=0). So installing the hook unconditionally at
    model load is safe; it costs one ``None`` carrier read per forward and nothing else.

  * The per-step mask is set BEFORE the model forward and cleared AFTER. We deliberately do this with
    the module-level carrier rather than threading a field through vLLM's frozen ``ForwardContext`` —
    the whole point of ``tidar_attn_metadata`` was to avoid a runner patch. ``run_block`` is the
    context manager that does set→(yield)→clear, so a caller cannot leak an active mask into the next
    forward.

  * For the eager correctness pass we use the FRESH-ALLOC builder (``build_tidar_mask_meta``). The
    in-place ``update_active_tidar_mask_`` is reserved for the later §31g FULL-cudagraph capture
    (fixed-address static buffer) and is intentionally NOT used here.

SCOPE (matches the coherence gate + design Parts D/E): SINGLE SEQUENCE. Batched decode (one qq_bias
per sequence) is a downstream §5 concern once the loop is on the real paged cca.py KV path.

§7.6 BOUNDARY (the one un-pinned off-by-one): building the FUSED single forward over
``[prefix | S | R_0..R_{block_len-1}]`` as model input requires ``position_ids`` for the mask-replica
tokens (design §7.6 — mask positions sit at ``prefix_len + block_len + j``; the exact convention the
diffusion FT trained on is NOT yet confirmed on-device). The standalone loop (``tidar_loop`` /
``coherence_gate``) SIDESTEPS this by re-deriving the next-block drafts from a *fresh causal forward*
each step (contiguous positions ``range(L+B)``) instead of the fused replica block. This module keeps
that same sidestep for the β=1 correctness path: ``run_block`` carries the qq_bias for a contiguous
``[prefix | S]`` (or ``[prefix | mask*B]``) block whose positions are unambiguous; it does NOT emit
replica position_ids. Wiring the fused replica forward into the live runner is gated on §7.6 being
pinned on the checkpoint — see ``replica_block_positions`` which raises rather than guess.
"""

from __future__ import annotations

import contextlib
import os

from tidar_attn_metadata import (
    TiDARMaskMeta,
    build_tidar_mask_meta,
    clear_active_tidar_mask,
    install_tidar_attn_hook,
    set_active_tidar_mask,
)
from tidar_loop import beta_verify


# --------------------------------------------------------------------------------------------- #
# Serve-time enable flag — a plain decode serve is byte-unchanged unless this is set.
# --------------------------------------------------------------------------------------------- #
ENV_BLOCK_LEN = "VLLM_TIDAR_BLOCK_LEN"


def tidar_block_len() -> int | None:
    """The configured TiDAR block_len (B) from ``VLLM_TIDAR_BLOCK_LEN``, or None if TiDAR serving is
    off. A missing / empty / non-positive value ⇒ None ⇒ the stock decode path (hook inert)."""
    raw = os.environ.get(ENV_BLOCK_LEN, "").strip()
    if not raw:
        return None
    try:
        b = int(raw)
    except ValueError:
        return None
    return b if b >= 1 else None


def tidar_serving_enabled() -> bool:
    return tidar_block_len() is not None


# --------------------------------------------------------------------------------------------- #
# Install point — additive, idempotent, null-safe. Call once at model load from the overlay.
# --------------------------------------------------------------------------------------------- #
def maybe_install_tidar_hook() -> bool:
    """Install the triton_attn mask hook so the standard ``self.attn`` path can honor an active TiDAR
    mask. ALWAYS safe to call: with no active mask on the carrier the wrapper is byte-identical to
    stock (gpu_validate Part E). Returns True if the hook is installed (or already was), False if the
    backend isn't importable (CPU-only env). Idempotent.

    We install unconditionally (not gated on the env flag) so that toggling
    ``VLLM_TIDAR_BLOCK_LEN`` at request time needs no re-install; the *carrier* — not the hook's
    presence — is what makes a step a TiDAR step. An installed-but-inert hook is a no-op.
    """
    return install_tidar_attn_hook()


# --------------------------------------------------------------------------------------------- #
# Per-step proposer — orchestrates the validated primitives for one decode step.
# --------------------------------------------------------------------------------------------- #
class TidarProposer:
    """Single-sequence TiDAR proposer. Holds the block_len and drives the per-step carrier set/clear
    + β=1 verify/commit, reusing the pinned primitives verbatim.

    This object owns NONE of the math. ``run_block`` places the mask on the carrier (the step-3
    bridge); ``verify_commit`` calls ``beta_verify`` (the step-2/3 sampler); ``evict_contract``
    documents how the accepted length routes to ``cca.py:_decode_verify_spec`` (the step-4 fold) —
    the runner reads conv/prev_hs column ``num_accepted-1`` to realize the evict, no truncation,
    no new rollback math (proven by ``cca_evict_gate.py``).
    """

    def __init__(self, block_len: int, *, replica_offset: int = 0, sampling_causal: bool = True):
        assert block_len >= 1
        self.block_len = block_len
        # The two §7.1 flags pinned by the passing β=1 coherence gate; defaults match it.
        self.replica_offset = replica_offset
        self.sampling_causal = sampling_causal

    # -- step 3: carrier set/clear around the model forward -------------------------------------
    def build_mask(self, prefix_len: int, *, device=None, dtype=None, want_square: bool = False) -> TiDARMaskMeta:
        """Fresh-alloc the per-step mask for ``[prefix | S | R_0..R_{B-1}]`` (eager correctness pass).

        Reuses ``build_tidar_mask_meta`` verbatim — the same builder gpu_validate Part E drives the
        kernel with — so this path inherits Part E's losslessness. ``want_square`` only for the
        contiguous attn_hip Route-A kernel (not used by the triton_attn carrier path).
        """
        import torch

        return build_tidar_mask_meta(
            prefix_len,
            self.block_len,
            device=device,
            dtype=dtype or torch.float32,
            replica_offset=self.replica_offset,
            sampling_causal=self.sampling_causal,
            want_square=want_square,
        )

    @contextlib.contextmanager
    def run_block(self, prefix_len: int, *, device=None, dtype=None, want_square: bool = False):
        """Context manager: set the active TiDAR mask for this forward, yield the meta, clear after.

        Use as::

            with proposer.run_block(prefix_len) as meta:
                logits = model_forward(...)   # the hooked self.attn injects meta.qq_bias

        The carrier is ALWAYS cleared on exit (even on exception) so a forward can never leak an
        active mask into the next (a stale mask would silently corrupt a plain decode). This is the
        runner-side set-before / clear-after boundary the plan calls for, done without a runner patch.
        """
        meta = self.build_mask(prefix_len, device=device, dtype=dtype, want_square=want_square)
        set_active_tidar_mask(meta)
        try:
            yield meta
        finally:
            clear_active_tidar_mask()

    @contextlib.contextmanager
    def run_with_mask(self, meta: TiDARMaskMeta):
        """Like ``run_block`` but with a caller-supplied meta (e.g. a pre-built or reused mask)."""
        set_active_tidar_mask(meta)
        try:
            yield meta
        finally:
            clear_active_tidar_mask()

    # -- step 2/3: β=1 verify/commit ------------------------------------------------------------
    def verify_commit(self, drafts, p_ar_logits, p_diff_logits=None, beta: float = 1.0):
        """Return ``(num_accepted, bonus_token)`` for this block via the pinned ``beta_verify``.

        ``p_ar_logits`` is the ``[block_len+1, V]`` AR slice (rows predict positions L..L+B). β=1
        ⇒ pure-AR ⇒ the committed stream is lossless by construction (the coherence gate's GATE B).
        No new sampler logic — this forwards to ``tidar_loop.beta_verify`` unchanged.
        """
        return beta_verify(drafts, p_ar_logits, p_diff_logits=p_diff_logits, beta=beta)

    # -- step 4: evict routing (documentation anchor; the math lives in cca.py) -----------------
    @staticmethod
    def evict_contract(num_accepted: int) -> int:
        """The rollback-column the runner reads to realize the TiDAR evict-on-reject.

        cca.py:_decode_verify_spec processes the whole ``(1 + num_spec)`` candidate window, writes
        the conv window + prev_hs ENDING at each candidate j to its rollback slot, and the NEXT step
        reads slot ``(num_accepted - 1)`` = the accepted-prefix end. Appending the rejected tail then
        reading the accepted column IS the evict — no physical truncation. This helper just names
        that column (``num_spec`` maps to ``block_len``); the actual scatter/read is cca.py's,
        proven equal to a from-scratch recompute by ``cca_evict_gate.py`` (max|Δ|=0). It exists so a
        caller/test can assert the index without re-deriving the rollback math.
        """
        # num_accepted is clamp(min=1) in cca.py (mamba2 selective_state_update init_token_idx
        # parity); mirror that so the documented column matches the kernel.
        return max(num_accepted, 1) - 1


def make_proposer_from_env() -> TidarProposer | None:
    """Build a ``TidarProposer`` from ``VLLM_TIDAR_BLOCK_LEN``, or None if TiDAR serving is off."""
    b = tidar_block_len()
    return None if b is None else TidarProposer(b)
