"""TiDAR structured attention mask — backend-neutral.

This is the single source of truth for the TiDAR mask, consumed identically by every
attention path so the math is defined once:

  (1) ``allow_matrix``     — boolean [q_len, kv_len] allow/deny. The dense reference /
                             ground truth used by unit tests and the SDPA correctness check.
  (2) ``additive_bias``    — float [q_len, kv_len], 0.0 where allowed, ``-inf`` where denied.
                             Added to the QK^T scores BEFORE softmax. Used identically by:
                               * triton_attn / SDPA (Route B): pass as attn bias.
                               * the attn_hip rocwmma kernel: add into the fp32 ``smem_S``
                                 scores in the per-row softmax loop (attn_kernels.hip:173-180,
                                 right where ``causal``/``sliding_window`` set ``s = -1e30f``).
  (3) ``MaskDescriptor``   — the compact layout integers, for the attn_hip *inline-predicate*
                             perf path (evaluate attend/deny from (q_pos, k_pos) with no
                             [q,kv] tensor materialized — cudagraph-clean, zero extra memory,
                             exactly how the kernel does ``causal`` today).

Because (1)/(2)/(3) are generated from one construction, the additive-bias path (cheap, stand
up first) and the inline-predicate path (fast, attn_hip) are guaranteed to agree, and the
parity test asserts it.

----------------------------------------------------------------------------------------------
LAYOUT  (one sequence; the prefix is the already-committed KV, cached & not re-queried)

  query/key new-token region, q_len = block_len * (1 + block_len):

    [ S : sampling block        | R_0 | R_1 | ... | R_{B-1} ]     (B = block_len)
      ^ B drafts to AR-verify     ^ B mask-token replicas, one per acceptance length

  - S (sampling block): the B draft tokens proposed last step. Verified left->right
    against p_AR this step (causal within S). These tokens are real keys (their KV commits
    on accept).
  - R_r (replica r): B mask tokens that pre-draft the NEXT block *conditioned on r drafts of
    S having been accepted*. Computing all B replicas in one forward is TiDAR's "pre-draft
    conditioned on every acceptance length" trick; after the verify yields accepted length
    k, the runner SELECTS replica R_k's outputs as the next block's drafts.

  Keys = [ prefix (<= max_seq_len, cached) | S | R_0 | ... | R_{B-1} ].

PREDICATE  (q attends k iff allow[q,k]):
  - prefix query     -> causal over prefix.
  - S[i] query       -> prefix (all) + S[j] for j<=i (causal).               NOT any R_r.
  - R_r[m] query     -> prefix (all) + S[j] for j < r (the r accepted drafts)
                        + R_r[m'] for all m' (bidirectional WITHIN its own replica).
                        NOT other replicas, NOT S[j>=r].

UNCERTAINTY (see docs/zaya/tidar-serving-design.md §7.1 — the paper gives no reproducible
mask_mod; this encodes the structural reading). The genuinely unresolved choices are FLAGS,
pinned by unit tests, to be confirmed against the conversion checkpoint:
  - ``replica_drafts(r) = r + replica_offset`` : whether replica r is conditioned on r or r+1
    accepted drafts. ``replica_offset=0`` -> replicas cover acceptance 0..B-1.
  - ``sampling_causal`` : S self-attention causal (paper: "clean tokens ... causally") vs bidir.
  - ``mask_sees_prefix`` : kept True (paper: mask tokens "along with the prefix").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

NEG_INF = float("-inf")


@dataclass(frozen=True)
class MaskDescriptor:
    """Compact layout integers for the inline-predicate (attn_hip perf) path.

    All positions are GLOBAL key-axis indices (prefix occupies [0, prefix_len)). The new
    region occupies [prefix_len, prefix_len + q_len). The query-axis index q in [0, q_len)
    maps to global key position prefix_len + q (queries are the new tokens).
    """

    prefix_len: int
    block_len: int
    replica_offset: int = 0
    sampling_causal: bool = True
    mask_sees_prefix: bool = True

    @property
    def q_len(self) -> int:
        return self.block_len * (1 + self.block_len)

    @property
    def kv_len(self) -> int:
        return self.prefix_len + self.q_len

    # --- segment boundaries on the NEW-token axis (0-based within the new region) ---
    @property
    def s_start(self) -> int:
        return 0

    @property
    def s_end(self) -> int:  # exclusive
        return self.block_len

    def replica_start(self, r: int) -> int:
        return self.block_len + r * self.block_len

    def replica_drafts(self, r: int) -> int:
        """How many sampling-block drafts replica r is conditioned on."""
        return r + self.replica_offset

    def seg_of(self, new_pos: int) -> tuple[str, int, int]:
        """Classify a new-region position: ('prefix'|'S'|'R', replica_idx_or_-1, local_idx)."""
        if new_pos < self.block_len:
            return ("S", -1, new_pos)
        rel = new_pos - self.block_len
        r = rel // self.block_len
        return ("R", r, rel % self.block_len)


def _allow_pair(d: MaskDescriptor, q_new: int, k_global: int) -> bool:
    """Reference predicate: does query (new-region index q_new) attend key (global k)?

    This is the human-readable definition the kernel inline-predicate must reproduce.
    """
    in_prefix = k_global < d.prefix_len
    k_new = k_global - d.prefix_len  # < 0 if key is in prefix

    q_seg, q_r, q_i = d.seg_of(q_new)

    if q_seg == "S":
        if in_prefix:
            return True  # sampling attends all prefix
        k_seg, k_r, k_i = d.seg_of(k_new)
        if k_seg == "S":
            return (k_i <= q_i) if d.sampling_causal else True  # causal within S
        return False  # S never attends replicas

    # q_seg == "R"
    if in_prefix:
        return d.mask_sees_prefix
    k_seg, k_r, k_i = d.seg_of(k_new)
    if k_seg == "S":
        return k_i < d.replica_drafts(q_r)  # conditioned on r accepted drafts
    # k in some replica: only the SAME replica, bidirectional
    return k_r == q_r


def build_allow_matrix(d: MaskDescriptor, device=None) -> torch.Tensor:
    """Dense boolean [q_len, kv_len] allow matrix (ground truth).

    Vectorised; equivalent to _allow_pair over every (q,k) — the test asserts the equality.
    """
    B, P = d.block_len, d.prefix_len
    ql, kl = d.q_len, d.kv_len

    # Classify every new-region index into (seg_id, replica, local).  seg_id: 0=S, 1=R
    new_idx = torch.arange(ql, device=device)
    is_S = new_idx < B
    rel = (new_idx - B).clamp(min=0)
    rep = torch.where(is_S, torch.full_like(new_idx, -1), rel // B)
    loc = torch.where(is_S, new_idx, rel % B)

    q_is_S = is_S.view(ql, 1)
    q_rep = rep.view(ql, 1)
    q_loc = loc.view(ql, 1)

    k_arange = torch.arange(kl, device=device)
    k_in_prefix = (k_arange < P).view(1, kl)
    k_new = (k_arange - P).clamp(min=0)
    k_is_S = ((k_arange >= P) & (k_arange < P + B)).view(1, kl)
    k_rel = (k_new - B).clamp(min=0)
    k_rep = torch.where(k_arange < P + B, torch.full_like(k_arange, -1), k_rel // B).view(1, kl)
    k_loc = torch.where(k_arange - P < B, (k_arange - P).clamp(min=0), k_rel % B).view(1, kl)

    allow = torch.zeros(ql, kl, dtype=torch.bool, device=device)

    # --- Sampling-block queries ---
    qS = q_is_S
    allow |= qS & k_in_prefix                                   # S -> all prefix
    if d.sampling_causal:
        allow |= qS & k_is_S & (k_loc <= q_loc)                 # S -> S causal
    else:
        allow |= qS & k_is_S

    # --- Replica (mask) queries ---
    qR = ~q_is_S
    if d.mask_sees_prefix:
        allow |= qR & k_in_prefix                               # R -> all prefix
    drafts = q_rep + d.replica_offset                           # replica_drafts(r)
    allow |= qR & k_is_S & (k_loc < drafts)                     # R_r -> first r drafts of S
    same_rep = (k_rep == q_rep) & (k_rep >= 0)
    allow |= qR & same_rep                                      # R_r -> own replica (bidir)

    return allow


def additive_bias(d: MaskDescriptor, dtype=torch.float32, device=None) -> torch.Tensor:
    """Float [q_len, kv_len] bias: 0 where allowed, -inf where denied. Add to QK^T pre-softmax."""
    allow = build_allow_matrix(d, device=device)
    bias = torch.zeros_like(allow, dtype=dtype)
    bias.masked_fill_(~allow, NEG_INF)
    return bias


def build_square_allow_matrix(d: MaskDescriptor, device=None) -> torch.Tensor:
    """Square [L, L] allow matrix over the FULL window L = prefix_len + q_len, where the prefix is
    also laid out as query rows. Needed by self-attention kernels (e.g. attn_hip.flash_prefill) whose
    query and key sequences are the same length. Prefix query rows attend the prefix causally; the
    new-region rows reuse the TiDAR predicate. Only the new-region output rows [prefix_len:] are used
    by the runner; prefix rows are computed-and-ignored.
    """
    P, ql = d.prefix_len, d.q_len
    L = P + ql
    allow = torch.zeros(L, L, dtype=torch.bool, device=device)
    # prefix queries: causal over prefix keys
    if P > 0:
        idx = torch.arange(P, device=device)
        allow[:P, :P] = idx[:, None] >= idx[None, :]
    # new-region queries: the [q_len, kv_len] block goes at rows [P:L], cols [0:L]
    allow[P:L, :] = build_allow_matrix(d, device=device)
    return allow


def square_additive_bias(d: MaskDescriptor, dtype=torch.float32, device=None) -> torch.Tensor:
    """Square [L, L] additive bias (0 / -inf) for self-attention kernels. See build_square_allow_matrix."""
    allow = build_square_allow_matrix(d, device=device)
    bias = torch.zeros_like(allow, dtype=dtype)
    bias.masked_fill_(~allow, NEG_INF)
    return bias


def select_next_drafts_row_range(d: MaskDescriptor, accepted_k: int) -> tuple[int, int]:
    """After verify yields ``accepted_k``, the q-axis row range of the replica to read as the
    next block's drafts. Maps acceptance length -> replica index (clamped to available)."""
    r = max(0, min(d.block_len - 1, accepted_k - d.replica_offset))
    start = d.replica_start(r)
    return (start, start + d.block_len)
