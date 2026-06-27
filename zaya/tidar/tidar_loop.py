"""TiDAR single-forward decode loop + β-rejection sampler — backend-neutral, weight-independent.

This is the §4.2/§4.3/§4.4 piece of the serving path (see docs/zaya/tidar-serving-design.md),
built and validated on a **random-weight stub** because there is no public ZAYA1-8B-Diffusion
checkpoint yet. The stub (``StubCCALM``) is a tiny deterministic causal LM that carries the two
CCA-relevant recurrent states so the rollback/evict logic is exercised for real:

  * a causal depthwise conv (kernel ``Kc`` -> needs ``Kc-1`` previous hidden rows) feeding q/k —
    the analogue of CCA's ``conv_qk`` / ``conv_states`` (cca.py); and
  * a previous-token term in v (``v[t] = h[t] Wv + h[t-1] Wv2``) — the analogue of CCA's second
    value stream off the previous hidden, i.e. the ``prev_hs`` rollback state.

The pieces here are exactly the runtime the production v1 model-runner will drive, just over the
stub instead of the real ZAYA forward:

  build [prefix | S | R_0..R_{B-1}]  ->  ONE fused forward (structured TiDAR mask, §3)
     ->  β-rejection-sample the sampling block S  ->  accepted length k
     ->  EVICT the B-k rejected positions (KV + conv state)  ->  commit k accepted + 1 bonus
     ->  pre-draft the next block from a mask replica  ->  loop.

LOSSLESSNESS (β=1): the committed token stream is **identical to greedy AR** — verify re-checks
every draft against the true AR distribution, so the drafts are only hints; correctness never
depends on them (this is what de-risks "diffusion models can't be lossless/KV-cached").

What is and isn't pinned here (consistent with tidar_mask.py + design §7.1):
  * The β sampler, the evict-on-reject == recompute property, and the per-replica segmented conv
    are weight-independent and FULLY pinned by test_tidar_loop.py.
  * The cross-step *replica reuse* (use this step's R_k instead of an extra forward) is validated
    as an equivalence (test E) but the exact bonus-token / position off-by-one (replica_offset,
    RoPE on mask tokens — design §7.1/§7.6) needs the conversion checkpoint. The decode loop here
    therefore re-derives next-block drafts from a fresh R_0 each step (simple, provably lossless);
    test E proves R_k == that fresh predraft so production can take the one-forward shortcut.

CPU, float64, no GPU lease. Reserve token id ``MASK_ID = 0`` as the diffusion ``[mask]`` token.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from tidar_mask import MaskDescriptor, additive_bias

MASK_ID = 0  # reserved [mask] token id
_DT = torch.float64


# ----------------------------------------------------------------------------------------------
# Stub model
# ----------------------------------------------------------------------------------------------
class StubCCALM:
    """Tiny deterministic causal LM with CCA-shaped recurrent state. float64 / CPU.

    Single attention head, no RoPE (positions enter only through the conv locality and the mask —
    the stub validates the *mechanism*; RoPE on mask tokens is design §7.6, pinned at checkpoint).
    """

    def __init__(self, vocab: int = 64, dim: int = 32, kc: int = 2, seed: int = 0):
        g = torch.Generator().manual_seed(seed)

        def r(*shape):
            return torch.randn(*shape, generator=g, dtype=_DT) * 0.5

        self.V, self.d, self.Kc = vocab, dim, kc
        self.emb = r(vocab, dim)
        self.cw = r(kc, dim)            # depthwise causal conv taps; cw[0] = current token
        self.Wq, self.Wk = r(dim, dim), r(dim, dim)
        self.Wv, self.Wv2 = r(dim, dim), r(dim, dim)  # Wv2 multiplies the previous-token hidden
        self.Wo = r(dim, dim)
        self.head = r(dim, vocab)
        self.scale = 1.0 / math.sqrt(dim)

    # --- primitives -------------------------------------------------------------------------
    def _embed(self, toks) -> torch.Tensor:
        return self.emb[torch.as_tensor(list(toks), dtype=torch.long)]  # [T, d]

    def _zeros(self, n: int) -> torch.Tensor:
        return torch.zeros(n, self.d, dtype=_DT)

    def _conv(self, h_seq: torch.Tensor, left: torch.Tensor | None) -> torch.Tensor:
        """Causal depthwise conv over ``h_seq`` given ``Kc-1`` rows of left context.

        out[i] = sum_j cw[j] * full[i - j],  full = [left (Kc-1 rows) ; h_seq].
        ``left=None`` -> zero left padding (sequence start).
        """
        Kc, T = self.Kc, h_seq.shape[0]
        if Kc == 1:
            return self.cw[0] * h_seq
        if left is None or left.numel() == 0:
            left = self._zeros(Kc - 1)
        full = torch.cat([left[-(Kc - 1):], h_seq], dim=0)  # [(Kc-1)+T, d]
        base = full.shape[0] - T  # == Kc-1
        out = self._zeros(T)
        for j in range(Kc):
            out = out + self.cw[j] * full[base - j: base - j + T]
        return out

    def _kv(self, hc: torch.Tensor, h_seq: torch.Tensor, prev_h: torch.Tensor):
        """q,k from the conv'd hidden; v from the raw hidden + previous-token hidden (prev_hs)."""
        q = hc @ self.Wq
        k = hc @ self.Wk
        v = h_seq @ self.Wv + prev_h @ self.Wv2
        return q, k, v

    def _heads(self, q, k_all, v_all, allow=None, bias=None):
        scores = (q @ k_all.t()) * self.scale
        if bias is not None:
            scores = scores + bias
        if allow is not None:
            scores = scores.masked_fill(~allow, float("-inf"))
        a = torch.softmax(scores, dim=-1) @ v_all
        o = a @ self.Wo
        return o @ self.head  # logits [q, V]

    # --- reference causal forward -----------------------------------------------------------
    def causal_forward(self, toks):
        """Plain causal LM forward. Returns dict(logits[T,V], k[T,d], v[T,d], h[T,d]).

        ``logits[t]`` is the next-token prediction *after* position ``t``.
        """
        h = self._embed(toks)
        hc = self._conv(h, None)
        prev = torch.cat([self._zeros(1), h[:-1]], dim=0)
        q, k, v = self._kv(hc, h, prev)
        T = h.shape[0]
        idx = torch.arange(T)
        allow = idx[:, None] >= idx[None, :]  # causal
        logits = self._heads(q, k, v, allow=allow)
        return {"logits": logits, "k": k, "v": v, "h": h}

    def greedy_decode(self, prompt, n_new: int):
        toks = list(prompt)
        for _ in range(n_new):
            toks.append(int(self.causal_forward(toks)["logits"][-1].argmax()))
        return toks

    # --- fused TiDAR single forward ---------------------------------------------------------
    def tidar_forward(self, committed, drafts, desc: MaskDescriptor):
        """ONE forward over [prefix(committed) | S(drafts) | R_0..R_{B-1}] with the structured
        TiDAR additive-bias mask. Replicas use a per-replica *segmented* conv (each R_r built from
        the conv context [committed + drafts[:r]] — design §1.1), exactly as cca.py's segmented
        spec conv does. Returns a dict with the new-region logits and the S block's k/v (for the
        incremental KV cache / evict path).
        """
        B, L = desc.block_len, desc.prefix_len
        drafts = list(drafts)
        assert L == len(committed) and len(drafts) == B

        # prefix (committed) -> cached keys/values
        ch = self._embed(committed) if L else self._zeros(0)
        chc = self._conv(ch, None) if L else self._zeros(0)
        cprev = torch.cat([self._zeros(1), ch[:-1]], dim=0) if L else self._zeros(0)
        _, ck, cv = self._kv(chc, ch, cprev) if L else (None, self._zeros(0), self._zeros(0))
        ctail = ch[-(self.Kc - 1):] if L else None
        clast = ch[-1:] if L else self._zeros(1)

        # sampling block S (causal, contiguous after the prefix -> conv context is just committed)
        sh = self._embed(drafts)
        shc = self._conv(sh, ctail)
        sprev = torch.cat([clast, sh[:-1]], dim=0)
        sq, sk, sv = self._kv(shc, sh, sprev)

        # replicas R_r: segmented conv with the per-replica accepted-draft context
        mask_h = self._embed([MASK_ID] * B)
        Rq, Rk, Rv = [], [], []
        for r in range(B):
            ctx = list(committed) + drafts[:r]
            cth = self._embed(ctx) if ctx else self._zeros(0)
            r_tail = cth[-(self.Kc - 1):] if ctx else None
            r_last = cth[-1:] if ctx else self._zeros(1)
            rhc = self._conv(mask_h, r_tail)
            rprev = torch.cat([r_last, mask_h[:-1]], dim=0)
            rq, rk, rv = self._kv(rhc, mask_h, rprev)
            Rq.append(rq); Rk.append(rk); Rv.append(rv)

        q = torch.cat([sq, *Rq], dim=0)            # [q_len, d]
        k_all = torch.cat([ck, sk, *Rk], dim=0)    # [L + q_len, d]
        v_all = torch.cat([cv, sv, *Rv], dim=0)
        bias = additive_bias(desc, dtype=_DT)      # [q_len, L + q_len]
        logits = self._heads(q, k_all, v_all, bias=bias)

        return {"logits": logits, "sk": sk, "sv": sv, "Rq": Rq, "Rk": Rk}

    def replica_drafts(self, fwd, desc: MaskDescriptor, r: int):
        """Argmax tokens of replica R_r (the next-block pre-draft conditioned on r accepts)."""
        s, e = desc.replica_start(r), desc.replica_start(r) + desc.block_len
        return fwd["logits"][s:e].argmax(dim=-1).tolist()


# ----------------------------------------------------------------------------------------------
# β-rejection sampler (§4.3)
# ----------------------------------------------------------------------------------------------
def beta_verify(drafts, p_ar_logits, p_diff_logits=None, beta: float = 1.0):
    """Speculative accept loop: accept ``drafts[i]`` while it equals
    ``argmax(β·logit_AR[i] + (1-β)·logit_diff[i])``; stop at the first mismatch.

    ``p_ar_logits`` is ``[B+1, V]`` — row ``i`` is the AR prediction for the i-th block position
    (so row ``k`` gives the *bonus* token committed after ``k`` accepts). ``p_diff_logits`` (the
    carried prior-step mask-block logits) is optional; β=1 ignores it (pure-AR = lossless).

    Returns ``(k, bonus_token)``: ``k`` accepted drafts and the AR/mixed token at position ``k``.
    """
    drafts = list(drafts)
    B = len(drafts)
    assert p_ar_logits.shape[0] == B + 1, (p_ar_logits.shape, B)

    def choose(i: int) -> int:
        lg = beta * p_ar_logits[i]
        if beta != 1.0:
            diff = p_ar_logits[i] if p_diff_logits is None else p_diff_logits[i]
            lg = lg + (1.0 - beta) * diff
        return int(lg.argmax())

    k = 0
    for i in range(B):
        if int(drafts[i]) == choose(i):
            k += 1
        else:
            break
    return k, choose(k)


# ----------------------------------------------------------------------------------------------
# Single-forward decode loop (§4.4) + exact-KV/conv evict-on-reject (§4.2)
# ----------------------------------------------------------------------------------------------
@dataclass
class TidarTrace:
    """Per-step record for the tests (accepted length, drafts, the running committed length)."""

    step: int
    L: int
    drafts: list
    k: int
    bonus: int
    committed_len: int


def tidar_decode(
    model: StubCCALM,
    prompt,
    n_new: int,
    block_len: int,
    beta: float = 1.0,
    replica_offset: int = 0,
    collect_trace: bool = False,
):
    """Drive the TiDAR single-forward loop until at least ``n_new`` new tokens are committed.

    Each step: one fused forward -> β-verify the sampling block -> commit k accepted + 1 bonus ->
    evict the B-k rejected positions -> re-derive next-block drafts from a fresh R_0. With β=1 the
    committed stream equals greedy AR (asserted in the tests). Returns the committed token list
    (and, if ``collect_trace``, the per-step trace).
    """
    B = block_len
    committed = list(prompt)
    assert len(committed) >= 1, "TiDAR loop needs a non-empty prompt (the first AR row is the prefix tail)"
    target = len(prompt) + n_new
    trace: list[TidarTrace] = []

    def fresh_drafts(ctx):
        d = MaskDescriptor(prefix_len=len(ctx), block_len=B, replica_offset=replica_offset)
        fwd = model.tidar_forward(ctx, [MASK_ID] * B, d)  # R_0 is independent of draft values
        return model.replica_drafts(fwd, d, 0)

    drafts = fresh_drafts(committed)
    step = 0
    while len(committed) < target:
        L = len(committed)
        desc = MaskDescriptor(prefix_len=L, block_len=B, replica_offset=replica_offset)
        # p_AR for verifying the block: row L-1..L+B-1 of a causal pass over [committed + drafts]
        # = predictions for positions L..L+B (B+1 rows; the last is the all-accepted bonus row).
        # (In production these come from the fused forward's S rows + the carried prefix-tail
        #  logit — test E proves the fused S rows equal this causal slice.)
        full = model.causal_forward(committed + drafts)["logits"]
        p_ar = full[L - 1: L + B]  # B+1 rows: predictions for positions L..L+B
        k, bonus = beta_verify(drafts, p_ar, beta=beta)
        committed = committed + drafts[:k] + [bonus]
        if collect_trace:
            trace.append(TidarTrace(step, L, list(drafts), k, int(bonus), len(committed)))
        step += 1
        if len(committed) >= target:
            break
        drafts = fresh_drafts(committed)

    return (committed, trace) if collect_trace else committed


# ----------------------------------------------------------------------------------------------
# Exact-KV / conv evict-on-reject reference (§4.2) — used by the tests
# ----------------------------------------------------------------------------------------------
class IncrementalKVConv:
    """Models the persistent state TiDAR must roll back on rejection: the paged KV (here a growing
    k/v cache) and the CCA conv/prev_hs recurrent state (here the running hidden tail). It appends
    a draft block's per-position state then *evicts* the rejected tail, and exposes the committed
    state so a test can assert it equals a from-scratch recompute of the accepted token stream.
    """

    def __init__(self, model: StubCCALM):
        self.m = model
        self.k = model._zeros(0)
        self.v = model._zeros(0)
        self.tokens: list[int] = []

    def commit_block(self, committed_before, sk, sv, drafts, k_accept, bonus):
        """Append all B drafts' KV, then evict the B-k rejected, then append the bonus token's KV.

        ``sk``/``sv`` are the sampling block's keys/values from ``tidar_forward`` (computed as if
        all B drafts were appended). This mirrors the runtime: state is produced for the whole
        speculative block, then truncated to the accepted prefix on verify.
        """
        # provisionally append the whole speculative block ...
        self.k = torch.cat([self.k, sk], dim=0)
        self.v = torch.cat([self.v, sv], dim=0)
        # ... then evict the rejected tail (keep only k accepted)
        keep = len(committed_before) + k_accept
        self.k = self.k[:keep]
        self.v = self.v[:keep]
        self.tokens = list(committed_before) + list(drafts[:k_accept])
        # commit the bonus token: recompute its KV with the (now committed) conv/prev context
        self._append_token(bonus)

    def _append_token(self, tok):
        # Append ONLY the new token's KV (its conv context = the already-committed tail). The
        # existing prefix rows are left untouched so a test comparing self.k to a fresh recompute
        # genuinely validates the incrementally-built + evicted state, not an overwrite.
        ctx = self.tokens + [int(tok)]
        f = self.m.causal_forward(ctx)
        self.k = torch.cat([self.k, f["k"][-1:]], dim=0)
        self.v = torch.cat([self.v, f["v"][-1:]], dim=0)
        self.tokens = ctx
