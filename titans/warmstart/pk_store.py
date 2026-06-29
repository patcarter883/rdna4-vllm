"""CAM step-3 — the shared product-key (PKM) sparse store + multi-head reads, keyed to the
canonical-Z hub (CANONICAL_BUILD_PLAN §1.3, dial #2; CAM_DESIGN §2.3 semantic tier).

The store is the capacity lever: addressing-sparsity (top-k product-key lookup) scales a large slot
bank to bounded per-token read cost and low interference. It lives in HUB space (d=4096), the
base-neutral canonical-Z geometry the committee atlas built — so the store is shared across bases and
each spoke reaches it only through its translator (the MAG tap, for the smoke's single base).

PRODUCT-KEY ADDRESSING (Lample 2019 / Memory-Layers 2412.09764):
  N slots = n_sub^2 via TWO sub-key codebooks C1,C2 in R[n_sub, d_hub/2]. A query q in R[d_hub] splits
  q=[q1;q2]; score each codebook (q1·C1, q2·C2), take top-k1 in each, form the k1^2 candidate product
  keys, keep the global top-k. Read cost ~ 2·n_sub (=2·sqrt(N)) not N. Each of the N slots owns a
  VALUE vector v_s in R[d_hub]; the read is the softmax(top-k scores)-weighted sum of selected values.

CANONICAL-Z ANCHORING: the sub-key codebooks are initialised from the canonical-Z atlas keys (split
into the two halves), and a light fidelity pull keeps the product-key grid near the committee geometry
so the store addresses in the same base-neutral space the translators land in. The store reads in hub
space; nothing here is base-specific.

WRITE PATH (error-correcting delta, NOT additive — non-negotiable #1): an episode's (key->value)
associations are written by, for each association, addressing the store with the key, and applying a
delta update v_s <- v_s + beta * sel * (value - v_s) to the selected slots (surprise = value-v_s). The
store VALUES are an episodic fast-weight state reset per episode (like DeepMemory's per-batch state) —
the trained params are the codebooks + read/write projections + the multi-head read/out projections.

MULTI-HEAD READS (factual / positional / recency — CAM_DESIGN §6, plan §1.3): heads specialise by
retrieval mode via per-head query/output projections over the SHARED store. factual = the plain
content read; positional = query biased by a learned position embedding; recency = query biased by a
learned recency code. They are cheap (extra d_hub x d_hub projections) and trained jointly; the smoke
measures whether per-purpose specialists even differentiate before any head is ever split out.

fp32 compute throughout (delta-write surprise + scores need precision); the additive update is cast
back to the base dtype only at the MAG tap (gated_tap.py owns that).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Plain RMSNorm (fp32) with a learnable per-feature gain. Used to STRIP MAGNITUDE from each
    read-head output so the store CONTENT (direction), not the read norm, drives the injection. This
    is the read-side fix for the step-3 cognitive bypass (read norms blew to ~5000 vs gate ~0.006)."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class ProductKeyStore(nn.Module):
    """Shared product-key sparse store in hub space + multi-head reads. Per-episode value state is an
    external tensor (init_state / write / read) so the store params stay batch-independent."""

    def __init__(self, d_hub, n_sub=32, topk=8, sub_topk=4, n_heads=3, anchor_keys=None,
                 write_beta=1.0):
        super().__init__()
        assert d_hub % 2 == 0, "product-key splits the query in half"
        self.d_hub = d_hub
        self.d_half = d_hub // 2
        self.n_sub = n_sub                  # codebook size per half -> N = n_sub^2 slots
        self.N = n_sub * n_sub
        self.topk = topk                    # global product-keys kept per query
        self.sub_topk = sub_topk            # top-k per half (candidates = sub_topk^2)
        self.n_heads = n_heads
        self.write_beta = write_beta

        # two product-key sub-codebooks C1,C2 : [n_sub, d_half]. Anchored to canonical-Z if given.
        c1 = torch.randn(n_sub, self.d_half)
        c2 = torch.randn(n_sub, self.d_half)
        if anchor_keys is not None:
            # seed the codebooks from the canonical-Z anchor directions (split into halves). The atlas
            # has 102 keys; tile/trim to n_sub rows. Keeps the product grid in committee geometry.
            ak = anchor_keys.float()
            idx = torch.arange(n_sub) % ak.shape[0]
            c1 = ak[idx, :self.d_half].clone()
            c2 = ak[idx, self.d_half:].clone()
        self.codebook1 = nn.Parameter(F.normalize(c1, dim=1))
        self.codebook2 = nn.Parameter(F.normalize(c2, dim=1))

        # write-path projections: an association's key/value (already in hub space) -> store query / value
        self.to_wkey = nn.Linear(d_hub, d_hub, bias=False)
        self.to_wval = nn.Linear(d_hub, d_hub, bias=False)

        # multi-head READ: per-head query projection (from the base-derived hub query) + output projection
        self.read_q = nn.ModuleList([nn.Linear(d_hub, d_hub, bias=False) for _ in range(n_heads)])
        self.read_o = nn.ModuleList([nn.Linear(d_hub, d_hub, bias=False) for _ in range(n_heads)])
        # READ NORMALIZATION (the bypass fix): RMSNorm the retrieved context BEFORE read_o, so a head
        # cannot dominate the gate by sheer norm. The store value's DIRECTION drives the output; the
        # gate (gamma) carries the magnitude. One norm per head over the d_hub context vector.
        self.read_norm = nn.ModuleList([RMSNorm(d_hub) for _ in range(n_heads)])
        # FINAL read-output norm: RMSNorm the summed multi-head read so the bank handed to the MAG tap
        # has bounded magnitude — the gate (gamma), not a read-norm blow-up, carries the injection
        # scale. Closes the read-side bypass loop with read_norm (per-head) + read_out_norm (summed).
        self.read_out_norm = RMSNorm(d_hub)
        # head biases: factual=0, positional=learned position code, recency=learned recency code.
        # head 0 (factual) has no bias; heads >=1 add a learned query bias (their "retrieval mode").
        self.head_bias = nn.Parameter(torch.zeros(n_heads, d_hub))

    # ---- per-episode value state -----------------------------------------
    def init_state(self, batch, device, dtype=torch.float32):
        """Episodic value bank V:[B, N, d_hub], zero (empty store). Reset per episode.

        STORAGE dtype is the knowledge-store-grade VRAM lever (step 3c): pass dtype=torch.bfloat16 to
        HALVE the value bank (the dominant term at large N: B*N*d_hub*2 B vs *4 B fp32). The bank is
        STORAGE only — every read/write casts the gathered slot values to fp32 for the delta-write
        surprise + score precision, then casts the net delta back to the bank dtype (the
        fp32-compute/cast-back pattern, mirroring the MAG tap). bf16's ~3-decimal-digit mantissa is
        ample for an L2-normalised hub value mix whose DIRECTION (not magnitude) drives the read."""
        return torch.zeros(batch, self.N, self.d_hub, device=device, dtype=dtype)

    # ---- product-key addressing ------------------------------------------
    def _address(self, q):
        """q:[B,Q,d_hub] -> (slot_idx[B,Q,topk] long, slot_w[B,Q,topk] softmax weights).

        Product-key top-k: score each half against its codebook, take sub_topk per half, expand to the
        sub_topk^2 candidate product slots, score = s1+s2, keep global topk, softmax over them."""
        B, Q, _ = q.shape
        q1, q2 = q[..., :self.d_half], q[..., self.d_half:]
        s1 = q1 @ self.codebook1.t()                          # [B,Q,n_sub]
        s2 = q2 @ self.codebook2.t()
        v1, i1 = s1.topk(self.sub_topk, dim=-1)               # [B,Q,sub_topk]
        v2, i2 = s2.topk(self.sub_topk, dim=-1)
        # candidate product scores [B,Q,sub_topk,sub_topk] and flat slot ids i1*n_sub + i2
        cand = v1.unsqueeze(-1) + v2.unsqueeze(-2)            # [B,Q,st,st]
        slot = (i1.unsqueeze(-1) * self.n_sub + i2.unsqueeze(-2)).reshape(B, Q, -1)  # [B,Q,st*st]
        cand = cand.reshape(B, Q, -1)
        w, sel = cand.topk(self.topk, dim=-1)                 # [B,Q,topk]
        slot_idx = torch.gather(slot, -1, sel)                # [B,Q,topk] global slot ids
        slot_w = torch.softmax(w, dim=-1)
        return slot_idx, slot_w

    # ---- write address / value projections (exposed for addressing supervision) ----
    def write_addr_val(self, keys, values):
        """keys/values:[B,A,d_hub] (hub embeds) -> (write addresses wk, stored values wv) [B,A,d_hub].
        wk = to_wkey(key) is the query that addressed the slot; wv = to_wval(value) is what was stored.
        The addressing-supervision loss aligns read queries to wk and read ctx to wv."""
        return self.to_wkey(keys), self.to_wval(values)

    def head_query(self, query, h=0):
        """read_q[h](query) + head_bias[h] — the per-head read query (the addressing target)."""
        return self.read_q[h](query) + self.head_bias[h]

    # ---- write (error-correcting delta into selected slots) --------------
    def write(self, V, keys, values):
        """Write a batch of associations into the episodic value bank V (functional, returns new V).
        keys/values:[B,A,d_hub] (A associations). For each association, address with to_wkey(key),
        then delta-update the selected slots toward to_wval(value): v_s += beta*w*(val - v_s).
        Done with a scatter-add of the net delta (sequential per-association folding is unrolled into
        one masked update — fine at smoke A; the store is a fast-weight state, parity vs sequential is a
        full-run concern). fp32."""
        B, A, _ = keys.shape
        bank_dtype = V.dtype                                  # bf16 (3c) or fp32 (smoke)
        wk = self.to_wkey(keys)                               # [B,A,d_hub]
        wv = self.to_wval(values)                             # [B,A,d_hub]
        slot_idx, slot_w = self._address(wk)                  # [B,A,topk]
        # gather current slot values, compute delta IN FP32, scatter-add the delta back in the bank
        # dtype. (Concurrent same-slot writes in one episode average via the scatter — acceptable for
        # the smoke; a strict sequential delta is the full-run / parity item.) The .float() lift makes
        # the surprise (wv - cur) precise even when the bank stores bf16; the cast-back keeps the bank
        # at storage dtype so VRAM stays halved.
        cur = torch.gather(V, 1, slot_idx.reshape(B, A * self.topk, 1).expand(-1, -1, self.d_hub)
                           ).reshape(B, A, self.topk, self.d_hub).float()
        delta = self.write_beta * slot_w.unsqueeze(-1) * (wv.unsqueeze(2).float() - cur)  # [B,A,topk,d_hub] fp32
        # V comes from init_state (fresh zeros, requires_grad=False, used nowhere else), so we can
        # scatter the (grad-carrying) delta IN PLACE — no V.clone(). At knowledge-store-grade N the
        # clone was a full DUPLICATE value bank (~10 GB @ N=100k bf16); dropping it halves write peak.
        Vnew = V if not V.requires_grad else V.clone()
        Vnew.scatter_add_(1, slot_idx.reshape(B, A * self.topk, 1).expand(-1, -1, self.d_hub),
                          delta.reshape(B, A * self.topk, self.d_hub).to(bank_dtype))
        return Vnew

    # ---- multi-head read -------------------------------------------------
    def read(self, V, query, return_ctx=False):
        """query:[B,Q,d_hub] hub-space read query -> (read_out[B,Q,d_hub], head_norms list).
        Each head: q_h = read_q[h](query) + head_bias[h]; address V; weighted-sum selected slot values
        -> RMSNorm (strip magnitude, the bypass fix) -> read_o[h]. Heads summed. Returns per-head
        output norms (specialisation diagnostic). If return_ctx, also returns the per-head retrieved
        context ctx[h] [B,Q,d_hub] (PRE read_o/norm) — used for the addressing-supervision loss."""
        B, Q, _ = query.shape
        out = query.new_zeros(B, Q, self.d_hub)
        head_norms, ctxs = [], []
        for h in range(self.n_heads):
            qh = self.read_q[h](query) + self.head_bias[h]
            slot_idx, slot_w = self._address(qh)             # [B,Q,topk]
            # gather selected slot values (bank dtype) and lift to fp32 for the weighted value mix —
            # the read context feeds the fp32 RMSNorm/read_o, so the bf16-stored bank reads losslessly.
            vals = torch.gather(V, 1, slot_idx.reshape(B, Q * self.topk, 1).expand(-1, -1, self.d_hub)
                                ).reshape(B, Q, self.topk, self.d_hub).float()
            ctx = (slot_w.unsqueeze(-1) * vals).sum(dim=2)   # [B,Q,d_hub]  retrieved value mix (fp32)
            oh = self.read_o[h](self.read_norm[h](ctx))      # RMSNorm strips magnitude
            out = out + oh
            head_norms.append(float(oh.detach().norm(dim=-1).mean()))
            if return_ctx:
                ctxs.append(ctx)
        out = self.read_out_norm(out)            # bound the bank norm; gate carries the scale
        if return_ctx:
            return out, head_norms, ctxs
        return out, head_norms
