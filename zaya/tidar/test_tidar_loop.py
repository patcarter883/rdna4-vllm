"""Unit tests for the TiDAR single-forward decode loop + β-rejection sampler + evict-on-reject
(weight-independent; CPU float64, no GPU lease). See docs/zaya/tidar-serving-design.md §4.2/4.3/4.4.

Run:  python zaya/tidar/test_tidar_loop.py      (prints OK per test)
 or:  python -m pytest zaya/tidar/test_tidar_loop.py -q

These pin, on a random-weight CCA-shaped stub:
  A — the β=1 sampler is LOSSLESS (accept loop on ground-truth p_AR == greedy AR);
  B — exact-KV/conv EVICT-on-reject leaves committed state == a from-scratch recompute (§4.2);
  C — the per-replica SEGMENTED conv == a fresh recompute (validates the §1.1 causal-conv reuse);
  D — the end-to-end loop with β=1 reproduces greedy AR EXACTLY (the headline lossless property);
  E — the in-forward replica R_k == a fresh predraft after k accepts (so production may take the
      one-forward shortcut instead of an extra forward; the bonus/position off-by-one is design
      §7.1, pinned at the conversion checkpoint, and does NOT affect losslessness).
"""

import torch

from tidar_loop import (
    MASK_ID,
    IncrementalKVConv,
    StubCCALM,
    beta_verify,
    tidar_decode,
)
from tidar_mask import MaskDescriptor

_TOL = 1e-9


def _model(seed=0, vocab=48, dim=24, kc=2):
    return StubCCALM(vocab=vocab, dim=dim, kc=kc, seed=seed)


def _prompt(model, length=5, seed=123):
    g = torch.Generator().manual_seed(seed)
    # avoid MASK_ID in the prompt so it reads as real committed tokens
    return (1 + torch.randint(0, model.V - 1, (length,), generator=g)).tolist()


# ---- A: sampler losslessness -----------------------------------------------------------------
def test_beta1_sampler_matches_greedy_ar():
    m = _model()
    prompt = _prompt(m)
    B = 4
    L = len(prompt)
    # Greedy AR truth for the next B+1 positions.
    greedy = m.greedy_decode(prompt, B + 1)[L:]
    full = m.causal_forward(prompt + greedy[:B])["logits"]
    p_ar = full[L - 1: L + B]  # predictions for positions L..L+B

    # (1) drafts == greedy -> accept all B, bonus == greedy[B]
    k, bonus = beta_verify(greedy[:B], p_ar, beta=1.0)
    assert k == B, k
    assert bonus == greedy[B], (bonus, greedy[B])

    # (2) corrupt draft j -> accept exactly j, bonus == greedy[j] (the AR token at the mismatch)
    for j in range(B):
        bad = list(greedy[:B])
        bad[j] = (bad[j] + 1) % m.V
        if bad[j] == MASK_ID:
            bad[j] = (bad[j] + 1) % m.V
        # ensure it's actually a mismatch (greedy token might equal the bumped value only if V tiny)
        if bad[j] == greedy[j]:
            continue
        k, bonus = beta_verify(bad, p_ar, beta=1.0)
        assert k == j, (j, k)
        assert bonus == greedy[j], (j, bonus, greedy[j])


# ---- D: end-to-end loop is lossless ----------------------------------------------------------
def test_loop_is_lossless_vs_greedy():
    for seed in (0, 1, 7):
        m = _model(seed=seed)
        prompt = _prompt(m, length=6, seed=seed + 50)
        for B in (1, 4, 8):
            n = 17
            got = tidar_decode(m, prompt, n_new=n, block_len=B, beta=1.0)
            ref = m.greedy_decode(prompt, len(got) - len(prompt))
            assert got == ref, (
                f"seed={seed} B={B}: TiDAR loop diverged from greedy AR at "
                f"{next((i for i,(a,b) in enumerate(zip(got,ref)) if a!=b), None)}"
            )
            assert len(got) >= len(prompt) + n


# ---- B: evict-on-reject == recompute ---------------------------------------------------------
def test_evict_on_reject_matches_recompute():
    m = _model(seed=3)
    prompt = _prompt(m, length=5, seed=99)
    B = 4
    cache = IncrementalKVConv(m)
    # seed the cache with the prompt's committed KV
    f0 = m.causal_forward(prompt)
    cache.k, cache.v, cache.tokens = f0["k"].clone(), f0["v"].clone(), list(prompt)

    committed, trace = tidar_decode(
        m, prompt, n_new=20, block_len=B, beta=1.0, collect_trace=True
    )

    # Replay the loop through the cache, asserting after EACH step that the committed KV/conv state
    # equals a from-scratch recompute of the committed tokens (i.e. eviction left no contamination).
    cur = list(prompt)
    for t in trace:
        assert t.L == len(cur)
        desc = MaskDescriptor(prefix_len=t.L, block_len=B)
        fwd = m.tidar_forward(cur, t.drafts, desc)
        cache.commit_block(cur, fwd["sk"], fwd["sv"], t.drafts, t.k, t.bonus)
        cur = cur + t.drafts[: t.k] + [t.bonus]
        assert cache.tokens == cur, (cache.tokens, cur)
        ref = m.causal_forward(cur)
        assert torch.allclose(cache.k, ref["k"], atol=_TOL), (cache.k - ref["k"]).abs().max()
        assert torch.allclose(cache.v, ref["v"], atol=_TOL), (cache.v - ref["v"]).abs().max()
    assert cur == committed


# ---- C: per-replica segmented conv == fresh recompute ----------------------------------------
def test_replica_segmented_conv_matches_recompute():
    """Each replica R_r's q/k come from a conv whose left context is [committed + drafts[:r]]
    (design §1.1). Assert the fused forward's replica keys equal those of a fresh causal pass over
    [committed + drafts[:r] + mask*B] at the mask rows — i.e. the segmentation is correct and the
    flattened row order does NOT leak conv context across replicas."""
    m = _model(seed=5, kc=2)
    prompt = _prompt(m, length=7, seed=11)
    B = 4
    drafts = (1 + torch.arange(B)).tolist()  # arbitrary distinct draft tokens
    desc = MaskDescriptor(prefix_len=len(prompt), block_len=B)
    fwd = m.tidar_forward(prompt, drafts, desc)

    # S block keys vs causal recompute of [prompt + drafts] at the S rows
    L = len(prompt)
    ref_s = m.causal_forward(prompt + drafts)["k"][L: L + B]
    assert torch.allclose(fwd["sk"], ref_s, atol=_TOL), (fwd["sk"] - ref_s).abs().max()

    # each replica's keys vs causal recompute of [prompt + drafts[:r] + mask*B] at the mask rows
    for r in range(B):
        ctx = list(prompt) + drafts[:r]
        ref = m.causal_forward(ctx + [MASK_ID] * B)["k"][len(ctx): len(ctx) + B]
        assert torch.allclose(fwd["Rk"][r], ref, atol=_TOL), (r, (fwd["Rk"][r] - ref).abs().max())


# ---- E: in-forward replica R_k == fresh predraft after k accepts -----------------------------
def test_replica_equals_fresh_predraft():
    """The 'pre-draft conditioned on every acceptance length' trick: replica R_k computed in ONE
    forward must equal a fresh mask-block forward after actually committing drafts[:k]. This is
    what lets production select R_k instead of paying an extra forward (design §2b/§7.1). The
    bonus-token/position off-by-one is separate (§7.1/§7.6) and pinned at the checkpoint."""
    m = _model(seed=8)
    prompt = _prompt(m, length=6, seed=22)
    B = 4
    g = torch.Generator().manual_seed(7)
    drafts = (1 + torch.randint(0, m.V - 1, (B,), generator=g)).tolist()
    desc = MaskDescriptor(prefix_len=len(prompt), block_len=B)
    fwd = m.tidar_forward(prompt, drafts, desc)

    for k in range(B):
        # replica R_k's logits (conditioned on k accepts) from the single forward
        r_logits = fwd["logits"][desc.replica_start(k): desc.replica_start(k) + B]
        # fresh predraft: commit drafts[:k], run R_0 of a new forward
        ctx = list(prompt) + drafts[:k]
        d2 = MaskDescriptor(prefix_len=len(ctx), block_len=B)
        fwd2 = m.tidar_forward(ctx, [MASK_ID] * B, d2)
        fresh = fwd2["logits"][d2.replica_start(0): d2.replica_start(0) + B]
        assert torch.allclose(r_logits, fresh, atol=_TOL), (k, (r_logits - fresh).abs().max())


# ---- β<1 mixing wiring (smoke) ---------------------------------------------------------------
def test_beta_mix_uses_both_streams():
    """β<1 mixes p_AR and the carried p_diff in logit space (§4.3). Smoke-check the wiring: with a
    p_diff that strongly prefers the draft token, a draft the pure-AR rule would reject can be
    accepted (and at β=1 p_diff is ignored)."""
    m = _model()
    B = 2
    V = m.V
    # p_AR prefers token 5 at position 0; draft is token 9.
    p_ar = torch.full((B + 1, V), -10.0, dtype=torch.float64)
    p_ar[:, 5] = 0.0
    p_diff = torch.full((B + 1, V), -10.0, dtype=torch.float64)
    p_diff[:, 9] = 50.0  # diffusion strongly prefers the draft token
    drafts = [9, 9]
    assert beta_verify(drafts, p_ar, beta=1.0)[0] == 0  # pure AR rejects immediately
    k, _ = beta_verify(drafts, p_ar, p_diff_logits=p_diff, beta=0.5)
    assert k == B  # mixed rule now accepts the draft


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nAll {len(fns)} TiDAR loop tests passed.")
