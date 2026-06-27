"""Unit tests pinning the TiDAR structured mask (weight-independent; CPU, no GPU lease).

Run:  python -m pytest zaya/tidar/test_tidar_mask.py -q
 or:  python zaya/tidar/test_tidar_mask.py     (runs the same checks, prints OK)

These tests are the contract the attn_hip inline-predicate kernel and the triton/SDPA
additive-bias path must both satisfy. They assert STRUCTURAL properties (causal sub-blocks
are causal, replicas are bidirectional and isolated, S never sees replicas, replica r is
conditioned on exactly r drafts) rather than paper-fidelity — see tidar_mask.py header and
docs/zaya/tidar-serving-design.md §7.1 for the uncertain choices these tests pin.
"""

import torch

from tidar_mask import (
    MaskDescriptor,
    _allow_pair,
    additive_bias,
    build_allow_matrix,
    select_next_drafts_row_range,
)


def _vectorised_matches_reference(d: MaskDescriptor):
    allow = build_allow_matrix(d)
    for q in range(d.q_len):
        for k in range(d.kv_len):
            ref = _allow_pair(d, q, k)
            assert bool(allow[q, k]) == ref, (
                f"mismatch at q={q} k={k}: vectorised={bool(allow[q, k])} ref={ref} "
                f"(B={d.block_len} P={d.prefix_len})"
            )


def test_vectorised_matches_reference():
    for B in (1, 2, 4, 8):
        for P in (0, 1, 5):
            _vectorised_matches_reference(MaskDescriptor(prefix_len=P, block_len=B))


def test_shapes():
    d = MaskDescriptor(prefix_len=7, block_len=4)
    assert d.q_len == 4 * (1 + 4) == 20
    assert d.kv_len == 7 + 20
    assert build_allow_matrix(d).shape == (20, 27)


def test_sampling_block_is_causal_and_blind_to_replicas():
    d = MaskDescriptor(prefix_len=3, block_len=4)
    allow = build_allow_matrix(d)
    B, P = d.block_len, d.prefix_len
    for i in range(B):  # S[i]
        # all prefix allowed
        assert allow[i, :P].all()
        # S->S strictly causal
        for j in range(B):
            assert bool(allow[i, P + j]) == (j <= i)
        # S sees no replica key
        assert not allow[i, P + B:].any()


def test_replica_is_bidirectional_and_isolated():
    d = MaskDescriptor(prefix_len=2, block_len=4)
    allow = build_allow_matrix(d)
    B, P = d.block_len, d.prefix_len
    for r in range(B):
        rstart = d.replica_start(r)
        for m in range(B):  # query R_r[m]
            q = rstart + m
            # bidirectional within own replica: attends ALL m'
            for mp in range(B):
                assert allow[q, P + d.replica_start(r) + mp]
            # isolated from OTHER replicas
            for r2 in range(B):
                if r2 == r:
                    continue
                for mp in range(B):
                    assert not allow[q, P + d.replica_start(r2) + mp]


def test_replica_conditioned_on_r_drafts():
    d = MaskDescriptor(prefix_len=0, block_len=4)
    allow = build_allow_matrix(d)
    B = d.block_len
    for r in range(B):  # replica r sees first r drafts of S
        q = d.replica_start(r)
        for j in range(B):
            assert bool(allow[q, j]) == (j < r), (r, j)


def test_mask_sees_all_prefix():
    d = MaskDescriptor(prefix_len=6, block_len=2)
    allow = build_allow_matrix(d)
    for q in range(d.block_len, d.q_len):  # every replica query
        assert allow[q, : d.prefix_len].all()


def test_bias_is_zero_or_neg_inf_and_matches_allow():
    d = MaskDescriptor(prefix_len=4, block_len=4)
    allow = build_allow_matrix(d)
    bias = additive_bias(d)
    assert torch.equal(bias == 0.0, allow)
    assert torch.equal(torch.isneginf(bias), ~allow)


def test_no_row_fully_masked():
    # Every query must attend at least one key (else softmax => nan). With prefix>0 the
    # mask block always sees prefix; sampling[i] always sees itself.
    for B in (1, 2, 4, 8):
        d = MaskDescriptor(prefix_len=1, block_len=B)
        allow = build_allow_matrix(d)
        assert allow.any(dim=1).all(), B


def test_select_next_drafts_row_range():
    d = MaskDescriptor(prefix_len=0, block_len=4)
    # accepted k -> replica k (offset 0), clamped to [0, B-1]
    assert select_next_drafts_row_range(d, 0) == (d.replica_start(0), d.replica_start(0) + 4)
    assert select_next_drafts_row_range(d, 3) == (d.replica_start(3), d.replica_start(3) + 4)
    assert select_next_drafts_row_range(d, 99) == (d.replica_start(3), d.replica_start(3) + 4)


def test_sdpa_equivalence_random_weights():
    """The additive bias drives a real (CPU) scaled-dot-product attention identically to a
    hand-masked reference — the weight-independent correctness gate for Route B."""
    torch.manual_seed(0)
    d = MaskDescriptor(prefix_len=5, block_len=4)
    ql, kl, dim = d.q_len, d.kv_len, 16
    q = torch.randn(ql, dim, dtype=torch.float64)
    k = torch.randn(kl, dim, dtype=torch.float64)
    v = torch.randn(kl, dim, dtype=torch.float64)
    scale = dim ** -0.5

    bias = additive_bias(d, dtype=torch.float64)
    scores = q @ k.t() * scale + bias
    out = torch.softmax(scores, dim=-1) @ v

    # Reference: mask by boolean, softmax over allowed keys only.
    allow = build_allow_matrix(d)
    ref_scores = q @ k.t() * scale
    ref_scores = ref_scores.masked_fill(~allow, float("-inf"))
    ref_out = torch.softmax(ref_scores, dim=-1) @ v

    assert torch.allclose(out, ref_out, atol=1e-12), (out - ref_out).abs().max()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nAll {len(fns)} TiDAR mask tests passed.")
