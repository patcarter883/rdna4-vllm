#!/usr/bin/env python3
"""REAL-MODEL evict-on-reject equivalence gate for the TiDAR CCA KV/conv rollback (step 4).

The serving production path folds the TiDAR decode loop / β sampler / evict-on-reject onto
``cca.py:_decode_verify_spec``'s KV + conv-state rollback. That code:

  * provisionally runs the WHOLE speculative block ``[current state | B candidate tokens]`` through
    the causal CCA conv (one ``_conv_qk_decode`` call), writing, for EACH spec position ``j``, the
    conv window ENDING at token ``j`` (``buf[:, :, j+1 : j+1+tp_pad]``) into the per-position rollback
    slot ``state_indices_2d[i, write_col[i, j]]``, and the token-``j`` hidden into ``prev_hs``;
  * the NEXT step reads slot ``blk_scheduled_prev + (num_accepted - 1)`` — i.e. the committed conv /
    prev_hs state is the one ENDING AT THE LAST ACCEPTED TOKEN. Appending the rejected tail and then
    reading the accepted-prefix column IS the evict-on-reject (no physical truncation needed).

This is the exact contract of the weight-independent stub ``IncrementalKVConv.commit_block``
(tidar_loop.py): append all B drafts' state, then keep only the accepted prefix == a from-scratch
recompute of the accepted token stream. ``test_tidar_loop.py`` test B pins it on a random-weight
stub. THIS gate pins the SAME invariant on the REAL ZAYA conv producer (the checkpoint's layer-0
``ZayaCCAProjection`` — the actual ``conv_qk`` depthwise+grouped causal conv + the delayed-v recurrent
state that cca.py caches as ``conv_states`` / ``prev_hs``).

Why this proves the fold is sound: the CCA conv is CAUSAL (left-pad only, ``F.pad(.., (k, 0))``,
cca.py:380 / modeling_zaya ZayaCCAProjection.forward). Therefore the q/k/v at position ``p`` AND the
conv window + recurrent state ENDING at ``p`` are independent of any token appended AFTER ``p``. So
the provisional-block-then-evict committed state (read at the accepted-prefix end) is bit-identical
to a fresh recompute of just ``[committed | accepted]``. The gate asserts exactly that, on real
weights, for several (prefix, block_len, num_accepted) layouts.

ISOLATION (the §7.5 fusion-contamination finding): this gate drives ONLY the CCA conv/recurrent
producer over ``[committed | B-draft block]`` — it NEVER feeds the ``B*B`` mask-replica scratch
tokens through the forward. The two-forward split keeps verification isolated from the replica
scratch (the fused single-forward is REFUTED on ZAYA). The conv producer is the per-token-cached
state path where the global op never sees the replicas, which is exactly why the fold fixes the
contamination.

Also re-runs the §1.1/§7.5 conv-causality confirmation directly on the real conv: appending tokens
after position p must NOT change position p's conv output (real-model analogue of stub test C).

Run (fork venv, CPU — 17.7 GB bf16 fits host RAM, not a 16 GB card → NO gpu-lease):
  docker run --rm -e HF_HOME=/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
    -v /home/pat/code/.venv-zaya-fork:/opt/zaya-fork-venv \
    -v /home/pat/code/vllm-gfx1201-tidar-serve/zaya/tidar:/work -w /work \
    --entrypoint bash vllm22-w4a8:dflash-rxf -lc '/opt/zaya-fork-venv/bin/python cca_evict_gate.py'
"""
import os
import sys

import torch

from serve_loader import load_tidar_zaya

CKPT = os.environ.get(
    "TIDAR_CKPT",
    "/root/.cache/huggingface/hub/models--pat883--zaya1-8b-tidar-experts/"
    "snapshots/e6f2ba2d904688059a9e4bd50531504554b02f6d",
)
DEVICE = os.environ.get("TIDAR_DEVICE", "cpu")
DT = torch.bfloat16


# --------------------------------------------------------------------------------------------- #
# Real CCA conv producer driver
#
# We drive the checkpoint's actual layer-0 ``ZayaCCAProjection`` (the qkv_proj that produces the
# q/k/v cca.py caches as conv_states/prev_hs) over a chosen token stream, statelessly. Statelessly
# is faithful here BECAUSE the conv is causal: feeding the explicit token prefix reproduces the
# cached conv-state window + recurrent v-state exactly (cca.py seeds the decode conv with the cached
# tp_pad columns = the last conv-kernel raw qk_states; identical to having processed those tokens).
# So a stateless forward over [tokens..p] yields q/k/v at p whose conv window + recurrent state are
# the committed state ENDING at p — the very thing the per-position rollback slot stores.
# --------------------------------------------------------------------------------------------- #
def _find_cca_proj(model):
    """Return the layer-0 ZayaCCAProjection (the real conv producer) from the loaded model."""
    # Zyphra fork: model.model.layers[i].self_attn.qkv_proj (ZayaCCAProjection).
    base = getattr(model, "model", model)
    layers = base.layers
    proj = layers[0].self_attn.qkv_proj
    assert proj.__class__.__name__ == "ZayaCCAProjection", proj.__class__.__name__
    return proj


@torch.no_grad()
def _embed_tokens(model, token_ids):
    """[1, T, hidden] embeddings for a token id list (the conv producer's input is hidden states)."""
    ids = torch.tensor([list(token_ids)], device=DEVICE)
    return model.get_input_embeddings()(ids)  # [1, T, hidden]


@torch.no_grad()
def _produce_qkv(proj, hs):
    """Run the real CCA conv producer STATELESSLY over hidden states ``hs`` ([1, T, hidden]).

    past_key_values=None → the conv seeds with the explicit left-pad (sequence start) and the
    recurrent-v state with a zero previous token, then advances causally token-by-token. Returns
    (q, k, v) each [1, T, n_heads, head_dim] — the per-position produced state. Position p's outputs
    are the committed conv/recurrent state ENDING at p.
    """
    q, k, v = proj(hs, past_key_values=None, padding_mask=None)
    return q, k, v


def _max_abs_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


# --------------------------------------------------------------------------------------------- #
# GATE: evict-on-reject == from-scratch recompute (the IncrementalKVConv test-B analogue, real model)
# --------------------------------------------------------------------------------------------- #
@torch.no_grad()
def gate_evict_eq_recompute(model, proj, committed, drafts, k_accept):
    """Assert the committed CCA conv/recurrent state after evicting the rejected tail equals a
    from-scratch recompute of just the accepted token stream.

    * INCREMENTAL+EVICT: produce q/k/v over the WHOLE provisional block ``[committed | all-B-drafts]``
      (one forward; mirrors cca.py running the full (1+num_spec) candidate window), then READ the
      committed state at the accepted-prefix END position ``len(committed)+k_accept-1`` (the
      rollback-slot read ``num_accepted-1``).
    * FROM-SCRATCH: produce q/k/v over only ``[committed | accepted-drafts]`` fresh, READ its last
      position.

    These index the same committed token's state via two different code paths; causality ⇒ they
    must be bit-identical (real-model evict == recompute).
    """
    accepted = list(committed) + list(drafts[:k_accept])
    provisional = list(committed) + list(drafts)  # full B-candidate block

    hs_prov = _embed_tokens(model, provisional)
    hs_acc = _embed_tokens(model, accepted)

    q_p, k_p, v_p = _produce_qkv(proj, hs_prov)
    q_a, k_a, v_a = _produce_qkv(proj, hs_acc)

    end = len(committed) + k_accept - 1  # accepted-prefix end (the num_accepted-1 read column)
    # incremental+evict committed state == provisional path read at the accepted-prefix-end column
    qi, ki, vi = q_p[:, end], k_p[:, end], v_p[:, end]
    # from-scratch committed state == fresh path's last position
    qr, kr, vr = q_a[:, -1], k_a[:, -1], v_a[:, -1]

    dq, dk, dv = _max_abs_diff(qi, qr), _max_abs_diff(ki, kr), _max_abs_diff(vi, vr)
    return dq, dk, dv


# --------------------------------------------------------------------------------------------- #
# CONV-CAUSALITY confirmation (§1.1/§7.5): appending tokens after p must not change p's conv output
# (real-model analogue of stub test C — the per-replica segmented-conv reuse premise).
# --------------------------------------------------------------------------------------------- #
@torch.no_grad()
def confirm_conv_causal(model, proj, prefix, tail):
    """Compare position p's q/k/v from ``prefix`` alone vs from ``prefix + tail`` (p = last prefix
    position). Causal conv ⇒ identical. This is the conv-mode confirmation the design asks for: the
    diffusion FT kept conv_qk CAUSAL at the K=2 (cca_time0/cca_time1) boundary."""
    p = len(prefix) - 1
    hs_short = _embed_tokens(model, prefix)
    hs_long = _embed_tokens(model, list(prefix) + list(tail))
    qs, ks, vs = _produce_qkv(proj, hs_short)
    ql, kl, vl = _produce_qkv(proj, hs_long)
    return (
        _max_abs_diff(qs[:, p], ql[:, p]),
        _max_abs_diff(ks[:, p], kl[:, p]),
        _max_abs_diff(vs[:, p], vl[:, p]),
    )


# --------------------------------------------------------------------------------------------- #
def main():
    print(f"[evict-gate] loading {CKPT} on {DEVICE} (bf16) …", flush=True)
    model, tok, mask_id, block_size, (missing, unexpected) = load_tidar_zaya(
        CKPT, device=DEVICE, dtype=DT, from_config=True
    )
    proj = _find_cca_proj(model)
    kc = proj.conv_kernel_size
    print(
        f"[evict-gate] mask_id={mask_id} block_size={block_size} conv_kernel_size={kc} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )

    # Deterministic token streams (avoid the mask id in committed/draft positions so they read as
    # real tokens; the gate is about STATE EVOLUTION, the token values just have to be valid ids).
    g = torch.Generator().manual_seed(0)
    V = model.get_input_embeddings().weight.shape[0]

    def rand_toks(n):
        t = (1 + torch.randint(0, V - 2, (n,), generator=g)).tolist()
        return [x if x != mask_id else x + 1 for x in t]

    # --- conv-causality confirmation (do FIRST — it underwrites the evict gate) -------------- #
    print("\n=== CONV-CAUSALITY (§1.1/§7.5): appending tokens does NOT change earlier positions ===",
          flush=True)
    causal_ok = True
    for plen, tlen in [(4, 4), (8, 8), (16, 4)]:
        prefix, tail = rand_toks(plen), rand_toks(tlen)
        dq, dk, dv = confirm_conv_causal(model, proj, prefix, tail)
        ok = max(dq, dk, dv) == 0.0
        causal_ok &= ok
        print(
            f"  prefix_len={plen:2d} tail_len={tlen}: max|Δ| q={dq:.3g} k={dk:.3g} v={dv:.3g} "
            f"-> {'CAUSAL ✓' if ok else 'NON-CAUSAL ✗'}",
            flush=True,
        )
    print(
        f"  conv_qk kept CAUSAL at the K=2 boundary by the diffusion FT: {causal_ok} "
        f"(no cca.py non-causal branch needed)",
        flush=True,
    )

    # --- evict-on-reject == from-scratch recompute (the real-model IncrementalKVConv test B) -- #
    print("\n=== EVICT == RECOMPUTE on the REAL ZAYA conv/prev_hs (step-4 fold equivalence) ===",
          flush=True)
    evict_ok = True
    B = block_size
    # Cover every acceptance count 0..B for a couple of prefix lengths.
    for plen in (4, 16):
        committed = rand_toks(plen)
        drafts = rand_toks(B)
        for k in range(0, B + 1):
            dq, dk, dv = gate_evict_eq_recompute(model, proj, committed, drafts, k)
            ok = max(dq, dk, dv) == 0.0
            evict_ok &= ok
            print(
                f"  prefix={plen:2d} B={B} k_accept={k}: max|Δ| q={dq:.3g} k={dk:.3g} v={dv:.3g} "
                f"-> {'EVICT==RECOMPUTE ✓' if ok else 'DIVERGED ✗'}",
                flush=True,
            )

    overall = causal_ok and evict_ok
    print(
        f"\n[evict-gate] STEP-4 FOLD EQUIVALENCE: {'PASS ✓' if overall else 'FAIL ✗'} "
        f"(conv-causal={causal_ok}, evict==recompute={evict_ok})",
        flush=True,
    )
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
