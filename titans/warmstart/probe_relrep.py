"""CAM step 1-2 — committee relative-representation PROBE harness (CANONICAL_BUILD_PLAN §1.1-§1.2).

The canonical-Z atlas is built by sampling each committee model's residual geometry at its
proportionally-mapped tap depth and ALIGNING the models via **relative representations**
(Moschella et al. 2209.15430 / RLSA 2311.06547) — a representation that is tokenizer- AND
dimension-agnostic, so models with different vocabs and different hidden sizes become comparable
without any learned map.

What the probe extracts, per base model:
  1. A FIXED anchor set of texts (shared across ALL committee members — this is the alignment key).
  2. Forward the frozen base on each anchor; grab the residual hidden state at the
     PROPORTIONALLY-MAPPED tap depth L = round(tap_frac * n_layers)  (tap_frac = 24/36, the v0
     depth, expressed as a fraction so it transfers across models of different depth).
  3. Mean-pool that hidden state over the anchor's tokens -> ONE d_base vector per anchor:
     the model's "absolute" embedding of the anchor at the tap.  -> A  [n_anchor, d_base]
  4. Relative representation: rel[i,j] = cos(A[i], A[j])  -> R  [n_anchor, n_anchor].
     R is INDEPENDENT of d_base and of the tokenizer (it is only pairwise-anchor geometry), so two
     models' R matrices live in the SAME [n_anchor, n_anchor] space and can be stacked/fused into
     the d=4096 canonical-Z atlas in the next increment (whiten -> spherical-code).

This file is the per-model decoupled extractor (forward-only). It dumps, per model, a card:
  { model, n_layers, tap_frac, tap_layer, d_base, anchors(meta), A (abs embeds), R (rel-rep) }
to ckpt/probe/<slug>.pt. The atlas builder (next increment) loads N such cards, stacks R, whitens,
spherical-codes -> Z. Zaya probes locally on gfx1201; the cloud members merge the same way.

Run (1 leased card; absolute arbiter path):
  /home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh -n 1 --name titans-probe -- \
    titans/warmstart/run_m2.sh titans-probe --entry warmstart/probe_relrep.py -- \
      --models Qwen/Qwen3.5-4B unsloth/gemma-3-4b-pt Qwen/Qwen3-0.6B
"""
import argparse
import gc
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from gated_tap import decoder_layers                                       # noqa: E402
from anchor_bank import ANCHORS, CATEGORIES, anchor_sha, save_bank         # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# v0's passing tap depth, expressed as a FRACTION of model depth so it maps proportionally across
# models with different layer counts (v0: L=24 of Qwen3.5-4B's 36 layers).
TAP_FRAC = 24.0 / 36.0

# The shared anchor set (the alignment key) now lives in anchor_bank.py — a fixed, content-diverse,
# ~100-anchor curated bank (factual/science/math/code/narrative/dialogue/structured/multilingual),
# reproducible by construction (static list). Every committee member (local + cloud) forwards the
# SAME ordered ANCHORS so their [n_anchor, n_anchor] rel-rep matrices stack into one atlas.


def load_base(model_id):
    """Load + freeze any HF causal LM (pure torch). Loader selection (CausalLM vs ImageTextToText)
    is isolated from the device move so a real GPU error surfaces (mirrors recall_v1.load_base).
    trust_remote_code=True lets members with bundled modeling code (e.g. Laguna's modeling_laguna.py
    + auto_map) load without registering their model_type in transformers."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    m, load_err = None, None
    for loader in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            m = loader.from_pretrained(
                model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
            )
            break
        except (ValueError, KeyError) as e:
            load_err = e
    if m is None:
        raise load_err
    m = m.to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    m.config.use_cache = False
    return m, tok


@torch.no_grad()
def probe_model(model_id, anchors, tap_frac):
    """Forward-only relative-rep probe of ONE base. Returns a card dict (CPU tensors)."""
    base, tok = load_base(model_id)
    layers = decoder_layers(base)
    n_layers = len(layers)
    tap_layer = max(0, min(n_layers - 1, round(tap_frac * n_layers)))

    # Capture the residual hidden state at the OUTPUT of decoder layer `tap_layer` (HF decoder layers
    # return a tuple; out[0] is the hidden state) — the exact stream v0's MAG tap injects into.
    captured = {}

    def hook(_mod, _inp, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = layers[tap_layer].register_forward_hook(hook)
    embed = base.get_input_embeddings()

    abs_vecs = []
    tok_counts = []
    try:
        for text in anchors:
            ids = tok(text, return_tensors="pt").input_ids.to(DEV)
            _ = base(input_ids=ids)                       # forward to fire the tap hook
            h = captured["h"][0]                          # [T, d_base]  (batch=1)
            v = h.float().mean(dim=0)                     # mean-pool over tokens -> [d_base]
            abs_vecs.append(v.cpu())
            tok_counts.append(int(ids.shape[1]))
    finally:
        handle.remove()

    A = torch.stack(abs_vecs)                              # [n_anchor, d_base], fp32 on CPU
    d_base = A.shape[1]
    # CENTER before cosine — the critical relative-rep normalization (Moschella 2209.15430). The raw
    # mean-pooled residual at a deep tap is dominated by a few "massive-activation" / rogue dimensions
    # (a near-constant per-model bias vector with norm in the thousands), so cos(A_i, A_j) on the RAW
    # vectors collapses toward 1.0 and barely distinguishes anchors. Subtracting the anchor-set mean
    # removes that shared bias and exposes the genuine per-anchor geometry; THIS centered cosine is the
    # tokenizer/dim-agnostic alignment key the atlas fuses. We keep R_raw too for diagnostics.
    Ac = A - A.mean(dim=0, keepdim=True)                   # center across anchors
    Acn = torch.nn.functional.normalize(Ac, dim=1)
    R = Acn @ Acn.t()                                      # [n_anchor, n_anchor] centered rel-rep
    An = torch.nn.functional.normalize(A, dim=1)
    R_raw = An @ An.t()                                    # uncentered (diagnostic only)

    card = {
        "model": model_id,
        "n_layers": n_layers,
        "tap_frac": float(tap_frac),
        "tap_layer": int(tap_layer),
        "d_base": int(d_base),
        "n_anchor": A.shape[0],
        "anchor_sha": anchor_sha(),                       # reproducibility witness (same bank?)
        "categories": list(CATEGORIES),                   # content axis per anchor (diagnostics)
        "tok_counts": tok_counts,
        "A": A,                                           # absolute tap embeds (per-anchor)
        "R": R,                                           # CENTERED relative-rep matrix (the atlas key)
        "R_raw": R_raw,                                   # uncentered cosine (diagnostic)
    }

    # free the base before the next model (16GB card; CONTINUANCE base-1-free pattern)
    handle = None
    del base, tok, layers, embed, captured, abs_vecs
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return card


def sanity(card):
    """Print + assert basic sanity: finite, anchors are distinguished (R off-diagonal < 1, spread)."""
    A, R = card["A"], card["R"]
    n = card["n_anchor"]
    finite = bool(torch.isfinite(A).all() and torch.isfinite(R).all())
    diag = R.diagonal()
    off = R[~torch.eye(n, dtype=torch.bool)]
    # are all anchors distinct?  min pairwise cos should be well below the self-similarity of 1.0
    max_off = float(off.max())
    mean_off = float(off.mean())
    min_off = float(off.min())
    a_norm = float(A.norm(dim=1).mean())
    diag_ok = bool(torch.allclose(diag, torch.ones(n), atol=1e-4))
    # After centering, distinct anchors must show REAL spread: the mean off-diagonal centered cosine
    # should be well away from 1.0 (a near-1 mean would mean the probe still collapses all anchors).
    distinct = mean_off < 0.9 and max_off < 0.999
    verdict = "OK" if (finite and diag_ok and distinct) else "CHECK"
    print(f"  [{card['model']}] L={card['tap_layer']}/{card['n_layers']} "
          f"d_base={card['d_base']} | A.norm~{a_norm:.2f} "
          f"| R off-diag min/mean/max {min_off:.3f}/{mean_off:.3f}/{max_off:.3f} "
          f"| finite={finite} diag1={diag_ok} distinct={distinct} -> {verdict}")
    return verdict == "OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="HF model ids to probe (cached)")
    ap.add_argument("--tap-frac", type=float, default=TAP_FRAC)
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "ckpt", "probe"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bank_path = save_bank(os.path.join(args.out_dir, "anchor_bank.pt"))
    print(f"[probe] device={DEV} n_anchor={len(ANCHORS)} tap_frac={args.tap_frac:.4f} "
          f"sha={anchor_sha()[:16]} ({len(args.models)} models) out={args.out_dir}")
    print(f"[probe] saved fixed anchor bank -> {bank_path}")

    all_ok = True
    cards = []
    for model_id in args.models:
        print(f"[probe] -> {model_id}")
        try:
            card = probe_model(model_id, ANCHORS, args.tap_frac)
        except Exception as e:                            # SKIP-and-record (OOM / unloadable / quant)
            print(f"  [SKIP] {model_id}: {type(e).__name__}: {str(e)[:200]}")
            gc.collect()
            if DEV == "cuda":
                torch.cuda.empty_cache()
            all_ok = False
            continue
        ok = sanity(card)
        all_ok = all_ok and ok
        slug = model_id.replace("/", "__")
        path = os.path.join(args.out_dir, f"{slug}.pt")
        torch.save(card, path)
        print(f"  saved {path}  (A {tuple(card['A'].shape)} / R {tuple(card['R'].shape)})")
        cards.append(card)

    # cross-model alignment smoke: the relative-rep matrices live in the SAME [n_anchor, n_anchor]
    # space regardless of d_base/tokenizer, so we can already measure cross-model R-correlation — the
    # quantity the atlas fuses. High (but <1) correlation = the Platonic-convergence signal the atlas
    # banks on; it must NOT be a trivial 1.0 (would mean the probe lost all model-specific geometry).
    if len(cards) >= 2:
        print("[probe] cross-model relative-rep correlation (off-diagonal, Pearson):")
        n = cards[0]["n_anchor"]
        mask = ~torch.eye(n, dtype=torch.bool)
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                ri = cards[i]["R"][mask]
                rj = cards[j]["R"][mask]
                rho = float(torch.corrcoef(torch.stack([ri, rj]))[0, 1])
                print(f"    {cards[i]['model']}  vs  {cards[j]['model']}:  rho={rho:+.3f}")

    print(f"[probe] {'ALL OK' if all_ok else 'SOME CHECK — inspect above'}: "
          f"{len(cards)} cards written to {args.out_dir}")


if __name__ == "__main__":
    main()
