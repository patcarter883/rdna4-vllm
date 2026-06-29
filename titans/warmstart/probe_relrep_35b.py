"""CAM step 1-2 (a) — 35B-A3B committee relative-rep probe (HF, 2-card device_map shard).

Dedicated probe for cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit (model_type qwen3_5_moe, a multimodal
GDN+MoE hybrid, compressed-tensors pack-quantized int4). It OOM'd at TP=1 on one 16GB card in the
prior fan-out, so this variant shards the frozen base across BOTH leased cards with
device_map="auto" (gpu-lease -n 2). Otherwise it mirrors probe_relrep.py EXACTLY:
  - same fixed 102-anchor bank (anchor_bank.py, sha 28a4acf8),
  - same proportional tap depth L = round((24/36) * n_layers) over the TEXT decoder layers,
  - mean-pool per anchor -> A[102, d_base],
  - same centered-cosine R (format-identical to the 5 existing cards),
PLUS R_zscore — the per-dim z-score-of-centered-A rel-rep the Zaya increment proved the atlas needs
(plain centering collapses on a single-rogue-dim model; z-scoring is a strict superset). The card
stores A so the atlas builder can recompute any normalization uniformly across all members.

Text-only forward (input_ids, no pixels) so the vision tower is bypassed — we probe the LM residual.

Run (BOTH leased cards; absolute arbiter path):
  /home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh -n 2 --name titans-probe35b -- \
    titans/warmstart/run_probe35b.sh titans-probe35b
"""
import gc
import hashlib
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from gated_tap import decoder_layers                                       # noqa: E402
from anchor_bank import ANCHORS, CATEGORIES, anchor_sha, save_bank         # noqa: E402

MODEL = os.environ.get("PROBE_MODEL", "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
TAP_FRAC = 24.0 / 36.0
OUT_DIR = os.path.join(_HERE, "ckpt", "probe")
SLUG = os.environ.get("PROBE_SLUG", "qwen36_35b_a3b")


def load_base_sharded(model_id):
    """Load + freeze the quantized GDN-MoE base, sharded across all visible cards (device_map=auto).
    Loader selection (CausalLM vs ImageTextToText) is isolated so a real GPU/quant error surfaces."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    m, load_err = None, None
    for loader in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            m = loader.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",            # shard the int4 GDN-MoE across BOTH leased cards
            )
            break
        except (ValueError, KeyError) as e:
            load_err = e
    if m is None:
        raise load_err
    m = m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    if hasattr(m, "config"):
        m.config.use_cache = False
    return m, tok


@torch.no_grad()
def probe(model_id, anchors, tap_frac, smoke_only=False):
    base, tok = load_base_sharded(model_id)
    layers = decoder_layers(base)
    n_layers = len(layers)
    tap_layer = max(0, min(n_layers - 1, round(tap_frac * n_layers)))
    print(f"[probe35b] {model_id}: n_layers={n_layers} tap_layer={tap_layer} "
          f"(frac {tap_frac:.4f})", flush=True)

    captured = {}

    def hook(_mod, _inp, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = layers[tap_layer].register_forward_hook(hook)
    # input embeddings live on some card; route input_ids there for the lookup.
    embed = base.get_input_embeddings()
    in_dev = next(embed.parameters()).device

    abs_vecs, tok_counts = [], []
    try:
        for i, text in enumerate(anchors):
            ids = tok(text, return_tensors="pt").input_ids.to(in_dev)
            _ = base(input_ids=ids)
            h = captured["h"][0]                          # [T, d_base]
            if i == 0:
                print(f"[probe35b] SMOKE anchor0 tapped residual shape={tuple(captured['h'].shape)} "
                      f"dtype={captured['h'].dtype} dev={captured['h'].device}", flush=True)
                if smoke_only:
                    break
            v = h.float().mean(dim=0).cpu()               # mean-pool over tokens -> [d_base]
            abs_vecs.append(v)
            tok_counts.append(int(ids.shape[1]))
            if (i + 1) % 20 == 0:
                print(f"[probe35b]   {i + 1}/{len(anchors)} anchors", flush=True)
    finally:
        handle.remove()

    if smoke_only:
        del base, tok, layers
        gc.collect()
        torch.cuda.empty_cache()
        return None

    A = torch.stack(abs_vecs)                              # [n_anchor, d_base] fp32 CPU
    d_base = A.shape[1]

    # R: centered cosine — format-identical to the 5 existing cards.
    Ac = A - A.mean(dim=0, keepdim=True)
    Acn = torch.nn.functional.normalize(Ac, dim=1)
    R = Acn @ Acn.t()
    # R_raw: uncentered (diagnostic).
    An = torch.nn.functional.normalize(A, dim=1)
    R_raw = An @ An.t()
    # R_zscore: per-dim z-score of the centered A, then cosine — the robust atlas key (Zaya fix).
    std = Ac.std(dim=0, keepdim=True).clamp_min(1e-6)
    Az = Ac / std
    Azn = torch.nn.functional.normalize(Az, dim=1)
    R_zscore = Azn @ Azn.t()

    card = {
        "model": model_id,
        "geometry": "MoE",                                # GDN+MoE hybrid; tag MoE like the roster
        "n_layers": n_layers,
        "tap_frac": float(tap_frac),
        "tap_layer": int(tap_layer),
        "d_base": int(d_base),
        "n_anchor": A.shape[0],
        "anchor_sha": anchor_sha(),
        "categories": list(CATEGORIES),
        "tok_counts": tok_counts,
        "A": A,
        "R": R,
        "R_raw": R_raw,
        "R_zscore": R_zscore,
    }
    del base, tok, layers
    gc.collect()
    torch.cuda.empty_cache()
    return card


def report(card):
    n = card["n_anchor"]
    mask = ~torch.eye(n, dtype=torch.bool)
    for key in ("R", "R_zscore"):
        M = card[key]
        off = M[mask]
        print(f"[probe35b] {key}: off-diag min/mean/max/std "
              f"{float(off.min()):+.3f}/{float(off.mean()):+.3f}/{float(off.max()):+.3f}/"
              f"{float(off.std()):.3f}  finite={bool(torch.isfinite(M).all())}", flush=True)
    print(f"[probe35b] A.norm~{float(card['A'].norm(dim=1).mean()):.2f} "
          f"d_base={card['d_base']} L={card['tap_layer']}/{card['n_layers']}", flush=True)

    # rho vs the existing committee cards, recomputed BOTH ways from each card's stored A.
    for fn in sorted(os.listdir(OUT_DIR)):
        if not fn.endswith(".pt") or fn in ("anchor_bank.pt",) or fn.startswith(SLUG):
            continue
        if fn in ("Zyphra__ZAYA1-8B.pt",):                # zaya.pt is the dedup'd copy; skip the twin
            continue
        try:
            other = torch.load(os.path.join(OUT_DIR, fn), map_location="cpu", weights_only=False)
        except Exception:
            continue
        if other.get("anchor_sha") != card["anchor_sha"] or other.get("n_anchor") != n:
            print(f"[probe35b]   (skip {fn}: anchor mismatch)", flush=True)
            continue
        oA = other["A"]
        def zscore_R(Am):
            c = Am - Am.mean(0, keepdim=True)
            z = c / c.std(0, keepdim=True).clamp_min(1e-6)
            zn = torch.nn.functional.normalize(z, dim=1)
            return zn @ zn.t()
        rz = zscore_R(oA)[mask]
        rho_z = float(torch.corrcoef(torch.stack([card["R_zscore"][mask], rz]))[0, 1])
        rho_c = float(torch.corrcoef(torch.stack([card["R"][mask], other["R"][mask]]))[0, 1])
        print(f"[probe35b]   vs {other.get('model', fn):40s} rho_zscore={rho_z:+.3f} "
              f"rho_centered={rho_c:+.3f}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    save_bank(os.path.join(OUT_DIR, "anchor_bank.pt"))
    print(f"[probe35b] cuda={torch.cuda.is_available()} n_gpu={torch.cuda.device_count()} "
          f"n_anchor={len(ANCHORS)} sha={anchor_sha()[:16]} model={MODEL}", flush=True)

    smoke = os.environ.get("PROBE_SMOKE", "0") == "1"
    card = probe(MODEL, ANCHORS, TAP_FRAC, smoke_only=smoke)
    if smoke:
        print("[probe35b] SMOKE OK — load + tap confirmed; re-run without PROBE_SMOKE for all 102.",
              flush=True)
        return

    report(card)
    out = os.path.join(OUT_DIR, f"{SLUG}.pt")
    torch.save(card, out)
    # also write the model-slug name so it stacks like the HF cards
    torch.save(card, os.path.join(OUT_DIR, MODEL.replace("/", "__") + ".pt"))
    print(f"[probe35b] saved {out}  (A {tuple(card['A'].shape)} / R {tuple(card['R'].shape)} / "
          f"R_zscore {tuple(card['R_zscore'].shape)})", flush=True)


if __name__ == "__main__":
    main()
