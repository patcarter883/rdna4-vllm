"""Print Zaya's cross-model rel-rep rho vs the 4 already-probed local committee members, and where
its CCA geometry sits (central vs outlier). CPU-only; run in any torch env over ckpt/probe/*.pt."""
import os
import sys

import torch

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt", "probe")
members = [
    ("Qwen/Qwen3.5-4B", "Qwen__Qwen3.5-4B.pt"),
    ("Qwen/Qwen3-0.6B", "Qwen__Qwen3-0.6B.pt"),
    ("unsloth/Llama-3.2-3B", "unsloth__Llama-3.2-3B.pt"),
    ("unsloth/gemma-3-4b-pt", "unsloth__gemma-3-4b-pt.pt"),
    ("Zyphra/ZAYA1-8B", "Zyphra__ZAYA1-8B.pt"),
]
cards = {}
for name, fn in members:
    p = os.path.join(D, fn)
    if os.path.exists(p):
        cards[name] = torch.load(p, map_location="cpu", weights_only=False)

names = list(cards)
n = cards[names[0]]["n_anchor"]
mask = ~torch.eye(n, dtype=torch.bool)
print(f"loaded {len(names)} cards (n_anchor={n}); shas: "
      + ", ".join(f"{nm.split('/')[-1]}={cards[nm]['anchor_sha'][:8]}" for nm in names))

# pairwise rho
rho = {nm: {} for nm in names}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        ri, rj = cards[a]["R"][mask], cards[b]["R"][mask]
        r = float(torch.corrcoef(torch.stack([ri, rj]))[0, 1])
        rho[a][b] = rho[b][a] = r
print("\npairwise rho (off-diagonal Pearson):")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        print(f"  {names[i].split('/')[-1]:16s} vs {names[j].split('/')[-1]:16s}: {rho[names[i]][names[j]]:+.3f}")

print("\nmean-rho-to-others (centrality; higher = more central):")
for nm in names:
    others = [rho[nm][o] for o in names if o != nm]
    print(f"  {nm.split('/')[-1]:16s}: {sum(others)/len(others):+.3f}")

z = "Zyphra/ZAYA1-8B"
if z in cards:
    c = cards[z]
    off = c["R"][mask]
    print(f"\nZAYA card: L={c['tap_layer']}/{c['n_layers']} d={c['d_base']} geom={c.get('geometry')} "
          f"A.norm~{float(c['A'].norm(dim=1).mean()):.1f} "
          f"R off-diag min/mean/max {float(off.min()):.3f}/{float(off.mean()):.3f}/{float(off.max()):.3f}")
