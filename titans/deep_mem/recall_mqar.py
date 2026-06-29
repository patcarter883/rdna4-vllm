"""Base-independent recall de-risk for the stage-3 graph-free DeepMemory (the bolt-on memory core).

WHY (not the frozen-base probe): the end-to-end frozen-Qwen probe (warmstart/recall_deepmem.py)
came back INCONCLUSIVE — it confounds the base's (weak ~0.4) in-context associative lookup with the
memory mechanism, so a null is uninterpretable. This probe strips the frozen base out entirely and
asks the one fundamental question the bolt-on rests on:

   Does DeepMemory RETAIN a planted key->value association across a segment/distractor boundary,
   and can a readout RETRIEVE it by key — above an ablated no-carry control and chance?

If NO, the bolt-on is dead regardless of the base. If YES, the (separate, harder) end-to-end
injection question is worth tuning a better base phrasing for. Mirrors the MQAR probe the project
already used to validate the linear-attn CELLS — applied here to the NEW deep-MLP memory.

## The task (MQAR — multi-query associative recall, abstract symbols)
Per example: M distinct (key, value) symbol pairs. "Study" segment = the interleaved pairs
[k1,v1,...,kM,vM]. Then `--distance` filler segments of unrelated symbols (distractors). Then RETRIEVE
the M keys (in random order) from the carried memory state -> readout -> predict each key's value.

The pairing is RE-RANDOMIZED every batch, so nothing about a specific k->v mapping can be memorized
in the trained params — the model MUST route the per-example binding through the memory STATE. The
trained params (embedding, the memory's to_k/v/q + gates, readout) learn the *mechanism*; the deep-MLP
memory weights are per-example test-time state folded by the surprise update.

## Conditions
  carry   : retrieve from the state that ingested study + all filler  -> recall possible
  ablated : retrieve from a FRESH init state (study never ingested)    -> floor (no carry)
Metric: exact-match accuracy (readout argmax == value), carry vs ablated vs chance (1/M among the
M in-example values). Verdict = carry clears ablated and chance by a clear margin.

Runs on CPU in the titans:dev container (small dims) — NO GPU lease needed.
  python -u /work/deep_mem/recall_mqar.py
"""
import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from deep_memory import DeepMemory  # noqa: E402


class MQARProbe(nn.Module):
    """Embedding -> DeepMemory (ingest study+filler, carry state) -> retrieve queries -> readout."""

    def __init__(self, vocab, dim, heads, chunk_size, expansion, conv_kernel=4, conv_init="identity"):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.mem = DeepMemory(dim=dim, heads=heads, chunk_size=chunk_size, expansion=expansion,
                              conv_kernel=conv_kernel, conv_init=conv_init)
        self.readout = nn.Linear(dim, vocab)
        nn.init.normal_(self.embed.weight, std=0.02)

    def study_embed(self, study_il, keys, vals, bind):
        """Build the study-segment input embeddings under the chosen binding.
          interleaved: [k1,v1,...,kM,vM] as 2M separate tokens (needs temporal mixing to bind —
                       DeepMemory has none, so this is the 'as-used-by-the-bolt-on' case).
          superposed : M tokens, each = embed(key)+embed(value) (binding is WITHIN-token — the pure
                       test of the memory's associative-storage capacity, no mixing required)."""
        if bind == "interleaved":
            return self.embed(study_il)                          # [B,2M,dim]
        return self.embed(keys) + self.embed(vals)               # [B,M,dim] superposed

    def forward(self, study_emb, filler_embs, query_emb, carry=True):
        """study_emb[B,Ls,dim], filler_embs list of [B,Lf,dim], query_emb[B,M,dim] -> logits[B,M,vocab].
        carry=False -> retrieve from a fresh init state (ablated floor)."""
        B = study_emb.shape[0]
        state = self.mem.init_state(B)
        if carry:
            state = self.mem(study_emb, state)
            for f in filler_embs:
                state = self.mem(f, state)
        retrieved = self.mem.retrieve(query_emb, state)          # [B,M,dim]
        return self.readout(retrieved)


def make_batch(g, B, M, n_key, n_val, n_fill, distance, fill_len, device):
    """Disjoint per-example key/value/filler symbols so nothing collides. Returns
    study[B,2M], fillers (distance x [B,fill_len]), queries[B,M], targets[B,M]."""
    KEY0, VAL0, FILL0 = 0, n_key, n_key + n_val
    study = torch.empty(B, 2 * M, dtype=torch.long, device=device)
    skeys = torch.empty(B, M, dtype=torch.long, device=device)   # study-order keys / vals
    svals = torch.empty(B, M, dtype=torch.long, device=device)
    queries = torch.empty(B, M, dtype=torch.long, device=device)
    targets = torch.empty(B, M, dtype=torch.long, device=device)
    for b in range(B):
        keys = KEY0 + torch.randperm(n_key, generator=g, device=device)[:M]
        vals = VAL0 + torch.randperm(n_val, generator=g, device=device)[:M]
        skeys[b], svals[b] = keys, vals
        study[b, 0::2] = keys                       # k1,v1,k2,v2,...
        study[b, 1::2] = vals
        order = torch.randperm(M, generator=g, device=device)
        queries[b] = keys[order]
        targets[b] = vals[order]
    fillers = [FILL0 + torch.randint(0, n_fill, (B, fill_len), generator=g, device=device)
               for _ in range(distance)]
    return study, skeys, svals, fillers, queries, targets


@torch.no_grad()
def evaluate(model, g, args, bind, device, n=2048):
    model.eval()
    accs = {"carry": 0, "ablated": 0}
    seen = 0
    while seen < n:
        B = min(args.batch, n - seen)
        study, skeys, svals, fillers, queries, targets = make_batch(
            g, B, args.M, args.n_key, args.n_val, args.n_fill, args.distance, args.fill_len, device)
        se = model.study_embed(study, skeys, svals, bind)
        fe = [model.embed(f) for f in fillers]
        qe = model.embed(queries)
        for cond in ("carry", "ablated"):
            logits = model(se, fe, qe, carry=(cond == "carry"))
            accs[cond] += (logits.argmax(-1) == targets).float().sum().item()
        seen += B
    model.train()
    return accs["carry"] / (seen * args.M), accs["ablated"] / (seen * args.M)


def train_eval(args, bind, device):
    """Fresh model trained + evaluated under one binding mode. Returns (carry_acc, ablated_acc)."""
    g = torch.Generator(device=device).manual_seed(args.seed)
    torch.manual_seed(args.seed)
    vocab = args.n_key + args.n_val + args.n_fill
    model = MQARProbe(vocab, args.dim, args.heads, args.chunk, args.expansion,
                      conv_kernel=args.conv_kernel, conv_init=args.conv_init).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"\n[mqar] === bind={bind} === ({sum(p.numel() for p in model.parameters())/1e3:.1f}K params)",
          flush=True)
    for step in range(args.steps):
        opt.zero_grad()
        study, skeys, svals, fillers, queries, targets = make_batch(
            g, args.batch, args.M, args.n_key, args.n_val, args.n_fill, args.distance,
            args.fill_len, device)
        se = model.study_embed(study, skeys, svals, bind)
        fe = [model.embed(f) for f in fillers]
        logits = model(se, fe, model.embed(queries), carry=True)
        loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == args.steps - 1:
            acc = (logits.argmax(-1) == targets).float().mean().item()
            print(f"[mqar]  step {step:4d} loss {loss.item():.3f} train_acc {acc:.3f}", flush=True)
    return evaluate(model, g, args, bind, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--expansion", type=float, default=2.0)
    ap.add_argument("--M", type=int, default=6, help="key->value pairs per example")
    ap.add_argument("--n-key", type=int, default=32, dest="n_key")
    ap.add_argument("--n-val", type=int, default=32, dest="n_val")
    ap.add_argument("--n-fill", type=int, default=32, dest="n_fill")
    ap.add_argument("--distance", type=int, default=2, help="filler segments between study and retrieve")
    ap.add_argument("--fill-len", type=int, default=16, dest="fill_len")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--conv-kernel", type=int, default=4, dest="conv_kernel",
                    help="depthwise causal conv kernel before the memory ingest (0/1 = no conv)")
    ap.add_argument("--conv-init", default="identity", dest="conv_init",
                    choices=["identity", "mix", "random"],
                    help="identity=passthrough(learn mixing); mix=box filter(value sees key at step0); random")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=20260624)
    ap.add_argument("--bind", nargs="+", default=["interleaved", "superposed"],
                    choices=["interleaved", "superposed"],
                    help="interleaved=[k,v,...] (needs mixing, = bolt-on as-is); "
                         "superposed=embed(k)+embed(v) per token (within-token binding, pure capacity test)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = args.n_key + args.n_val + args.n_fill
    chance = 1.0 / args.M
    print(f"[mqar] device={device} dim={args.dim} heads={args.heads} chunk={args.chunk} "
          f"exp={args.expansion} | M={args.M} distance={args.distance} fill_len={args.fill_len} "
          f"vocab={vocab} | chance acc={chance:.3f}", flush=True)

    res = {}
    for bind in args.bind:
        res[bind] = train_eval(args, bind, device)

    print("\n" + "=" * 60, flush=True)
    print(f"[mqar] held-out recall (n=2048 examples x {args.M} queries, fresh pairings):", flush=True)
    print(f"{'bind':>12} {'carry':>8} {'ablated':>8} {'chance':>8}", flush=True)
    print("-" * 40, flush=True)
    for bind in args.bind:
        c, a = res[bind]
        print(f"{bind:>12} {c:>8.3f} {a:>8.3f} {chance:>8.3f}", flush=True)
    print("=" * 60, flush=True)

    sup = res.get("superposed", (0, 0))[0]
    il = res.get("interleaved", (0, 0))[0]
    if "superposed" in res and sup > 0.6:
        if "interleaved" in res and il > 0.6:
            print(f"[verdict] => DeepMemory RECALLS across distractors in BOTH bindings ({il:.0%} il / "
                  f"{sup:.0%} sup). The memory core is sound; the e2e bolt-on null is a base/injection "
                  f"issue, not the memory.", flush=True)
        else:
            print(f"[verdict] => MEMORY CORE IS SOUND ({sup:.0%} superposed recall) but it CANNOT bind "
                  f"interleaved [k,v] ({il:.0%}). Root cause: DeepMemory ingests context-free tokens "
                  f"with NO temporal mixing — the bolt-on needs a causal conv before the memory (as real "
                  f"GDN has) OR must ingest base hidden states, not raw embeddings. Actionable arch fix.",
                  flush=True)
    elif "superposed" in res:
        print(f"[verdict] => Even WITHIN-TOKEN binding fails ({sup:.0%} superposed). The deep-memory "
              f"store/retrieve itself can't hold associations at this scale — investigate the surprise/"
              f"scan/capacity (dim, heads, expansion, steps) before any bolt-on claim.", flush=True)


if __name__ == "__main__":
    main()
