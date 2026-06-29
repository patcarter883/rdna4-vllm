"""Fix A — decoupled bolt-on: DIRECT-train the adapter (binding + injection), then test whether the
FROZEN base emits the recalled answer from the injected prefix.

WHY (the diagnosis that led here):
  - MQAR / diagnostic D: the DeepMemory BINDS frozen Qwen embeddings at ~0.95 with a DIRECT loss + input
    normalization. The memory core is sound.
  - e2e (train the adapter only by LM-backprop through the frozen 32-layer base): FAILS — the through-base
    gradient is too weak/indirect to learn the binding from scratch, with or without normalization.
  => Decouple training from the base. Qwen3.5-4B has TIED embeddings, so the input-embedding table doubles
     as the output unembedding. Train the WHOLE adapter with a DIRECT tied-unembed readout:
        logits = out_proj(pooled_retrieval) @ embed.T ;  CE vs the answer token        (NO base in the loop)
     This gives both the binding (in_proj/norm/mem) AND the injection (out_proj) a strong, short-path
     gradient. Because embeddings are tied, an out_proj output that the unembed maps to the answer token
     IS (a vector in) the answer token's input-embedding direction — exactly what we prepend at inference.

THE ONE REMAINING UNKNOWN, tested by generative eval through the FROZEN base:
   Given the adapter now delivers the answer into the prefix, does the frozen base actually EMIT it when
   that prefix is prepended (input-embeds injection efficacy)?
  - generative memory >> no_memory and approaching local_control  => the bolt-on WORKS end to end.
  - generative memory ≈ no_memory despite high DIRECT acc        => input-embeds prefix can't drive the
       frozen base (binding+delivery are fine) => escalate the injection point (multi-layer KV), NOT the memory.

Generalization = MQAR criterion: FRESH random pairings every batch, SAME single-token vocab train/eval.

Run (1 leased card):
  scripts/gpu-lease.sh -n 1 --name titans-bolta -- \
    titans/warmstart/run_m2.sh titans-bolta --entry warmstart/recall_boltA.py -- --steps 3000
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "deep_mem"))
from m2_adapter import MODEL, DEV, load_frozen_base  # noqa: E402
from recall_deepmem import NAME_CANDIDATES, CARGO_CANDIDATES, single_token_ids, DocBuilder  # noqa: E402
from deep_memory import DeepMemory  # noqa: E402

LN2 = math.log(2.0)


class BoltAdapter(nn.Module):
    """Frozen Qwen embed -> in_proj -> norm -> DeepMemory; query-conditioned retrieve at the QA cargo;
    attn-pool (readout_q) -> out_proj -> K prefix vectors in base-embedding space. Trained DIRECTLY via
    the tied unembedding (no base); evaluated generatively through the frozen base."""

    def __init__(self, embed_weight, base_hidden, mem_dim, heads, chunk, expansion, k):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)   # FROZEN, tied table
        self.in_proj = nn.Linear(base_hidden, mem_dim, bias=False)
        self.norm = nn.LayerNorm(mem_dim)                                      # diagnostic-D necessity
        self.mem = DeepMemory(dim=mem_dim, heads=heads, chunk_size=chunk, expansion=expansion)
        self.readout_q = nn.Parameter(torch.randn(k, mem_dim) * 0.02)
        self.out_proj = nn.Linear(mem_dim, base_hidden, bias=False)
        self.mem_dim = mem_dim
        self.register_buffer("unembed", embed_weight.t().contiguous())        # [base_hidden, vocab] tied

    def _e(self, ids):
        return self.norm(self.in_proj(self.embed(ids).float()))               # [B,L,mem_dim] normalized

    def inject(self, ids, seg_len, qa_start, answer_pos, carry=True):
        """-> K prefix vectors [B,K,base_hidden] (out_proj of the query-conditioned pooled retrieval)."""
        emb = self._e(ids)
        B = emb.shape[0]
        state = self.mem.init_state(B)
        if carry:
            for s in range(0, qa_start, seg_len):
                state = self.mem(emb[:, s:s + seg_len], state)               # ingest pre-QA context
        q = emb[:, qa_start:answer_pos]                                      # cargo query (leak-free)
        retrieved = self.mem.retrieve(q, state)                             # [B,Lq,mem_dim]
        pq = self.readout_q.unsqueeze(0).expand(B, -1, -1)
        attn = torch.softmax(pq @ retrieved.transpose(1, 2) / (self.mem_dim ** 0.5), dim=-1)
        pooled = attn @ retrieved                                           # [B,K,mem_dim]
        return self.out_proj(pooled)                                        # [B,K,base_hidden]

    def direct_logits(self, prefix):
        """tied-unembed readout of the injected prefix (mean over K) -> vocab logits. Training signal."""
        return prefix.mean(dim=1) @ self.unembed                            # [B,vocab]


def labels_of(ans, name_idx, device):
    return torch.tensor([name_idx[int(t)] for t in ans], device=device)


@torch.no_grad()
def eval_direct(model, builder, rng, args, n=2048):
    """Held-out DIRECT acc: does the injected prefix encode the answer (tied-unembed argmax)? carry vs ablated."""
    model.eval()
    acc = {"carry": 0, "ablated": 0}
    seen = 0
    while seen < n:
        cur = min(args.batch, n - seen)
        ids, ans, apos = builder.build(rng, cur, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        for cond in ("carry", "ablated"):
            pref = model.inject(ids, args.seg_len, builder.qa_start, apos, carry=(cond == "carry"))
            pred = model.direct_logits(pref).argmax(-1)
            acc[cond] += (pred == ans).float().sum().item()
        seen += cur
    model.train()
    return acc["carry"] / seen, acc["ablated"] / seen


@torch.no_grad()
def eval_generative(base, model, builder, rng, args, n=512):
    """Through the FROZEN base. memory/no_memory: prepend the injected prefix to [header ; query] (header =
    FORMAT only, no bindings) and read the answer logits. local_control: base sees the full in-context doc,
    no prefix (ceiling). Returns dict cond -> (nll_bits, acc)."""
    base_embed = base.get_input_embeddings()
    hlen = len(builder.bos) + len(builder.header)
    res = {c: [0.0, 0] for c in ("local_control", "memory", "no_memory")}
    nbits = {c: [] for c in res}
    seen = 0
    while seen < n:
        cur = min(args.batch, n - seen)
        ids, ans, apos = builder.build(rng, cur, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        # local_control: full doc in-context (header+bindings+query), no prefix
        full_emb = base_embed(ids[:, :apos])
        lc = base(inputs_embeds=full_emb).logits[:, -1].float()
        # header (format, no binding) + query tokens -> the base context for the injected conditions
        ctx_ids = torch.cat([ids[:, len(builder.bos):hlen], ids[:, builder.qa_start:apos]], dim=1)
        ctx_emb = base_embed(ctx_ids)
        for cond, carry in (("memory", True), ("no_memory", False)):
            pref = model.inject(ids, args.seg_len, builder.qa_start, apos, carry=carry).to(ctx_emb.dtype)
            inp = torch.cat([pref, ctx_emb], dim=1)
            res_logits = base(inputs_embeds=inp).logits[:, -1].float()
            lp = F.log_softmax(res_logits, -1)
            nbits[cond].extend((-lp.gather(-1, ans[:, None]).squeeze(-1) / LN2).tolist())
            res[cond][1] += (res_logits.argmax(-1) == ans).float().sum().item()
        lp = F.log_softmax(lc, -1)
        nbits["local_control"].extend((-lp.gather(-1, ans[:, None]).squeeze(-1) / LN2).tolist())
        res["local_control"][1] += (lc.argmax(-1) == ans).float().sum().item()
        seen += cur
    return {c: (float(np.mean(nbits[c])), res[c][1] / seen) for c in res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seg-len", type=int, default=32, dest="seg_len")
    ap.add_argument("--qa-seg", type=int, default=2, dest="qa_seg")
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--mem-dim", type=int, default=512, dest="mem_dim")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--expansion", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260625)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    base, tok = load_frozen_base()
    H = base.config.get_text_config().hidden_size
    embed_weight = base.get_input_embeddings().weight.detach().float().clone()

    names = single_token_ids(tok, NAME_CANDIDATES)
    cargo = single_token_ids(tok, CARGO_CANDIDATES, prefix="")          # dict: line-initial no-space cargo
    name_idx = {tid: i for i, (_, tid) in enumerate(names)}             # (only used by ablation sanity)
    builder = DocBuilder(tok, names, cargo, args.M, args.seg_len, args.qa_seg, phrasing="dict")
    print(f"[boltA] {MODEL} | names={len(names)} cargo={len(cargo)} | M={args.M} K={args.k} "
          f"mem_dim={args.mem_dim} steps={args.steps} | chance acc={1/args.M:.3f}", flush=True)

    model = BoltAdapter(embed_weight, H, args.mem_dim, args.heads, args.chunk, args.expansion, args.k).to(DEV)
    train_params = [p for n, p in model.named_parameters() if p.requires_grad]
    print(f"[boltA] trainable params (excl frozen embed): "
          f"{sum(p.numel() for p in train_params)/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(train_params, lr=args.lr)

    # ---- DIRECT training (no base in the loop): tied-unembed readout of the injected prefix ----
    for step in range(args.steps):
        opt.zero_grad()
        ids, ans, apos = builder.build(rng, args.batch, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        pref = model.inject(ids, args.seg_len, builder.qa_start, apos, carry=True)
        loss = F.cross_entropy(model.direct_logits(pref), ans)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            acc = (model.direct_logits(pref).argmax(-1) == ans).float().mean().item()
            print(f"[boltA] step {step:4d} loss {loss.item():.3f} direct_train_acc {acc:.3f}", flush=True)

    # ---- held-out DIRECT acc (binding + delivery into the prefix) ----
    d_carry, d_abl = eval_direct(model, builder, rng, args)
    print(f"\n[boltA] held-out DIRECT (prefix tied-unembed): carry {d_carry:.3f} | ablated {d_abl:.3f} | "
          f"chance {1/args.M:.3f}", flush=True)

    # ---- GENERATIVE eval through the FROZEN base (the injection-efficacy test) ----
    gen = eval_generative(base, model, builder, rng, args)
    print("\n[boltA] === generative through frozen base ===", flush=True)
    print(f"{'condition':>14} {'NLL (bits)':>12} {'accuracy':>10}", flush=True)
    for c in ("local_control", "memory", "no_memory"):
        print(f"{c:>14} {gen[c][0]:>12.3f} {gen[c][1]:>10.3f}", flush=True)

    # ---- verdict ----
    lc_acc = gen["local_control"][1]
    m_acc, nm_acc = gen["memory"][1], gen["no_memory"][1]
    m_nll, nm_nll = gen["memory"][0], gen["no_memory"][0]
    print("\n" + "=" * 64, flush=True)
    print(f"[verdict] direct binding+delivery: carry {d_carry:.3f} (chance {1/args.M:.3f})", flush=True)
    print(f"[verdict] generative: memory acc {m_acc:.3f} / no_memory {nm_acc:.3f} / ceiling {lc_acc:.3f}; "
          f"ΔNLL {nm_nll - m_nll:+.3f} bits", flush=True)
    if d_carry < 0.6:
        print("[verdict] => DIRECT binding/delivery failed — unexpected given diagnostic D; check tied "
              "unembed / training before reading the generative result.", flush=True)
    elif m_acc > nm_acc + 0.15 and m_acc > 0.5:
        print("[verdict] => BOLT-ON WORKS END-TO-END: the adapter delivers the binding AND the frozen base "
              "emits it from the input-embeds prefix. Recall premise holds → green-light M3 / kernel.", flush=True)
    elif m_acc > nm_acc + 0.10 or (nm_nll - m_nll) > 0.5:
        print("[verdict] => PARTIAL: real but weak injection. The prefix helps but doesn't fully drive the "
              "frozen base — tune K / prefix or consider a stronger injection point.", flush=True)
    else:
        print("[verdict] => INJECTION-MECHANISM WALL: binding+delivery are SOLVED (direct carry high) but the "
              "frozen base does NOT emit the answer from the input-embeds prefix (memory ≈ no_memory). The "
              "input-embeds prefix is the blocker → escalate the injection point (multi-layer KV), NOT the "
              "memory.", flush=True)


if __name__ == "__main__":
    main()
