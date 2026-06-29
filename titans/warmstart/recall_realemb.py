"""Diagnostic D — does the adapter's DeepMemory bind FROZEN Qwen embeddings with a DIRECT loss?

WHY: the e2e bolt-on (recall_deepmem.py) fails — memory NLL worse than no_memory, the answer is not even
decodable from the injection (decode ≈ chance) — yet the memory CORE passes MQAR at 0.90. The two setups
differ in two ways that this probe isolates:
  - MQAR: DIRECT loss readout(retrieve(query))->CE (strong short gradient) on FREELY-LEARNED embeddings.
  - e2e:  loss is LM backprop through the FROZEN 32-layer base (long attenuated gradient) on FROZEN Qwen
          embeddings fed through a learned in_proj.
This probe = MQAR's direct-loss setup but with the REAL frozen Qwen embedding table + the adapter's
in_proj/readout_q machinery, and NO base in the gradient path. It answers exactly one question:

   Can in_proj -> DeepMemory -> readout learn to bind cargo->name across a distractor boundary, using
   FROZEN Qwen token embeddings, when trained with a direct readout loss?

PASS (carry >> ablated/chance) => the memory binds real frozen embeddings; the e2e failure is the
   weak-gradient-through-the-frozen-base => fix = direct-pretrain the adapter to bind, then attach to the base.
FAIL => the frozen embeddings aren't separable for this memory => the bolt-on needs a deeper rethink.

Generalization criterion matches MQAR (the test the core passed): FRESH random pairings every batch,
SAME single-token name/cargo vocab for train and eval (so the readout-over-names units are all trained).
This isolates 'binding a new association', not 'transfer to unseen symbols' (a separate, harder question).

Run (1 leased card; loads the base only to lift its embedding table, then frees it):
  scripts/gpu-lease.sh -n 1 --name titans-realemb -- \
    titans/warmstart/run_m2.sh titans-realemb --entry warmstart/recall_realemb.py -- --steps 3000
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


class RealEmbProbe(nn.Module):
    """Frozen Qwen embed -> in_proj -> DeepMemory (ingest context, carry state) -> query-conditioned
    retrieve at the QA cargo -> attn-pool (readout_q) -> readout over the NAME set. No base."""

    def __init__(self, embed_weight, n_names, mem_dim, heads, chunk, expansion, k):
        super().__init__()
        H = embed_weight.shape[1]
        self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)   # FROZEN Qwen table
        self.in_proj = nn.Linear(H, mem_dim, bias=False)
        # normalize the memory's input: frozen Qwen embeddings are not std~0.02 like MQAR's learned ones,
        # so unnormalized in_proj output saturates the surprise MLP's gelu -> near-zero gradient (flat loss)
        self.norm = nn.LayerNorm(mem_dim)
        self.mem = DeepMemory(dim=mem_dim, heads=heads, chunk_size=chunk, expansion=expansion)
        self.readout_q = nn.Parameter(torch.randn(k, mem_dim) * 0.02)
        self.readout = nn.Linear(mem_dim, n_names)
        self.mem_dim = mem_dim

    def forward(self, ids, seg_len, qa_start, answer_pos, carry=True):
        emb = self.norm(self.in_proj(self.embed(ids).float()))   # [B,S,mem_dim] normalized memory input
        B = emb.shape[0]
        state = self.mem.init_state(B)
        if carry:
            for s in range(0, qa_start, seg_len):            # ingest the context before the QA query
                state = self.mem(emb[:, s:s + seg_len], state)
        q = emb[:, qa_start:answer_pos]                      # cargo query tokens (leak-free: pre-answer)
        retrieved = self.mem.retrieve(q, state)              # [B,Lq,mem_dim] surfaced values
        pq = self.readout_q.unsqueeze(0).expand(B, -1, -1)   # [B,K,mem_dim] pool queries
        attn = torch.softmax(pq @ retrieved.transpose(1, 2) / (self.mem_dim ** 0.5), dim=-1)  # [B,K,Lq]
        pooled = (attn @ retrieved).mean(dim=1)              # [B,mem_dim]
        return self.readout(pooled)                          # [B,n_names]


@torch.no_grad()
def evaluate(model, builder, name_idx, rng, args, n=2048):
    model.eval()
    acc = {"carry": 0, "ablated": 0}
    seen = 0
    while seen < n:
        cur = min(args.batch, n - seen)
        ids, ans, apos = builder.build(rng, cur, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        labels = torch.tensor([name_idx[int(t)] for t in ans], device=DEV)
        for cond in ("carry", "ablated"):
            logits = model(ids, args.seg_len, builder.qa_start, apos, carry=(cond == "carry"))
            acc[cond] += (logits.argmax(-1) == labels).float().sum().item()
        seen += cur
    model.train()
    return acc["carry"] / seen, acc["ablated"] / seen


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
    embed_weight = base.get_input_embeddings().weight.detach().float().clone()   # lift the frozen table
    del base                                                                     # free the 4B
    torch.cuda.empty_cache() if DEV == "cuda" else None

    # SAME vocab for train & eval (MQAR criterion: fresh pairings, not unseen symbols). dict cargo = no-space.
    names = single_token_ids(tok, NAME_CANDIDATES)
    cargo = single_token_ids(tok, CARGO_CANDIDATES, prefix="")
    name_idx = {tid: i for i, (_, tid) in enumerate(names)}
    print(f"[realemb] {MODEL} | names={len(names)} cargo={len(cargo)} | M={args.M} seg_len={args.seg_len} "
          f"qa_seg={args.qa_seg} K={args.k} mem_dim={args.mem_dim} steps={args.steps} | "
          f"chance acc={1/args.M:.3f} (1/M) / {1/len(names):.3f} (1/names)", flush=True)

    builder = DocBuilder(tok, names, cargo, args.M, args.seg_len, args.qa_seg, phrasing="dict")
    model = RealEmbProbe(embed_weight, len(names), args.mem_dim, args.heads, args.chunk,
                         args.expansion, args.k).to(DEV)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[realemb] trainable params (excl frozen embed): {trainable:.2f}M", flush=True)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    for step in range(args.steps):
        opt.zero_grad()
        ids, ans, apos = builder.build(rng, args.batch, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        labels = torch.tensor([name_idx[int(t)] for t in ans], device=DEV)
        logits = model(ids, args.seg_len, builder.qa_start, apos, carry=True)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            acc = (logits.argmax(-1) == labels).float().mean().item()
            print(f"[realemb] step {step:4d} loss {loss.item():.3f} train_acc {acc:.3f}", flush=True)

    carry, ablated = evaluate(model, builder, name_idx, rng, args)
    print("\n" + "=" * 56, flush=True)
    print(f"[realemb] held-out (fresh pairings, n=2048): carry {carry:.3f} | ablated {ablated:.3f} | "
          f"chance {1/args.M:.3f}", flush=True)
    print("=" * 56, flush=True)
    if carry > 0.6 and carry > ablated + 0.2:
        print("[verdict] => PASS: DeepMemory BINDS frozen Qwen embeddings with a direct loss. The e2e "
              "failure is the weak gradient through the frozen base, NOT the memory/embeddings → fix = "
              "direct-pretrain the adapter to bind, then attach to the frozen base.", flush=True)
    elif carry > ablated + 0.1:
        print("[verdict] => PARTIAL: some binding of frozen embeddings but well below MQAR. The frozen "
              "embedding space is harder for the memory; tune (mem_dim/heads/chunk/lr/steps) before "
              "concluding — but the e2e gradient path is still a likely compounding factor.", flush=True)
    else:
        print("[verdict] => FAIL: even with a direct loss the memory cannot bind frozen Qwen embeddings "
              "(carry ≈ ablated ≈ chance). The frozen embedding space / in_proj is the blocker → the "
              "bolt-on needs a rethink (separability, richer projection, or a different memory input).",
              flush=True)


if __name__ == "__main__":
    main()
