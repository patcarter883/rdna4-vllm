"""CAM v0 — the MAG falsifier. Does an additive ZERO-INIT Memory-as-Gate tap deliver the validated
DeepMemory binding through the FROZEN base, where boltA's Memory-as-Context prefix hit the wall
(memory ≈ no_memory)?  Full spec: titans/V0_SPEC.md.

Stage 1 (binding): train the BoltAdapter by the direct tied-unembed loss (reused from recall_boltA);
                    freeze it. This is the validated 0.86-carry binding.
Stage 2 (delivery): freeze base + memory; train ONLY the GatedMemoryTap(s) by LM-loss-through-the-
                    frozen-base on the recall task. Eval mirrors boltA: local_control / memory /
                    no_memory (NLL bits + accuracy). Default = sweep each tap depth independently.

Run (1 leased card; absolute arbiter path):
  /home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh -n 1 --name titans-magv0 -- \
    titans/warmstart/run_m2.sh titans-magv0 --entry warmstart/recall_mag.py -- \
      --bind-steps 3000 --steps 3000 --tap-layers 8,16,24
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "deep_mem"))
from m2_adapter import MODEL, DEV, load_frozen_base                       # noqa: E402
from recall_deepmem import NAME_CANDIDATES, CARGO_CANDIDATES, single_token_ids, DocBuilder  # noqa: E402
from recall_boltA import BoltAdapter, eval_direct                         # noqa: E402
from gated_tap import MAGInjector                                         # noqa: E402

LN2 = math.log(2.0)


# ---- memory bank: K query-conditioned pooled retrieval vectors (pre out_proj), mirrors BoltAdapter.inject
def memory_bank(adapter, ids, seg_len, qa_start, answer_pos, carry=True):
    """[B,K,mem_dim] — the leak-free memory bank fed to the MAG taps (mem_dim, NOT base-embed space)."""
    emb = adapter._e(ids)                                                 # frozen embed->in_proj->norm
    B = emb.shape[0]
    state = adapter.mem.init_state(B)
    if carry:
        for s in range(0, qa_start, seg_len):
            state = adapter.mem(emb[:, s:s + seg_len], state)             # ingest pre-QA context
    q = emb[:, qa_start:answer_pos]                                       # cargo query (leak-free)
    retrieved = adapter.mem.retrieve(q, state)                           # [B,Lq,mem_dim]
    pq = adapter.readout_q.unsqueeze(0).expand(B, -1, -1)
    attn = torch.softmax(pq @ retrieved.transpose(1, 2) / (adapter.mem_dim ** 0.5), dim=-1)
    return attn @ retrieved                                              # [B,K,mem_dim] pooled


def _leakfree_ctx(base, builder, ids, apos):
    """header (FORMAT only, no bindings) + query tokens -> base inputs_embeds. Same context boltA used."""
    hlen = len(builder.bos) + len(builder.header)
    ctx_ids = torch.cat([ids[:, len(builder.bos):hlen], ids[:, builder.qa_start:apos]], dim=1)
    return base.get_input_embeddings()(ctx_ids)


# ---- stage 1: bind (direct tied-unembed; no base in the loop) ----------------------------------
def bind_adapter(adapter, builder, rng, args):
    train_params = [p for p in adapter.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=args.lr)
    for step in range(args.bind_steps):
        opt.zero_grad()
        ids, ans, apos = builder.build(rng, args.batch, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        pref = adapter.inject(ids, args.seg_len, builder.qa_start, apos, carry=True)
        loss = F.cross_entropy(adapter.direct_logits(pref), ans)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        if step % 200 == 0 or step == args.bind_steps - 1:
            acc = (adapter.direct_logits(pref).argmax(-1) == ans).float().mean().item()
            print(f"[mag] bind step {step:4d} loss {loss.item():.3f} direct_acc {acc:.3f}", flush=True)
    d_carry, d_abl = eval_direct(adapter, builder, rng, args)
    print(f"[mag] binding held-out: carry {d_carry:.3f} | ablated {d_abl:.3f} | chance {1/args.M:.3f}",
          flush=True)
    for p in adapter.parameters():
        p.requires_grad_(False)
    adapter.eval()
    return d_carry


# ---- stage 2: train the MAG tap(s) by LM-loss through the frozen base ---------------------------
def train_taps(base, adapter, injector, builder, rng, args, tag):
    injector.attach().train()
    opt = torch.optim.AdamW(injector.parameters(), lr=args.lr)
    for step in range(args.steps):
        opt.zero_grad()
        ids, ans, apos = builder.build(rng, args.batch, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        with torch.no_grad():
            bank = memory_bank(adapter, ids, args.seg_len, builder.qa_start, apos, carry=True)
        injector.set_bank(bank)                                          # memory frozen -> bank detached
        ctx_emb = _leakfree_ctx(base, builder, ids, apos)
        logits = base(inputs_embeds=ctx_emb).logits[:, -1].float()
        loss = F.cross_entropy(logits, ans)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(injector.parameters()), 1.0)
        opt.step()
        if step % 200 == 0 or step == args.steps - 1:
            acc = (logits.argmax(-1) == ans).float().mean().item()
            print(f"[mag][{tag}] step {step:4d} loss {loss.item():.3f} acc {acc:.3f} "
                  f"gate {injector.gate_stats()}", flush=True)
    injector.set_bank(None)


@torch.no_grad()
def eval_generative_mag(base, adapter, injector, builder, rng, args, n=512):
    base_embed = base.get_input_embeddings()
    res = {c: [0.0, 0] for c in ("local_control", "memory", "no_memory")}
    nbits = {c: [] for c in res}
    injector.eval()
    seen = 0
    while seen < n:
        cur = min(args.batch, n - seen)
        ids, ans, apos = builder.build(rng, cur, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        injector.set_bank(None)                                          # tap OFF -> ceiling
        lc = base(inputs_embeds=base_embed(ids[:, :apos])).logits[:, -1].float()
        ctx_emb = _leakfree_ctx(base, builder, ids, apos)
        for cond, carry in (("memory", True), ("no_memory", False)):
            bank = memory_bank(adapter, ids, args.seg_len, builder.qa_start, apos, carry=carry)
            injector.set_bank(bank)
            lg = base(inputs_embeds=ctx_emb).logits[:, -1].float()
            lp = F.log_softmax(lg, -1)
            nbits[cond].extend((-lp.gather(-1, ans[:, None]).squeeze(-1) / LN2).tolist())
            res[cond][1] += (lg.argmax(-1) == ans).float().sum().item()
        injector.set_bank(None)
        lp = F.log_softmax(lc, -1)
        nbits["local_control"].extend((-lp.gather(-1, ans[:, None]).squeeze(-1) / LN2).tolist())
        res["local_control"][1] += (lc.argmax(-1) == ans).float().sum().item()
        seen += cur
    return {c: (float(np.mean(nbits[c])), res[c][1] / seen) for c in res}


# ---- checkpoint: persist the frozen v0 memory front-end (BoltAdapter) + a passing GatedMemoryTap ----
# so v1 reuses ONE fixed memory across bases instead of re-binding it each run. The bank fed to the
# taps ([B,K,mem_dim]) is base-AGNOSTIC (DeepMemory's own mem_dim space), so the same checkpoint drives
# any base; only the per-base translator/tap geometry differs.
def save_ckpt(path, adapter, injector, tap_layer, args, d_carry):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # drop the frozen tied embed/unembed (~3GB) — rebuilt from base-1's table on load
    asd = {k: v for k, v in adapter.state_dict().items()
           if not (k.startswith("embed.") or k == "unembed")}
    torch.save({
        "adapter": asd,
        "taps": injector.taps.state_dict(),
        "tap_layer": tap_layer,
        "tap_heads": args.tap_heads,
        "mem_dim": args.mem_dim, "heads": args.heads, "chunk": args.chunk,
        "expansion": args.expansion, "k": args.k, "d_carry": d_carry,
    }, path)
    print(f"[mag] saved v0 memory checkpoint -> {path} (tap L={tap_layer}, carry {d_carry:.3f})", flush=True)


def load_ckpt(path, embed_weight, base, dev):
    """Rebuild the frozen BoltAdapter + GatedMemoryTap from a checkpoint and freeze them. Returns
    (adapter, injector, tap_layer, meta)."""
    ck = torch.load(path, map_location=dev, weights_only=False)
    H = base.config.get_text_config().hidden_size
    adapter = BoltAdapter(embed_weight, H, ck["mem_dim"], ck["heads"], ck["chunk"],
                          ck["expansion"], ck["k"]).to(dev)
    # embed/unembed are not in the ckpt (rebuilt from base-1's table); load the rest strictly-ish
    missing, unexpected = adapter.load_state_dict(ck["adapter"], strict=False)
    assert not unexpected, f"unexpected ckpt keys: {unexpected}"
    assert all(k.startswith("embed.") or k == "unembed" for k in missing), \
        f"unexpected MISSING adapter keys: {missing}"
    for p in adapter.parameters():
        p.requires_grad_(False)
    adapter.eval()
    L = ck["tap_layer"]
    injector = MAGInjector(base, [L], ck["mem_dim"], n_heads=ck["tap_heads"]).to(dev)
    injector.taps.load_state_dict(ck["taps"])
    for p in injector.parameters():
        p.requires_grad_(False)
    injector.eval()
    print(f"[mag] loaded v0 memory checkpoint <- {path} (tap L={L}, carry {ck.get('d_carry', float('nan')):.3f})",
          flush=True)
    return adapter, injector, L, ck


def verdict(tag, d_carry, gen, chance):
    lc = gen["local_control"][1]
    m_acc, nm_acc = gen["memory"][1], gen["no_memory"][1]
    m_nll, nm_nll = gen["memory"][0], gen["no_memory"][0]
    print(f"\n[mag][{tag}] === generative through frozen base ===", flush=True)
    print(f"{'condition':>14} {'NLL(bits)':>11} {'acc':>7}", flush=True)
    for c in ("local_control", "memory", "no_memory"):
        print(f"{c:>14} {gen[c][0]:>11.3f} {gen[c][1]:>7.3f}", flush=True)
    print(f"[mag][{tag}] memory acc {m_acc:.3f} / no_memory {nm_acc:.3f} / ceiling {lc:.3f}; "
          f"ΔNLL {nm_nll - m_nll:+.3f} bits", flush=True)
    if m_acc > nm_acc + 0.15 and m_acc > 0.5:
        v = "MAG WORKS — greenlight v1 (translator + 2nd base)"
    elif m_acc > nm_acc + 0.10 or (nm_nll - m_nll) > 0.5:
        v = "PARTIAL — go multi-layer / data-dependent gate / unfreeze memory gates"
    else:
        v = "WALL at this depth — escalate to multi-layer; if all depths fail, frozen-base premise is the limit"
    print(f"[mag][{tag}] => {v}\n" + "=" * 64, flush=True)
    return m_acc, nm_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind-steps", type=int, default=3000, dest="bind_steps")
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
    ap.add_argument("--tap-heads", type=int, default=8, dest="tap_heads")
    ap.add_argument("--tap-layers", type=str, default="", dest="tap_layers",
                    help="comma list of decoder layers to tap; empty -> [n_layers//2]")
    ap.add_argument("--multi", action="store_true",
                    help="train ALL --tap-layers together (escalation) instead of sweeping each")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260628)
    ap.add_argument("--save-ckpt", type=str, default="", dest="save_ckpt",
                    help="after a single-layer run, save the frozen BoltAdapter+tap to this path")
    ap.add_argument("--load-ckpt", type=str, default="", dest="load_ckpt",
                    help="reload a saved v0 memory checkpoint instead of re-binding; reproduces V0")
    ap.add_argument("--save-anyway", action="store_true", dest="save_anyway",
                    help="save the ckpt even if the tap didn't pass (smoke-test plumbing only)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    base, tok = load_frozen_base()
    H = base.config.get_text_config().hidden_size
    n_layers = base.config.get_text_config().num_hidden_layers
    embed_weight = base.get_input_embeddings().weight.detach().float().clone()

    names = single_token_ids(tok, NAME_CANDIDATES)
    cargo = single_token_ids(tok, CARGO_CANDIDATES, prefix="")
    builder = DocBuilder(tok, names, cargo, args.M, args.seg_len, args.qa_seg, phrasing="dict")

    # ---- RELOAD path: reuse a fixed v0 memory checkpoint (no re-bind) and reproduce the V0 eval ----
    if args.load_ckpt:
        adapter, injector, L, ck = load_ckpt(args.load_ckpt, embed_weight, base, DEV)
        print(f"[mag] {MODEL} | H={H} n_layers={n_layers} | RELOAD tap L={L} | "
              f"K={ck['k']} mem_dim={ck['mem_dim']} | chance acc={1/args.M:.3f}", flush=True)
        injector.attach()
        gen = eval_generative_mag(base, adapter, injector, builder, rng, args)
        m_acc, nm_acc = verdict(str(L), ck.get("d_carry", float("nan")), gen, 1 / args.M)
        injector.detach()
        print("\n[mag] RELOAD SANITY (tap -> memory / no_memory / ceiling):", flush=True)
        print(f"  L={L:>8}  {m_acc:.3f} / {nm_acc:.3f} / {gen['local_control'][1]:.3f}", flush=True)
        print("[mag] PASS if memory ≫ no_memory (reproduces V0 on base-1 from the saved memory).", flush=True)
        return

    layers = ([int(x) for x in args.tap_layers.split(",") if x != ""]
              if args.tap_layers else [n_layers // 2])
    print(f"[mag] {MODEL} | H={H} n_layers={n_layers} | tap_layers={layers} multi={args.multi} | "
          f"K={args.k} mem_dim={args.mem_dim} | chance acc={1/args.M:.3f}", flush=True)

    # ---- stage 1: bind once ----
    adapter = BoltAdapter(embed_weight, H, args.mem_dim, args.heads, args.chunk, args.expansion, args.k).to(DEV)
    d_carry = bind_adapter(adapter, builder, rng, args)

    # ---- stage 2: MAG delivery ----
    configs = [layers] if args.multi else [[L] for L in layers]
    summary = []
    for cfg in configs:
        tag = "+".join(map(str, cfg))
        injector = MAGInjector(base, cfg, args.mem_dim, n_heads=args.tap_heads).to(DEV)
        train_taps(base, adapter, injector, builder, rng, args, tag)
        gen = eval_generative_mag(base, adapter, injector, builder, rng, args)
        m_acc, nm_acc = verdict(tag, d_carry, gen, 1 / args.M)
        summary.append((tag, m_acc, nm_acc, gen["local_control"][1]))
        # save the FIRST passing single-layer tap as the reusable v0 memory checkpoint
        if args.save_ckpt and len(cfg) == 1 and (args.save_anyway or (m_acc > nm_acc + 0.15 and m_acc > 0.5)):
            save_ckpt(args.save_ckpt, adapter, injector, cfg[0], args, d_carry)
            args.save_ckpt = ""  # save only once (the first passing depth)
        injector.detach()

    print("\n[mag] SUMMARY (tap -> memory / no_memory / ceiling):", flush=True)
    for tag, m, nm, lc in summary:
        print(f"  L={tag:>8}  {m:.3f} / {nm:.3f} / {lc:.3f}", flush=True)
    print(f"[mag] boltA reference (MAC): memory ≈ no_memory ≈ 0.000 (the wall this run tests against).",
          flush=True)


if __name__ == "__main__":
    main()
