"""CAM build step 3 (SMOKE) — train the product-key sparse store + multi-head reads against the
canonical-Z hub, on the generalized-DocBuilder associative-recall task, with MAG-tap injection, on the
v0 base (Qwen3.5-4B). CANONICAL_BUILD_PLAN §3 (the dense + 2:4 arms), §1.3 (PKM + multi-head + MAG),
§6 (pin step-time / steps-to-converge / store size / VRAM — the 1-hr smoke).

PIPELINE (one episode = one batch of recall docs):
  1. Build a content-diverse, capacity-stressed associative-recall doc (generalized DocBuilder: M
     name->cargo bindings, leak-free; M>v0's 3 stresses store capacity / interference).
  2. Derive HUB-space (key,value) per binding from the FROZEN base embeddings via a learned hub
     in-projection (base_embed d_base -> d_hub=4096, the smoke's "into-hub translator"): key=cargo,
     value=name. WRITE them into the per-episode product-key store (error-correcting delta).
  3. Derive the HUB-space read query from the QA cargo (leak-free) and READ the store with the three
     specialised heads (factual/positional/recency) -> a d_hub read vector.
  4. Project the read back to base space (d_hub -> d_base) and INJECT it through the FROZEN base via the
     zero-init MAG tap (gated_tap.GatedMemoryTap; reused verbatim — its bank is the [B,K,mem_dim] read).
  5. LM-loss through the frozen base on the answer token. Only the store + hub projections + tap train.

The canonical-Z hub is loaded from ckpt/atlas/canonical_z_v1_local6.pt: the store's product-key
sub-codebooks are SEEDED from the atlas anchor keys, so the store addresses in the committee's
base-neutral geometry. The frozen v0 memory (cam_v0_L24.pt) is NOT reused here: step 3 trains a FRESH
store (the v0 DeepMemory front-end is a different mechanism; the plan's "reuse v0" is the tap/translator
scaffold + the de-risked 2:4 masker, both reused). [recorded choice — see CONTINUANCE.]

TWO ARMS: dense (nn.Linear serve weights) + s24 (Mask24Linear 2:4-by-design SR-STE). Smoke the dense
arm first; s24 added if cheap (it is — SR-STE is plain torch, step-0 proved ~0 extra train cost).

Gate discipline: gamma-ALONE zero-init (GatedMemoryTap), NaN/grad skip guard, fp32-compute/cast-back.

Smoke-first-within-the-smoke: --steps 1 prints loss/shapes/store-size before any real run.

Run (1 leased card; absolute arbiter):
  /home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh -n 1 --name titans-memc -- \
    titans/warmstart/run_m2.sh titans-memc --entry warmstart/train_mem_canonical.py -- \
      --atlas warmstart/ckpt/atlas/canonical_z_v1_local6.pt --steps 200 --arms dense,s24
"""
import argparse
import gc
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "deep_mem"))
from m2_adapter import MODEL, DEV, load_frozen_base                       # noqa: E402
from recall_deepmem import NAME_CANDIDATES, CARGO_CANDIDATES, single_token_ids, DocBuilder  # noqa: E402
from gated_tap import GatedMemoryTap, decoder_layers                      # noqa: E402
from sparse24 import Mask24Linear                                         # noqa: E402
from pk_store import ProductKeyStore                                      # noqa: E402

LN2 = math.log(2.0)


def lin(in_f, out_f, arm):
    """dense -> nn.Linear (bias-free); s24 -> Mask24Linear (2:4-by-design). in_f%4 enforced by Mask."""
    if arm == "s24" and in_f % 4 == 0:
        return Mask24Linear(in_f, out_f)
    return nn.Linear(in_f, out_f, bias=False)


class CanonicalMemoryFrontEnd(nn.Module):
    """The trainable step-3 front-end: hub in/out projections + the product-key store + the MAG tap.
    Frozen base is external; this module reads base embeddings (for key/value/query) and emits the
    tap (a GatedMemoryTap reading the store's hub-space output as its [B,K,mem_dim=d_hub] bank)."""

    def __init__(self, d_base, d_hub, anchor_keys, arm="dense", n_sub=32, topk=8, sub_topk=4,
                 n_heads=3, tap_heads=8, k_inject=8, bank_dtype=torch.float32):
        super().__init__()
        self.d_base, self.d_hub, self.k_inject = d_base, d_hub, k_inject
        # value-bank STORAGE dtype (3c): bf16 halves the dominant B*N*d_hub term at large N; reads/writes
        # still compute in fp32 (pk_store casts). Smoke kept fp32; knowledge-store-grade N wants bf16.
        self.bank_dtype = bank_dtype
        # base-embed -> hub: the smoke's "into-hub translator". key/value/query share this in-proj.
        self.to_hub = lin(d_base, d_hub, arm)
        self.store = ProductKeyStore(d_hub, n_sub=n_sub, topk=topk, sub_topk=sub_topk,
                                     n_heads=n_heads, anchor_keys=anchor_keys)
        # the store read (d_hub) -> the K-slot bank the MAG tap consumes (mem_dim = d_hub). We pool the
        # read across query tokens into K learned slots (mirrors BoltAdapter.readout_q).
        self.readout_q = nn.Parameter(torch.randn(k_inject, d_hub) * 0.02)
        # MAG tap: hub-space bank (mem_dim=d_hub) -> base residual. Zero-init gate (gamma alone).
        self.tap = GatedMemoryTap(d_base, d_hub, n_heads=tap_heads)
        if arm == "s24":
            for nm in ("to_q", "to_k", "to_v", "to_o"):
                old = getattr(self.tap, nm)
                new = Mask24Linear(old.in_features, old.out_features)
                setattr(self.tap, nm, new)

    def _hub(self, emb):
        """frozen base embeds [B,L,d_base] (fp32) -> hub [B,L,d_hub]."""
        return self.to_hub(emb.float())

    def write_episode(self, base_embed, ids, builder, apos, return_assoc=False):
        """Address-and-write the M bindings of each doc into a fresh store. Returns the value bank V
        (and, if return_assoc, the hub keys/vals [B,M,d_hub] and the binding cargo token ids [B,M]
        for the addressing-supervision loss).
        key = cargo-token hub embed, value = name-token hub embed, read from the binding block."""
        B = ids.shape[0]
        V = self.store.init_state(B, ids.device, dtype=self.bank_dtype)
        # binding block positions: after bos+header, M bindings of bind_len; dict layout is
        # [cargo, ':', name, '\n'] -> cargo at offset 0, name at offset 2 within each binding.
        hstart = len(builder.bos) + len(builder.header)
        keys_pos, vals_pos = [], []
        for m in range(builder.M):
            base = hstart + m * builder.bind_len
            keys_pos.append(base)                 # cargo token
            vals_pos.append(base + 1 + len(builder.colon))   # name token
        emb = base_embed(ids).float()             # [B,S,d_base]
        keys = self._hub(emb[:, keys_pos])        # [B,M,d_hub]
        vals = self._hub(emb[:, vals_pos])
        Vnew = self.store.write(V, keys, vals)
        if return_assoc:
            cargo_ids = ids[:, keys_pos]          # [B,M] binding cargo token ids (to match the query)
            wk, wv = self.store.write_addr_val(keys, vals)   # [B,M,d_hub] write addresses / stored vals
            return Vnew, wk, wv, cargo_ids
        return Vnew

    def read_inject(self, base_embed, ids, builder, apos):
        """Read the store with the QA cargo (leak-free) -> pool to K slots -> set the tap bank.
        Returns head_norms (specialisation diagnostic). Sets self.tap._bank for the base forward."""
        B = ids.shape[0]
        q_ids = ids[:, builder.qa_start:apos]     # the cargo query tokens (before the answer)
        q_emb = base_embed(q_ids).float()         # [B,Lq,d_base]
        q_hub = self._hub(q_emb)                  # [B,Lq,d_hub]
        return q_hub

    def make_bank(self, V, q_hub, return_ctx=False):
        """store read of the QA query -> attn-pool to K hub-space slots -> tap bank [B,K,d_hub].
        If return_ctx, also returns the per-head retrieved value-mix ctxs (PRE read_o) for the
        addressing-supervision loss."""
        if return_ctx:
            read, head_norms, ctxs = self.store.read(V, q_hub, return_ctx=True)   # [B,Lq,d_hub]
        else:
            read, head_norms = self.store.read(V, q_hub)      # [B,Lq,d_hub]
        B = read.shape[0]
        pq = self.readout_q.unsqueeze(0).expand(B, -1, -1)    # [B,K,d_hub]
        attn = torch.softmax(pq @ read.transpose(1, 2) / (self.d_hub ** 0.5), dim=-1)  # [B,K,Lq]
        bank = attn @ read                                    # [B,K,d_hub]
        if return_ctx:
            return bank, head_norms, ctxs
        return bank, head_norms

    def store_params(self):
        return sum(p.numel() for p in self.store.parameters())


def _leakfree_ctx(base, builder, ids, apos):
    """header (FORMAT only, no bindings) + QA query tokens -> base inputs_embeds (mirrors recall_mag)."""
    hlen = len(builder.bos) + len(builder.header)
    ctx_ids = torch.cat([ids[:, len(builder.bos):hlen], ids[:, builder.qa_start:apos]], dim=1)
    return base.get_input_embeddings()(ctx_ids)


def attach_tap(base, fe, L):
    base_embed = base.get_input_embeddings()
    def hook(module, inp, out):
        if isinstance(out, tuple):
            return (fe.tap(out[0]),) + tuple(out[1:])
        return fe.tap(out)
    return decoder_layers(base)[L].register_forward_hook(hook), base_embed


def run_step(base, base_embed, fe, builder, ids, apos, addr_loss=False):
    """one forward: write episode -> read+inject -> LM loss on the answer token. Returns (logits,
    head_norms, addr) where addr is an addressing-supervision loss (or None). Store write/read in
    fp32; the tap casts back.

    ADDRESSING SUPERVISION (the bypass fix #2): the read QUERY for the queried binding must address
    the SLOT its key wrote to (and the retrieved value must match what was written). Two differentiable
    InfoNCE terms over the M bindings (no hard top-k gradient gap):
      (a) ADDRESS: read_q[0](q_hub) close to the queried binding's write-address wk, far from others.
      (b) VALUE:   the factual head's retrieved value-mix close to the queried binding's STORED value
          wv (= to_wval(name)), far from others.
    This forces reads to DEPEND on store content / addresses, which the LM-loss-only bypass shortcuts.
    The queried binding is found by matching the QA cargo token id (at qa_start) to the binding cargo
    ids."""
    if addr_loss:
        V, wk, wv, cargo_ids = fe.write_episode(base_embed, ids, builder, apos, return_assoc=True)
    else:
        V = fe.write_episode(base_embed, ids, builder, apos)
    q_hub = fe.read_inject(base_embed, ids, builder, apos)
    if addr_loss:
        bank, head_norms, ctxs = fe.make_bank(V, q_hub, return_ctx=True)
    else:
        bank, head_norms = fe.make_bank(V, q_hub)
    fe.tap.set_bank(bank)
    ctx_emb = _leakfree_ctx(base, builder, ids, apos)
    logits = base(inputs_embeds=ctx_emb).logits[:, -1].float()
    fe.tap.set_bank(None)

    addr = None
    if addr_loss:
        # which binding was queried? match QA cargo token (ids[:, qa_start]) to the binding cargo ids.
        q_tok = ids[:, builder.qa_start].unsqueeze(1)            # [B,1]
        tgt = (cargo_ids == q_tok).float().argmax(dim=1)         # [B] queried binding index
        # factual read query, pooled over the QA query tokens -> [B,d_hub]
        rq = fe.store.head_query(q_hub, h=0).mean(dim=1)         # [B,d_hub]
        ctx_fac = ctxs[0].mean(dim=1)                            # [B,d_hub] retrieved value-mix
        # (a) address InfoNCE: read query vs the M write addresses wk
        sa = torch.einsum("bd,bmd->bm", F.normalize(rq, dim=-1), F.normalize(wk, dim=-1)) / 0.1
        # (b) value InfoNCE: retrieved ctx vs the M STORED values wv (= to_wval(name))
        sv = torch.einsum("bd,bmd->bm", F.normalize(ctx_fac, dim=-1), F.normalize(wv, dim=-1)) / 0.1
        addr = F.cross_entropy(sa, tgt) + F.cross_entropy(sv, tgt)
    return logits, head_norms, addr


@torch.no_grad()
def evaluate(base, base_embed, fe, builder, rng, args, n=512, L=24):
    """memory / no_memory / ceiling, mirroring eval_generative_mag. no_memory = empty store (bank from
    a never-written V). ceiling = full in-context doc, tap off.

    The MAG tap injects ONLY via a forward hook on layer L — train_arm removes its hook before
    returning, so evaluate MUST re-attach one or the tap is a silent no-op (set_bank with no hook =
    memory==no_memory EXACTLY; this was the step-3 eval bug). The tap self-no-ops when its bank is
    None (the ceiling pass), so one hook for the whole eval is correct."""
    fe.eval()
    handle, _ = attach_tap(base, fe, L)
    res = {c: [[], 0] for c in ("ceiling", "memory", "no_memory")}
    eval_batch = args.eval_batch if args.eval_batch > 0 else args.batch
    seen = 0
    while seen < n:
        cur = min(eval_batch, n - seen)
        ids, ans, apos = builder.build(rng, cur, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        # ceiling: pure base, full in-context doc, tap OFF
        fe.tap.set_bank(None)
        lc = base(inputs_embeds=base_embed(ids[:, :apos])).logits[:, -1].float()
        ctx_emb = _leakfree_ctx(base, builder, ids, apos)
        q_hub = fe.read_inject(base_embed, ids, builder, apos)
        for cond in ("memory", "no_memory"):
            if cond == "memory":
                V = fe.write_episode(base_embed, ids, builder, apos)
            else:
                V = fe.store.init_state(cur, ids.device, dtype=fe.bank_dtype)   # empty store
            bank, _ = fe.make_bank(V, q_hub)
            fe.tap.set_bank(bank)
            lg = base(inputs_embeds=ctx_emb).logits[:, -1].float()
            lp = F.log_softmax(lg, -1)
            res[cond][0].extend((-lp.gather(-1, ans[:, None]).squeeze(-1) / LN2).tolist())
            res[cond][1] += (lg.argmax(-1) == ans).float().sum().item()
        fe.tap.set_bank(None)
        lp = F.log_softmax(lc, -1)
        res["ceiling"][0].extend((-lp.gather(-1, ans[:, None]).squeeze(-1) / LN2).tolist())
        res["ceiling"][1] += (lc.argmax(-1) == ans).float().sum().item()
        seen += cur
    handle.remove()
    return {c: (float(np.mean(res[c][0])), res[c][1] / seen) for c in res}


def train_arm(base, base_embed, fe, builder, rng, args, arm, L, time_budget=0.0):
    handle, _ = attach_tap(base, fe, L)
    fe.train()
    opt = torch.optim.AdamW(fe.parameters(), lr=args.lr)
    t0 = time.time()
    conv_step, acc_hist, n_done = None, [], 0
    addr_w = args.addr_weight
    for step in range(args.steps):
        if time_budget and (time.time() - t0) > time_budget:
            print(f"[memc][{arm}] time budget {time_budget:.0f}s hit at step {step}", flush=True)
            break
        opt.zero_grad()
        ids, ans, apos = builder.build(rng, args.batch, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        logits, head_norms, addr = run_step(base, base_embed, fe, builder, ids, apos,
                                            addr_loss=(addr_w > 0))
        lm = F.cross_entropy(logits, ans)
        loss = lm + (addr_w * addr if addr is not None else 0.0)
        if not torch.isfinite(loss):
            print(f"[memc][{arm}] step {step}: non-finite loss -> skip", flush=True)
            continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(fe.parameters(), 1.0)
        if not torch.isfinite(gn):
            print(f"[memc][{arm}] step {step}: non-finite grad -> skip", flush=True)
            opt.zero_grad()
            continue
        opt.step()
        n_done = step + 1
        acc = (logits.argmax(-1) == ans).float().mean().item()
        acc_hist.append(acc)
        if conv_step is None and len(acc_hist) >= 10 and np.mean(acc_hist[-10:]) >= args.conv_acc:
            conv_step = step
        if step % args.log_every == 0 or step == args.steps - 1:
            gate = float(fe.tap.last_gate)
            hn = "/".join(f"{x:.2f}" for x in head_norms)
            astr = f" addr {addr.item():.3f}" if addr is not None else ""
            print(f"[memc][{arm}] step {step:4d} loss {lm.item():.3f}{astr} acc {acc:.3f} "
                  f"gate {gate:.4f} head_norms(F/P/R) {hn}", flush=True)
    sec_step = (time.time() - t0) / max(n_done, 1)
    win_acc = float(np.mean(acc_hist[-50:])) if acc_hist else 0.0
    handle.remove()
    return n_done, sec_step, conv_step, win_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="warmstart/ckpt/atlas/canonical_z_v1_local6.pt")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seg-len", type=int, default=32, dest="seg_len")
    ap.add_argument("--qa-seg", type=int, default=2, dest="qa_seg")
    ap.add_argument("--M", type=int, default=8, help="bindings per doc (>v0's 3 = capacity stress)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--arms", type=str, default="dense,s24")
    ap.add_argument("--n-sub", type=int, default=32, dest="n_sub", help="codebook size/half -> N=n_sub^2 slots")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--sub-topk", type=int, default=4, dest="sub_topk")
    ap.add_argument("--read-heads", type=int, default=3, dest="read_heads")
    ap.add_argument("--tap-heads", type=int, default=8, dest="tap_heads")
    ap.add_argument("--k-inject", type=int, default=8, dest="k_inject")
    ap.add_argument("--tap-layer", type=int, default=24, dest="tap_layer")
    ap.add_argument("--addr-weight", type=float, default=1.0, dest="addr_weight",
                    help="weight of the write->read addressing-supervision (InfoNCE) loss; 0 disables")
    ap.add_argument("--conv-acc", type=float, default=0.90, dest="conv_acc")
    ap.add_argument("--bank-dtype", type=str, default="fp32", dest="bank_dtype",
                    choices=["fp32", "bf16"], help="value-bank STORAGE dtype; bf16 halves VRAM at large N")
    ap.add_argument("--eval-batch", type=int, default=0, dest="eval_batch",
                    help="eval batch (0 = use --batch); a SMALL eval batch caps the value-bank VRAM at large N")
    ap.add_argument("--eval-n", type=int, default=512, dest="eval_n")
    ap.add_argument("--log-every", type=int, default=25, dest="log_every")
    ap.add_argument("--time-budget", type=float, default=0.0, dest="time_budget")
    args = ap.parse_args()

    atlas_path = args.atlas if os.path.isabs(args.atlas) else os.path.join(os.path.dirname(_HERE), args.atlas)
    atlas = torch.load(atlas_path, map_location="cpu", weights_only=False)
    Z = atlas["Z"]                                # [102, 4096] unit-norm canonical keys
    d_hub = atlas["d_hub"]
    print(f"[memc] atlas <- {atlas_path}: Z {tuple(Z.shape)} d_hub={d_hub} anchor_sha "
          f"{atlas['anchor_sha'][:12]} relrep={atlas['relrep']}", flush=True)

    torch.manual_seed(args.seed)
    base, tok = load_frozen_base()
    d_base = base.config.get_text_config().hidden_size
    n_layers = base.config.get_text_config().num_hidden_layers
    base_embed = base.get_input_embeddings()
    L = min(args.tap_layer, n_layers - 1)

    names = single_token_ids(tok, NAME_CANDIDATES)
    cargo = single_token_ids(tok, CARGO_CANDIDATES, prefix="")
    builder = DocBuilder(tok, names, cargo, args.M, args.seg_len, args.qa_seg, phrasing="dict")
    print(f"[memc] {MODEL} d_base={d_base} n_layers={n_layers} tap L={L} | M={args.M} "
          f"(v0 was 3) | names {len(names)} cargo {len(cargo)} | chance {1/args.M:.3f} | "
          f"store N={args.n_sub**2} slots topk={args.topk} heads={args.read_heads} | arms={args.arms} "
          f"steps={args.steps} batch={args.batch} budget={args.time_budget}s", flush=True)

    bank_dtype = torch.bfloat16 if args.bank_dtype == "bf16" else torch.float32
    bank_bytes = 2 if args.bank_dtype == "bf16" else 4
    eval_batch = args.eval_batch if args.eval_batch > 0 else args.batch
    print(f"[memc] value-bank dtype={args.bank_dtype} ({bank_bytes}B) | N={args.n_sub**2} | "
          f"train-batch {args.batch} -> bank {args.batch*args.n_sub**2*d_hub*bank_bytes/1e6:.0f}MB | "
          f"eval-batch {eval_batch} -> bank {eval_batch*args.n_sub**2*d_hub*bank_bytes/1e6:.0f}MB", flush=True)

    rows = []
    for arm in [a for a in args.arms.split(",") if a]:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        fe = CanonicalMemoryFrontEnd(d_base, d_hub, Z, arm=arm, n_sub=args.n_sub, topk=args.topk,
                                     sub_topk=args.sub_topk, n_heads=args.read_heads,
                                     tap_heads=args.tap_heads, k_inject=args.k_inject,
                                     bank_dtype=bank_dtype).to(DEV)
        n_params = sum(p.numel() for p in fe.parameters())
        store_params = fe.store_params()
        # smoke-first: one step, print shapes/loss/store-size before the run
        ids, ans, apos = builder.build(rng, args.batch, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        handle, _ = attach_tap(base, fe, L)
        with torch.no_grad():
            logits, head_norms, addr0 = run_step(base, base_embed, fe, builder, ids, apos,
                                                 addr_loss=(args.addr_weight > 0))
            loss0 = F.cross_entropy(logits, ans).item()
        handle.remove()
        addr0s = f" addr {addr0.item():.3f}" if addr0 is not None else ""
        print(f"[memc][{arm}] SMOKE step0: logits {tuple(logits.shape)} loss {loss0:.3f}{addr0s} "
              f"head_norms {[round(x,3) for x in head_norms]} | trainable {n_params/1e6:.2f}M "
              f"(store {store_params/1e6:.2f}M, value-bank/episode "
              f"{args.batch*args.n_sub**2*d_hub*bank_bytes/1e6:.0f}MB {args.bank_dtype})",
              flush=True)
        # reseed so training starts from the same stream as the smoke consumed
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        n_steps, sec_step, conv_step, win_acc = train_arm(
            base, base_embed, fe, builder, rng, args, arm, L, time_budget=args.time_budget)
        gen = evaluate(base, base_embed, fe, builder, rng, args, n=args.eval_n, L=L)
        m_acc, nm_acc, ceil = gen["memory"][1], gen["no_memory"][1], gen["ceiling"][1]
        m_nll, nm_nll = gen["memory"][0], gen["no_memory"][0]
        passed = (m_acc > nm_acc + 0.15 and m_acc > 0.5)
        if DEV == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1e9
        else:
            peak = 0.0
        print(f"\n[memc][{arm}] === step-3 generative through frozen base ===", flush=True)
        for c in ("ceiling", "memory", "no_memory"):
            print(f"  {c:>10} NLL {gen[c][0]:8.3f}  acc {gen[c][1]:.3f}", flush=True)
        print(f"[memc][{arm}] memory {m_acc:.3f} / no_memory {nm_acc:.3f} / ceiling {ceil:.3f}; "
              f"ΔNLL {nm_nll - m_nll:+.3f} bits; train {n_steps} steps @ {sec_step:.3f}s/step; "
              f"converged@{conv_step}; peak {peak:.2f}GB; {'PASS' if passed else 'FAIL'}", flush=True)
        rows.append((arm, m_acc, nm_acc, ceil, nm_nll - m_nll, n_steps, sec_step, conv_step, peak, passed))
        del fe, gen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    print("\n[memc] === STEP-3 SMOKE SUMMARY (canonical-Z PKM store + multi-head + MAG) ===", flush=True)
    print(f"{'arm':>6} {'memory':>7} {'no_mem':>7} {'ceiling':>8} {'ΔNLL':>9} {'steps':>6} "
          f"{'s/step':>7} {'conv@':>6} {'peakGB':>7} {'verdict':>7}", flush=True)
    for arm, m, nm, c, dn, ns, ss, cs, pk, p in rows:
        print(f"{arm:>6} {m:7.3f} {nm:7.3f} {c:8.3f} {dn:9.3f} {ns:6d} {ss:7.3f} "
              f"{str(cs):>6} {pk:7.2f} {'PASS' if p else 'FAIL':>7}", flush=True)
    d = {a: m for a, m, *_ in rows}
    if "dense" in d and "s24" in d:
        print(f"[memc] FIDELITY DELTA (s24 − dense) memory-acc = {d['s24'] - d['dense']:+.3f}", flush=True)
    print("[memc] PASS rule: memory > no_memory+0.15 and > 0.5. Smoke pins step-time/conv/store/VRAM.",
          flush=True)


if __name__ == "__main__":
    main()
