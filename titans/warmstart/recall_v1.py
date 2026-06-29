"""CAM v1 — the base-agnostic translator FALSIFIER (CAM_DESIGN §2.2, §6 v1).

V0 proved: a zero-init MAG tap delivers the validated DeepMemory binding through a FROZEN base
(Qwen3.5-4B). v1 asks the product question: does ONE frozen memory serve a SECOND base (different
d_base) through only a TINY learned translator?

Pipeline:
  1. Load the frozen v0 memory checkpoint (BoltAdapter + a passing GatedMemoryTap @ L=8 or L=24).
     The bank it produces ([B,K,mem_dim]) is base-AGNOSTIC — DeepMemory's own mem_dim retrieval, built
     from the adapter's OWN frozen base-1 embedding, never base-2's space. Same memory, any base.
  2. Load base-2 (frozen, different hidden dim) via load_frozen_base2().
  3. Fit a TINY affine translator (A: d_base2->d_base1, B: d_base1->d_base2 + zero-init gamma2) that
     stitches base-2's residual stream into the frozen tap and back. Train ONLY the translator by
     LM-loss through frozen base-2 on the same recall task. base-2 + memory + tap all frozen.
  4. Eval = same memory/no_memory/ceiling on base-2. PASS = memory >> no_memory with only a tiny fit.

Run (1 leased card; absolute arbiter path):
  /home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh -n 1 --name titans-v1 -- \
    titans/warmstart/run_m2.sh titans-v1 --entry warmstart/recall_v1.py -- \
      --load-ckpt /work/warmstart/ckpt/cam_v0_L24.pt --steps 3000
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
from m2_adapter import MODEL, DEV                                          # noqa: E402
from recall_deepmem import NAME_CANDIDATES, CARGO_CANDIDATES, single_token_ids, DocBuilder  # noqa: E402
from recall_mag import memory_bank, load_ckpt                             # noqa: E402
from translator import TranslatedInjector, save_translator                # noqa: E402

LN2 = math.log(2.0)
# 2nd base, overridable via --base2. Default = the v1 same-family base (Qwen3-0.6B, d=1024).
# Cross-family falsifier base = unsloth/Llama-3.2-3B (d=3072, Llama tiktoken vocab, bos=128000,
# plain LlamaForCausalLM) — a genuinely DIFFERENT tokenizer + architecture, the decisive test that
# the translator isn't exploiting Qwen-family vocab/embedding similarity.
MODEL2 = "Qwen/Qwen3-0.6B"


def load_base(model_id):
    """Load + freeze any HF causal LM (pure torch). Loader selection (CausalLM vs
    ImageTextToText) is isolated from the device move / grad-ckpt setup so a real GPU error
    (OOM, HIP) surfaces instead of being masked as a bogus 'unrecognized config' fallback."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    m, load_err = None, None
    for loader in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            m = loader.from_pretrained(model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True)
            break
        except (ValueError, KeyError) as e:  # config not recognized by THIS AutoModel -> try the next
            load_err = e
    if m is None:
        raise load_err
    m = m.to(DEV).eval()                                    # device move OUTSIDE the loader fallback
    for p in m.parameters():
        p.requires_grad_(False)
    m.config.use_cache = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return m, tok


def _leakfree_ctx(base2, builder2, ids2, apos2):
    """header (FORMAT only, no bindings) + query tokens -> base-2 inputs_embeds (base-2 vocab)."""
    hlen = len(builder2.bos) + len(builder2.header)
    ctx_ids = torch.cat([ids2[:, len(builder2.bos):hlen], ids2[:, builder2.qa_start:apos2]], dim=1)
    return base2.get_input_embeddings()(ctx_ids)


def train_translator(base2, adapter, injector, builder1, builder2, rng, args):
    injector.attach().train()
    opt = torch.optim.AdamW(injector.A_params(), lr=args.lr)
    for step in range(args.steps):
        opt.zero_grad()
        # SAME random recall instance for base-1 (bank) and base-2 (context): build with shared rng
        # state so the bindings/query match; each base tokenizes with its own DocBuilder.
        seed = int(rng.integers(0, 2**31 - 1))
        r1 = np.random.default_rng(seed); r2 = np.random.default_rng(seed)
        ids1, ans1, apos1 = builder1.build(r1, args.batch, local=False)
        ids2, ans2, apos2 = builder2.build(r2, args.batch, local=False)
        ids1, ids2, ans2 = ids1.to(DEV), ids2.to(DEV), ans2.to(DEV)
        with torch.no_grad():
            bank = memory_bank(adapter, ids1, args.seg_len, builder1.qa_start, apos1, carry=True)
        injector.set_bank(bank)                                          # memory frozen -> bank detached
        ctx_emb = _leakfree_ctx(base2, builder2, ids2, apos2)
        logits = base2(inputs_embeds=ctx_emb).logits[:, -1].float()
        loss = F.cross_entropy(logits, ans2)
        if not torch.isfinite(loss):                                     # NaN/Inf guard: skip the step
            print(f"[v1] step {step:4d} NON-FINITE loss -> skip", flush=True)
            opt.zero_grad(); continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(injector.A_params(), 1.0)
        if not torch.isfinite(gn):                                       # NaN grad guard
            print(f"[v1] step {step:4d} NON-FINITE grad -> skip", flush=True)
            opt.zero_grad(); continue
        opt.step()
        with torch.no_grad():                                           # gamma guard against divergence
            injector.tt.gamma2.clamp_(-6.0, 6.0)
        if step % 200 == 0 or step == args.steps - 1:
            acc = (logits.argmax(-1) == ans2).float().mean().item()
            print(f"[v1] step {step:4d} loss {loss.item():.3f} acc {acc:.3f} "
                  f"gate {injector.gate_stat():.4f} |g|grad {float(gn):.2f}", flush=True)
    injector.set_bank(None)


@torch.no_grad()
def eval_v1(base2, adapter, injector, builder1, builder2, rng, args, n=512):
    base_embed = base2.get_input_embeddings()
    res = {c: [0.0, 0] for c in ("local_control", "memory", "no_memory")}
    nbits = {c: [] for c in res}
    injector.eval()
    seen = 0
    while seen < n:
        cur = min(args.batch, n - seen)
        seed = int(rng.integers(0, 2**31 - 1))
        r1 = np.random.default_rng(seed); r2 = np.random.default_rng(seed)
        ids1, ans1, apos1 = builder1.build(r1, cur, local=False)
        ids2, ans2, apos2 = builder2.build(r2, cur, local=False)
        ids1, ids2, ans2 = ids1.to(DEV), ids2.to(DEV), ans2.to(DEV)
        injector.set_bank(None)                                          # tap OFF -> ceiling on base-2
        lc = base2(inputs_embeds=base_embed(ids2[:, :apos2])).logits[:, -1].float()
        ctx_emb = _leakfree_ctx(base2, builder2, ids2, apos2)
        for cond, carry in (("memory", True), ("no_memory", False)):
            bank = memory_bank(adapter, ids1, args.seg_len, builder1.qa_start, apos1, carry=carry)
            injector.set_bank(bank)
            lg = base2(inputs_embeds=ctx_emb).logits[:, -1].float()
            lp = F.log_softmax(lg, -1)
            nbits[cond].extend((-lp.gather(-1, ans2[:, None]).squeeze(-1) / LN2).tolist())
            res[cond][1] += (lg.argmax(-1) == ans2).float().sum().item()
        injector.set_bank(None)
        lp = F.log_softmax(lc, -1)
        nbits["local_control"].extend((-lp.gather(-1, ans2[:, None]).squeeze(-1) / LN2).tolist())
        res["local_control"][1] += (lc.argmax(-1) == ans2).float().sum().item()
        seen += cur
    return {c: (float(np.mean(nbits[c])), res[c][1] / seen) for c in res}


def verdict(gen, chance):
    lc = gen["local_control"][1]
    m_acc, nm_acc = gen["memory"][1], gen["no_memory"][1]
    m_nll, nm_nll = gen["memory"][0], gen["no_memory"][0]
    print(f"\n[v1] === one frozen memory, SECOND base ({MODEL2}) via tiny affine translator ===",
          flush=True)
    print(f"{'condition':>14} {'NLL(bits)':>11} {'acc':>7}", flush=True)
    for c in ("local_control", "memory", "no_memory"):
        print(f"{c:>14} {gen[c][0]:>11.3f} {gen[c][1]:>7.3f}", flush=True)
    print(f"[v1] memory acc {m_acc:.3f} / no_memory {nm_acc:.3f} / ceiling {lc:.3f}; "
          f"ΔNLL {nm_nll - m_nll:+.3f} bits (chance {chance:.3f})", flush=True)
    if m_acc > nm_acc + 0.15 and m_acc > 0.5:
        v = "TRANSLATOR WORKS — one memory serves TWO bases. v1 PASSES (Modular Memory Organ proven)."
    elif m_acc > nm_acc + 0.10 or (nm_nll - m_nll) > 0.5:
        v = "PARTIAL — affine translator helps but doesn't fully transfer; try a wider/nonlinear translator."
    else:
        v = "WALL — affine residual-stitch did not transfer the memory to base-2; escalate translator."
    print(f"[v1] => {v}\n" + "=" * 64, flush=True)
    return m_acc, nm_acc


def main():
    global MODEL2
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-ckpt", type=str, required=True, dest="load_ckpt")
    ap.add_argument("--base2", type=str, default=MODEL2,
                    help="2nd (frozen) base; default Qwen3-0.6B same-family, "
                         "or unsloth/Llama-3.2-3B for the cross-family falsifier")
    ap.add_argument("--save-translator", type=str, default="", dest="save_translator",
                    help="save the fitted translator card (A,B,gamma2 + meta) to this path")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seg-len", type=int, default=32, dest="seg_len")
    ap.add_argument("--qa-seg", type=int, default=2, dest="qa_seg")
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260629)
    args = ap.parse_args()
    MODEL2 = args.base2

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # base-1 ONLY supplies its frozen embedding table to rebuild the adapter; we never run base-1 fwd.
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok1 = AutoTokenizer.from_pretrained(MODEL)
    m1 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                              low_cpu_mem_usage=True).to(DEV).eval()
    for p in m1.parameters():
        p.requires_grad_(False)
    embed_weight = m1.get_input_embeddings().weight.detach().float().clone()
    n1 = m1.config.get_text_config().num_hidden_layers   # base-1 depth (for the proportional tap map)

    # Build the v0 memory front-end (adapter + tap) against base-1's embedding, THEN free base-1's full
    # weights BEFORE loading base-2 — base-1 forward is never used at v1, and a large cross-family base-2
    # (e.g. Llama-3.2-3B ~6GB) won't fit on a 16GB card alongside base-1 (~8GB). load_ckpt only reads
    # base-1's hidden_size off the model, so m1 is sufficient here.
    adapter, injector_tap, tap_layer, ck = load_ckpt(args.load_ckpt, embed_weight, m1, DEV)
    frozen_tap = injector_tap.taps[str(tap_layer)]
    # CRITICAL: MAGInjector.__init__ did `self.layers = decoder_layers(base)` — it holds references to
    # base-1's decoder modules, so the whole Qwen-4B (~8GB) stays alive unless we drop the injector too.
    # At v1 the injector_tap is never attached (we only need the standalone frozen tap); cut its base ref.
    injector_tap.layers = None
    del injector_tap
    del m1                              # free base-1 weights NOW (forward never used at v1)
    del embed_weight                   # the adapter holds its own fp32 self.embed copy now
    # the adapter's tied UNEMBED buffer (~1.5GB fp32) is ONLY used by the DIRECT bind loss, never at v1
    # (memory_bank reads adapter.embed -> in_proj -> mem; it never unembeds). Drop it so a large
    # cross-family base-2 fits on a 16GB card alongside the adapter's fp32 embed.
    if hasattr(adapter, "unembed"):
        adapter.unembed = None
    torch.cuda.empty_cache()

    # base-2: the SECOND base (different d_base), frozen — loaded AFTER base-1 is freed
    base2, tok2 = load_base(MODEL2)
    H2 = base2.config.get_text_config().hidden_size
    n2 = base2.config.get_text_config().num_hidden_layers
    # map the base-1 tap depth to a base-2 depth proportionally (cards carry the tap-layer as metadata)
    tap_layer2 = min(int(round(tap_layer / n1 * n2)), n2 - 1)

    print(f"[v1] base-1={MODEL} (d_base1={frozen_tap.H}, tap L={tap_layer}) | "
          f"base-2={MODEL2} (d_base2={H2}, n_layers={n2}, tap L={tap_layer2}) | "
          f"K={ck['k']} mem_dim={ck['mem_dim']} | chance {1/args.M:.3f}", flush=True)

    # two DocBuilders over the SAME single-token NAME/CARGO words, each in its own base's vocab
    names1 = single_token_ids(tok1, NAME_CANDIDATES); cargo1 = single_token_ids(tok1, CARGO_CANDIDATES, prefix="")
    names2 = single_token_ids(tok2, NAME_CANDIDATES); cargo2 = single_token_ids(tok2, CARGO_CANDIDATES, prefix="")
    builder1 = DocBuilder(tok1, names1, cargo1, args.M, args.seg_len, args.qa_seg, phrasing="dict")
    builder2 = DocBuilder(tok2, names2, cargo2, args.M, args.seg_len, args.qa_seg, phrasing="dict")

    injector = TranslatedInjector(base2, frozen_tap, tap_layer2).to(DEV)
    nparam = sum(p.numel() for p in injector.A_params())
    print(f"[v1] translator trainable params: {nparam/1e6:.3f}M (A {H2}->{frozen_tap.H}, "
          f"B {frozen_tap.H}->{H2}, gamma2 {H2})", flush=True)

    train_translator(base2, adapter, injector, builder1, builder2, rng, args)
    gen = eval_v1(base2, adapter, injector, builder1, builder2, rng, args)
    m_acc, nm_acc = verdict(gen, 1 / args.M)
    if args.save_translator:
        save_translator(args.save_translator, injector, {
            "base2": MODEL2, "tap_layer2": tap_layer2, "steps": args.steps, "lr": args.lr,
            "memory_acc": m_acc, "no_memory_acc": nm_acc, "ceiling": gen["local_control"][1],
        })
    injector.detach()
    print(f"\n[v1] base-2 ({MODEL2}) SUMMARY: memory {m_acc:.3f} / no_memory {nm_acc:.3f} / "
          f"ceiling {gen['local_control'][1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
