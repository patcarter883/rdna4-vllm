"""Training-free base-capability + scoring sanity for the recall de-risk.

The first recall_deepmem.py run came back INCONCLUSIVE: local_control failed (the frozen base could
not do the lookup even with the binding in-context), so the memory comparison is meaningless. Before
spending another training run, isolate the question with NO adapter, NO segmentation, NO memory:
in ONE context window, can frozen Qwen3.5-4B answer an associative-recall query, and is my
answer-token scoring aligned? Sweep a few phrasings and report acc / NLL / the base's top-5 guesses.

If some phrasing gives high acc -> task+scoring are sound; fix recall_deepmem's local_control to use
that phrasing (and prepend nothing), then judge memory. If NONE works -> the frozen base can't do
this lookup format at all and the whole probe needs rethinking (or a different base / instruct model).
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
from m2_adapter import MODEL, DEV, load_frozen_base  # noqa: E402
from recall_deepmem import NAME_CANDIDATES, CARGO_CANDIDATES, single_token_ids  # noqa: E402

LN2 = math.log(2.0)


def make_doc(rng, names, cargo, M, phrasing):
    """Return (prompt_str, answer_str). M random bindings, query one. answer is space-prefixed so it
    is a single clean token. Several phrasings to find one the base actually completes."""
    n_idx = rng.choice(len(names), size=M, replace=False)
    c_idx = rng.choice(len(cargo), size=M, replace=False)
    nm = [names[i][0] for i in n_idx]
    cg = [cargo[i][0] for i in c_idx]
    q = int(rng.integers(0, M))
    name_q, cargo_q = nm[q], cg[q]

    if phrasing == "manifest":
        body = " ".join(f"{n} carries {c}." for n, c in zip(nm, cg))
        prompt = f"The manifest lists the following ships. {body} Question : which ship carries {cargo_q} ? Answer :"
        return prompt, f" {name_q}"
    if phrasing == "natural":
        body = " ".join(f"{n} carries {c}." for n, c in zip(nm, cg))
        prompt = f"{body} The ship that carries {cargo_q} is"
        return prompt, f" {name_q}"
    if phrasing == "dict":
        body = "\n".join(f"{c}: {n}" for c, n in zip(cg, nm))
        prompt = f"Cargo to ship:\n{body}\n{cargo_q}:"
        return prompt, f" {name_q}"
    if phrasing == "qa":
        body = " ".join(f"{n} carries {c}." for n, c in zip(nm, cg))
        prompt = f"{body}\nQ: Who carries {cargo_q}?\nA:"
        return prompt, f" {name_q}"
    raise ValueError(phrasing)


@torch.no_grad()
def eval_phrasing(base, tok, names, cargo, rng, M, phrasing, n, batch, show=1):
    # BPE context-merges make prompt lengths RAGGED across rows (the old constant-length assert
    # crashed here). Score per-row at each row's own answer position: right-pad sequences to a common
    # length (causal LM -> pad tokens follow each answer position and never affect the read logits)
    # and gather logits[row, prompt_len_row - 1].
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (
        tok.eos_token_id if tok.eos_token_id is not None else 0)
    nlls, accs, shown = [], [], 0
    done = 0
    while done < n:
        cur = min(batch, n - done)
        fulls, golds, ans_pos = [], [], []
        for _ in range(cur):
            p, a = make_doc(rng, names, cargo, M, phrasing)
            pid = tok(p, add_special_tokens=True).input_ids
            fid = tok(p + a, add_special_tokens=True).input_ids
            fulls.append(fid)
            golds.append(fid[len(pid)])          # first answer token
            ans_pos.append(len(pid))             # predict this token from logits[len(pid)-1]
        Lmax = max(len(f) for f in fulls)
        x = torch.full((cur, Lmax), pad_id, dtype=torch.long, device=DEV)
        for r, f in enumerate(fulls):
            x[r, :len(f)] = torch.tensor(f, device=DEV)
        all_logits = base(input_ids=x).logits.float()               # [cur, Lmax, V]
        rows = torch.arange(cur, device=DEV)
        ap = torch.tensor(ans_pos, device=DEV)
        logits = all_logits[rows, ap - 1]                           # [cur, V] per-row answer logits
        gold = torch.tensor(golds, device=DEV)
        logp = F.log_softmax(logits, dim=-1)
        nlls.extend((-logp.gather(-1, gold[:, None]).squeeze(-1) / LN2).tolist())
        accs.extend((logits.argmax(-1) == gold).float().tolist())
        if shown < show:                          # show one worked example with the base's top-5
            top = logits[0].topk(5).indices.tolist()
            print(f"  [{phrasing}] eg prompt={tok.decode(fulls[0][:ans_pos[0]])!r}", flush=True)
            print(f"           gold={tok.decode([golds[0]])!r} top5={[tok.decode([t]) for t in top]}",
                  flush=True)
            shown += 1
        done += cur
    return float(np.mean(nlls)), float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260624)
    ap.add_argument("--phrasings", nargs="+",
                    default=["manifest", "natural", "dict", "qa"])
    args = ap.parse_args()

    base, tok = load_frozen_base()
    names = single_token_ids(tok, NAME_CANDIDATES)
    cargo = single_token_ids(tok, CARGO_CANDIDATES)
    print(f"[basecheck] {MODEL} | single-token names={len(names)} cargo={len(cargo)} | "
          f"M={args.M} n={args.n} | chance acc={1/args.M:.3f} nll={math.log2(args.M):.2f} bits\n",
          flush=True)

    print(f"{'phrasing':>10} {'NLL (bits)':>12} {'accuracy':>10}", flush=True)
    print("-" * 36, flush=True)
    best = None
    for ph in args.phrasings:
        rng = np.random.default_rng(args.seed)
        nll, acc = eval_phrasing(base, tok, names, cargo, rng, args.M, ph, args.n, args.batch)
        print(f"{ph:>10} {nll:>12.3f} {acc:>10.3f}\n", flush=True)
        if best is None or acc > best[1]:
            best = (ph, acc, nll)
    print("=" * 36, flush=True)
    print(f"[basecheck] best phrasing = {best[0]!r} acc={best[1]:.3f} nll={best[2]:.3f} bits", flush=True)
    if best[1] > 0.6:
        print(f"[basecheck] => base CAN do the in-context lookup with {best[0]!r}. Task+scoring sound; "
              f"rebuild the recall probe on this phrasing, then judge memory.", flush=True)
    else:
        print("[basecheck] => frozen base FAILS the in-context lookup on all phrasings. The probe "
              "format (or this base) is the problem — rethink before any memory claim.", flush=True)


if __name__ == "__main__":
    main()
