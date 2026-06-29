"""Phase 1a — tiny from-scratch convergence run for Titans MAC on enwik8 (byte-level).

Goal: prove the architecture CONVERGES on real data on gfx1201 (not just overfits a
batch), and produce a small checkpoint + config.json that the serving pipeline
(Phases 2-4) can load. ~30M params by default; time-bounded for a few-hour run.

RDNA4 flags (use_accelerated_scan / use_flex_attn = False) come from titans_common.

Run via gpu-lease (foreground job; lease releases when it exits), checkpoints to a
mounted host dir:
  scripts/gpu-lease.sh -n 1 -- bash -c 'docker run ... titans:dev -lc "... \
    python train_enwik8.py --out /ckpt --max-minutes 180"'
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from titans_common import (
    build_mac_model,
    materialize_overlapping_params,
    dedup_params,
    default_config,
)


def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_tokens(tokens):
    return "".join(chr(max(32, int(t))) for t in tokens)


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, device):
        self.data, self.seq_len, self.device = data, seq_len, device

    def __getitem__(self, index):
        i = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        return self.data[i: i + self.seq_len + 1].long().to(self.device)

    def __len__(self):
        return self.data.size(0) // self.seq_len


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/work/ref-titans-pytorch/data/enwik8.gz")
    p.add_argument("--out", default="/ckpt")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512, dest="seq_len")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4, dest="grad_accum")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-minutes", type=float, default=180.0, dest="max_minutes")
    p.add_argument("--max-steps", type=int, default=100000, dest="max_steps")
    p.add_argument("--val-every", type=int, default=200, dest="val_every")
    p.add_argument("--gen-every", type=int, default=1000, dest="gen_every")
    p.add_argument("--save-every", type=int, default=1000, dest="save_every")
    p.add_argument("--gen-len", type=int, default=256, dest="gen_len")
    p.add_argument("--prime-len", type=int, default=128, dest="prime_len")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)

    print(f"[train] torch={torch.__version__} hip={getattr(torch.version,'hip',None)} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    # ---- config + model ----
    cfg = default_config(dim=args.dim, depth=args.depth, seq_len=args.seq_len)
    model = build_mac_model(cfg).to(device)
    n_mat = materialize_overlapping_params(model)
    params = dedup_params(model)
    n_params = sum(t.numel() for t in params)
    cfg["param_count"] = n_params
    print(f"[train] params={n_params/1e6:.2f}M materialized={n_mat} "
          f"mem_layers={cfg['neural_memory_layers']}", flush=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ---- data ----
    with gzip.open(args.data) as fh:
        raw = np.frombuffer(fh.read(int(95e6)), dtype=np.uint8).copy()
    tr, va = np.split(raw, [int(90e6)])
    tr, va = torch.from_numpy(tr), torch.from_numpy(va)
    train_loader = cycle(DataLoader(TextSamplerDataset(tr, args.seq_len, device),
                                    batch_size=args.batch, shuffle=True))
    val_loader = cycle(DataLoader(TextSamplerDataset(va, args.seq_len, device),
                                  batch_size=args.batch, shuffle=True))

    # ---- optimizer (reference's AdoptAtan2; params are contiguous after materialize) ----
    try:
        from adam_atan2_pytorch import AdoptAtan2
        optim = AdoptAtan2(params, lr=args.lr)
        print("[train] optimizer=AdoptAtan2", flush=True)
    except Exception as e:  # pragma: no cover
        optim = torch.optim.Adam(params, lr=args.lr)
        print(f"[train] optimizer=Adam (AdoptAtan2 unavailable: {e})", flush=True)

    LN2 = 0.6931471805599453
    log_path = os.path.join(args.out, "train_log.jsonl")
    log = open(log_path, "a")
    t0 = time.time()
    best_val = float("inf")

    def save(tag):
        path = os.path.join(args.out, f"model_{tag}.pt")
        torch.save({"state_dict": model.state_dict(), "config": cfg, "step": step}, path)
        print(f"[train] saved {path}", flush=True)

    step = 0
    while step < args.max_steps:
        model.train()
        optim.zero_grad()
        for _ in range(args.grad_accum):
            loss = model(next(train_loader), return_loss=True)
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        optim.step()

        if step % 25 == 0:
            mins = (time.time() - t0) / 60
            print(f"[train] step {step:6d} loss {loss.item():.4f} "
                  f"bpc {loss.item()/LN2:.3f} t {mins:.1f}m", flush=True)

        if step % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                vloss = model(next(val_loader), return_loss=True).item()
            rec = dict(step=step, train_loss=loss.item(), val_loss=vloss,
                       val_bpc=vloss / LN2, minutes=(time.time() - t0) / 60)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"[train] === val step {step}: loss {vloss:.4f} bpc {vloss/LN2:.3f} ===",
                  flush=True)
            if vloss < best_val:
                best_val = vloss
                save("best")

        if args.gen_every and step % args.gen_every == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                prime = next(val_loader)[0, : args.prime_len]
                # sample()'s 2nd arg is TOTAL length (prime + new), and it returns only
                # the continuation (out[..., prompt_seq_len:]).
                sample = model.sample(prime[None, ...], args.prime_len + args.gen_len,
                                      use_cache=False, show_progress=False)
            print(f"[train] --- gen @ {step} ---\nPRIME: {decode_tokens(prime)}\n"
                  f"CONT : {decode_tokens(sample[0])}\n---", flush=True)

        if step % args.save_every == 0 and step > 0:
            save("last")

        if (time.time() - t0) / 60 >= args.max_minutes:
            print(f"[train] hit time budget {args.max_minutes}m at step {step}", flush=True)
            break
        step += 1

    save("last")
    print(f"[train] DONE step={step} best_val_bpc={best_val/LN2:.3f} "
          f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)
    log.close()


if __name__ == "__main__":
    main()
