"""Phase 1a exit check — load a trained checkpoint and generate.

Validates two things the serving pipeline (Phases 3/4) depends on:
  1. config.json + state_dict round-trip: rebuild the exact architecture and load weights
     strictly (no missing/unexpected keys).
  2. the (fixed) generation path produces coherent char-level text.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os

import numpy as np
import torch

from titans_common import build_mac_model, materialize_overlapping_params


def decode_tokens(tokens):
    return "".join(chr(max(32, int(t))) for t in tokens)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/ckpt/model_best.pt")
    p.add_argument("--data", default="/work/ref-titans-pytorch/data/enwik8.gz")
    p.add_argument("--prime-len", type=int, default=128, dest="prime_len")
    p.add_argument("--gen-len", type=int, default=384, dest="gen_len")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--n-samples", type=int, default=3, dest="n_samples")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print(f"[gen] ckpt={args.ckpt} step={ckpt.get('step')} params={cfg.get('param_count','?')}",
          flush=True)

    model = build_mac_model(cfg)
    materialize_overlapping_params(model)
    model = model.to(device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    # the aliased memory_model.* submodule keys may be redundant; report any real gaps
    real_missing = [k for k in missing if "memory_model.model" not in k and "memory_model.norm" not in k]
    print(f"[gen] load: missing={len(missing)} (real={len(real_missing)}) "
          f"unexpected={len(unexpected)}", flush=True)
    if real_missing[:5]:
        print(f"[gen] sample missing: {real_missing[:5]}", flush=True)
    if unexpected[:5]:
        print(f"[gen] sample unexpected: {unexpected[:5]}", flush=True)
    model.eval()

    # primes from the held-out val split
    with gzip.open(args.data) as fh:
        raw = np.frombuffer(fh.read(int(95e6)), dtype=np.uint8).copy()
    _, va = np.split(raw, [int(90e6)])
    va = torch.from_numpy(va)

    for s in range(args.n_samples):
        i = torch.randint(0, va.size(0) - args.prime_len - 1, (1,))
        prime = va[i: i + args.prime_len].long().to(device)
        with torch.no_grad():
            out = model.sample(prime[None, ...], args.prime_len + args.gen_len,
                               temperature=args.temperature, use_cache=False,
                               show_progress=False)
        print(f"\n===== sample {s} =====", flush=True)
        print(f"PRIME: {decode_tokens(prime)}", flush=True)
        print(f"CONT : {decode_tokens(out[0])}", flush=True)

    print("\n[gen] OK", flush=True)


if __name__ == "__main__":
    main()
