#!/usr/bin/env python3
"""Checkpoint-phase PREP smoke: prove the TiDAR ZAYA checkpoint LOADS + a forward runs.

Not the coherence gate — just structural validation before that work:
  1. the fork constructs ZAYA1-8B,
  2. load_state_dict(model_latest.pt) matches keys (report missing/unexpected),
  3. a short causal forward produces finite logits, and a greedy argmax is sane.

CPU by default (host RAM ~17.7 GB model + ckpt; no 16 GB-card OOM, no GPU lease).

Run (fork venv inside the dflash-rxf container, CPU-only):
  docker run --rm -e HF_HOME=/root/.cache/huggingface \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
    -v /home/pat/code/.venv-zaya-fork:/opt/zaya-fork-venv \
    -v /home/pat/code/vllm-gfx1201-tidar-serve/zaya/tidar:/work -w /work \
    --entrypoint bash vllm22-w4a8:dflash-rxf -lc \
    '/opt/zaya-fork-venv/bin/python smoke_load.py'
"""
import os
import sys

import torch

from serve_loader import load_tidar_zaya

CKPT = os.environ.get(
    "TIDAR_CKPT",
    "/root/.cache/huggingface/hub/models--pat883--zaya1-8b-tidar-experts/"
    "snapshots/e6f2ba2d904688059a9e4bd50531504554b02f6d",
)
DEVICE = os.environ.get("TIDAR_DEVICE", "cpu")


def main():
    print(f"[smoke] loading {CKPT} on {DEVICE} …", flush=True)
    model, tok, mask_id, block_size, (missing, unexpected) = load_tidar_zaya(
        CKPT, device=DEVICE, dtype=torch.bfloat16, from_config=True
    )
    print(f"[smoke] mask_id={mask_id} block_size={block_size}", flush=True)
    print(f"[smoke] missing keys: {len(missing)}  unexpected keys: {len(unexpected)}")
    if missing:
        print("  missing[:10]:", missing[:10])
    if unexpected:
        print("  unexpected[:10]:", unexpected[:10])

    emb = model.get_input_embeddings().weight
    mask_row_norm = emb[mask_id].float().norm().item()
    print(f"[smoke] mask-embedding row norm = {mask_row_norm:.4f} (must be > 0 = trained)")

    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(DEVICE)
    print(f"[smoke] prompt ids shape {tuple(ids.shape)}", flush=True)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    logits = out.logits
    finite = torch.isfinite(logits).all().item()
    nxt = int(logits[0, -1].float().argmax())
    print(f"[smoke] logits shape {tuple(logits.shape)} finite={finite}")
    print(f"[smoke] greedy next token id {nxt} -> {tok.decode([nxt])!r}")

    # A few greedy steps as a coherence eyeball (NOT the β=1 gate).
    cur = ids
    gen = []
    for _ in range(12):
        with torch.no_grad():
            lg = model(input_ids=cur, use_cache=False).logits[0, -1]
        t = int(lg.float().argmax())
        gen.append(t)
        cur = torch.cat([cur, torch.tensor([[t]], device=DEVICE)], dim=1)
    print(f"[smoke] greedy continuation: {tok.decode(gen)!r}")

    ok = finite and not missing and mask_row_norm > 0
    print(f"[smoke] RESULT: {'PASS' if ok else 'CHECK'} "
          f"(finite={finite}, no-missing={not missing}, mask-trained={mask_row_norm>0})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
