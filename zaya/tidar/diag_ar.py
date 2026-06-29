#!/usr/bin/env python3
"""DIAGNOSTIC: is full-RECOMPUTE AR == the model's native CACHED AR (generate)?

The β=1 gate uses full-recompute forwards. If the model isn't recompute-stable (it
showed last-position + mask-token drift), the recompute AR reference itself may differ
from the model's true cached AR — meaning the gate must use production KV-cache
semantics, not recompute. This isolates "my methodology" from "TiDAR verify logic".

  native_ar  = model.generate(greedy, use_cache=True)   # stock causal, NO mask patch
  recompute  = full-recompute greedy with the causal-bias monkeypatch (the gate's path)
"""
import os
import torch

import zaya_mask_patch as zmp
from serve_loader import load_tidar_zaya

CKPT = os.environ.get(
    "TIDAR_CKPT",
    "/root/.cache/huggingface/hub/models--pat883--zaya1-8b-tidar-experts/"
    "snapshots/e6f2ba2d904688059a9e4bd50531504554b02f6d",
)
DEVICE = "cpu"
DT = torch.bfloat16
N = 8
PROMPT = "In the beginning was the"


def recompute_ar(model, ids, n):
    ids = list(ids)
    for _ in range(n):
        L = len(ids)
        idx = torch.arange(L)
        b = torch.zeros(L, L, dtype=DT)
        b.masked_fill_(~(idx[:, None] >= idx[None, :]), torch.finfo(DT).min)
        zmp.set_bias(b.view(1, 1, L, L))
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids]), attention_mask=None,
                        position_ids=torch.arange(L).view(1, -1), use_cache=False)
        ids.append(int(out.logits[0, -1].float().argmax()))
    return ids[-n:]


def main():
    model, tok, mask_id, B, _ = load_tidar_zaya(CKPT, device=DEVICE, dtype=DT, from_config=True)
    pid = tok(PROMPT, return_tensors="pt", add_special_tokens=True)["input_ids"]

    # 1) NATIVE cached AR via generate() — NO patch installed (stock causal path).
    with torch.no_grad():
        gen = model.generate(pid, max_new_tokens=N, do_sample=False, use_cache=True,
                             num_beams=1)
    native = gen[0, pid.shape[1]:].tolist()

    # 2) full-recompute AR with the causal monkeypatch (the gate's reference path).
    zmp.install()
    recomp = recompute_ar(model, pid[0].tolist(), N)

    print(f"native  cached AR : {native}")
    print(f"recompute    AR  : {recomp}")
    print(f"MATCH: {native == recomp}")
    print(f"native  text: {tok.decode(native)!r}")
    print(f"recomp  text: {tok.decode(recomp)!r}")


if __name__ == "__main__":
    main()
