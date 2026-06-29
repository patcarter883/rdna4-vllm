#!/usr/bin/env python3
"""β=1 COHERENCE GATE on the real TiDAR ZAYA checkpoint (design §9 NEXT step 2).

The losslessness pin: TiDAR single-forward decode at β=1 must equal plain AR-greedy
decode of the same model + prompts, token-for-token. This empirically pins the mask
+ sampling_causal + the verify/commit bookkeeping on a genuinely TiDAR-trained model.

Two checks (CPU, bf16, no GPU lease — the 17.7 GB model fits host RAM but not a 16 GB card):

  GATE B (headline, PASS/FAIL): the β=1 TiDAR loop == AR-greedy.
    * diffusion drafts are produced TRAINING-STYLE: a [committed | mask*B] forward with the
      trainer's block bias (causal ∪ intra-block-bidirectional, == train_tidar_zaya.build_block_bias);
      read the B mask-position logits. This is exactly the distribution the model was trained on.
    * verification is a plain causal AR forward over [committed | drafts]; β=1 accepts draft i while
      it equals argmax(p_AR[i]) (reusing tidar_loop.beta_verify), commits k accepted + 1 bonus.
    * with β=1 the committed stream is lossless BY CONSTRUCTION (verify is the true AR argmax) — so a
      mismatch vs AR-greedy means a real plumbing/bookkeeping bug, which is what this catches.

  GATE A (mask, justifies the production single-forward): the FUSED structured-mask forward over
    [committed | S=drafts | R_0..R_{B-1}] produces S-block logits identical to the causal AR forward
    over [committed | drafts]. Proves the replicas + structured bias don't corrupt the verify rows,
    so production can verify from the one fused forward (design §3, §7.2).

Also reports the β=1 acceptance rate (drafts accepted / step) as the speed proxy — NOT a gate.

Run (fork venv, CPU):
  docker run --rm -e HF_HOME=/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
    -v /home/pat/code/.venv-zaya-fork:/opt/zaya-fork-venv \
    -v /home/pat/code/vllm-gfx1201-tidar-serve/zaya/tidar:/work -w /work \
    --entrypoint bash vllm22-w4a8:dflash-rxf -lc '/opt/zaya-fork-venv/bin/python coherence_gate.py'
"""
import os
import sys

import torch

import zaya_mask_patch as zmp
from serve_loader import load_tidar_zaya
from tidar_loop import beta_verify
from tidar_mask import MaskDescriptor, square_additive_bias

CKPT = os.environ.get(
    "TIDAR_CKPT",
    "/root/.cache/huggingface/hub/models--pat883--zaya1-8b-tidar-experts/"
    "snapshots/e6f2ba2d904688059a9e4bd50531504554b02f6d",
)
DEVICE = os.environ.get("TIDAR_DEVICE", "cpu")
DT = torch.bfloat16
N_NEW = int(os.environ.get("TIDAR_N_NEW", "8"))
PROMPTS = [
    "The capital of France is",
    "In the beginning was the",
    "The game was released in",
    "He was born in the city of",
]


# --------------------------------------------------------------------------- #
# masks
# --------------------------------------------------------------------------- #
def causal_bias_4d(L):
    idx = torch.arange(L, device=DEVICE)
    allow = idx[:, None] >= idx[None, :]
    b = torch.zeros(L, L, dtype=DT, device=DEVICE)
    b.masked_fill_(~allow, torch.finfo(DT).min)
    return b.view(1, 1, L, L)


def block_bias_4d(P, B):
    """Trainer's block bias: causal UNION intra-block bidirectional (train_tidar_zaya.build_block_bias)."""
    L = P + B
    idx = torch.arange(L, device=DEVICE)
    causal = idx[:, None] >= idx[None, :]
    in_block = idx >= P
    intra = in_block[:, None] & in_block[None, :]
    keep = causal | intra
    b = torch.where(keep, torch.zeros((), dtype=DT, device=DEVICE),
                    torch.full((), torch.finfo(DT).min, dtype=DT, device=DEVICE))
    return b.view(1, 1, L, L)


# --------------------------------------------------------------------------- #
# forwards (all go through the create_causal_mask monkeypatch -> set_bias)
# --------------------------------------------------------------------------- #
def _logits(model, ids_list, bias_4d, pos_list):
    ids = torch.tensor([ids_list], device=DEVICE)
    pos = torch.tensor([pos_list], device=DEVICE)
    zmp.set_bias(bias_4d)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=None, position_ids=pos, use_cache=False)
    return out.logits[0]  # [L, V]


def causal_logits(model, ids_list):
    L = len(ids_list)
    return _logits(model, ids_list, causal_bias_4d(L), list(range(L)))


def block_predict(model, committed, B, mask_id):
    """Training-style diffusion draft: [committed | mask*B] read at the mask positions."""
    L = len(committed)
    ids = list(committed) + [mask_id] * B
    lg = _logits(model, ids, block_bias_4d(L, B), list(range(L + B)))
    return lg[L:L + B].float().argmax(-1).tolist()


# --------------------------------------------------------------------------- #
# decoders
# --------------------------------------------------------------------------- #
def ar_greedy(model, prompt_ids, n_new):
    ids = list(prompt_ids)
    for _ in range(n_new):
        ids.append(int(causal_logits(model, ids)[-1].float().argmax()))
    return ids[len(prompt_ids):]


def tidar_beta1(model, prompt_ids, n_new, B, mask_id):
    committed = list(prompt_ids)
    L0 = len(committed)
    target = L0 + n_new
    accepts = []
    while len(committed) < target:
        L = len(committed)
        drafts = block_predict(model, committed, B, mask_id)
        full = causal_logits(model, committed + drafts)        # [L+B, V]
        p_ar = full[L - 1: L + B].float()                      # B+1 rows
        k, bonus = beta_verify(drafts, p_ar, beta=1.0)
        committed = committed + drafts[:k] + [bonus]
        accepts.append(k)
    return committed[L0:target], accepts


# --------------------------------------------------------------------------- #
# GATE A — fused structured single-forward S-rows == causal
# --------------------------------------------------------------------------- #
def gate_a(model, committed, drafts, B, mask_id, replica_offset=0):
    L = len(committed)
    d = MaskDescriptor(prefix_len=L, block_len=B, replica_offset=replica_offset)
    seq = list(committed) + list(drafts) + [mask_id] * (B * B)
    pos = list(range(L + B))
    for _ in range(B):
        pos += list(range(L + B, L + 2 * B))                   # replicas at the next-block positions
    assert len(seq) == d.kv_len == len(pos), (len(seq), d.kv_len, len(pos))
    bias = square_additive_bias(d, dtype=DT, device=DEVICE).view(1, 1, d.kv_len, d.kv_len)
    fused_S = _logits(model, seq, bias, pos)[L:L + B].float()
    caus_S = causal_logits(model, list(committed) + list(drafts))[L:L + B].float()
    same = bool((fused_S.argmax(-1) == caus_S.argmax(-1)).all())
    return same, float((fused_S - caus_S).abs().max())


# --------------------------------------------------------------------------- #
def main():
    print(f"[gate] loading {CKPT} on {DEVICE} (bf16) …", flush=True)
    model, tok, mask_id, block_size, (missing, unexpected) = load_tidar_zaya(
        CKPT, device=DEVICE, dtype=DT, from_config=True)
    B = block_size
    print(f"[gate] mask_id={mask_id} block_size(B)={B} missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    zmp.install()

    # ---- GATE B (the NEXT-step-2 losslessness gate): β=1 two-forward loop == AR-greedy ----
    # This is the provably-lossless loop tidar_loop.py itself uses (verify against a causal forward,
    # drafts from a separate diffusion forward) — the single-forward fusion is Gate A below.
    gate_b_ok = True
    print("\n=== GATE B [step-2 losslessness]: β=1 TiDAR decode == AR-greedy ===", flush=True)
    for p in PROMPTS:
        pid = tok(p, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
        ar = ar_greedy(model, pid, N_NEW)
        td, accepts = tidar_beta1(model, pid, N_NEW, B, mask_id)
        ok = ar == td
        gate_b_ok &= ok
        avg_acc = sum(accepts) / len(accepts) if accepts else 0.0
        print(f"  prompt {p!r}", flush=True)
        print(f"    AR-greedy : {ar}")
        print(f"    TiDAR β=1 : {td}")
        print(f"    decode  : {'IDENTICAL ✓' if ok else 'DIVERGED ✗'}  "
              f"steps={len(accepts)} accepts/step={accepts} avg={avg_acc:.2f} (of {B})")
        print(f"    text AR   : {tok.decode(ar)!r}")

    # ---- GATE A (DIAGNOSTIC, not pass/fail): is the fully-FUSED single-forward viable on ZAYA? ----
    # Production wants ONE forward to both verify (S rows) and pre-draft (R replicas). On ZAYA the B²
    # mask-replica tokens contaminate the S verify rows via a sequence-GLOBAL op (NOT attention — S is
    # masked from the replicas), so the fusion does not hold as-is. Reported to inform steps 3-4.
    print("\n=== GATE A [diagnostic]: fused single-forward S-rows == causal? ===", flush=True)
    fusion_ok = True
    for p in PROMPTS[:2]:
        pid = tok(p, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
        drafts = block_predict(model, pid, B, mask_id)
        same, maxd = gate_a(model, pid, drafts, B, mask_id)
        fusion_ok &= same
        print(f"  prompt {p!r}: S-row argmax identical={same}  max|Δlogit|={maxd:.4g}")
    print(f"  fused single-forward viable on ZAYA: {fusion_ok} "
          f"(False ⇒ verify must be isolated from mask-replica scratch — design §7.5/§1.1)")

    print(f"\n[gate] β=1 LOSSLESSNESS (step 2): {'PASS ✓' if gate_b_ok else 'FAIL ✗'}", flush=True)
    sys.exit(0 if gate_b_ok else 1)


if __name__ == "__main__":
    main()
