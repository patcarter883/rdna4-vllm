#!/usr/bin/env python3
"""TiDAR throughput on the REAL converted checkpoint (§7.6 throughput run).

Answers: does two-forward β=1 TiDAR beat plain AR-greedy on this ZAYA1-8B-Diffusion checkpoint, and
by how much? The lossless production path is the TWO-FORWARD form (a [committed|mask*B] diffusion-draft
forward + a [committed|drafts] causal-verify forward per step), proven token-for-token == AR-greedy by
coherence_gate.py. It uses CONTIGUOUS positions, so the §7.6 replica-position_ids unknown (which only
blocks the §7.5-contaminated FUSED single forward) does NOT apply here.

The speedup is cache-independent and exact via FORWARD COUNT:
  - AR-greedy           : 1 model forward per committed token.
  - two-forward TiDAR   : 2 forwards per step (draft + verify) producing (k+1) committed tokens
                          (k = accepted drafts) -> 2/(k+1) forwards/token.
  => throughput speedup = (forwards/token)_AR / (forwards/token)_TiDAR = (avg_accept + 1) / 2,
     PROVIDED each forward costs ~the same. At batch-1 decode an 8B model is memory-bandwidth bound
     (reloading the weights dominates), so a B-query block forward ~= a 1-query decode forward. PART 1
     measures exactly this on-device; PART 2 measures the forward-count ratio + acceptance on the real
     weights. (The coherence loop itself is no-KV-cache O(L^2) recompute, so wall-clock of the loop is
     NOT used — only the cache-independent forward count, validated by the equal-per-forward-cost fact.)

Run on 2 GPUs (the 16.5GB bf16 model needs both): accelerate device_map across cuda:0/1.
  scripts/gpu-lease.sh -n 2 -- bash -c 'docker run --rm --device /dev/kfd --device /dev/dri \
    --group-add video --security-opt seccomp=unconfined --security-opt label=disable --ipc host \
    --shm-size 16gb -e HF_HUB_OFFLINE=1 \
    -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
    -v /home/pat/code/.venv-zaya-fork:/opt/zaya-fork-venv \
    -v /home/pat/code/vllm-gfx1201-tidar-serve/zaya/tidar:/work \
    -v /home/pat/code/zaya1-8b-tidar-serve:/ckpt \
    -w /work --entrypoint bash vllm22-w4a8:dflash-rxf -lc \
    "/opt/zaya-fork-venv/bin/python throughput_tidar.py --ckpt /ckpt"'
"""
import argparse
import time

import torch

import zaya_mask_patch as zmp
from serve_loader import load_tidar_zaya
from tidar_loop import beta_verify

DEV = "cuda:0"          # input/embedding device; device_map places later layers on cuda:1
DT = torch.bfloat16
_NFWD = 0               # global model-forward counter

# The forward-COUNT ratio + acceptance (PART 2) are determined by the model's OUTPUTS, not the
# hardware, so they are identical on CPU and GPU — the honest, device-independent speedup signal.
# --device cpu runs PART 2 only (no device_map, no lease); --device cuda:0 additionally dispatches
# across 2 GPUs and runs the PART 1 memory-bound latency check.


def causal_bias_4d(L):
    idx = torch.arange(L, device=DEV)
    b = torch.zeros(L, L, dtype=DT, device=DEV)
    b.masked_fill_(~(idx[:, None] >= idx[None, :]), torch.finfo(DT).min)
    return b.view(1, 1, L, L)


def block_bias_4d(P, B):
    L = P + B
    idx = torch.arange(L, device=DEV)
    causal = idx[:, None] >= idx[None, :]
    in_block = idx >= P
    keep = causal | (in_block[:, None] & in_block[None, :])
    b = torch.where(keep, torch.zeros((), dtype=DT, device=DEV),
                    torch.full((), torch.finfo(DT).min, dtype=DT, device=DEV))
    return b.view(1, 1, L, L)


def _logits(model, ids_list, bias_4d, pos_list):
    global _NFWD
    ids = torch.tensor([ids_list], device=DEV)
    pos = torch.tensor([pos_list], device=DEV)
    zmp.set_bias(bias_4d)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=None, position_ids=pos, use_cache=False)
    _NFWD += 1
    return out.logits[0]


def causal_logits(model, ids_list):
    return _logits(model, ids_list, causal_bias_4d(len(ids_list)), list(range(len(ids_list))))


def block_predict(model, committed, B, mask_id):
    L = len(committed)
    ids = list(committed) + [mask_id] * B
    lg = _logits(model, ids, block_bias_4d(L, B), list(range(L + B)))
    return lg[L:L + B].float().argmax(-1).tolist()


def ar_greedy(model, prompt_ids, n_new):
    ids = list(prompt_ids)
    for _ in range(n_new):
        ids.append(int(causal_logits(model, ids)[-1].float().argmax()))
    return ids[len(prompt_ids):]


def tidar_beta1(model, prompt_ids, n_new, B, mask_id):
    committed = list(prompt_ids)
    L0, target, accepts = len(committed), len(prompt_ids) + n_new, []
    while len(committed) < target:
        L = len(committed)
        drafts = block_predict(model, committed, B, mask_id)
        full = causal_logits(model, committed + drafts)
        k, bonus = beta_verify(drafts, full[L - 1:L + B].float(), beta=1.0)
        committed = committed + drafts[:k] + [bonus]
        accepts.append(k)
    return committed[L0:target], accepts


def dispatch_2gpu(model):
    from accelerate import dispatch_model, infer_auto_device_map
    layer_cls = type(model.model.layers[0]).__name__
    dmap = infer_auto_device_map(
        model, max_memory={0: "13GiB", 1: "13GiB"}, dtype=DT,
        no_split_module_classes=[layer_cls],
    )
    ndev = len({v for v in dmap.values() if isinstance(v, int)})
    print(f"[dispatch] no_split={layer_cls}  devices_used={ndev}  "
          f"(sample {dict(list(dmap.items())[:3])} … {dict(list(dmap.items())[-2:])})", flush=True)
    return dispatch_model(model, device_map=dmap)


def time_forward(model, prefix_len, q_len, mask_id, iters=5):
    """Median wall-time of ONE forward over [prefix(prefix_len) | q_len new tokens], contiguous pos.
    Confirms the memory-bound regime: t should be ~flat in q_len for small q_len."""
    ids = list(range(100, 100 + prefix_len)) + [mask_id] * q_len
    bias = block_bias_4d(prefix_len, q_len)
    pos = list(range(prefix_len + q_len))
    ids_t = torch.tensor([ids], device=DEV)
    pos_t = torch.tensor([pos], device=DEV)
    zmp.set_bias(bias)
    ts = []
    for _ in range(iters + 1):
        torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad():
            model(input_ids=ids_t, attention_mask=None, position_ids=pos_t, use_cache=False)
        torch.cuda.synchronize()
        ts.append(time.time() - t)
    ts = sorted(ts[1:])  # drop first (warm)
    return ts[len(ts) // 2]


def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ckpt")
    ap.add_argument("--n-new", type=int, default=48)
    ap.add_argument("--device", default="cuda:0", help="cpu (PART 2 only) or cuda:0 (+ 2-GPU PART 1)")
    args = ap.parse_args()
    DEV = args.device
    on_gpu = DEV.startswith("cuda")

    print(f"[load] {args.ckpt} on {'CPU then dispatch to 2 GPUs' if on_gpu else 'CPU (single device)'} "
          f"(bf16, eager) …", flush=True)
    zmp.install()
    model, tok, mask_id, B, (miss, unexp) = load_tidar_zaya(args.ckpt, device="cpu", dtype=DT)
    print(f"[load] mask_id={mask_id} B={B} missing={len(miss)} unexpected={len(unexp)}", flush=True)
    if on_gpu:
        model = dispatch_2gpu(model)
    model.eval()

    # PART 1 — per-forward cost vs query length (memory-bound check; GPU only)
    if on_gpu:
        print("\n=== PART 1: per-forward latency vs query length (prefix=256) ===", flush=True)
        base = None
        for q in (1, B, 2 * B, B * (1 + B)):
            t = time_forward(model, 256, q, mask_id)
            base = base or t
            print(f"  q_len={q:3d}: {t*1000:7.1f} ms  ({t/base:.2f}x the q=1 forward)", flush=True)
        print("  (≈flat ⇒ memory-bound: a B-query block forward costs ~the same as a 1-token decode)",
              flush=True)
    else:
        print("\n=== PART 1 skipped (CPU mode); memory-bound regime confirmed by the 27 tok/s "
              "baseline: 16GB/37ms ≈ 430 GB/s ≈ card bandwidth ⇒ t_block ≈ t_decode ===", flush=True)

    # PART 2 — forward-count ratio + acceptance over real prompts
    print("\n=== PART 2: AR vs two-forward TiDAR — forwards per committed token ===", flush=True)
    prompts = [
        "The capital of France is",
        "In the beginning God created the heavens and the",
        "Q: What is 2+2? A:",
        "The mitochondria is the powerhouse of the",
    ]
    tot_ar_f = tot_ti_f = tot_tok = 0
    all_acc = []
    for p in prompts:
        pid = tok(p, return_tensors=None)["input_ids"]
        global _NFWD
        _NFWD = 0
        ar = ar_greedy(model, pid, args.n_new)
        f_ar = _NFWD
        _NFWD = 0
        td, acc = tidar_beta1(model, pid, args.n_new, B, mask_id)
        f_ti = _NFWD
        match = (ar == td)
        avg_acc = sum(acc) / len(acc)
        all_acc += acc
        tot_ar_f += f_ar; tot_ti_f += f_ti; tot_tok += args.n_new
        print(f"  [{'LOSSLESS' if match else 'DIVERGED!'}] {p[:34]!r:36s} "
              f"AR_fwd={f_ar} TiDAR_fwd={f_ti} avg_accept={avg_acc:.2f}/{B} "
              f"speedup={f_ar/f_ti:.2f}x", flush=True)
    avg_accept = sum(all_acc) / len(all_acc)
    ratio = tot_ar_f / tot_ti_f
    print(f"\n=== RESULT ===")
    print(f"  avg accepted drafts/step = {avg_accept:.2f} / {B}")
    print(f"  AR forwards/token   = {tot_ar_f/tot_tok:.3f}")
    print(f"  TiDAR forwards/token= {tot_ti_f/tot_tok:.3f}")
    print(f"  forward-count speedup (≈throughput, memory-bound) = {ratio:.2f}x")
    print(f"  predicted (avg_accept+1)/2 = {(avg_accept+1)/2:.2f}x")
    print(f"  => two-forward TiDAR vs ~27 tok/s AR ≈ {27*ratio:.1f} tok/s")


if __name__ == "__main__":
    main()
