"""Phase 0 — Titans reference bring-up on RDNA4 (gfx1201).

Self-contained de-risk: stand up the lucidrains `MemoryAsContextTransformer` (MAC
variant) with the RDNA4-hostile perf flags turned OFF and confirm the whole
test-time-memory machinery RUNS and LEARNS on this box:

  * `use_accelerated_scan=False`  — the CUDA `accelerated_scan` pkg is not ROCm-buildable;
    force the pure-torch `AssocScan` path (this is the path we'll port to a kernel later).
  * `use_flex_attn=False`         — FlexAttention/torch.compile attention is unreliable on
    gfx1201; use the plain SDPA path.

It does NOT need a corpus: we OVERFIT A SINGLE FIXED RANDOM BATCH and assert the loss
falls. If fwd+bwd through the `torch.func grad`/`vmap` neural-memory update works on
gfx1201, the loss on one batch must drop toward zero. That is the Phase-0 exit signal.

Run (CPU first, then GPU via gpu-lease):
  python phase0_smoke.py --device cpu  --steps 30
  scripts/gpu-lease.sh -n 1 -- ... python phase0_smoke.py --device cuda --steps 60
"""

from __future__ import annotations

import argparse
import sys

import torch


def build_model(args):
    from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

    # linear-ish deep memory: a small MLP is the per-sequence STATE that gets
    # test-time-updated. depth=2 == the default 2-layer TTT memory MLP.
    neural_memory_model = MemoryMLP(dim=args.mem_dim, depth=args.mem_depth)

    model = MemoryAsContextTransformer(
        num_tokens=args.num_tokens,
        dim=args.dim,
        depth=args.depth,
        segment_len=args.window,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=4,
        neural_memory_layers=args.mem_layers,
        neural_memory_segment_len=args.mem_segment,
        neural_memory_batch_size=args.mem_batch,
        neural_mem_weight_residual=True,
        neural_memory_qkv_receives_diff_views=True,
        use_flex_attn=False,            # RDNA4: no FlexAttention
        sliding_window_attn=args.sliding_window,
        neural_memory_model=neural_memory_model,
        neural_memory_kwargs=dict(
            dim_head=args.mem_dim,
            heads=4,
            attn_pool_chunks=True,
            qk_rmsnorm=True,
            momentum=True,
            momentum_order=1,
            default_step_transform_max_lr=1e-1,
            use_accelerated_scan=False,  # RDNA4: pure-torch AssocScan
            per_parameter_lr_modulation=True,
            per_head_learned_parameters=args.per_head_learned,
        ),
    )
    return model


def materialize_overlapping_params(model):
    """The per-head learned memory init is registered as an einops.repeat (stride-0
    broadcast) that ALIASES the memory submodule's weights — a single Parameter whose
    storage overlaps itself, which in-place optimizers (Adam addcdiv_) reject. Replace
    each such ParameterList entry with an independent contiguous clone so every head
    gets a real, separately-updatable init. Returns count replaced."""
    import torch.nn as nn

    n = 0
    for module in model.modules():
        plist = getattr(module, "memory_model_parameters", None)
        if not isinstance(plist, nn.ParameterList):
            continue
        for i in range(len(plist)):
            p = plist[i]
            if not p.is_contiguous() or torch._debug_has_internal_overlap(p) != 0:
                plist[i] = nn.Parameter(p.detach().clone().contiguous(),
                                        requires_grad=p.requires_grad)
                n += 1
    return n


def dedup_params(model):
    """Optimize one Parameter per underlying storage (drops the now-dead memory_model
    submodule aliases), all contiguous."""
    seen, out = set(), []
    for p in model.parameters():
        key = p.data_ptr()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--num-tokens", type=int, default=256, dest="num_tokens")
    p.add_argument("--seq-len", type=int, default=256, dest="seq_len")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--mem-dim", type=int, default=64, dest="mem_dim")
    p.add_argument("--mem-depth", type=int, default=2, dest="mem_depth")
    p.add_argument("--mem-layers", type=int, nargs="+", default=(2, 4), dest="mem_layers")
    p.add_argument("--mem-segment", type=int, default=4, dest="mem_segment")
    p.add_argument("--mem-batch", type=int, default=128, dest="mem_batch")
    p.add_argument("--sliding-window", action="store_true", dest="sliding_window")
    p.add_argument("--per-head-learned", action="store_true", dest="per_head_learned",
                   help="keep upstream per-head learned memory init (materialized to "
                        "contiguous params); off => simpler tied init")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"[phase0] torch={torch.__version__} hip={getattr(torch.version, 'hip', None)} "
          f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"[phase0] gpu={torch.cuda.get_device_name(0)} count={torch.cuda.device_count()}",
              flush=True)

    model = build_model(args).to(device)
    n_mat = materialize_overlapping_params(model)
    params = dedup_params(model)
    n_params = sum(p.numel() for p in params)
    print(f"[phase0] model params={n_params/1e6:.2f}M  mem_layers={tuple(args.mem_layers)}  "
          f"materialized={n_mat} per_head_learned={args.per_head_learned}", flush=True)

    # one FIXED random batch to overfit (the de-risk signal)
    batch = torch.randint(0, args.num_tokens, (args.batch, args.seq_len + 1), device=device)

    optim = torch.optim.Adam(params, lr=args.lr)

    first_loss = None
    last_loss = None
    for step in range(args.steps):
        model.train()
        loss = model(batch, return_loss=True)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        last_loss = loss.item()
        if first_loss is None:
            first_loss = last_loss
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"[phase0] step {step:4d}  loss {last_loss:.4f}", flush=True)

    drop = first_loss - last_loss
    print(f"[phase0] first={first_loss:.4f} last={last_loss:.4f} drop={drop:.4f}", flush=True)

    # exit criterion: overfitting one batch must reduce the loss meaningfully
    ok = last_loss < first_loss * 0.6
    print(f"[phase0] {'PASS' if ok else 'FAIL'}: single-batch overfit "
          f"{'reduced' if ok else 'did NOT reduce'} loss on {device.type}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
