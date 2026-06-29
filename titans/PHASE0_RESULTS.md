# Phase 0 — Titans reference bring-up on RDNA4: RESULTS

**Status: PASS (2026-06-22) on gfx1201 (RX 9070, card 1 via gpu-lease).**

## What was validated
The lucidrains `MemoryAsContextTransformer` (MAC variant) — the full test-time neural-memory
machinery — **runs and learns on this box**:
- per-token/chunk surprise via `torch.func` `grad` + `vmap` through the memory MLP,
- momentum + data-dependent decay via the **pure-torch `AssocScan`** (not the CUDA
  `accelerated_scan` pkg),
- per-head learned memory init.

Single fixed random batch, overfit test: **loss 5.67 → 1.44 over 80 steps** (drop 4.23). Mechanism
confirmed; this is the Phase-0 exit signal (no corpus needed).

## RDNA4 settings that matter (carry forward to Phase 1)
- `use_accelerated_scan=False` — the CUDA `accelerated_scan` pkg is not ROCm-buildable; the
  pure-torch `AssocScan` is the path we will eventually port to a kernel (Phase 2).
- `use_flex_attn=False` — FlexAttention/torch.compile attention is unreliable on gfx1201; SDPA path.
- **Overlapping-param gotcha:** with `per_head_learned_parameters=True` the per-head memory init is
  registered as an `einops.repeat` (stride-0 broadcast) aliasing the submodule weights → plain
  in-place Adam (`addcdiv_`) rejects it on CPU. `.to('cuda')` materializes the copy so it's a non-
  issue on-device, but `materialize_overlapping_params()` in `phase0_smoke.py` is the portable fix.

## How to reproduce
```
docker build -f titans/Dockerfile.titans -t titans:dev .       # combined + titans deps
# GPU (gfx1201):
scripts/gpu-lease.sh -n 1 -- bash -c 'docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add CAP_SYS_PTRACE \
  --ipc host --shm-size 16g \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -v /home/pat/code/vllm-gfx1201-titans/titans:/work \
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
  -e PYTHONPATH=/work/ref-titans-pytorch \
  --entrypoint bash titans:dev -lc "source /app/.venv/bin/activate && cd /work && \
    python phase0_smoke.py --device cuda --steps 80 --dim 192 --depth 4 --seq-len 256 \
    --batch 2 --mem-layers 2 4 --per-head-learned"'
```

## Artifacts
- `titans/phase0_smoke.py` — parametrized overfit smoke (CPU + CUDA).
- `titans/Dockerfile.titans` — `titans:dev` = combined image + titans-pytorch deps.
- `titans/ref-titans-pytorch/` — vendored lucidrains reference (clone).

## Next: Phase 1 (from-scratch training) — needs a scope/budget decision before kicking off
(dataset, model size, how long to train on 2 cards). Flagged as the budget-dominant phase.
