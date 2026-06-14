# Het-TP handoff — consolidate onto the combined image

**Status:** the het-TP work is currently stranded. The helper + the integration *sketch* are
here (`patches/het_tp.py`, `patches/HET_TP_PATCH.md`), but the actual vLLM **source edits** live
uncommitted in the **zaya** tree (`code/zaya/vllm-therock/vllm/…`) — built against the wrong vLLM
and entangled with the CCA changes. We are consolidating all serving onto the **combined image**,
which uses a *third* vLLM (**0.22.69**, editable at `/app/vllm` in `tcclaviger/vllm22:dev`).

**This repo (`vllm-gfx1201`) + the combined image are now the canonical home.** Please rebase the
het-TP work here and stop diverging in the zaya tree.

## What's already scaffolded for you (so you only do the re-target)

- **Apply slot** in `Dockerfile.combined` (step 2b), gated `--build-arg WITH_HET_TP=1` (default 0,
  so current builds are unaffected). When enabled it:
  1. installs `patches/het_tp.py` → `/app/vllm/vllm/distributed/het_tp.py`,
  2. `patch -p1 < patches/het_tp_vllm.patch` from `/app/vllm`,
  3. parse-checks `linear.py`.
- **Runtime env** documented in `docker-compose.combined.yml`: `VLLM_TP_CU_WEIGHTS="64,56"`
  (commented; uncomment once built with het-TP). Helper fail-safes to even split if unset.
- `patches/het_tp_vllm.patch` is a **placeholder** — the build refuses `WITH_HET_TP=1` until it
  has real `@@` hunks.

## What you deliver (two files in `patches/`)

1. **`patches/het_tp.py`** — the helper (`partition_sizes`, `get_cu_weights`, `het_eligible`).
   Keep it the *canonical* copy; sync from your zaya `vllm/distributed/het_tp.py`. Pure Python,
   CPU-unit-testable (`python patches/het_tp.py`).

2. **`patches/het_tp_vllm.patch`** — a **unified diff of ONLY the het-TP source edits**, generated
   so `patch -p1` applies from `/app/vllm`, and **re-targeted to vLLM 0.22.69** (not zaya's
   0.22-therock — line numbers and surrounding code differ). The edits to port (currently in zaya,
   *isolated from* `mamba/cca.py` and other CCA changes):
   - `vllm/model_executor/layers/linear.py` (Column/Row/Merged/QKV ParallelLinear sizing + param stamp)
   - `vllm/model_executor/layers/fused_moe/config.py`
   - `vllm/model_executor/layers/fused_moe/routed_experts.py`
   - `vllm/model_executor/parameter.py`
   - `vllm/model_executor/layers/vocab_parallel_embedding.py` (only if it's load-side; per the
     scope note, lm_head/vocab stays *even* — confirm this file is actually needed)

   **Correctness invariant (from `HET_TP_PATCH.md`):** `gate_up` output split and `down` input
   split must index the *same* intermediate channels → call `partition_sizes(intermediate,
   weights, align=group_size)` identically for both (same for MoE `w13`/`w2`). Keep
   `align=group_size` (128 for AWQ-INT4). Attention heads + lm_head/vocab stay even.

## Validate (when both are in)

```
docker build -f Dockerfile.combined -t vllm22-w4a8:combined \
  --build-context w4a8_src=/home/pat/code/vllm-rocm714-gfx1250/vllm/csrc/quantization/w4a8_fp8_wmma \
  --build-arg WITH_HET_TP=1 .
python patches/het_tp.py            # CPU apportionment unit test
# then a TP=2 run with VLLM_TP_CU_WEIGHTS="64,56" on both cards (needs a 2-GPU window;
# coordinate via Pat) — expect the rank0/rank1 COMM imbalance (the sync bubble) to shrink.
```

## Coordination notes

- I (the consolidation session) **cannot message you directly** — sync is via this repo, `memory/`,
  and Pat relaying. Drop the two files in `patches/` and ping Pat; I'll wire the build + validate.
- Bring your het-TP **tests** here too (`patches/test_het_packing.py` is present;
  `test_het_loader.py`, `het_e2e_check.py` need re-targeting to the combined image if image-specific).
- Once landed, the zaya `vllm/distributed/het_tp.py` + the zaya source edits should be considered
  **superseded** — develop het-TP here against the combined image only.
