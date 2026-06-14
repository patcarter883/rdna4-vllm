# ZAYA1-8B + CCA — consolidation into the combined repo

**Status (2026-06-14):** ZAYA1-8B + CCA development now lives **here**
(`vllm-gfx1201`), on branch `zaya-consolidation`. The old checkout
`code/zaya/vllm-therock/` (a vLLM 0.22-therock overlay) is **retired** — see its
`LEGACY-MOVED.md`. This is the staging pass: **files + wiring are in place and
config-validated; the image is NOT built yet** (a build + serve coherence gate is
a coordinated GPU window — checklist below).

## What ZAYA is (one paragraph)
A hybrid model: 80 layers alternating **CCA (Compressed Convolutional Attention)**
with **MoE** (16 experts, top-1), carrying conv + temporal recurrent state (not a
plain KV cache). The differentiator is a **fused CCA decode HIP kernel**
(`cca_hip/cca_kernel.hip`, `cca_decode_qk`) that replaces ~1.4M tiny eager ATen
launches per decode step. Current kernel: **coalesced-`w1` transpose + `#pragma
unroll 16` = 142µs→8µs/call on the 56-CU 9070 (17.6×), bit-exact**, gated
`ZAYA_CCA_HIP=1` (eager fallback otherwise). A twin **PREFILL kernel**
(`cca_prefill_qk`) refuses the eager flat-conv prefill path (scatter → MIOpen
Conv1d → gather → eager means/norm → state scatter, ~25-30 ATen ops/layer) into
one launch: **2.4-3x faster, bit-exact** (`test_cca_prefill_qk.py`,
`bench_cca_prefill_qk.py`), gated `ZAYA_CCA_HIP_PREFILL=1` (default off, pure-
prefill batches only). CCA has **no TP>1** (per-head RMSNorm + grouped-mean state
don't column-split) — multi-card is DP=2 + expert-parallel (model replicated per
rank), never TP.

## Layout (what landed where)
| Piece | Path |
|---|---|
| NEW vLLM overlay files (model/config/parser/backend) | `zaya/overlay/vllm/...` |
| CCA HIP kernel csrc (built in-image) | `zaya/cca_hip/` |
| 9 additive registration hooks (0.22.69) | `zaya/zaya_vllm_0.22.69.patch` |
| Derived image build | `Dockerfile.zaya` (`FROM vllm22-w4a8:combined`) |
| Run config | `docker-compose.zaya.yml` (profile `zaya`) |
| Test-time-compute (RSA), quant tooling, benches, docs | `rsa/  quant/  bench/  docs/zaya/` |

The overlay split = **12 new files** (copied) + **9 edits** (the patch). The patch
is pure additive registration (a model, a config, a tool parser, a CCA attention
backend, one `splitting_ops` entry, the `MambaStateShapeCalculator.cca_*` funcs)
**plus** a generic torch.compile-safety guard in `fused_moe.py` that 0.22.69
lacks. Re-derived against vLLM 0.22.69 and **`patch -p1 --dry-run` clean (16/16
hunks, exit 0)** inside `tcclaviger/vllm22:dev`.

## Why a derived image, not a `WITH_ZAYA` gate in Dockerfile.combined
Mirroring W4A8/het-TP, ZAYA *should* eventually be a gated step inside
`Dockerfile.combined`. It is a **derived image** for now because, at landing time,
`Dockerfile.combined` and `docker-compose.yml` were being **refactored in a
parallel session (uncommitted WIP)** — editing a file mid-rewrite is exactly the
Act IX "wrong-tree" footgun. The derived image touches **none** of their in-flight
files and is fully functional. **To fold into the single combined image later**
(once their refactor commits), drop this gated step after Dockerfile.combined's
step 3 and add `WITH_ZAYA: ${WITH_ZAYA:-0}` to the compose build args:

```dockerfile
# --- 4. (optional) ZAYA1-8B hybrid-CCA support ---
ARG WITH_ZAYA=0
COPY zaya/ /tmp/zaya/
RUN set -eu; \
    if [ "$WITH_ZAYA" = "1" ]; then \
      . /app/.venv/bin/activate; export PATH="/opt/rocm-7.2.1/lib/llvm/bin:$PATH"; \
      VV=/app/vllm/vllm; \
      cp -r /tmp/zaya/overlay/vllm/. "$VV"/; \
      cp -r /tmp/zaya/cca_hip "$VV"/model_executor/layers/mamba/; \
      cd "$VV"/model_executor/layers/mamba/cca_hip && find . -name '*.so' -delete && \
        find . -name '*_hip.*' -delete && GPU_ARCHS=gfx1201 python setup.py build_ext --inplace; \
      cd /app/vllm && patch -p1 < /tmp/zaya/zaya_vllm_0.22.69.patch; \
      echo "[combined] ZAYA applied"; \
    else echo "[combined] ZAYA slot present but disabled"; fi; rm -rf /tmp/zaya
```
(That block is just `Dockerfile.zaya`'s RUN, gated. Same source, same patch.)

## Build + validate (GPU window — NOT done yet)
Prereq: the base `vllm22-w4a8:combined` image exists (build from `docker-compose.yml`).
```bash
# 1. build the derived ZAYA image (CCA kernel cross-compiles; no GPU needed here)
docker build -f Dockerfile.zaya -t vllm22-zaya:combined .
# 2. coherence gate (1 GPU). Put ZAYA1-8B-fp8 under $ZAYA_MODELS_DIR (default ~/models)
cp .env.template .env   # set HF_HOME, ZAYA_MODELS_DIR
docker compose -f docker-compose.zaya.yml --profile zaya up
curl -s localhost:8001/v1/completions -d '{"model":"model","prompt":"17*23=","max_tokens":64}'
```
**GPU-window checklist (the unchecked boxes):**
- [ ] image builds (overlay applies, CCA kernel compiles + loads, patch applies).
- [ ] coherence gate: one chat returns sane text; tool/reasoning parsers load.
- [ ] CCA DECODE kernel A/B: `ZAYA_CCA_HIP=1` vs `=0` — both coherent; `=1` faster.
- [ ] CCA PREFILL kernel A/B: `ZAYA_CCA_HIP_PREFILL=1` vs `=0` — coherence must be
      identical (exact refusion of the eager path; standalone bit-exact qk rel
      ~6e-7, 2.4-3x faster). Default OFF until this passes. Fires only for
      pure-prefill batches today; mixed prefill+decode -> eager (follow-up).
- [ ] add + validate the multi-card (DP=2 + EP) profile — topology re-confirmed on
      the combined base, not staged blind.

## Open items / out of scope this pass
- `quant/` docker-composes still reference the old TheRock image paths — retarget
  to the combined image when the FP8/INT8 quant flow is next exercised.
- het-TP is already consolidated here (`patches/het_tp*`); **not** re-migrated.
- Fold the derived image into `Dockerfile.combined` once its refactor commits.
