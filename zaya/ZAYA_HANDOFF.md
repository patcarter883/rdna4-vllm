# ZAYA1-8B + CCA — consolidation into the combined repo

**Status (2026-06-14):** ZAYA1-8B + CCA is **merged to `main`**, built, and serving coherently on the
combined base — and now **folded into the single combined image** (`Dockerfile.combined` `WITH_ZAYA`,
default on; run via the `zaya` profile in `docker-compose.yml`). The standalone `Dockerfile.zaya` /
`docker-compose.zaya.yml` from the staging pass are **retired**. The old checkout
`code/zaya/vllm-therock/` (a vLLM 0.22-therock overlay) is also retired — see its `LEGACY-MOVED.md`.
The remaining open item is the multi-card (DP=2 + EP) profile (below).

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
| Image build | `Dockerfile.combined` step 4, gated `WITH_ZAYA` (default on) |
| Run config | `docker-compose.yml` (profiles `zaya` + `rsa`) |
| Test-time-compute (RSA), quant tooling, benches, docs | `rsa/  quant/  bench/  docs/zaya/` |

The overlay split = **12 new files** (copied) + **9 edits** (the patch). The patch
is pure additive registration (a model, a config, a tool parser, a CCA attention
backend, one `splitting_ops` entry, the `MambaStateShapeCalculator.cca_*` funcs)
**plus** a generic torch.compile-safety guard in `fused_moe.py` that 0.22.69
lacks. Re-derived against vLLM 0.22.69 and **`patch -p1 --dry-run` clean (16/16
hunks, exit 0)** inside `tcclaviger/vllm22:dev`.

## Derived image → folded into the combined image (DONE)
ZAYA first shipped as a **derived image** (`Dockerfile.zaya`, `FROM vllm22-w4a8:combined`) because, at
landing time, `Dockerfile.combined` and `docker-compose.yml` were being **refactored in a parallel
session** — editing a file mid-rewrite is the Act IX "wrong-tree" footgun. Once that refactor
committed, the derived image was **folded into `Dockerfile.combined` as step 4**, gated `WITH_ZAYA`
(default `1`), with `WITH_ZAYA: ${WITH_ZAYA:-1}` in the compose build args and `zaya`/`rsa` profiles
added to `docker-compose.yml`. The gated step is exactly `Dockerfile.zaya`'s RUN block — same source,
same patch — and the only two shared-file patch hunks (a torch.compile-safety guard in `fused_moe.py`;
a `vllm::cca` `splitting_ops` entry) are no-ops for the 35B path. The standalone `Dockerfile.zaya` /
`docker-compose.zaya.yml` are retired.

## Build + validate (DONE — commands for reference)
The combined image bakes ZAYA (`WITH_ZAYA=1`). Put `ZAYA1-8B-fp8` under `$ZAYA_MODELS_DIR` (default `~/models`).
```bash
cp .env.template .env   # set HF_HOME, ZAYA_MODELS_DIR
docker compose --profile zaya up --build
curl -s localhost:8001/v1/completions -d '{"model":"model","prompt":"17*23=","max_tokens":64}'
```
**GPU-window checklist:**
- [x] image builds (overlay applies, CCA kernel compiles + loads, patch applies).
- [x] coherence gate: chat returns sane text; tool/reasoning parsers load (first coherent serve
      needed the MoE loader ported from `RoutedExperts` to the base's factory `FusedMoE`).
- [x] CCA DECODE kernel A/B: `=1` vs `=0`, both coherent; `=1` **+38% chat / +49% RSA** (cudagraphs
      on — `docs/zaya/cca-kernel-perf.md`).
- [x] CCA PREFILL/mixed kernel A/B: coherence identical, bit-exact, throughput-neutral on an
      exclusive card — now **default ON** (`ZAYA_CCA_HIP_PREFILL=1`).
- [ ] add + validate the multi-card (DP=2 + EP) profile — topology re-confirmed on the combined
      base, not staged blind.

## Open items / out of scope this pass
- `quant/` docker-composes still reference the old TheRock image paths — retarget
  to the combined image when the FP8/INT8 quant flow is next exercised.
- het-TP is already consolidated here (`patches/het_tp*`); **not** re-migrated.
- ✅ Folded into `Dockerfile.combined` (`WITH_ZAYA`) + `docker-compose.yml` (`zaya`/`rsa` profiles).
