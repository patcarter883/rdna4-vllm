# Het-TP revalidation — re-run recipe (for the next dev image)

The het-TP equivalence + COMM-bubble validation is reproducible against any image that
carries the het-TP patch. Re-run this whenever the dev/base image changes (e.g. once the
consolidation lands and the latest W4A8 work is in the dev image).

## 1. Rebuild the het image on the new base

```
docker build -f Dockerfile.combined -t vllm22-w4a8:hettp \
  --build-arg WITH_HET_TP=1 \
  --build-arg BASE_IMAGE=<new dev image> \
  --build-context w4a8_src=<latest w4a8 csrc tree> .
```

**Drift check (the one real risk):** the patch (`patches/het_tp_vllm.patch`) targets vLLM
**0.22.69** source for 3 files: `model_executor/layers/linear.py`, `model_executor/parameter.py`,
`model_executor/layers/fused_moe/layer.py`. If the new dev image bumps the vLLM version or
edits those files, `patch -p1` rejects on context mismatch and the build **fails loud** (the
WITH_HET_TP step). That's the signal to re-target: extract the 3 files from the new image
(`docker create` + `docker cp /app/vllm/...`), re-apply the edits, regenerate the diff. The
helper `patches/het_tp.py` is version-independent (pure Python, installed separately) and
shouldn't need changes. See patches/HET_TP_HANDOFF.md for the extraction/re-target method.

## 2. Greedy-equivalence (het ≡ even, math-preserving)

```
# fast dense smoke (7B, ~5 min):
HET_W4A8=0 bash patches/run_het_e2e_combined.sh Qwen/Qwen2.5-Coder-7B-Instruct-AWQ 2
# real MoE target (35B, slower — pays GDN/Triton compile on a cold cache):
HET_W4A8=0 HET_LIMIT_MM=1 HET_MAXLEN=2048 HET_GPUUTIL=0.90 \
  bash patches/run_het_e2e_combined.sh cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit 2
```
Each loads the model twice (even: `VLLM_TP_CU_WEIGHTS` unset; het: `64,56`) and diffs greedy
token ids — they MUST be identical. Knobs: `HET_IMG` (image tag, default vllm22-w4a8:hettp),
`HET_W4A8` (0=stock loaders/isolates het sharding; 1=shipped W4A8 config), `HET_MAXTOK`.

## 3. COMM-bubble profile (the perf payoff)

Serve the 35B TP=2 with `VLLM_TORCH_PROFILER_DIR=/profiles`, once even and once het=64,56;
drive a decode wave (`profiling/drive_tp2_profile.py`), then per-rank bucket the kineto
traces (`profiling/analyze_torch_trace.py`) and compare the "all-reduce / collective (TP)"
bucket + decode tok/s. Even split => rank imbalance (the bigger card spin-waits at the
barrier); het=64,56 should shrink that imbalance and lift decode tok/s toward the ~5% ceiling.

## Notes / gotchas baked into the runner
- The combined image bakes `HIP_VISIBLE_DEVICES=ROCR_VISIBLE_DEVICES=0,1,2,3`; they MUST be
  overridden together to `0,1` (mismatch -> "Disabling Triton" + a model-inspect subprocess
  crash). The runner does this; replicate it for any new invocation.
- Run via `bash -lc 'source /app/.venv/bin/activate && ...'` (not `--entrypoint python`) so the
  venv PATH is set for Triton's JIT.
- 35B is a GDN hybrid: first cold run pays a ~20 min FLA-GDN/attention Triton compile. Mount a
  persistent Triton cache (`.triton-cache-combined`) so it's one-time. The het run recompiles
  only the FFN/MoE GEMM kernels (uneven shapes), not the cached GDN/attention kernels.
