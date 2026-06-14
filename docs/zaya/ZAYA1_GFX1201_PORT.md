# ZAYA1 support for the gfx1201 TheRock vLLM server (handoff)

**Branch:** `zaya1-therock` in `/home/pat/code/zaya/vllm` (worktree:
`/home/pat/code/zaya/vllm-therock`). It is `rocm/micah/therock-nightly`
(`912bcb62c`, the branch the vllm-rocm714-gfx1250 sessions build from) plus the
ZAYA1/RSA commits rebased on top.

```bash
# From /home/pat/code/vllm-rocm714-gfx1250/vllm:
git fetch /home/pat/code/zaya/vllm zaya1-therock
```

## What it adds (all pure Python — no csrc, no recompile)

- **ZAYA1-8B model support**: `ZayaForCausalLM` (hybrid CCA/MoE arch, Zyphra
  ships no `transformers` code), CCA mamba-style layer + v1 attention backend,
  `ZayaConfig`, `zaya_xml` tool parser, registry/config plumbing. 14 files
  under `vllm/`.
- **Client-side tooling** at the repo root (not part of the wheel): `rsa/`
  (RSA test-time-compute proxy), `bench/` (RSA latency/accuracy harness),
  `quant/` (offline quantizers + investigation record).

Port adaptations for the ~383 upstream commits since the original base are in
commit `zaya: adapt port to post-May upstream API changes` — `mamba_type` now
returns a `MambaAttentionBackendEnum` (new `CCA` member),
`IsHybrid.get_mamba_state_copy_func` is implemented, and a layer-count
attribute fix.

## Verified

Import smoke test **inside `vllm-gfx1201:latest`** (CPU-only, no GPU taken):
all 14 files bind-mounted over the installed vllm 0.22.0 wheel; the model
imports, `ModelRegistry` resolves `ZayaForCausalLM`, the CCA backend enum and
state-copy funcs resolve, `ZayaConfig` and the `zaya_xml` tool parser load.

Not yet verified (needs a GPU window): actual serving on gfx1201, CCA/MoE
kernel numerics on RDNA4, AITER on/off comparison, quantized variants.

## Three ways to consume it

1. **Runtime overlay (zero rebuild, what the smoke test used).** The root
   `docker-compose.yml` on the branch bind-mounts the 14 files over
   `vllm-gfx1201:latest`'s site-packages and serves `Zyphra/ZAYA1-8B`.
2. **Bake into the image.** Generate the wheel-relative patch and add a
   `RUN patch` step (or COPY the files) after the wheel install in
   `docker/Dockerfile.rocm.gfx1201-inject`:
   ```bash
   git diff rocm/micah/therock-nightly..zaya1-therock -- 'vllm/*.py' \
     > zaya1-vllm-overlay.patch
   # in the Dockerfile, after the vllm wheel install:
   #   COPY zaya1-vllm-overlay.patch /tmp/
   #   RUN cd /opt/python/lib/python3.12/site-packages && patch -p1 < /tmp/zaya1-vllm-overlay.patch
   ```
   A pre-generated copy is at `/home/pat/code/zaya/zaya1-vllm022-overlay.patch`.
3. **Merge the branch** into the therock line and build the wheel as usual —
   the model files are additive; the only shared-file edits are small
   (registry entries, a fused_moe torch.compile-tracing guard, config plumbing).

## Serve flags that matter for ZAYA

`--mamba-cache-dtype float32` (CCA state precision), `--trust-remote-code`,
`--tool-call-parser zaya_xml --enable-auto-tool-choice`,
`--reasoning-parser qwen3`. Prefix caching is unsupported by the model and
must stay off. `VLLM_ROCM_USE_AITER=0` for first bring-up — the fork enables
AITER on gfx1201, but ZAYA's MoE/CCA path is unvalidated under it.

## Why this exists

The RSA (recursive self-aggregation) test-time-compute work drives N=16
parallel rollouts per query against a ZAYA1-8B server and is latency-bound on
batch capacity. The perf plan needs the model running on the new gfx1201 stack
(quantized experts → more VRAM → bigger rollout waves) — see `quant/README.md`
and `bench/README.md` on this branch.
