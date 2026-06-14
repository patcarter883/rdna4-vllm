# CLAUDE.md — instructions for all agents working in this repo

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the stack contract (what-lives-where, the ABI
rule) and [`DIARY.md`](DIARY.md) for why. This file is the **operational protocol every agent
must follow when building/testing containers**, so we stop wasting each other's GPU time.

## Container testing protocol (MANDATORY)

### 1. Always test against the *latest* image — never a stale tag
- The canonical stack is `vllm22-w4a8:combined`, built from `Dockerfile.combined`
  (`FROM tcclaviger/vllm22:dev`, W4A8 kernel built in-image from `w4a8_fp8_wmma/`).
- **Before a test campaign**, make sure the image reflects current sources: rebuild if the
  base image, `w4a8_fp8_wmma/` csrc, or any applied patch changed since the tag was built.
  A stale image is exactly what bit us before (adapter calling v10/v11 against a v5-only `.so`).
- Build variants with a **distinct tag** so the known-good `:combined` is preserved, and tell
  the others which tag is current. Het-TP example:
  `docker build -f Dockerfile.combined -t vllm22-w4a8:hettp --build-arg WITH_HET_TP=1 \
   --build-context w4a8_src=<latest w4a8 csrc> .`
- Legacy tags (`vllm-gfx1201-w4a8bench:latest`, the retired wheel images) are **not** the
  stack — only use them for an explicit legacy A/B, never as "the image."

### 2. Always mount the *same* Triton cache — one dir per image/toolchain
The 35B is a GDN hybrid: a cold boot pays a ~15-30 min FLA-GDN + attention autotune compile.
A shared, persistent Triton cache makes that **one-time** and is reused across runs, models,
and configs. So **every** container run mounts the repo cache to `/root/.triton`:
```
-v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton   # combined image
```
- **One cache dir PER image/toolchain.** `.triton-cache-combined` for the combined image;
  `.triton-cache-old` for the legacy bench image. The cache is keyed by kernel-source hash +
  shapes, so different **models/shapes on the same image** safely add entries — sharing is the
  whole point (e.g. the het-TP equivalence runs warmed the cache so the het profile server
  skipped the cold GDN compile).

### 3. When NOT to reuse the cache (the caveat — start a fresh dir instead)
Reusing across an **incompatible toolchain** can load a stale/wrong kernel and crash or
silently misbehave. Use a new cache dir (or wipe the per-image one) when:
- The image's **vLLM / Triton / ROCm / torch / GPU-arch** changed (new base image, version
  bump). The combined-vs-old split exists for exactly this reason — keep them separate.
- You suspect **cache corruption**: a kernel crash that disappears after `rm -rf` the cache dir.
- **Concurrent compiles** into one dir can race on writes. Sequential runs sharing a dir is
  safe and normal; for runs that compile in parallel, pre-warm the cache once then run, or give
  each concurrent run its own dir.

### 4. Container-run gotchas (combined image)
- It bakes `HIP_VISIBLE_DEVICES=ROCR_VISIBLE_DEVICES=0,1,2,3`. Override **both together** to
  your device set (e.g. `-e HIP_VISIBLE_DEVICES=0,1 -e ROCR_VISIBLE_DEVICES=0,1`). A mismatch
  disables Triton and crashes model inspection.
- To run a script (not `vllm serve`), use
  `--entrypoint bash IMG -lc 'source /app/.venv/bin/activate && exec python <script>'`
  so the venv PATH is set for Triton's JIT — not `--entrypoint python`.
- Mount HF cache: `-v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1`.
- Shared **2× gfx1201** box: check `rocm-smi` and coordinate a GPU window before any GPU
  workload (CPU/build work is fine anytime).

Reference runners that already follow all of the above: `patches/run_het_e2e_combined.sh`
(equivalence), `profiling/run_het_profile.sh` (profiling), `profiling/run_compare.sh`.
