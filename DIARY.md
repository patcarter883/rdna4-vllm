# Development diary — vLLM on AMD RDNA4 (gfx1201)

A field journal of getting Qwen3.6-35B-A3B (AWQ-INT4 MoE) serving on two consumer
Radeon RX 9070-class cards, and building a W4A8-FP8-WMMA MoE kernel along the way.
It's written so the next person hits the walls already knowing where the doors are.

---

## Why this was hard before it was easy

gfx1201 is **RDNA4** — the RX 9070 XT (`0x7550`, 64 CU) and RX 9070 (`0x7551`,
56 CU). It has WMMA-w32 and FP8, but it is **not** a datacenter part and it is
**not** gfx1250. The stock `rocm/vllm-dev` image builds aiter for `gfx942;gfx950`
and `sed`-strips every `gfx1xxx` from flash-attention. So out of the box there is
**zero** gfx1201 support in the two kernel libraries vLLM leans on. The whole
project is the story of putting that support back, then discovering that "the
model loads" and "the model runs" are separated by three more brick walls.

The crux that shaped every kernel decision: **gfx1201 has no TDM** (tensor-descriptor
async load — that's gfx1250-only), only wave32, only single-row DPP. Early on the
mental model was "gfx1250 and gfx1201 are near-identical ISA, so reuse the gfx1250
kernels." That was wrong and cost time. Every hot gfx1250 Gluon kernel hard-codes
`gl.amd.gfx1250.tdm.*`; widening its arch selector to gfx1201 doesn't fall back
gracefully, it **crashes**. The right framing, learned the hard way: for *kernel
reuse*, the performance model that matters is sync-vs-async memory + WMMA tile shape
+ wave width — and on that axis **gfx1201 ≈ RDNA3.5 (gfx1100/gfx1151), not gfx1250**.
That single reframing is why the W4A8 kernel was scaffolded from a gfx1151 kernel,
not a gfx1250 one.

---

## Act I — building the three wheels

### The environment trap
The container base (`rocm/vllm-dev:nightly-therock714`) is **TheRock**: ROCm 7.14
ships as *pip packages* (`rocm-sdk-devel`), hipcc lives inside `_rocm_sdk_devel`,
and there is no `/opt/rocm`. The dev host *did* have a `/opt/rocm` — version 7.2.4,
totally mismatched. **Pitfall:** building against the host `/opt/rocm` produces
binaries that won't load in the container. **Solution:** a container-matched
`uv` venv pinned to the exact torch (`2.10.0+rocm7.14`), triton (`3.6.0`), and SDK
the image ships — captured in `activate-build-env.sh`. One venv builds the wheels
*and* runs host-GPU op tests.

### aiter
- **`PREBUILD_KERNELS=1` is a trap on gfx1201.** It makes FlyDSL prebuild a hard-coded
  **gfx950** MoE matrix (ignoring `GPU_ARCHS`), and every one of 671 kernels fails on
  an `ld.lld` serialize bug in flydsl 0.2.0 × ROCm 7.14. **Solution:** `PREBUILD_KERNELS=0`
  — kernels JIT at runtime on the actual gfx1201, which is the correct dev behavior anyway.
- **The iGPU footgun.** The host has a third device, a gfx1036 display iGPU. aiter's JIT
  defaults to `--offload-arch=native`, which enumerates *all* agents including gfx1036,
  and the fp8 kernels (`v_cvt_pk_fp8_f32`) fail to compile for it. `HIP_VISIBLE_DEVICES`
  does **not** fix the build arch list. **Solution:** always `GPU_ARCHS=gfx1201`.
- Add `$ROCM/lib/llvm/bin` to PATH or FlyDSL can't find `ld.lld`. Don't `pkill -f` your
  build by pattern — the pattern matches the kill command and you SIGKILL your own shell
  (yes, that happened).

### flash-attention
- Its **own** CK submodule (`csrc/composable_kernel`) ships unpopulated. `git submodule
  update --init csrc/composable_kernel` *before* building or it won't compile.
- The build is ~2 hours / 2669 CK FMHA steps at `-j16`. Verified afterwards that all
  2662 embedded code objects were pure gfx1201 (no stray gfx950).
- **The test that wasted an afternoon.** Smoke-testing flash_attn against
  `F.scaled_dot_product_attention` as the "reference" — PyTorch's bundled AOTriton flash
  backend intermittently launches a block-(0,0,0) kernel on gfx1201 →
  `hipErrorInvalidValue`. Both calls were in one try-block, so the AOTriton failure got
  blamed on flash_attn for hours. A gdb backtrace finally showed the throw was inside
  SDPA, not flash_attn. **Lesson:** use a *manual* eager `softmax(QKᵀ/√d)V` as the
  gfx1201 attention reference, never SDPA. This AOTriton hazard comes back to bite in Act II.

### vLLM
- The fork's version string says `0.9.2rc2.dev9980` but the *code* is upstream v0.22.0-level
  — a stale setuptools_scm base tag. **Pitfall:** that label sorts *below* `0.9.2`, so any
  `vllm.__version__ >= "0.x"` gate silently misbehaves. **Solution:** build with
  `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.22.0`.

Three wheels, all gfx1201, all pinned to py3.12 + torch 2.10+rocm7.14. They are
ABI-locked: they will not load on any other arch/python/torch.

---

## Act II — the model that fought back

`facebook/opt-125m` ran end-to-end (single-GPU and TP=2, coherent text) almost
immediately once the wheels were injected. The real target, Qwen3.6-35B-A3B-AWQ-4bit,
hit **three sequential blockers, none of them hardware**, each masking the next.

### Blocker 1 — `import tilelang` aborts the process
The model's linear-attention (MHC) layers route through vLLM's tilelang kernel. Loading
the model imports tilelang, which **aborts** with `TypeAttr __ffi_repr__ is already
registered for type index 130`. The base image ships `apache-tvm-ffi 0.1.12`, which
double-registers a TVM-FFI type against tilelang 0.1.10's bundled `libtvm_compiler.so`
static init. The abort surfaces as the cryptic *"Rust cannot catch foreign exceptions,
aborting"* because it unwinds through pydantic_core's Rust frame during config
construction — **no Python traceback at all**. gdb (`catch throw`) found the real
culprit. **Solution:** pin `apache-tvm-ffi==0.1.10` (still satisfies tilelang's
constraint; the issue was an ABI/registration skew, not a version-range miss). Naively
`pip uninstall`-ing it doesn't work — tilelang *imports* `tvm_ffi` from that package.

### Blocker 2 — `'RoutedExperts' object has no attribute 'tp_size'`
With tilelang fixed, model load crashes here. AWQ MoE layers fall back to the WNA16 MoE
kernel, whose weight loader reads `layer.tp_size` — but the current expert container
`RoutedExperts` exposes TP only via `layer.moe_config.tp_size` (the legacy
`FusedMoE.tp_size` is gone). **This is a generic upstream vLLM bug, not gfx1201-specific.**
**Solution:** a one-line fallback (`getattr(layer,"tp_size",None) or layer.moe_config.tp_size`).
It's owed back to vLLM as a PR.

### Blocker 3 — the vision encoder hangs memory profiling
Now it loads, then *hangs* — both workers at 100% CPU, GPU idle, no progress for 11
minutes. The model is a VLM; vLLM's memory profiler runs a dummy forward through the ViT
"with 1 image of max feature size." The ViT uses Torch-SDPA → **the same AOTriton
block-(0,0,0) hazard from Act I**, now as a CPU-bound spin. Diagnosed by ruling out
compilation (no clang children) and GPU work (`rocm-smi` idle). **Solution for a
text-throughput benchmark:** `--limit-mm-per-prompt {"image":0,"video":0}` skips the ViT
dummy forward entirely. (Real vision support means routing the ViT through flash_attn or
disabling the AOTriton SDPA backend — deliberately out of scope.)

### The 4th thing that *looks* like blocker 4 but isn't
After mm is disabled, the first run sits at 100% CPU / idle GPU for **15–30 minutes**.
This is *not* a hang — it's the cold Triton autotune of the FLA-GDN linear-attention
kernels (e.g. `chunk_scaled_dot_kkt_fwd` has 27 configs, each a slow LLVM-AMDGCN compile,
in-process via libtriton so there are no spawned clang children to see). It looks
*identical* to a hang. **How to tell them apart:** `py-spy dump` (needs `SYS_PTRACE`) —
a healthy compile shows `make_amdgcn → autotuner → chunk_*`. **Solution:** mount a
persistent Triton cache so it's a one-time cost; subsequent boots load binaries and start
in 1–2 minutes. A single config's `make_amdgcn` was measured at ~5 minutes.

---

## Act III — the VRAM wall and a benchmark

Each gfx1201 card is **15.9 GiB**. The MoE keeps *all* 35B params resident (~23.25 GiB)
even though only ~3B are active per token. 23.25 > 15.9, so **TP=1 OOMs at load** — the
model fundamentally needs *both* cards. TP=2 splits it to ~11.5 GiB/card and fits.

TP=2 here is *heterogeneous* (64 CU + 56 CU). aiter asserts all visible GPUs share one CU
count, and — a subtle one — **`rocminfo` ignores `HIP_VISIBLE_DEVICES`**; only
`ROCR_VISIBLE_DEVICES` filters it. So unmasked auto-detect crashes (it sees 64, 56, and
the iGPU's 2). The recipe: `ROCR_VISIBLE_DEVICES=0,1` + explicit `CU_NUM=56` (the lower).
Compute is then bottlenecked by the 56-CU card; best *aggregate* perf would be one model
per card, but this model can't be split that way.

**The number:** with a warm Triton cache, a warmup-generate then a timed generate,
TP=2 delivered **298 decode tok/s / 1887 total (prefill+decode) tok/s** on
Qwen3.6-35B-A3B-AWQ-4bit. That figure is the baseline the W4A8 kernel has to beat.

**A trick that half-worked.** To share GPUs with another agent, we tried warming the
Triton cache on *one* card (CPU-offloading weights to fit), betting the compiled binaries
are content-addressed by IR + block sizes + arch, not by the sharded runtime dims. Partly
true — 136 binaries transferred — but Triton *also* specializes on argument
shape/stride/divisibility, and the TP=2-sharded FLA-GDN shapes produce different cache
keys. So the first *real* TP=2 run still paid a ~15–30 min cold-ish compile. There is no
way to fully prewarm TP=2 from one GPU; the binaries are genuinely shape-specific.

---

## Act IV — the W4A8-FP8-WMMA MoE kernel

The differentiator: a HIP kernel that expands packed INT4 expert weights to FP8 e4m3
**in-register** and feeds RDNA4's FP8 WMMA units. Scaffolded from a gfx1151 kernel (per
the Act I reframing). The v5 WMMA path hit ~48 TFLOP/s and beats the Triton compute-bound
path; dense AWQ+GPTQ works end-to-end.

**How it engages without touching your serving script:** a `vllm.general_plugins` entry
point. vLLM loads general plugins in *every* process — including the EngineCore worker
where the model actually loads — so `register()` runs there and monkeypatches the MoE
oracle. Three hooks cover three routing paths (AWQ, compressed-tensors, and the WNA16
fallback). For Qwen3.6-35B-A3B the live one is **`register_moe_wna16`** (AWQ-Marlin →
WNA16 fallback). E2E validated: coherent text with `WNA16 MoE -> grouped FP8-WMMA` in
both workers.

**Pitfalls earned in blood:**
- **ABI, ABI, ABI.** The `.so` must be compiled against the *container's* torch. A
  host-built `.so` mounted in → `ImportError: undefined symbol _ZN3c103hip...`. That's why
  the Dockerfile builds the kernel *inside* the image, never ships a prebuilt `.so`.
- **sys.path shadowing.** Run tests from a dir *without* the source package, or Python's
  `sys.path[0]` puts the source copy (lacking a fresh `_C.so`) ahead of the installed build.
- **The MoE apply OOMs at the profiling batch.** The apply scratch is O(M·top_k)
  padded-sorted; at vLLM's 8192-token profiling batch it blows the KV cache on a 16 GB
  card. **Solution:** the `tp2-w4a8` profile *must* run low —
  `--max-model-len 2048 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.92`.
- Bisect flags saved hours: `VLLM_ROCM_W4A8_FP8_WMMA_MOE=0` (Marlin baseline),
  `_MOE_VERSION=0` (scalar golden kernel — isolates WMMA numerics from wiring bugs),
  `VLLM_ROCM_USE_W4A8_FP8_WMMA=0` (master off).

---

## Act V — making it reproducible (this repo)

A working stack on one bench is not a deliverable. The packaging problem: three
245 MB-ish ABI-locked wheels, a base image, two source patches, a HIP kernel, and a pile
of non-obvious run flags — collapsed into `git clone … && docker compose up`.

Decisions that shaped the repo:
- **Wheels live in a GitHub Release, not git.** The default Dockerfile `curl`s them at
  build time. Git stays small; the build is ~5–10 min, not ~4 h. (One gotcha checked and
  cleared: GitHub *preserved* the `+` in the wheel filenames, and the `+` resolves fine in
  the download URL, so the exact-name fetch works.)
- **The kernel is baked in, gated, and toggleable.** `WITH_W4A8=1` builds it (ABI-correct,
  in-image); `VLLM_ROCM_USE_W4A8_FP8_WMMA=0` disables it at runtime with no rebuild.
- **One image, three compose profiles** — `tp2-baseline` / `tp2-w4a8` / `single` — so the
  heterogeneous-TP flags, the persistent Triton cache mount, and the W4A8 low-batch caps
  are *encoded*, not memorized.
- **Two build paths:** fast (Release wheels) by default; fully-from-source
  (`Dockerfile.fromsource` clones the patched forks and compiles, ~2–4 h) for anyone who
  won't run someone else's binaries.
- **Credit is not optional.** The serving engine, the kernels, the base image, and the
  models are overwhelmingly other people's work; `NOTICE` says so plainly and names them.

---

## The short list of things that cost the most time

1. Assuming gfx1250 kernels would port to gfx1201. They don't — no TDM.
2. A C++ abort unwinding through Rust with no Python traceback (× 2: tilelang, AOTriton).
   When you see *"Rust cannot catch foreign exceptions,"* reach for gdb immediately.
3. Mistaking a slow in-process Triton compile for a hang. py-spy is the arbiter.
4. ABI mismatches from host-built artifacts. Build inside the target image.
5. Letting an iGPU into the build/detect arch list.

## What actually works, today

Qwen3.6-35B-A3B-AWQ-4bit serving on two RX 9070-class cards, TP=2, coherent text,
**298 dec / 1887 total tok/s** on the stock path; the W4A8-FP8-WMMA MoE kernel validated
end-to-end on the same model; all of it reproducible from a clone and a `docker compose up`.
