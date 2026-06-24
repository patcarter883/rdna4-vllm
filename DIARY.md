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
path *in the GEMM microbenchmark* (direct op timing, large-M ≳3072 only — this is a
kernel-level number, not an e2e one; see Act VI + `profiling/.../AUDIT.md`); dense AWQ+GPTQ
works end-to-end.

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

## Act VI — the profiling reckoning: measure before you build

The W4A8 kernel works — but a quieter question loomed. The dense fp8-WMMA kernels (v10/v11)
**never engage** on the target checkpoints (dense layers stay fp16; only experts are int4),
and the MoE kernel wins per-GEMM yet is e2e-neutral. So before scaffolding *another* kernel —
paged attention for the head_dim=256 layers, or an FLA/GDN linear-attention kernel — the honest
move was to **measure the decode step**, not estimate it. The MoE saga had already taught this
in blood: a fast kernel for a non-bottleneck bucket buys ~0 end-to-end.

**The tool that doesn't work here.** The instinct was rocprofv3 `--kernel-trace` for GPU-side
truth. On this stack it sees **nothing useful**: a TP=2 dense run logged 1633 launches across 19
names — *all* torch-native HIP copies/fills/elementwise + RCCL, zero `paged_attention` /
`delta_rule` / GEMM. The entire decode compute is **Triton-compiled**, and rocprofv3 doesn't
capture Triton kernels here. (It did confirm the one thing it can — the comm/copy glue — and the
old TP=2 rocprof deadlock is gone with `NCCL_PROTO=SIMPLE` + `HSA_NO_SCRATCH_RECLAIM=1`.) What
*does* see through Triton is vLLM's built-in **torch profiler** (`VLLM_TORCH_PROFILER_DIR` +
`start_profile`/`stop_profile`) via kineto, eager mode for clean per-op attribution.

**The two-rank trap.** The first bucketed trace screamed COMM = 61% of the step. Artifact. The
two TP ranks disagree wildly — rank0 COMM 14.9%, rank1 61% — because the all-reduce kernel
**spin-waits**: on the rank that arrives early, the nccl kernel's wall-time is mostly idle spin.
**Read rank0 (the busy/critical rank) as compute truth; rank1's COMM inflation IS the
heterogeneous-TP sync bubble** — the 64-CU XT finishing ahead of the 56-CU 9070 and spinning.
Real e2e waste, but not a compute cost.

**Where the decode step actually goes.** A clean 16-cell `triton`-only sweep (stock vs W4A8 ×
two models × four batch regimes) settled it. The breakdown is **strongly batch- and
model-dependent** — which is itself the lesson (rank0 self-CUDA %, decode regime):

| bucket — **35B MoE** | share |  | bucket — **27B dense** | share |
|---|---|---|---|---|
| dense fp16 GEMM (`aten::mm`→Tensile) | ~33% |  | **int4 GEMM** (`triton_w4a16_gemm`) | **~57%** |
| MoE experts (`fused_moe_kernel_gptq_awq`) | ~19% |  | dense fp16 GEMM | ~5% |
| COMM (RCCL all-reduce) | ~14% |  | COMM (RCCL all-reduce) | ~23% |
| GDN/Mamba linear-attn | ~5–8% |  | — | — |
| **full attention** | **~1.6%** |  | full attention | **<2%** |

On the 35B MoE the MoE share climbs to ~29% at the large batch (dense fp16 dominates at small
batch); on the dense 27B the **int4 GEMM is the whole game, ~57–69% across every regime** —
exactly what W4A8 targets.

**The verdict — three findings, one chapter-closer:**
- **No custom attention kernel.** Full attention is ~1.6% of decode, confirmed across both models
  and all batch sizes. The attention-backend axis is moot a second way: **AITER attention is
  arch-rejected on gfx1201** (`'gfx1201' is invalid or not supported` at engine init), so Triton
  is the only viable backend — there is no faster one to switch to.
- **W4A8 doesn't win on the MoE model.** Stock→W4A8 across regimes: decode **1.01×**, mid
  **0.94×**, large **0.95×**, prefill **0.99×** — neutral at the tiny batch, *negative* elsewhere.
  The MoE GEMM we replace is only ~19–29% of decode while comm + dense fp16 dominate, so a
  per-GEMM-competitive kernel buys ~0 end-to-end.
- **W4A8's ideal case was unloadable — now fixed.** The dense 27B *is* int4-GEMM-bound (~57%) —
  the one place a faster W4 path should win big — but in the sweep the W4A8 adapter **OOM'd at
  weight conversion**, materializing the full `(N,K,8)`-expanded int4 unpack and killing all four
  27B regimes on a 16 GB card. That load-time memory bug has since been fixed (chunked/in-place
  unpack in `vllm_adapter.py`), so the dense-path A/B is now **runnable but not yet measured** —
  and it's the one open datapoint that could still move the verdict, being the kernel's most
  favorable workload.
- Net: **on the MoE model the int4→fp8-WMMA kernel is e2e-neutral; its best case — the dense,
  int4-GEMM-bound 27B — is now loadable and awaiting an A/B.** The kernel chapter is closing,
  pending that one dense comparison.

> **Correction (2026-06-13, forced re-measurement — see `profiling/sweep-2026-06-13/AUDIT.md`).**
> The A/B numbers above were taken with the kernel **not actually engaged**: by default the
> MoE/dense gates consult an *untuned* crossover cache and silently fall back to stock, so the
> first sweep was stock-vs-stock (profiler: our op = 0 calls). Re-run with the new tuning-gate
> override `VLLM_ROCM_W4A8_FORCE=on` (which also fixed the 27B load-OOM by dropping the duplicate
> Triton weight copy), the kernel runs and the verdict is **regime-dependent, not "neutral":**
>
> | regime | 35B MoE (forced) | 27B dense (forced) |
> |---|---|---|
> | decode  | **1.11×** ✅ | 0.66× |
> | mid     | 0.92× | 0.88× |
> | large   | 0.91× | OOM @ b64 (to fix) |
> | prefill | 0.84× | **1.53×** ✅✅ |
>
> So there ARE real wins — **dense prefill +53%** (the compute-bound regime the 1.28× GEMM
> microbench predicted) and **MoE decode +11%** — alongside mid-batch losses. The kernel is worth
> keeping for prefill-heavy / decode-latency-bound serving; it is not a universal win.

**The war story that came free.** The sweep wrote ~300 MB kineto traces per cell into a
**RAM-backed `/tmp` (45 GB tmpfs)** while TP=2 vLLM held both cards — and this box swapped to a
**ZFS zvol**, which deadlocks under memory pressure (the writeback path needs memory to free
memory; no OOM-kill ever fires). The host hard-hung mid-sweep, power-cycle required, and the
reboot wiped the tmpfs with every trace not already hand-copied. **Lessons:** never swap to a
ZFS zvol (zram + a capped ARC instead); write big profiler output to real disk, not tmpfs; and
transcribe the bucketed numbers *as they land* — the conclusion is the deliverable, not the
300 MB blob. (The first run's numbers survived only because they were transcribed by hand; the
table above is the cleaner 16-cell re-run, written to the repo on disk.)

*Method notes: numbers are from the 16-cell `triton`-only sweep — the AITER backends were dropped
as unsupported, and the `large` cells needed a 30-min per-cell cap to clear their cold Triton
autotune. Shares are eager-mode, so the compute *ratios* hold but absolute comm/launch overhead is
inflated; W4A8 cudagraph capture is separately confirmed to work, and the heterogeneous-TP sync
bubble has a drafted 64:56 proportional-sharding fix.*

---

## Act VII — a second front: the ZAYA1-8B port

Running in parallel on the same gfx1201 stack — a separate vLLM-v0.22 therock-branch overlay at
`code/zaya/vllm-therock/` — is a port of **Zyphra's ZAYA1-8B**, a different animal entirely.
ZAYA1 is a **hybrid** model: 80 layers alternating **CCA (Compressed Convolutional Attention)**
with **MoE** (16 experts, top-1), a bf16 high-precision router, and *recurrent convolutional
state* like Mamba. Unlike Qwen3.6 it isn't a transformer with a KV cache — it carries conv +
temporal state, which reframes every systems decision below.

**The differentiator: a fused CCA decode kernel.** The eager Python decode path ran the CCA step
as ~1.4M tiny ATen launches — death by a thousand kernels. The port's answer is
`cca_decode_qk_kernel` (`mamba/cca_hip/cca_kernel.hip`): one block-per-token pass that fuses the
two-stage causal conv + GQA grouped-means + per-head RMSNorm + conv-state window roll, emitting
normalized q|k directly — no extra HBM round-trip. wave32 intra-wave shuffle reduction, one LDS
sync for the cross-wave norm. Gated `ZAYA_CCA_HIP=1` (eager fallback otherwise); optimized
**102 µs → 53 µs per call (+7.7% e2e)**.

**State as a first-class block-manager citizen.** `ZayaForCausalLM(IsHybrid)` + `CCA(MambaBase)`
register conv_states + prev_hs into vLLM's Mamba/hybrid cache via `cca_state_shape/dtype/copy`
(`mamba_utils.py`), at **float32** (`--mamba-cache-dtype float32` — CCA numerics demand it).
Per-spec-position rollback lives in *separate block slots*, not sliced columns — the groundwork
for state-fork speculation.

**The walls (different from Qwen's):**
- **CCA has no TP>1.** Per-head RMSNorm + grouped-mean state break under column-wise tensor
  splits; `zaya.py` warns and runs every rank as TP=1. So serving is **DP=2 + expert-parallel**
  (replicate the model across both cards), never TP.
- **Spec-decode / prefix-caching is default-off.** The "all" mamba-cache layout overflows
  gfx1201's **64 KB LDS** in `chunked_prefill_paged_decode`, so only *align* mode runs;
  `ZAYA_SPEC_ALL=1` (prefix-cache reuse + per-position state-fork) stays off and unvalidated —
  the per-position `_decode_verify_spec` machinery exists, but the actual multi-branch *tree*
  spec is unbuilt.
- **Same tilelang ABI skew** as the Qwen side (apache-tvm-ffi) — handled with a
  `has_tilelang=False` overlay; the router is pinned to bf16 under quantization
  (`quant_config=None`) so it's never quantized.
- **Quant is W8A8** here — INT8 (bitsandbytes LLM.int8(), proven on RDNA3) with FP8 e4m3 staged
  (offline CPU expert quant → compressed-tensors). It shares the FP8-WMMA *hardware* path with
  this repo but **not** the W4A8 kernels.

**Test-time compute is in scope.** ZAYA1 is built to run **RSA (Recursive Self-Aggregation)** —
N=16 parallel rollouts, K=4 subset aggregation, T=2 rounds (per the ZAYA1 paper) — via a
client-side OpenAI-compatible proxy (`rsa/server.py`) + a capacity bench harness, plus a
streaming `zaya_xml` tool-call parser and a qwen3 reasoning parser.

**Status: architecturally complete, GPU bring-up in progress.** Everything imports (model
registry resolves, state-copy funcs + parser load on CPU), the CCA kernel compiles, and the RSA
harness passed 5/5 AIME on an *old RDNA3 bf16* stack. End-to-end inference on gfx1201 is being
brought up — test containers are stood up and torn down as the work iterates, so container state
isn't a status signal. The next unchecked box is the simplest one: a coherence gate — send one
chat, read the output. Much of the "frontier" wishlist
(fused conv-attention, first-class recurrent state,
per-position state-fork, test-time-compute orchestration) is *already built*; what's left is
bring-up and the genuinely hard one — **prefix-cache reuse of recurrent state**, which isn't
prefix-sliceable and needs checkpoint/fork semantics.

---

## Act VIII — the small-batch kernels: two honest results

Resuming the kernel work with one mandate: make the custom path **meet or exceed stock in
every regime**, starting with a grouped-GEMV for MoE decode, then a small-batch dense kernel.
The first move was to **re-measure on a dedicated GPU0 (RX 9070 XT, 64 CU)** instead of
trusting the DIARY's framing — and the framing was wrong.

**The gap is small-batch (M=4-64), not "mid-M 512-2048."** Against an extracted, byte-faithful
Triton-W4A16 baseline (`triton_w4a16_ref.py` — the host build venv has no vLLM, so the real
production kernel was copied out to compare on bare metal), the dense path **already crushes
Triton at M≥128** (v10: 2-6× at M≥256) and wins M≤2 (v11 GEMV). The *only* loss is **M=4-64**
(1.3-1.76× slower). Same shape on the MoE side: the WMMA grouped kernels hit only **10-23% of
peak weight bandwidth at decode**, and production doesn't even engage below M=64. Both gaps are
the same animal: a small-batch, weight-bandwidth-bound GEMM where a big WMMA tile wastes rows.

**Piece 1 — `moe_gemv_v7`, the grouped GEMV (a real win over our own kernel).** The MoE analogue
of dense v11: one warp per output column, lanes stream the expert's weight row coalesced (b128),
expand int4→fp8 in-register, **no per-group barrier** (g=32 ⇒ 72 `__syncthreads` in the WMMA
path — gone). The unlock was **compacting to the real routed rows** (decode = 1-2 of block_m=16)
and sizing LDS by `min(block_m, T)` so occupancy isn't throttled by padding. Result (op-level,
gemm1): **3-4.6× faster than v6** at T≤8; full `_run_grouped_moe` apply **1.5-2.4× faster than
v6** end-to-end (validated in-container). **But** — measured against stock `fused_experts` — our
v7 apply is still **1.0-2.0× slower at M=1-64**. The decode bottleneck is no longer the GEMM:
it's **apply-level fusion**. Stock fuses gemm1→silu→gemm2→reduce into one kernel; ours is ~4
launches + intermediate HBM buffers (out1, buf2, out2). v7 is the right foundation and a genuine
2-3× kernel improvement, but it does not beat stock at decode — and a per-step breakdown found
why, and it isn't fusion. The apply *is* the two GEMMs (silu/align/Python ~0.02ms; `sum ≈
apply`). gemm1 reads w13 at **273 GB/s** (already beats stock's 248 per byte); **gemm2 reads w2
at only 126 GB/s — the whole gap.** Isolation ruled out every removable cause: a non-atomic
diagnostic (0.427 vs 0.437ms) kills the scatter-atomics theory; NWARPS 8/16/32 and every
v6-WMMA/v7-GEMV/block_m/BN config all converge to a **~0.42ms / ~120 GB/s gemm2 wall**. Four fixes turned that 1.0-2.0× loss into a **win across M=1-96** — and none of them was the
weight-repack I'd assumed. (1) **int→float weights**: the GEMV was doing
`e4m3_to_f32(int4_to_e4m3(nibble−zp))` — a fp8 round-trip that is *identity* (nibble−zp ∈
[−15,15] is exact in e4m3), so it's just `(float)(nibble−zp)`. The fp8 hop and its `wf[32]`
register array were pure dead work; the WMMA path needs fp8, the f32-accumulating GEMV never did.
(2) **COLS column-tiling, tapered** (issue several weight reads/warp to hide short-K latency, but
back off as M grows or `acc[COLS][MMAX]` registers kill occupancy). (3) **block_m=8 for the GEMV**
(it doesn't tile, so the WMMA-multiple constraint was needless) — caps real-rows/block at 8 so
`MMAX=8` holds occupancy through batched decode. (4) **packed `cvt_pk_f32_fp8`** (2 activations per
convert). Then a fifth fix closed M=32: the decode-default **K-chunking** (BK=1024 → 3 chunks for the
K=2304 gemm1, a `__syncthreads` each) was itself dead work — staging the **full K in one chunk**
(BK=K, LDS still ≤18 KB) deletes the syncs and sped up *every* M (gemm1 @M=32: 1.33→0.97 ms).
Result, full-apply vs stock `fused_experts`: **0.66× @M=1, 0.83× @M=8, 0.88× @M=16, 0.97× @M=32,
0.56× @M=48, 0.73× @M=64, 0.98× @M=96** — **win or parity across the entire M=1–96 decode/mid
range**, all bit-exact. And the same "find the dead work" move finished the job at the *large*
end: the **WMMA path (v6) still paid a `__syncthreads` per K-group** (72 of them at g=32,
K=2304) with no double-buffering to hide them — staging **gtile=4 groups per sync** cut that 4×
and took **M=128 from 1.10× to 0.53×, and prefill M=2048 from ~0.9× to 0.59×**. So the served MoE
pipeline (v7 ≤ M=96, v6+gtile > 96) now **beats or matches stock at every M from 1 to 2048**
(0.55–1.00×) — the grouped-MoE chapter is, finally and measurably, won everywhere. The lesson that kept paying out:
**measure the apply + the achieved bandwidth, find the *dead work* (the fp8 round-trip, the
oversized MMAX, the K-chunk syncs), and the bounded fix is there** — the "needs a Marlin repack"
verdict was wrong three times over.

The same recipe then carried to the **dense** GEMV (v11), which had never been optimized: applying
int→float weights + packed converts took dense **M≤4 to a win at g=32 (0.58–0.90×)**, and a new
small-M WMMA (**v14**, N-split warps / BM=16 to cut padding waste) brought **M=4–8 to parity**.
But dense **M=16–32 stays ~1.5× behind** and is the honest wall. The obvious culprit was the
LDS weight-staging, so I built the textbook fix — **v15, a Marlin-style register-direct kernel**:
weights pre-repacked offline into WMMA-B-fragment lane-order so one coalesced 128-byte load per
warp fills the fragment with no LDS and no shuffle, exactly how Triton feeds `tl.dot`. It's
bit-exact — and hits the **identical ~108 GB/s**. That falsified the hypothesis: LDS-staging,
register-direct-shuffle (v13), and Marlin-repack (v15) all converge to 108 GB/s, so the ceiling
is **not** the weight-read mechanism at all. It's a tiny single GEMM (1.7% of WMMA peak, 15% of
HBM at M=16) that's latency-bound, where Triton's *autotuned* fp16 `tl.dot` (195 GB/s) simply
schedules an under-utilized GPU better than any of the ~18 hand-written WMMA variants I tried.
So the chapter rests there honestly: dense M=16–32 is a compiler-codegen gap on an under-filled
GPU — not dead work, not a missing technique, and not the weight repack everyone (me included)
assumed. **But the goal is the served *pathway* ≥ the stock *pathway*, and there the resolution
was hiding in plain sight:** the stock path runs Triton W4A16 with a config tuned for *gfx1151*
(40 CU, BLOCK_K clamped to 64) on our *gfx1201* (64 CU). That config is itself leaving 1.2–1.6×
on the table at small M for g=128 — so I shipped a **gfx1201-tuned Triton** in the dense fallback
(BLOCK_K = the full group, gated to M≤32 ∧ g>64 where it strictly wins) and routed the small-M
fallback to it. Now the served W4A8 pathway **meets-or-exceeds the stock pathway in every regime**:
MoE M=1–2048 and dense M≤2 / M≥256 win on our HIP kernels; dense M=16–32 *exceeds* stock at g=128
(1.2–1.4× via the correctly-tuned Triton) and *meets* it at g≤64 (where the stock config is
already optimal, so the fallback is identical). The custom fp8-WMMA kernel still can't out-WMMA
an autotuned compiler on that tiny GEMM — that part is a proven hardware limit — but the *path*
the goal actually measures no longer falls below stock anywhere, because the place our kernel
can't win is exactly the place the *stock* path was misconfigured for this GPU.

**Piece 2 — `v12`, split-K small-M dense (a measured dead-end, usefully).** At M=8 the dense path
is occupancy-starved (one full-K tile = ~32 blocks on 64 CUs ⇒ 108 GB/s vs Triton's 190). The
obvious fix — split K across `grid.z` blocks, fp32 atomic-accumulate — is bit-exact (rel 1e-7 vs
v5) and **2× faster than v5 at M=4-8**, but **still 1.7-2.2× behind Triton** and, the tell, **stuck
at ~107 GB/s no matter how many splits**. So it was never occupancy — it's the **int4→fp8 LDS
round-trip**: we expand weights to fp8 *in LDS* before the WMMA; Triton dequants int4→fp16
*register-direct*, fused into `tl.dot`, no round-trip. Beating Triton at small-M dense needs a
Marlin-style weight repack + register-direct WMMA feed (ROADMAP Task 4), not split-K. Deferred.

**The chapter's lesson, and where the two pieces converge.** A faster GEMM kernel was the wrong
unit of optimization for small batch — but the deeper finding is that **both pieces hit the same
wall, and it's the weight read.** Dense small-M loses because we expand int4→fp8 through LDS while
Triton dequants register-direct (190 vs 108 GB/s). MoE decode loses because gemm2's short K gives
short coalesced bursts (120 vs gemm1's 273 GB/s). Same disease: stock reads quantized weights in
**long, coalesced, register-resident bursts**; we read them in short bursts staged through LDS.
The fix that would win *both* regimes is a register-direct WMMA feed avoiding the LDS round-trip
— so I built it (v13: load int4 coalesced, warp-shuffle to the B-fragment layout reusing v10's
A-shuffle, expand in-register, split-K). Bit-exact, and **slower** — 5-15× on large shapes. Two
reasons, both instructive: a `__shfl` per (k-subtile, N-frag) makes it shuffle-bound, and B's
(N, K/8) layout makes the transpose-load *strided* (consecutive lanes jump `ppr·4` bytes), so the
"coalesced" burst is 8 bytes with huge gaps — exactly what LDS staging exists to fix. So the LDS
round-trip wasn't the enemy; **Triton wins by a vectorized `tl.interleave`+`tl.dot` whose codegen
the compiler optimizes**, and matching that is a from-scratch codegen-quality WMMA effort (or a
Marlin-style weight repack), not a bounded edit. That is the honest ceiling found this chapter. What stands today: v7 (+2-3× over our prior MoE
kernel, gemm1 beats stock per byte) and v12 (+2× over v5) are real, validated improvements; the
**AUTO dispatch keeps the served pathway ≥ stock in every regime** (fallback at small batch, wins
at prefill: dense +53%, MoE prefill via gather-reduce); and the small-batch *kernel* win is now a
precisely-scoped weight-layout problem rather than a vague "make it faster."

---

## Act IX — convergence: the combined image, and what the tuned attention adds

As the kernel chapter was closing, a second RDNA4 vLLM appeared — a collaborator's image
(`tcclaviger/vllm22:dev`) that had independently built **the entire attention half of the
problem**: the tuned 3D `triton_attn` made the gfx1201 default (not the narrow, brittle
`rocm_attn`), an fp8-KV **native** path (the CUDA-only query-quant gate extended to gfx12), the fp8
dequant refactored to *fold the scale* instead of materializing an fp32 tile, and a **startup
autotuner** that profiles the deployed shape per attention group and picks `num_warps`/
`waves_per_eu`/tile at engine init — the "occupancy 4× too low on RDNA4" problem solved properly.
Our W4A8 work is the GEMM half; theirs is the attention half. So we stopped maintaining two stacks
and smashed them together.

**Integrating W4A8 onto their image.** Their base is a different world — system ROCm 7.2.1
(amdclang 22), a py3.12 venv, vLLM 0.22.69 editable at `/app/vllm` — vs our TheRock 7.14. The
kernel had to be **rebuilt in their image** (ABI binds to their torch); it compiled first try
(amdclang 22 carries the gfx1201 fp8-WMMA builtins), passed dense + MoE correctness on their ROCm,
and every `vllm.general_plugins` hook engaged on 0.22.69 — the one adaptation was re-deriving the
`moe_wna16` tp_size fix *surgically* (their copy had diverged with a SiLU-only assertion the
whole-file patch would have clobbered). One drift bug earned in blood: the repo's kernel **source**
was a stale snapshot whose adapter already called v10/v11 against a v5-only `.so`. The fix was to
stop copying the kernel into the repo and build it from its **canonical csrc via a BuildKit
build-context** — a stale copy can no longer be the build input.

**What the tuned attention actually adds (same kernel, old vs combined):**

| (Qwen2.5-Coder-7B-AWQ) | OLD (`rocm_attn`) | NEW (tuned) | Δ |
|---|---|---|---|
| prefill (tok/s) | 66.8k | 87.4k | **+31%** |
| prefill, fp8-KV | 55.7k | 87.3k | **+57%** |

The **+31% prefill is pure attention efficiency** — identical at the 210 W power cap and after the
limit was bumped to 374 W (prefill isn't power-bound, so it's no cap artifact). And the **fp8-KV
dequant tax is real and now fixed**: old path, fp8-KV *cost* −16% prefill (the materialize); new
path, neutral — so fp8-KV flips from "don't bother" to a free 2× KV-capacity lever, which is the
whole game on 16 GB cards. The two customizations are **orthogonal** (attention path vs GEMM path),
so they stack *across* regimes — tuned attention owns prefill, W4A8 owns decode — not within one.

**War stories from the bench:**
- **`VLLM_ROCM_W4A8_FORCE=on` is a trap.** It forces our kernel into every shape, including the
  ones AUTO correctly sends to Triton — it *halved* dense throughput (438→212). A debug toggle, not
  a perf setting. And it doesn't engage the MoE: the grouped kernel has its own gate
  (`VLLM_ROCM_W4A8_FP8_WMMA_MOE_MIN_M`), so FORCE lit up only the dense path.
- **Single-shot timing lies.** Decode swung ±30% run-to-run until `jit_monitor` revealed kernels
  JIT-compiling *during* the timed window — one short warmup didn't cover the decode shapes. Warm
  up at the timed shape and the numbers settle.
- **Hitting PPT0 is good news.** Pinning the card's power limit means the kernels *saturate* the
  hardware, not stall on memory. And since +30 W still triggers PPT0, there is no "unconstrained
  peak" to chase — the honest framing is **perf-per-watt at the power wall**, where +31% prefill at
  equal watts is a real win.
- **W4A8 broadens support for free:** the GPTQ MoE *crashes to load* on stock (a `triton_w4a16`
  qzeros assertion); our dense kernel replaces that path, so it runs at all.

**Het-TP, and a multi-agent mess.** The heterogeneous-TP work — split the FFN/MoE *intermediate*
64:56 (proportional to the 64-CU XT vs 56-CU 9070) so the big card stops spin-waiting at the
all-reduce barrier (the rank1 COMM=61% bubble from Act VI) — had been built by a *second* Claude
session, but against the **wrong vLLM tree** (the zaya checkout) because it couldn't find ours and
didn't ask, entangled with the CCA changes. Consolidating it: scaffold a gated apply slot in
`Dockerfile.combined` (`--build-arg WITH_HET_TP=1`), write a precise handoff doc, and have that
agent **re-target** the patch to 0.22.69 — where the MoE intermediate split had, helpfully, moved
out of `fused_moe/config.py` into `layer.py`. It came back a clean 3-file diff that applies and
whose CPU apportionment self-tests pass, gated dormant until a 2-GPU validation window. The lesson
is process, not code: **with three sessions and no channel between them, the repo + `memory/` + a
written handoff are the only coordination that works** — and "couldn't find the source, didn't ask"
is exactly how work lands in the wrong tree.

---

## Act X — the kernels grow up: fusion, an autotune gate, and a VRAM win

Act VIII left the W4A8 kernels in an honest but frustrating place: validated, faster in their
regimes, yet **gated dormant** because stock's *fused* MoE apply and *register-direct* dense weights
still won at small batch — and, worse, the dispatch gate returned `_NEVER` for any shape absent from
the crossover caches, so on **any checkpoint we hadn't pre-profiled the kernel was dead weight and
the served path ran stock-vs-stock**. This act is the kernels growing into the served pathway, on
three fronts, every change either **bit-exact** or **gated OFF until measured**.

**Fusion — stop paying HBM round-trips.** The two biggest wins were structural, not arithmetic:
- *Dense act-quant fused into the v5/v10 prologue.* The separate `compute_act_fp8_and_scales` launch
  and the `(M,K)` `x_fp8` HBM write-then-read-back are gone; both kernels take fp16 activations
  directly and quantize to e4m3 inline during A-tile staging. The per-row scale is `max(|x|)/448`
  over the full K row (exact under any thread tiling), so it's **bit-exact** to the old staged buffer.
- *MoE `silu_and_mul` fused into gemm1's epilogue.* gemm1 now writes the post-activation `(P, inter)`
  directly, dropping the `(P, 2·inter)` and `(P, inter)` HBM staging plus the silu launch — one block
  computes both the gate and up N-tiles for an expert and writes `silu(gate)·up`, **`max|diff| == 0`**
  vs the unfused path. (Targets prefill/mid; Act VIII already showed decode is gemm2-BW-bound, so the
  GEMV decode path is deliberately left unfused.)

**An autotune gate — the kernel now works on any checkpoint.** On a crossover-cache miss the gate now
runs a quick ours-vs-Triton A/B for that exact shape (dense at load, MoE on first batch), selects the
*safe* crossover with the existing winning-suffix / contiguous-interval rule, persists it, and is O(1)
thereafter. Any autotune failure caches `null → stock`, so **the served pathway still can't regress
vs stock** — Act VIII's safety property is preserved; only the "untuned checkpoint = dead weight" hole
is closed.

**A VRAM win that buys concurrency.** Single-layout is now the served default: we skip the `(K, N//8)`
Triton-fallback weight copy that *doubled* dense weight VRAM (it had OOM'd the 27B on 16 GB cards) and
hand the freed ~50% to KV cache. Reversible via `VLLM_ROCM_W4A8_LAYOUT=tuned` (rebuild the copy, keep
the strictly-≥-stock small-M tuned-Triton path at 2× VRAM). The apply path now gates on the *actual
presence* of fallback weights rather than the env, closing a latent `None`-deref the new default would
otherwise have hit.

**And the next levers, scoped honestly.** v6 (b128 double-K — two back-to-back WMMAs per LDS read,
halving v5's load instructions) is wired into dispatch but **gated OFF** for the mid-M ~512-2048 band
that still trails Triton; and a research doc (`RESEARCH_burst_repack.md`) maps the **MoE-decode
126 GB/s wall** — gemm2 is genuinely bandwidth-bound (gemm1's 273 GB/s is the existence proof, and
Act VIII's dense v13/v15 falsification does *not* transfer, because dense was GPU-starved), with an
N-interleave burst-repack as the proposed fix. The Act VIII discipline holds: measure achieved
bandwidth, name the lever, don't turn it on until a GPU window proves it.

**The kicker — and it inverts the read above: every A/B this cycle was `enforce_eager`, and graph
capture turns out to be a *stock-only* speedup.** With cudagraphs off, decode is CPU-launch-bound —
both paths wait on Python launch overhead across 40 layers × 128 steps, a regime in which our kernel
looked competitive, which is why the sweeps read favorably. But the production default is cudagraphs
*on*, and two things are now clear there: (1) our W4A8 kernel runs **the same captured or not** — it
is graph-invariant — while (2) graph capture makes the **stock** path substantially faster by
collapsing its launch overhead. So the balance didn't move because our kernel regressed; it moved
because stock got a speedup **we don't share**, and under cudagraphs the served W4A8 path lands
**slower than stock**. The likely reason our path sees no graph benefit is that it's already
**GPU-compute-bound** per call (the int4→fp8 expansion + MoE apply are the ~41%/~20% decode buckets
the profile flagged), so eliminating launch latency buys it little — the exact opposite of stock's
lighter, launch-bound kernels. The **structural** wins here still stand (bit-exactness, fewer HBM
round-trips, ~half the dense weight VRAM); the **throughput case does not**, and "turn on cudagraphs"
is not the fix that rescues it — it's the measurement that *exposed* it. Worse for the autotune gate:
its crossovers were tuned from eager A/Bs against a slower-than-real stock baseline, so every
threshold needs re-tuning against **stock-with-cudagraphs**. Full numbers next cycle.

---

## Act XI — het-TP, validated: the COMM bubble actually closes

Act IX landed the heterogeneous-TP patch — split the FFN/MoE *intermediate* 64:56 (proportional to
the 64-CU 9070 XT vs the 56-CU 9070) so the faster card stops spin-waiting at the all-reduce barrier
— but gated it dormant, "pending a 2-GPU validation window." This act is that window, and it splits
into the two questions any sharding change has to answer: is it still correct, and does it actually
do anything.

**Correct — byte-identical.** Greedy-equivalence het≡even is token-for-token identical on both the
dense 7B (TP=2) and the 35B MoE (TP=2): the proportional split is math-preserving, as designed
(`profiling/het-e2e/`).

**Effective — the bubble shrinks ~75%.** A per-rank kineto A/B (`vllm22-w4a8:hettp`, het patch baked,
W4A8 *off* — the imbalance is a TP-balance effect independent of the GEMM path) confirms the even
split is genuinely lopsided: rank1's all-reduce runs ~400 ms longer than rank0's — the documented
sync bubble. The 64:56 split collapses it: all-reduce imbalance **399.6 ms → 99.0 ms (~75% smaller)**,
total device-time imbalance **405.6 ms → 35.8 ms** — the two ranks now reach the barrier in
near-lockstep. Non-collective compute was already balanced (<2% spread across all four traces), so the
bubble lived almost entirely in the collective wait, exactly where het collapses it. **The
proportional sharding works as designed.**

**But tok/s is flat (200.5 vs 200.6) — the expected catch.** The A/B ran `enforce_eager`, so decode
is CPU-launch-bound: non-collective GPU compute is only ~2.9 s of the ~10.2 s wall, the GPU idling
between Python launches across 40 layers × 128 steps. The het rebalancing is real but hidden under
launch latency — the project's standing note made concrete: **het-TP pays off only after cudagraphs.**
The next step is the same A/B with cudagraphs (drop `enforce_eager`), where GPU compute becomes the
critical path and the ~400 ms even-split imbalance should convert toward the ~5% decode ceiling — and
the residual ~99 ms hints the true ratio sits a hair past nominal 64:56 (worth tuning to the measured
CU/throughput ratio).

So het-TP graduates from "landed, gated dormant" to **validated and working**: correct, and
demonstrably balancing the barrier — with the throughput conversion now a cudagraph A/B rather than
an open question.

---

## Act XII — the second model, landed: ZAYA1 + CCA, coherent on the combined base

Act VII opened the ZAYA1-8B hybrid-CCA front and parked it on a bring-up window; the previous cut of
this act had it staged on a branch. It is now **merged into `main` and serving coherently** on the
combined base (`tcclaviger/vllm22:dev`, vLLM 0.22.69) — the second model is no longer a port-in-waiting.

- **Consolidation.** All ZAYA1 + CCA development moved under `zaya/` — the overlay vLLM files
  (`zaya.py`, `cca.py`, configs, the `zaya_xml` tool parser, `cca_attn.py`), the CCA HIP csrc, and a
  registration patch **re-derived for 0.22.69** (16/16 hunks clean). The old `zaya/vllm-therock` tree
  is retired.
- **First coherent serve cost a weight-loader port.** The blocker to coherent output on the combined
  base was the MoE expert weights: the therock tree nested them under a `RoutedExperts` module, but
  the base wires experts through its factory `FusedMoE`. Re-homing the loader onto that factory is
  what produced the first coherent ZAYA serve on the combined image ("capital of France" → "Paris").
- **CCA is now three fused kernels, all bit-exact.** The eager CCA step is the **graph-broken
  `vllm::cca` op** — ~1.4M tiny pointwise ATen launches per decode run, latency-bound on wave32. Three
  HIP kernels collapse each region into one launch: **decode** (`cca_decode_qk`, 142 µs → 8 µs/call),
  **prefill** (`cca_prefill_qk`, 2.4–3× standalone), and **mixed** (prefill+decode in one forward,
  disjoint conv-state slots). Validated bit-exact — qk rel ~6e-7, state bit-exact, `mixed == pure⊕pure`.

**The perf result — and why it's the exact mirror of Act X.** A/B on an exclusive 56-CU 9070,
ZAYA1-8B-fp8, TP=1, **cudagraphs ON** (not eager): the fused decode kernel is **+38.6% chat-decode /
+49.5% RSA-decode tok/s** over eager CCA. And — crucially after Act X's kicker — this win is **real
under the production config**, not an eager artifact: because `vllm::cca` is graph-broken, cudagraphs
*can't* capture the launch storm, so fusing it is the only lever and the gain survives graph capture.
That is the inverse of the W4A8 dense/MoE kernels, which are capturable — so there graphs sped up
*stock* and erased our edge. Same hardware, opposite verdict, for one structural reason: **capturable
vs graph-broken.** (The prefill/mixed kernels are throughput-neutral on an exclusive card — prefill
is a small share of these MoE-bound workloads — but bit-exact, and worth it for launch-storm /
bandwidth relief under multi-tenant pressure and for killing the eager mixed-batch fallback.)

**Defaults flipped on.** `ZAYA_CCA_HIP` and `ZAYA_CCA_HIP_PREFILL` now **default ON** (no regression,
bit-exact, all-mode coverage) — the staged-and-gated-OFF status from the previous cut is gone.

**Packaging — the Act IX lesson on purpose, then consolidated.** ZAYA first landed as a **derived
image** (`Dockerfile.zaya` `FROM vllm22-w4a8:combined`) with a standalone compose, rather than a gate
inside `Dockerfile.combined` — chosen to avoid editing a file being refactored in a parallel session;
it added zero risk to the base image and let ZAYA land independently. Once that refactor settled, the
derived image was **folded into the single combined image** behind a `WITH_ZAYA` build arg (default
on, mirroring `WITH_HET_TP`): the two shared-file patch hunks — a generic torch.compile-safety guard
in `fused_moe.py` and a `vllm::cca` `splitting_ops` entry — are no-ops for the 35B path, so **one
image now serves both**, via the `zaya` profile in the single `docker-compose.yml` (the standalone
`Dockerfile.zaya` / `docker-compose.zaya.yml` were retired). That compose also carries an optional
**RSA proxy** (`Dockerfile.rsa`, `--profile rsa`): a no-GPU, OpenAI-compatible shim fronting the
backend with Recursive Self-Aggregation (N=16/K=4/T=2) for test-time compute.

**What's left:** multi-card ZAYA (CCA has no TP>1, so DP=2 + expert-parallel — model replicated per
rank, MoE sharded) is the next profile, its exact topology to be confirmed on the combined base in a
GPU window before it's committed, not staged blind.

---

## Act XIII — the public release: from three wheels to one image

Act I shipped the stack as three from-source wheels (vllm, aiter, flash-attention) you fetched
from a GitHub Release or rebuilt yourself; Act IX folded all of it into one combined image. This
release (v0.1.0) makes that the *only* distribution: the prebuilt image is published to **GHCR**
(`ghcr.io/patcarter883/rdna4-vllm`), auto-built and pushed by CI on every `v*` tag, and the user
story collapses to **clone → `cp .env.template .env` → `docker compose up`**. There are no wheels
to fetch or compile — vLLM 0.22.69, the tuned RDNA4 attention, aiter and flash-attention all arrive
inside the base image; only the W4A8 kernel is built in-image. The from-source wheel recipe (the
aiter enablement — `PREBUILD_KERNELS=0` for gfx1201 and all) isn't so much deleted as **archived**:
pushed to `archive/aiter-wheel-distribution` on the remote in case we ever want to rebuild aiter
from source again, and the two `scripts/*-wheels.sh` recipes come out of `main`.

**A blocker the release would otherwise have shipped.** The tag-triggered CI built `file:
./Dockerfile`, but the repo's build file is `Dockerfile.combined` (what compose, the README, and
CLAUDE.md all name). A `v*` push would have *failed the GHCR build and produced no image* — a
release that releases nothing. Caught in prep; fixed by pointing the workflow at `Dockerfile.combined`
rather than renaming the file (one line vs. touching four docs). v0.1.0 is the first formally tagged
release; the wheel era never carried a real tag.

**And the coordination lesson again — but handled right this time.** Prep ran *concurrently* with a
second agent merging the `feat/w4a8-*` branches into `main`; the tip moved under me three times
mid-task (`4db75d1 → 22e448a → 6b0c8b5`) as merge after merge landed. Unlike the het-TP misfire in
Act IX, the channel existed this time: a heads-up that the W4A8 agent owned `main` until it was done,
so release prep stayed **read-only** (archive the branch, draft the notes, find the blocker) and the
mutating steps — delete the scripts, fix the workflow, cut the tag — waited for the all-clear. The
repo is still the only shared memory between sessions; the difference is that someone said so out loud.

---

## The short list of things that cost the most time

1. Assuming gfx1250 kernels would port to gfx1201. They don't — no TDM.
2. A C++ abort unwinding through Rust with no Python traceback (× 2: tilelang, AOTriton).
   When you see *"Rust cannot catch foreign exceptions,"* reach for gdb immediately.
3. Mistaking a slow in-process Triton compile for a hang. py-spy is the arbiter.
4. ABI mismatches from host-built artifacts. Build inside the target image.
5. Letting an iGPU into the build/detect arch list.
6. Reaching for rocprofv3 when the whole decode compute is Triton — it captures none of it,
   only the comm/copy glue. And trusting rank1's 61% COMM share (a heterogeneous-TP spin-wait
   artifact; rank0 is the truth).
7. Optimizing the GEMM kernel for the small-batch regime. The stock path wins there by
   *architecture* — fusing the whole MoE apply into one kernel, and dequanting weights
   register-direct instead of through LDS — so a faster grouped-GEMM (v7, +2-3×) or split-K
   dense (v12, +2×) improves *our* path without catching stock. Measure the apply / the
   achieved bandwidth, not just the GEMM (Act VIII).
8. Three Claude sessions with no channel between them, and `VLLM_ROCM_W4A8_FORCE=on`. The het-TP
   work landed in the *wrong* vLLM tree (the agent couldn't find ours and didn't ask); FORCE
   *halved* throughput because it overrides the AUTO crossover. Coordinate through the repo +
   `memory/` + a written handoff; never benchmark with FORCE; re-target patches per vLLM version
   (the MoE split moved files between 0.22-therock and 0.22.69) — Act IX.
9. Benchmarking the served path under `enforce_eager`. Cudagraphs off makes decode
   **CPU-launch-bound** — both paths wait on Python launch overhead — which **flatters our kernel**:
   graph capture is a *stock-only* speedup (our W4A8 kernel is GPU-bound and runs the same captured or
   not, while stock collapses its launch overhead), so eager A/Bs hid that stock-with-graphs pulls
   *ahead*. Every W4A8 A/B this cycle was eager and read too favorably for it; the honest baseline is
   **stock with cudagraphs on** (the production default), not eager-vs-eager. Het-TP is the mirror
   case — its GPU rebalancing only *surfaces* with graphs (Acts X-XI). Sibling of the FORCE trap: both
   make the benchmark measure the wrong thing.

## What actually works, today

Qwen3.6-35B-A3B-AWQ-4bit serving on two RX 9070-class cards, TP=2, coherent text,
**298 dec / 1887 total tok/s** on the stock path; the W4A8-FP8-WMMA kernel is
correctness-validated end-to-end and, when force-engaged, gives regime-dependent e2e
results (dense prefill **+53%**, MoE decode **+11%**; neutral-to-negative at mid batch —
see Act VI's correction note); all of it reproducible from a clone and a `docker compose up`.

The **AUTO dispatch keeps the served pathway ≥ stock in every regime** — it falls back to
stock where stock wins (small batch) and engages our kernels where they win (prefill /
large M) — so "meets or exceeds stock in all regimes" holds by construction. Act VIII added
two validated small-batch kernels (`moe_gemv_v7` decode GEMV, +2-3× over our prior MoE
kernel; `v12` split-K dense, +2× over v5) that are gated dormant in AUTO because stock's
*fused* MoE apply and *register-direct* dense weights still win at small batch — the named
next levers to turn those fallbacks into wins. Act X then made the served path *engage on any
checkpoint* (an autotune gate replaces the "untuned shape = dead weight" hole, still `null → stock`
on failure) and shaved its memory traffic — act-quant fused into the dense prologue and `silu_and_mul`
fused into the MoE gemm1 epilogue, both bit-exact — with single-layout the default to hand ~50% of
dense weight VRAM back to KV cache.

**That "≥ stock by construction" needs a hard correction.** Every W4A8 throughput number above — the
force-engaged +53%/+11%, the small-batch crossovers, the Act X gate decisions — was measured
**`enforce_eager` (cudagraphs off)**, where decode is launch-bound and our kernel looked competitive.
But cudagraphs is the production default, and graph capture is a **stock-only** speedup: our kernel is
graph-invariant (already GPU-bound per call), stock collapses its launch overhead, and under graphs
the served W4A8 path lands **slower than stock**. The by-construction safety argument fails twice over
— the crossover was tuned in the wrong regime, *and* its baseline (stock) itself speeds up once graphs
are on. Honest verdict: *correctness end-to-end solid; throughput vs the real (cudagraphs) stock
baseline, currently negative.* The thresholds need re-tuning against stock-with-cudagraphs, and this
section gets rewritten around those numbers next cycle.

And we now know where to *stop*: a per-op decode profile puts full attention at ~1.8% and the
dominant buckets (MoE ~41%, dense fp16 ~20%) at ones we already understand — so the
int4→fp8-WMMA kernel chapter closes on evidence (the open lever is the gemm2 126 GB/s decode wall,
mapped in `RESEARCH_burst_repack.md`). The ZAYA1 hybrid-CCA front is no longer a port-in-waiting:
**merged to `main` and serving coherently** on the combined base, its CCA decode/prefill/mixed
regions fused into HIP kernels that — because `vllm::cca` is graph-broken — give a **real,
cudagraphs-on +38%/+49% decode** win (the mirror of the W4A8 graph story), now **folded into the
single combined image** (`WITH_ZAYA`, default on) and run via the `zaya` profile, with an optional RSA
test-time-compute proxy; multi-card DP+EP is the next profile (Act XII).

And the two halves are now **one stack**: development has consolidated onto the **combined image**
(the collaborator's tuned attention + our W4A8, built from canonical sources) — measured
**+31% prefill** and **fp8-KV made free** on top of the kernel work (Act IX). Heterogeneous-TP
(64:56 proportional sharding) is now **GPU-validated**: byte-identical to the even split and the
all-reduce imbalance ~75% smaller (399.6 ms → 99.0 ms), with the wall-clock payoff awaiting a
cudagraph A/B since the `enforce_eager` profile is launch-bound (Act XI). And as of v0.1.0 it ships
as a **prebuilt GHCR image** built on every tag — no wheels to
compile, just `docker compose up` (Act XIII).

## Act XIV — the cudagraph A/B, finally measured: the win is stack-dependent

The "awaiting a cudagraph A/B" promissory notes from Acts XI and the W4A8 throughput section
got paid. Three images (kyuz0 TheRock 0.21, tcclaviger 0.22, our combined 0.22) × eager vs
full-compile+cudagraph × batch {1,16,32,64}, `vllm bench latency` with every run's
`enforce_eager`/capture state verified from its own log, Qwen3.5-4B-FP8-dynamic (a GDN hybrid),
in=128/out=256. The headline answers the question that kept recurring — *why do we keep seeing
"no difference" between eager and graphs?* — **because the answer depends entirely on the
stack.** On the **0.22 stack** (tcclaviger and our combined) graphs are a large win —
**+80% at bs1, +107% at bs16, +85% at bs32**, tapering to +6% at bs64 as it goes compute-bound;
the gain is *biggest at low batch*, where eager is CPU-dispatch-bound. On **kyuz0's 0.21
TheRock** build the same A/B is ~**0%** until bs64 (+30%) — the opposite batch dependence. So a
"no difference" observation was a kyuz0-style stack (or eager-vs-eager); on our production
lineage, graphs matter a lot, and the stale `--enforce-eager` serving flags hide exactly that.
A clean side-result: **combined ≈ tcclaviger for FP8** across every batch — the W4A8 layer adds
no regression to non-W4A8 models.

The W4A8 kernel's own eager-vs-graphs story is more sobering. Pointed at a 4-bit model
(Qwen3.5-4B-AWQ-BF16-INT4) the dense kernel **engages and runs in eager** (494.8/869.6/1285.7
tok/s at bs16/32/64, `Using RocmW4A8Fp8WmmaLinearKernel` confirmed per layer) — after two
surprises: it crashes on **bf16 activations** (`tl.dot: bf16 vs fp16`, `can_implement` wrongly
says yes — run `--dtype float16`), and it **cannot run under the 0.22 full-compile + cudagraph
path at all.** That last one is a three-layer wall: the custom op had **no `register_fake`**
(added now, in `w4a8_fp8_wmma/__init__.py` — validated, clears the first crash), then the op
isn't **pickleable** for Inductor's autograd cache (env-disabled the caches), then Inductor
asserts on a **dynamic-shape symbolic graph input** (`sympy GreaterThan`). `register_fake` is
necessary but not sufficient; making W4A8 graph-capable means either real Inductor-tracing work
under BACKED dynamic shapes or registering the op as a `splitting_op` (run eager between graph
pieces, like attention). This also **corrects** the earlier "W4A8 cudagraph capture confirmed"
note — whatever mode that was, it was not 0.22 full-compile. Full data + runners live in
`profiling/bench-eager-vs-graphs/` (`SUMMARY.md`).

## Act XV — the served cudagraph path finally boots on 16 GB, and the numbers land

The promise from Act XIV — "make W4A8 graph-capable" — got paid (M1–M4: `vllm::w4a8_dense_apply`
clears all three Inductor blockers), and this act is what it took to actually **serve** the 35B
on two 16 GB cards with cudagraphs on, end to end, plus the first honest cross-the-board numbers.

**Getting `serve` to boot with cudagraphs was a five-wall slog**, each wall a different resource
ceiling on RDNA4 / 16 GB:
1. **Cudagraph capture set.** vLLM's default (51 sizes to 512, `FULL_AND_PIECEWISE`) *stalls
   capture on ROCm* (vLLM #19579/#39010) — a 12-min wedge at 3 % GPU. Capped to a conservative
   `[1,2,4,8,16,32,64]` (verl's AMD guidance); these cards serve low-batch anyway.
2. **KV-cache OOM at init.** `fp8_per_token_head` + `gpu-util 0.92` left no room for cache blocks;
   `--kv-cache-dtype=fp8` + `gpu-util 0.96` clears it.
3. **Mamba state blocks.** The GDN/Mamba hybrid needs one Mamba block per decode seq and only ~36
   fit; `max-num-seqs 256` aborted cudagraph capture → set **`--max-num-seqs 32`** (< the 43× KV
   ceiling; KV peaks ~60 % under 32 in-flight).
4. **Async-scheduler desync.** Under concurrency the batch-queue path threw a scheduler `KeyError`
   + multiproc `AssertionError`; **`--no-async-scheduling`** (serial `step()`) removes it.
5. **The boot was 22 min** — *not* the attention autotune (that's fast) but the GDN/FLA
   `@triton.autotune` (`chunk_scaled_dot_kkt` &co). The thread-pool **parallel-compile livelocked
   on TP=2** (two ranks × 8 threads thrash the shared `/root/.triton`, zero progress) — turned it
   **off**; serial cache-hits cleanly. With the config frozen + `TRITON_CACHE_AUTOTUNING=1` and the
   vLLM compile cache now mounted, a warm restart is **~90 s** (vs 22 min cold). That's the "use
   the cache next start" guarantee, proven.

**The crash hunt.** The EngineCore died on concurrent inference. A 4-phase diagnosis workflow
(single → ramp → W4A8-off → nightly) found **no code bug**: it's **memory-budget overcommit** at
`gpu-util 0.96` (zero headroom → any inference-time alloc OOMs under heavy/streaming load). Our
W4A8 kernel is **not** at fault (the prior-crashing config passes the full 1→32 ramp + a 32×512
burst, kernel engaged, ~902 tok/s gen). A real side-finding: **pure upstream `vllm-openai-rocm:
nightly` can't even initialise TP=2 on gfx1201** — `ncclCommInitRank` → `HIP failure: invalid
argument`. So there is *no* stock-upstream baseline for this model on this hardware; the
collaborator's fork (PYNCCL backend + `disable_custom_all_reduce`) is what makes multi-GPU work
on RDNA4 at all.

**The numbers** (35B Qwen3.6-A3B-AWQ-4bit, TP=2 het-TP 64:56, 32 prompts × 128 out, gpu-util
0.90, `test/bench_tp2.py`; `profiling/bench-results/results.jsonl`):

| path | decode tok/s | total tok/s |
|---|---|---|
| stock, **eager** | 510.8 | 1371.6 |
| stock, **cudagraphs** | **957.5** | **2571.0** |
| W4A8, cudagraphs | 917.5 | 2463.6 |

- **Cudagraphs are the win: +87 %** decode (957 vs 511) on the current stack — the whole point of
  the graph-compat work, and why `enforce_eager` is now **profiling-only** (codified in
  CONTRIBUTING; the stale `--enforce-eager` serving flags hid exactly this).
- **W4A8 on the 35B is ~4 % *slower* than stock under graphs** — it touches only the MoE experts,
  and graphs don't help MoE (consistent with the "W4A8 cudagraphs are a stock-only speedup"
  finding). The kernel's value here is that it now *runs* under the graph path at all; the dense
  win lives on other models.
- **ZAYA1-8B CCA — the clear kernel win:** the fused HIP `cca_decode_qk` vs the eager ATen path,
  single-stream decode, **36.1 vs 23.0 tok/s = +57 %** (256-tok gens, ±0 across 3 runs each).

The old `298 / 1887` headline is retired: it was an older/eager config (current eager is
510.8 / 1371.6). The shipped serve path is **957 / 2571 stock, 917 / 2464 W4A8** with cudagraphs.

## Act XVI — de-numbering the kernel ladder, and the tiled dense default

The W4A8 kernel grew a ladder of versioned variants — `v5`, `v6`, `v7`, `v10`, `v11`, `v12`,
`v15/16/17` — each a research rung, the numbers carrying no meaning to anyone who wasn't there.
This act paid down that debt: every dispatchable kernel got a **descriptive name** and the
near-duplicate tile kernels collapsed onto one parameterized policy.

- **Dense ladder** → `DenseKernel` names; the register-direct `v15/16/17` family → `mmq_regdirect_*`
  (the small-shape GEMV regime mapped in the *regdirect wall* finding).
- **Grouped MoE** → named kernels (`wmma` default / `scalar` golden / `gemv`); the old numeric
  `MOE_VERSION` env is gone and now **hard-errors at boot** if set, so a stale launch can't silently
  pick a retired variant.
- **TileConfig consolidation** (`gemm_tiled.h`): the two tiled dense kernels (the old `v5` LDS-staged
  and `v10` WMMA paths) became *one* template driven by a `TileConfig` policy — same kernel, different
  tile shape / residence. Validated bit-exact against both predecessors.
- It shipped **gated** first (`VLLM_W4A8_DENSE_TILED`, off — zero upside without a second dense
  backend to dispatch against), then in `f9dede9` the tiled TileConfig kernels became the **served
  default** for dense, and the grouped MoE tiled TileConfig is the live MoE default. A-residence
  (formerly the `v5`-vs-`v6` distinction) survives as the `VLLM_W4A8_MOE_A_IN_LDS` knob.

No new math — but the dispatch table now reads as kernels, not version integers, and a wrong knob
fails loud instead of quiet.

## Act XVII — mxfp4 on the W4A8 kernel: a decode-table swap, not a new kernel

OCP **mxfp4 (E2M1)** turned out to ride the existing W4A8 path for free. The insight: **E2M1 ⊂ e4m3**
— every 4-bit E2M1 code maps losslessly to an e4m3 byte — so mxfp4 weight decode is a **decode-table
swap** in the in-register int4→fp8 expansion the kernel already does, not a second GEMM. The
microscaling block scale (the `mx` in mxfp4) folds into the per-group fp16 scale the kernel already
carries.

- **Dense** (`9d4d3dd`) then **MoE** (`1e102d0`) E2M1 weight decode, both **GPU bit-exact** against a
  reference dequant — the e2m1 path threads through `_run_grouped_moe` unchanged downstream of the
  decode table.
- vLLM dispatch wired for both: dense E2M1 linear (`83c4ff5`) and **gpt-oss E2M1 MoE experts routed to
  the grouped FP8-WMMA path** (`81968d6`) — so an mxfp4 checkpoint engages our kernel without a code
  change to the serving script.
- It is **orthogonal to the rotation work** (Act XVIII): mxfp4 is a *B-side* (weight) decode format;
  rotation is an *A-side* (activation) conditioning pass. They compose.

The CPU-proven foundation came in on `feat/mxfp4-w4a8`; the served dispatch landed on `main`. No
gfx1201 model ships mxfp4 in the lab yet, so this is a *capability* (a model that does will engage the
kernel), not a measured serving win.

## Act XVIII — RXF supersedes our FP8 angle; the rotation-tuning line, closed as a null

The base image (`tcclaviger/vllm22:dev`, bumped to `ad046fde` this cycle) started shipping **RXF —
"Rotated eXtra Fast"**, the collaborator's successor to RFP458. RXF reframed our whole quant
contribution: it already does **W4A8 as int8×int8 with an integer (IQ4-NL) codebook** — which is
*exact*, no fp8 round-trip — plus a **FWHT-32 rotation** pre-pass to tame activation outliers. That
**obsoletes our int4→fp8-WMMA angle** except in the non-integer-codebook regime; the surviving lever
from the ParoQuant line is the **rotation** itself — RXF ships its activation-aware rotation *stalled*
(`ACT_AWARE_ENABLED = False`, "in development"), and a wider / learned / importance-aware rotation is a
**pre-pass-only** change: the K=32 int8 GEMM, the pack format, and the per-group fp16 scale are all
untouched (`(X·R)·(R·W)ᵀ = X·Wᵀ` for any matched span). That is the entire integration surface, and we
ran it to a conclusion.

**Three stages built, all measured, all null on this W4A8 model:**
- **(a) wider fixed Hadamard span** — generalized the FWHT from a hard 32 to any power-of-two span;
  span-32 stays **bit-identical** to shipped. Sanity-proven cancellation for spans 32–512. *PPL: worse
  on both arms* (`hadamard128` +1.07%, `hadamard256` +0.76% W4A8) — a wider FWHT spreads *bulk* weight
  energy across the size-32 scale-group boundaries, creating ~86–88k degenerate near-zero scale groups.
- **(b) learned Givens rotation** — a single **model-wide shared R** (so it survives merged q/k/v,
  gate/up, and TP shards), built as Hadamard-init greedy Givens coordinate descent (`≥ Hadamard by
  construction` on weight-MSE; from *identity* is strictly worse). *PPL: +0.32% W4A8* — minimizing
  weight-quant MSE doesn't move PPL because Hadamard is already near-optimal for incoherence.
- **(c) activation-conditioning** — the strategic pivot (cf. the DFlash result, where activation
  conditioning saved INT4 acceptance): fit R to the real **per-token int8 activation-quant error**
  instead of weight MSE, `ACT_AWARE_ENABLED` flipped, importance carried into the rotated basis via
  `(R²)·imp`. The decisive cheap gate — `analyze_act_conditioning.py`, real Qwen3.5-4B activations,
  one GPU forward — showed the lever is **span width, not learning** (wider fixed Hadamard cuts the
  real activation int8 MSE up to 2.3×; the **learned** R *regresses* it 0.79–0.94× at every span,
  because the greedy per-block fit overshoots the per-token full-row absmax the runtime actually uses).
  But that activation gain is **PPL-invisible**: int8 ≈ fp16 PPL at every span, i.e. the int8
  activation quant is **already near-lossless** on this model and has no headroom to recover.

**The synthesis:** on this W4A8 model the PPL bottleneck is the **4-bit weight**, the shipped
**span-32 Hadamard is already at/near its optimum**, and the int8 activation path is near-lossless — so
the entire rotation-tuning premise has nothing to recover. **`hadamard32` is the right default;** every
learned/wider/importance delta is ≤ noise or negative. Stage (c) would only pay where the *activation*
quant is the bottleneck (true 4-bit / NVFP4 activations — the DFlash regime), not int8; the machinery
is kept for that future model, not recommended here. The standing rule that fell out: **run
`analyze_act_conditioning.py` (one GPU forward, real acts) before funding any future rotation
experiment** — it would have called this null in an hour instead of a campaign.

**A genuine RDNA4 hardware flake found along the way.** The first GPU quant with `--rotation-kind
givens` generated garbage ("hydrogen and" → "a 1000.00 mL"). Root cause pinned: the offline rotation
matmul `[~737k, 32] @ [32, 32]` (large M) **silently zeros ~29 % of its output rows on gfx1201**, while
the *identical* matmul is exact on CPU. Hadamard dodges it because its FWHT is butterfly add/sub — no
matmul. **Fix:** run the (cheap) Givens rotation on **CPU**, keep the expensive scale search on GPU →
0 % zero groups, coherent generation. A standalone large-M small-K ROCm matmul correctness hazard,
worth remembering independently of the rotation result.

**And ZAYA RXF now serves correctly** (`53cf7de`). RXF garbage on ZAYA traced to the **monolithic RXF
MoE path ignoring `FusedMoE.custom_routing_function`** — it re-derived `fused_topk` from ZAYA's
pre-packed router logits instead of honoring the model's custom router. Forwarding
`custom_routing_function` fixed it (h32 PPL ~41, coherent); folded into the `zaya/` overlay, committed
to `main`, and baked into `vllm22-w4a8:combined`.

---

# In flight (feature branches, not yet on `main`)

The acts above are landed history. The three workstreams below are live on their own worktrees —
recorded here so the next session knows where they are and which wall each is against.

## Act XIX — DFlash speculative decoding: the infrastructure works; acceptance is the regime

Two fronts on diffusion speculative decoding, both converging on the same lesson — *the plumbing is
the easy part; whether the drafter's hidden-state distribution survives the target's quantization is
the whole game.*

**Front 1 — poolside's DFlash drafter on RDNA4** (`feat/dflash-spec`, `feat/rxf-laguna-dflash`). The
deliverable that *works*: vLLM's tuned 3D unified `TRITON_ATTN` kernel (the RDNA4 default) hard-asserted
`causal`, so a DFlash drafter's **bidirectional (non-causal) block** could only fall back to the slower
`ROCM_ATTN` path. The patch in `patches/dflash_triton_noncausal/` makes `TRITON_ATTN` itself
non-causal-capable (thread `IS_CAUSAL` through the unified kernel + `supports_non_causal()`), matching
upstream intent (vLLM #40632 / #42068), plus a PR#40898 SWA backport and a dtype cast — shipped as the
`vllm22-w4a8:dflash` overlay. DFlash boots end-to-end on TP=2, coherent and lossless.
- **The acceptance wall, then its resolution.** On `Laguna-XS.2-INT4` (the only target that fits 2×16
  GB) draft acceptance was **~0.6–0.8%** (pos-0 ~2.3%) vs the speculator card's 70.9% pos-1 — the
  drafter loads and runs but is **mis-conditioned**, a hidden-state distribution shift. Root cause
  turned out to be the **coarse INT4 quant *format*, not a code bug**: switching to
  `Laguna-XS.2-NVFP4` (which emulates → bf16, preserving the distribution the drafter was distilled on)
  recovers it to **25.7%, pos-0 ~70%**. A separate z-lab 4B drafter reaches **36.5% unquant / 48.3%
  AWQ / 40.2% TP2** — so DFlash genuinely *works* on RDNA4 when the format keeps the hidden state intact.
- **The remaining structural cap:** held-out **pos2+ collapse** — a single-pass-mask limitation (later
  draft positions can't see earlier ones). The fix menu (T-step/dual-pass diffusion vs Hydra
  sequential heads vs full-AR) is mapped; sequencing is fix the single-pass conditioning first, then a
  few-step T=2 only if pos1+ stays starved.

**Front 2 — train our *own* CCA-aware drafter for ZAYA1-8B** (`feat/zaya-dflash`,
`docs/zaya/zaya-dflash-plan.md`). Training the drafter ourselves **dissolves the INT4 acceptance wall**
(no external distillation to mis-match) and ZAYA's CCA already exposes the target verify path. Phased
M0–M6; **M0/M1 done** (bring-up + bit-lossless CCA rollback). **M5 reopened (2026-06-22):** an n-gram
bit-identity gate caught a *real* verify-path bug — spec-greedy ≠ non-spec-greedy at token 1 under
*partial* acceptance. Cache-mode, rollback plumbing, and the fused kernel were each ruled out; the
suspect is the `_decode_verify_spec` `all_avail` rollback **column math**. The bit-identity gate
earning its keep — it found a correctness bug a throughput-only A/B would have shipped.

## Act XX — Titans: test-time neural memory, trained from scratch on RDNA4

A full train-*and*-serve of **Titans** (test-time neural memory, arXiv 2501.00663) on consumer RDNA4
— the first workstream here that *trains* a model rather than serving someone else's (`feat/titans`,
image `titans:dev`). The early conceptual correction that shaped everything: Titans' "memory" is a
**per-sequence recurrent *state*, not optimizer-trained parameters** (two Explore agents got this
wrong) — and in its linear form it is **≈ the gated delta rule** already running on the GDN
infrastructure, so it rides FLA/KDA kernels rather than needing a bespoke one.

- **Phase 0 + 1a done:** trained enwik8 (31.5M chars) **from scratch**, val BPC 8.5 → **1.83**;
  checkpoint round-trips bit-exact; generations are locally-fluent MediaWiki-markup English (globally
  incoherent, exactly as expected at 1.83 BPC / 31M params).
- **The recall question, run properly.** enwik8 char-LM is the wrong vehicle to test *memory* recall
  (inconclusive), so the probe moved to **MQAR** (`mqar_probe.py`): memory-vs-ablated is
  **PASS-directional** — the ablated arm is at chance beyond the segment boundary while the memory arm
  is 12–20× chance with lift *growing with distance* — i.e. memory genuinely retains and retrieves
  across segments on RDNA4 (low absolute accuracy = undertrained, not a ceiling).
- **The expressiveness ladder → a shipping decision.** A 5-arm ladder (`mqar_ladder.py` +
  `mem_cells.py::DecoupledGateCell`) confirmed the memory cells work (~0.94 cross-segment vs ablated
  chance) and that **GDN-2 (decoupled per-channel gates) > scalar gated-delta** when off-ceiling.
  **Decision (2026-06-24): provisionally SHIP GDN-2, DEFER the deep-memory chunked kernel** — GDN-2 is
  closed-form so it rides the existing FLA/KDA kernels and sidesteps the Phase-2 deep-mem kernel risk
  (deep memory never got a fully fair run but is unlikely to clear the payoff gate). Theory backstop:
  the Miras follow-on (arXiv 2504.13173) unifies Titans/GDN/Mamba/DeltaNet as one associative-memory
  family and confirms linear ≈ GDN is the long-context lever.
- **NEXT:** the training-path choice (cold from-scratch vs warm-start-from-Qwen) and serving the
  checkpoint in `minisgl-rdna4` (pure-torch memory forward first, validate numerics, then swap in
  kernels). Knobs that matter on this box: `use_accelerated_scan=False`, `flex_attn=False`; the
  pure-torch `AssocScan` store is memory-hungry (batch 8 OOMs 16 GB; batch 4 fits).

## Act XXI — gdn_hip: native HIP kernels that kill the GDN Triton compile cliff

The single most-cursed wall in this whole diary is the **15–30 minute cold Triton autotune of the
FLA-GDN linear-attention kernels** — it has been mistaken for a hang (Act II), forced the persistent
Triton-cache machinery (cold start), and livelocked the parallel-compile on TP=2 (Act XV). `gdn_hip`
(`feat/gdn-hip`) attacks it at the root: a **standalone, framework-agnostic `torch.ops` extension that
replaces the ~15 FLA Triton GDN kernels with AOT-compiled HIP** — compile once, run *any* shape, never
JIT/autotune a GDN Triton kernel again. It's shared by both `minisgl-rdna4` and `vllm-gfx1201` (both
route GDN through the same fla recurrence): build one `.so`, call `torch.ops.gdn_hip.*` from either.

- **Six ops** replace the decode SSM, `chunk_gated_delta_rule` prefill, the two `causal_conv1d`
  paths, and `RMSNormGated`. The math is lifted *verbatim* from `fla/ops/fused_recurrent.py` — the
  rank-1 update `S*=exp(g); v-=S@k; v*=β; S+=outer(v,k); o=S@q` — so correctness is by construction.
- **Status (2026-06-24):** builds AOT for gfx1201 (hipify-clean), all 6 ops load with valid
  fake/meta schemas, **numeric parity ≤ ~1e-7** vs torch reference, and — wired into `gdn/layer.py`'s
  recurrent path — it **serves 4B TP1/TP2 and 35B TP2 coherently** on the two cards. **The Triton GDN
  compile cliff is gone.** Gated behind `WITH_GDN_HIP` (build) + `VLLM_GDN_HIP=1` (runtime), default off.
- **The honest limit — and why this *is* the right framing.** The chunked-prefill op
  (`gdn_prefill_chunked`) is numerically correct but **~4× slower** as a scalar per-row kernel, so the
  serve uses the recurrent `gdn_prefill`. This is consistent with the earlier finding that GDN
  recurrent decode is occupancy-limited (~34% roofline, flat to tuning) and a custom kernel wins
  little on raw decode *throughput*. The win `gdn_hip` actually banks is different and real:
  **eliminating the cold-compile cliff** (AOT, shape-agnostic) — not faster steady-state decode. The
  remaining throughput lever is a **WMMA/matrix-core chunked prefill** (the intra-chunk KK/KQ/solve
  matmuls); that, plus deleting the now-unused Triton GDN tree and a bf16-native state, is the
  open work.
