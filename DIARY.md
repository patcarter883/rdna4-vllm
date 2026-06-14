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
next levers to turn those fallbacks into wins.

And we now know where to *stop*: a per-op decode profile puts full attention at ~1.8% and the
dominant buckets (MoE ~41%, dense fp16 ~20%) at ones we already understand — so the
int4→fp8-WMMA kernel chapter closes on evidence. The ZAYA1 hybrid-CCA port is the open front,
waiting on a gfx1201 bring-up window.

And the two halves are now **one stack**: development has consolidated onto the **combined image**
(the collaborator's tuned attention + our W4A8, built from canonical sources) — measured
**+31% prefill** and **fp8-KV made free** on top of the kernel work — with heterogeneous-TP
(64:56 proportional sharding) landed but gated dormant, a 2-GPU validation the next step. See
Act IX.
