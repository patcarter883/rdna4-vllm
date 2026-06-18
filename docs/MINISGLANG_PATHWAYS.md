# mini-SGLang → RDNA4: two pathways

**Date:** 2026-06-17
**Question:** Compare (A) forking mini-SGLang and stripping all NVIDIA specifics to build a
native high-performance RDNA4 serving engine, vs (B) mining mini-SGLang's ideas to push our
existing vLLM-gfx1201 modifications further. Plan an implementation pathway for each.

---

## 0. What mini-SGLang actually is (grounding)

- ~5,000 lines, ~81% Python, type-annotated, *educational reference* implementation of SGLang.
- **NVIDIA CUDA-only**, benchmarked on H200. README: depends on "Linux-specific CUDA kernels."
- Modular seams: `attention/`, `distributed/`, `kvcache/` (radix), `scheduler/` (overlap),
  `moe/`, `engine/` (cudagraph runner), `layers/`, `server/`, `tokenizer/`, `kernel/`.

### The NVIDIA surface is concentrated and well-isolated
This is the single most important fact for the comparison. The CUDA dependency lives in a small
number of clearly-bounded places, and the *systems* code above it is hardware-agnostic Python.

| Layer | NVIDIA-bound? | mini-SGLang location |
|---|---|---|
| Attention kernels | **Yes** | `attention/fa.py` (FlashAttention), `fi.py` (FlashInfer), `trtllm.py` (TRT-LLM) |
| Quant / dense GEMM, fused norm/rope/sampling, grouped-MoE GEMM | **Yes** | `sgl_kernel` (CUTLASS), `flashinfer-python`, `quack-kernels` deps + `kernel/csrc`, `kernel/triton` |
| Collectives | **Yes (NCCL)** | `kernel/pynccl.py` |
| CUDA-graph capture | **Yes (HIP-equiv needed)** | `engine/` runner |
| Radix prefix cache | No — pure Python | `kvcache/`, `kernel/radix.py` |
| Overlap scheduler, chunked prefill, continuous batching | No — pure Python | `scheduler/` |
| TP sharding logic (column/row parallel) | No — pure Python | `distributed/impl.py`, `info.py` |
| Server / OpenAI API / tokenizer / message bus | No — pure Python | `server/`, `tokenizer/`, `message/` |

**Roughly:** ~2k lines of replaceable NVIDIA kernel surface, ~3k lines of portable serving
systems. Path A's whole feasibility rests on that 3k being genuinely hardware-agnostic.

### What we already own that maps onto the NVIDIA surface
Every NVIDIA-bound box above has a proven RDNA4 counterpart in *this* repo:

| mini-SGLang NVIDIA piece | Our RDNA4 replacement | Memory / source |
|---|---|---|
| FlashAttn/FlashInfer/TRT-LLM attention | tuned 3D unified **triton_attn** kernel + fp8-KV fixes | `rdna4-attention-fast-path-is-narrow` |
| `sgl_kernel` dense/quant GEMM | **W4A8 fp8 WMMA** kernel | `w4a8_fp8_wmma/`, `w4a8-single-source-of-truth` |
| `sgl_kernel` grouped-MoE GEMM | our **grouped W4A8 MoE** kernel (g32) | `w4a8-moe-kernel-does-not-execute...` (corrected: it runs) |
| flashinfer/quack norm·rope·sampling | Triton kernels (portable) | standard |
| `pynccl` (NCCL) | **RCCL** + decoupled-barrier custom allreduce | `allreduce-backend-fixed-pynccl-gfx1201` |
| CUDA-graph runner | HIP-graph capture (proven capturable) | `w4a8-cudagraph-capture-confirmed`, `w4a8-eager-only-under-full-compile` |
| cold autotune | 2-accelerator parallel compile | `cold-boot-parallel-compile` |

We are not starting from zero on any kernel. The gap is *integration*, not *invention*.

---

## PATH A — Fork → native RDNA4 engine (`minisgl-rdna4`)

**Thesis:** inherit mini-SGLang's clean ~3k lines of hardware-agnostic serving systems
(radix cache, overlap scheduler, chunked prefill, continuous batching, TP orchestration, server)
and swap its ~2k-line NVIDIA kernel surface for our proven RDNA4 stack. End state: a fully-owned,
~5k-line engine where our W4A8 kernels and het-TP are *first-class*, not patches on a 200k-line host.

### Phases
- **Phase 0 — ROCm boot spike (days, decisive).** Boot mini-SGLang on ONE gfx1201, smallest dense
  model, *slowest-correct* path: a naive Torch/Triton attention backend, `torch.matmul` GEMM (no
  quant), single-GPU no-op collectives. **Goal: prove the Python systems layer (radix + scheduler +
  engine + server) runs unmodified on ROCm.** If yes, everything else is kernel swap-in. This is the
  go/no-go gate — cheap, and it de-risks the entire bet.
- **Phase 1 — Attention backend.** Implement `attention/triton_rdna4.py` behind `base.py`'s
  interface using our tuned 3D unified triton_attn kernel + fp8-KV. Match mini-SGLang's page-table
  metadata contract.
- **Phase 2 — Dense W4A8.** Wire our fp8 WMMA kernel into `layers/` linear. Carry the load-time
  unpack-OOM chunking fix (`w4a8-dense-kernel-oom-at-load-on-27b`).
- **Phase 3 — MoE W4A8.** Grouped g32 kernel + the routed-expert model def. (35B quantizes only
  routed experts — `35b-only-quantizes-moe-experts`.)
- **Phase 4 — TP.** RCCL collectives + decoupled-barrier custom allreduce; then implement het-TP
  proportional sharding cleanly in the small `distributed/` layer (where it *belongs*, vs our vLLM
  layer.py patch).
- **Phase 5 — Capture + tune.** HIP-graph capture, 2-accelerator cold-boot parallel autotune,
  perf parity A/B vs `vllm22-w4a8:combined`.

### The killer risk: model coverage (the 35B GDN hybrid)
mini-SGLang is educational — small model zoo. Our flagship 35B is a **GDN/SSD hybrid** whose
FLA-GDN + attention autotune is the 15–30 min cold-compile beast the whole cache protocol exists
to amortize. **mini-SGLang almost certainly does not model GDN/SSD at all.** Porting a
Mamba/GDN-hybrid into a minimal engine is a major effort and is the single biggest Path-A blocker.
Path A is *much* cheaper if scoped to dense + standard-MoE transformers and the 35B stays on vLLM.

### Other costs
- Re-solving for free things vLLM gives us: compressed-tensors loading, HF integration, sampling
  params, full OpenAI API surface, LoRA, broad model coverage.
- **Permanent fork maintenance** of a fast-moving research repo.

### Upside
- A clean, ~5k-line, *fully-understood* RDNA4-native engine. Iteration velocity on kernels/het-TP
  potentially far higher than fighting vLLM internals. Excellent research demonstrator and a place
  to prototype ideas that are painful to isolate in vLLM.

---

## PATH B — Mine mini-SGLang to push vLLM-gfx1201 forward

**Thesis:** keep vLLM (model coverage, quant, the GDN hybrid, ecosystem) and import mini-SGLang's
*ideas and clean reference implementations* as a teaching aid to accelerate specific vLLM mods.

### Concrete imports
1. **Het-TP is ALREADY DONE in vLLM — do not re-prove it.** The real 0.22.69 patch is landed
   (`patches/het_tp_vllm.patch`, 13 hunks across `linear.py`/`parameter.py`/`fused_moe/layer.py`),
   greedy-equivalence is tested (het `64,56` ≡ even, bit-identical token ids via
   `run_het_e2e_combined.sh`), and the COMM-bubble profiling recipe exists
   (`HET_TP_REVALIDATE.md`). The only residual question mini-SGLang could touch is *generality*
   — is the 64:56 win a gfx1201/PCIe artifact or portable to other interconnects? That's a
   "nice to know," NOT a driver. (The older `HET_TP_HANDOFF.md` "stranded/placeholder" language
   is stale — it predates the patch landing.)
2. **W4A8 × radix/APC composition (the one genuinely new perf pathway).** vLLM already has prefix
   caching; mini-SGLang's `kvcache/` + `kernel/radix.py` are the clean reference for the
   match/evict logic. **Verify our W4A8 served path composes with APC on repeated-prefix
   workloads** — an axis we've never tested (`w4a8-served-parity-at-low-concurrency` only covered
   steady-state low-concurrency). Hardware-portable, no kernel changes, attacks a *different* axis
   than anything we've measured → plausibly additive.
3. **Overlap scheduler.** mini-SGLang's CPU/GPU overlap loop is a clean reference for spotting
   where vLLM's scheduler stalls; informs CPU-overhead reduction on our served path *if* profiling
   shows CPU stalls.
4. **Attention-backend abstraction.** `base.py` is a clean model for how a backend interface
   *should* look — informs how we register the triton_attn / fp8-KV fixes in vLLM.
5. **pynccl cross-check.** Both projects ship a pynccl; theirs is a clean reference to validate our
   decoupled-barrier custom-allreduce design against.

### Phases
- **Phase 0 — Read & map (days).** Produce an internal "mini-SGLang → vLLM idea map." Cheap.
- **Phase 1 — Het-TP clean proof** in mini-SGLang (isolated, measurable) → confidence into the
  vLLM patch.
- **Phase 2 — W4A8 × APC composition test** in vLLM. The concrete new win.
- **Phase 3 — Overlap-inspired CPU-overhead reduction** in the vLLM served path (profiling-gated).

### Cost / risk
Low. No fork, no maintenance burden, keep all vLLM coverage. Upside bounded by vLLM's architecture
— we keep fighting its internals for deep kernel/graph work.

---

## Recommendation

**Default to Path B; treat Path A as an option you buy into, not a starting fork.**

- **Path B is high-ROI / low-risk and should start now.** The het-TP clean proof and the
  W4A8 × radix/APC composition test are concrete, near-term wins; the latter is the only pathway
  here that is both hardware-portable *and* attacks an axis we've left untested.
- **Path A is a bigger bet, justified only if** one of: (a) we want a research/demonstrator vehicle
  we fully control; (b) vLLM internals become a real velocity bottleneck for kernel/het-TP work; or
  (c) we accept dropping the 35B GDN-hybrid from this engine (the single biggest Path-A blocker).

### Sequence
Het-TP is already shipped (correctness-proven), so it is **not** part of the remaining work — it
drops out of Path B entirely. That leaves **B2 (W4A8 × radix/APC composition)** as the one genuinely
new, hardware-portable win, and Path A's case is *weaker* than first framed (one of its supposed
clean-room prizes — isolating het-TP — is moot). So:

1. Do **B2 now**: audit how vLLM's automatic prefix caching interacts with the W4A8 served path and
   design the repeated-prefix test. Portable, no kernels, untested axis.
2. Run **A0** (the ROCm boot spike) only as an *option probe* — does mini-SGLang's Python systems
   layer (radix + overlap scheduler) run clean on gfx1201 with a naive backend? Its value is the
   radix/overlap *machinery*, not het-TP.
3. **Decide the fork only then**, with two facts in hand: (i) the systems layer runs on ROCm, and
   (ii) whether we're willing to port the 35B GDN hybrid (the biggest Path-A blocker). Until both
   hold, keep shipping via vLLM.

---

## A0 result — ROCm boot-spike scoping (CPU-only, DONE 2026-06-17)

Cloned to `/home/pat/code/mini-sglang` (8.1k py lines). Full static audit of the NVIDIA surface.

**Verdict: feasible, low-risk to start.** Every NVIDIA-bound import (`flashinfer`, `sgl_kernel`,
`trtllm`, `tvm_ffi`/`.cu`, `pynccl`) is **lazy** — inside function bodies or behind the
attention/MoE backend registries. `import minisgl` and the entire systems layer (radix cache,
overlap scheduler, engine loop, server) load with **no CUDA kernel present**. There are **zero**
top-level NVIDIA imports that would break import on ROCm.

- **Default-backend gotcha:** `utils/arch.py` → on ROCm `torch.version.cuda is None` ⇒ `is_sm90/100`
  both False ⇒ `auto` attention silently resolves to `fi` (flashinfer). Must register a new backend
  AND force-select it (extend `engine.py` `_adjust_config`).
- **Attention interface is tiny:** `attention/base.py` needs only `forward()` + `prepare_metadata()`
  for first-token (the 3 cudagraph methods are skipped when `--cuda-graph-max-bs 0`). `page_size=1`
  makes the page table a flat slot map → a naive **gather-then-`F.scaled_dot_product_attention`**
  backend is ~80–120 lines. `prepare_metadata` lifts almost verbatim from `attention/fa.py:67-105`.
- **Everything else = torch one-liners:** RMSNorm/RoPE/silu·gelu (flashinfer, lazy at layer
  construction), greedy sampling already bypasses flashinfer (`engine/sample.py:73-74`), KV-store
  `.cu` → `k_cache[out_loc]=k`, vocab gather → `weight[idx]`, radix `fast_compare_key` → torch compare.
- **Collectives:** default is `TorchDistributedImpl` = **RCCL on ROCm out-of-the-box**; TP=1 never
  touches `pynccl.cu`; TP>1 has a built-in `--disable-pynccl` escape. Zero work for the spike.
- **cudagraph:** standard `torch.cuda.graph` API → HIP graphs transparently (consistent with our
  prior W4A8 capture finding).
- **Effort:** dense model, greedy, eager, TP=1 = **~4–6 days, dominated by the one attention backend.**
- **35B GDN hybrid OUT OF SCOPE:** model zoo is qwen2/qwen3/qwen3_moe/mistral/llama only; KV cache is
  strictly `MHAKVCache`. No SSM/GDN state machinery. Porting the 35B would mean new model defs + a
  recurrent-state cache type + FLA-GDN kernels — a separate, much larger effort. **This is the
  decisive Path-A constraint: a mini-SGLang fork serves dense/standard-MoE, NOT our flagship.**

## B2 result — W4A8 × prefix-cache audit (CPU-only, DONE 2026-06-17)

Audited vLLM 0.22.69 source (extracted read-only from `vllm22-w4a8:combined`). **An earlier
hypothesis was corrected.**

- **CORRECTION: the 35B GDN-hybrid DOES support prefix caching** — via vLLM's GDN **"align" mode**,
  which reuses **both** attention KV blocks (`single_type_kv_cache_manager.py:879-897`) **and** GDN
  recurrent state (10-entry side cache keyed by blake2b over prefix tokens, `gdn_side_cache_manager.py`).
  Earlier "hybrids can't prefix-cache" was wrong.
- **But it's default-OFF / opt-in:** `ModelConfig.is_prefix_caching_supported` returns False for
  `attn_type=="hybrid"` (`config/model.py:1828-1884`), so APC is off unless `--enable-prefix-caching`
  is passed (auto-selects align; `--mamba-cache-mode=all` is explicitly NotImplemented for Qwen3.5,
  `qwen3_5.py:459-463`). Reachable only on the V1 runner (not `VLLM_USE_V2_MODEL_RUNNER=1`).
- **W4A8 is orthogonal — confirmed at source:** the adapter has zero references to prefix caching /
  cache_config / KV. AND on the 35B, W4A8 quantizes **only MoE experts** while APC caches
  attention+GDN → **they never co-act.** So a 35B run is an *additive* TTFT measurement
  ("APC win alongside W4A8"), **not** a W4A8×APC interaction test.
- **The true composition test = a non-hybrid AWQ model:** `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`
  (`single` profile, TP=1) — pure decoder, APC **default-on**, and W4A8 quantizes the *attention*
  linears that produce the cached KV. Only here do W4A8 numerics and KV-block reuse genuinely interact.
- **Runnable experiment (needs a GPU window):**
  - *Composition (1 card):* `single` profile, 7B-AWQ, 2×2 {W4A8 on/off}×{APC on/off}, shared long
    system prompt + varied suffixes, **metric = TTFT** (+ cache-hit rate, throughput). Confirms W4A8
    KV stays coherent under reuse. **Fits GPU[1] now.**
  - *Additive (2 cards):* `serve` profile, 35B, add `--enable-prefix-caching` (align), shared- vs
    unique-prefix TTFT A/B at W4A8=1. Needs a 2-card window (TP=2 het 64:56).
  - Needs a **served HTTP TTFT driver** (streaming first-token timestamp) — `bench_tp2.py` is offline,
    don't repurpose it. Mount `.triton-cache-combined`, HF cache, override HIP+ROCR visible devices,
    use the compose profile (add the flag to its `command`).

**Net:** B2 is a real, hardware-portable win, but its framing must be exact — "APC additive on the
35B" vs "W4A8×APC compose on the 7B." The 7B composition cell is the one that actually validates
the headline and **fits a single free card now.**
