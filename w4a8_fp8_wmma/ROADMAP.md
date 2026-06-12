# W4A8-FP8-WMMA kernel — status & roadmap

Custom RDNA4 (gfx1201) kernel: expand packed int4 weights to FP8 (e4m3) in
registers and feed the FP8 WMMA units, instead of int4→f16→f16-WMMA. Goal: run
4-bit quantized models faster than vLLM's default Triton W4A16 path on the
RX 9070 XT (VRAM/compute-limited research context).

**Reference docs/repos/headers used: see `REFERENCES.md` (AMD RDNA4 WMMA guides,
ISA XML, transpose-load builtins, scaffold repo, vLLM internals, models).**

**▶ Continuing the MoE work? Read `TASK6_HANDOFF.md` first** — it's the cold-start
guide for the (written, pending HW-validation) MoE vLLM wiring: build/run steps,
the apply math, the AWQ-layout conversion, and a triage list of likely failures.

Package: `vllm/csrc/quantization/w4a8_fp8_wmma/`. Iterate kernel on bare-metal host
`blue` (`source activate-build-env.sh`, `GPU_ARCHS=gfx1201`,
`.venv-therock714`); validate vLLM integration in container
`kyuz0/vllm-therock-gfx1201` (torch 2.13 — rebuild the .so in-container, ABI).

## ✅ Achieved

- **Concept proven on hardware.** int4→fp8→FP8-WMMA validated; ISA confirms
  `v_wmma_f32_16x16x16_fp8_fp8` + `v_cvt_pk_fp8_f32`. int4 in [-8,7] is exact in
  e4m3 → weights lossless, only activations lose precision.
- **Kernel optimized 1 → 48 TFLOP/s** (4096³). Best kernel = **v5** (raw
  `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` + direct K-major `int2`
  b64 loads). Key wins: LDS **bank-conflict padding** (stride BK→BK+8; +80%),
  BM=256 staging amortization, in-register per-group scaling via probed accumulator
  layout (`col=lane&15, row=(lane>>4)*8+e`). FP8 WMMA peak is **370 TFLOP/s**
  (wmma_peak.hip) so there's headroom left.
- **Beats Triton in the compute-bound regime** — up to **1.28×** (large M); crosses
  over ~M=2560–3072. Crossover shifts with N (down) and K (up) — characterized.
- **vLLM integration**: `MPLinearKernel` adapter (`vllm_adapter.py`) + general-
  plugin registration (`register.py`, runs in the EngineCore subprocess). Handles
  GPTQ weight repack `(K//8,N)→(N,K//8)` and symmetric-zero (uint4b8) layers.
- **M-adaptive dispatch, AOT Profile & Cache** (no startup cost): `tune_crossover.py`
  (offline) → `crossover_cache.json` → O(1) `_crossover_for()` lookup. Unknown
  shapes default to Triton → **pathway is always ≥ Triton**, faster where we win.
- **End-to-end verified (GPTQ)**: `Qwen2.5-0.5B-Instruct-GPTQ-Int4` generates
  coherently through the integrated pathway.
- **AWQ support DONE + end-to-end verified (2026-06-12)** — see Task 1 below for
  the full write-up. `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` (asym uint4, zp, g128)
  routes AWQMarlinLinearMethod→choose_mp_linear_kernel→our kernel and generates
  coherently in BOTH dispatch regimes (default Triton fallback / forced v5).
- **Grouped MoE kernel (Task 2 kernel half) DONE + validated (2026-06-12)** —
  `moe_kernel.hip` (`mmq_fp8_moe_gemm` op, grouped-v0 golden + grouped-v5 WMMA),
  20/20 in `test_moe_correctness.py`.
- **MoE vLLM wiring (Task 6) WRITTEN + STATIC-AUDITED (2026-06-12) — pending
  in-container HW validation.** `moe_experts.py` adds
  `W4A8Fp8WmmaExperts(mk.FusedMoEExpertsModular)` + the AWQ-MoE→op-layout weight
  conversion + an oracle monkeypatch (`register_moe()`, wired into `register()`).
  `apply` composes the whole gated MoE from the one grouped op: moe_align →
  grouped GEMM(w13) → `silu_and_mul` → grouped GEMM(w2, identity-gather top_k=1) →
  fp32 topk scatter-reduce. New tests: `test_moe_experts.py` (conversion +
  composition vs fp8 ref) and `moe_model_test.py` (e2e). **Second-session audit
  (vs in-repo vLLM source): fixed a hard-crash bug** — `_patched_make`'s param was
  `experts_cls_arg` but `make_wna16_moe_kernel` is called with the keyword
  `experts_cls=` → `TypeError` at model load. All other runtime signatures + the
  `(M,K)`/`TopKWeightAndReduceNoOP` output contract verified against
  `MarlinExperts`. **AWQ→op-layout conversion proven bit-exact off-HW**
  (`test_moe_conversion_numpy.py`, numpy, max|diff|=0.0 over 5 shapes incl.
  g=16/64/128). **Still not run on gfx1201** — needs an in-container .so rebuild + a
  small cached AWQ MoE model; remaining unknowns are HW-only (kernel numerics, the
  full apply composition). See Task 6 below + `TASK6_HANDOFF.md`.
- Kernel versions kept: v0 scalar ref (golden), v2 (simple tiled), v5 (optimized);
  MoE: grouped-v0 + grouped-v5 in moe_kernel.hip. vLLM wiring: vllm_adapter.py
  (dense MPLinear) + moe_experts.py (modular-MoE experts + oracle patch), both
  installed by register.py. Tools: wmma_peak.hip, probe_layout.hip, probe_tr.hip,
  bench_vs_triton.py, sweep_crossover.py, tune_crossover.py. test_correctness.py
  (now incl. asym/AWQ cases), test_triton_awq_fallback.py, test_moe_correctness.py
  (single op), test_moe_experts.py (conversion + apply composition),
  container_test.py, model_test.py, awq_model_test.py, moe_model_test.py.

## 🎯 FINAL GOAL

**`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` runs faster on this kernel than the default
Triton path, end-to-end.** This model is **MoE** (A3B ≈ 3B active / 35B total) and
**AWQ-4bit, group_size 32** (the model the gfx1151 INT8 reference targeted). So the
goal requires three things we don't have yet: AWQ support, MoE grouped GEMM, and
fused ops. Success criterion: higher tokens/s (and/or lower latency) than the
stock Triton-served model on gfx1201, at matched output quality.

## 🚧 Remaining work (ordered toward the goal)

> **Sequencing (recommended):** finish **Task 1 (AWQ) on the dense
> `MPLinearKernel` path first**, validated end-to-end, BEFORE attempting Task 2
> (MoE grouped GEMM). MoE adds *both* asymmetric math *and* irregular memory
> routing — don't debug them together.

### 1. AWQ support (asymmetric uint4 + zero points) — ✅ DONE (2026-06-12)
**Result: AWQ works end-to-end with a ~3-line adapter fix.** Most of the work was
already in place; the investigation found the real gap was in the decode path.

- **Routing (verified, this checkout AND kyuz0 vllm 0.21.1):** on ROCm an AWQ-4bit
  `zero_point=True` checkpoint is converted to **`awq_marlin`** (`is_awq_marlin_
  compatible` passes because `query_marlin_supported_quant_types(has_zp=True)`
  returns `[uint4]` on ROCm — the `device_capability<75` guard is skipped for
  ROCm). `AWQMarlinLinearMethod.create_weights` then calls **`choose_mp_linear_
  kernel`**, which consults `_POSSIBLE_KERNELS[ROCM]` — where our plugin sits at
  pos 0. So **no extra hook needed**; our kernel is offered every AWQ dense layer.
  (MoE experts still go through `AWQMarlinMoEMethod`/fused_moe — that's Task 2.)
- **Layout (verified):** `AWQMarlinLinearMethod.process_weights_after_loading`
  runs `_convert_awq_to_standard_format` BEFORE delegating to our kernel. That
  hands us **GPTQ-standard** tensors: qweight `(K//8, N)` (std bit order) and
  qzeros `(N//8, K//group)` (N-packed, std nibble order). The qweight is exactly
  our existing AutoGPTQ branch; **the qzeros are already in our op's `w_zeros`
  convention** (`zp[an//8, g]`, nibble `an%8`). Zero-point convention `(q−zp)`
  also matches. So the **large-M FP8-WMMA path needed NO change** — the kernel
  already unpacks `w_zeros` correctly (all of v0/v1/v2/v4/v5 do).
- **The one real bug (fixed):** the small-M **Triton W4A16 fallback** (the decode
  path, M=1) was passing `qzeros=None` → it dropped the per-group zeros → wrong
  AWQ outputs during generation. Fix in `vllm_adapter.py`: build
  `layer._w4a8_tri_zp` = our `(N//8,K//group)` zeros **transposed** to
  `(K//group, N//8)` (same N-packing/nibble order, exactly what `triton_w4a16_
  gemm`'s `HAS_ZP` path wants) and pass it (with `zp_bias=0`) for asymmetric
  layers; symmetric uint4b8 keeps `None` (implicit `zp_bias=8`).
- **Validation (all on gfx1201 / RX 9070 XT, in docker):**
  - Kernel asym correctness: `test_correctness.py` asym cases — v0 (golden) and
    v5 are **bit-identical** to each other and match the fp8 reference (mean-tol
    made magnitude-relative since AWQ |q−zp|≤15 → ~2× larger outputs than sym).
  - Decode-path fix: `test_triton_awq_fallback.py` runs the EXACT triton_w4a16
    kernel with the transposed zeros → `(q−zp)·scale`, rel err ~2e-4.
  - End-to-end: `awq_model_test.py` + `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`
    (g128, zp) generates coherent text in BOTH the default (Triton fallback) and
    forced-v5 (`VLLM_ROCM_W4A8_FP8_WMMA_MIN_M=1`) regimes; log shows
    `Using RocmW4A8Fp8WmmaLinearKernel for AWQMarlinLinearMethod`.
- **Notes for later:** |q−zp| ≤ 15 stays **exact in e4m3** (weights lossless).
  The runtime int32 subtract `(nibble − zp)` is already how every kernel builds
  the fp8 byte; no VGPR-pressure problem observed (no need for the marlin-style
  fold-zp-at-repack). AWQ model shapes aren't in `crossover_cache.json` yet, so by
  default AWQ runs entirely on the Triton fallback (pathway ≥ Triton); re-tune
  with `tune_crossover.py` for the target model's shapes to engage v5 at large M.

### 2. MoE grouped GEMM (the big one — where we truly beat Triton)
**✅ Standalone grouped kernel DONE + hardware-validated (2026-06-12); vLLM wiring
WRITTEN (Task 6, below) — pending in-container HW validation.** `moe_kernel.hip`
adds a grouped GEMM op
`torch.ops.w4a8_fp8_wmma.mmq_fp8_moe_gemm(x, w_packed(E,N,K/8), scales(E,N,K/g),
w_zeros(E,N/8,K/g), sorted_token_ids, expert_ids, num_tokens_post_padded, top_k,
block_m, version)` matching the fused_moe contract: per-block expert pick, A-gather
by `sorted_token_ids//top_k`, padding-sentinel guard, per-expert weight slab.
grouped-v0 (scalar golden) + grouped-v5 (raw fp8 WMMA, bank-padded LDS, in-register
scale+AWQ zp; **tile M = block_m** 16/32/64 so each tile = one expert). Validated by
`test_moe_correctness.py`: 20/20 cases pass (v0 & v5, sym & asym/AWQ, incl. T=1
decode, top_k=4) vs an fp8 per-expert reference. **Task 6 (vLLM wiring) WRITTEN
(`moe_experts.py`); the routing/interface map below stands and is what the wiring
implements. Remaining = build the .so in-container and run the two MoE tests.**

**Integration mapped (2026-06-12).** Concrete routing for an
AWQ MoE model on ROCm gfx1201 (traced through this checkout's vllm):
- AWQ MoE experts route to **`AWQMarlinMoEMethod`** (`FusedMoEMethodBase`), NOT our
  MPLinear plugin — so our dense `_POSSIBLE_KERNELS` hook is never consulted here.
- It builds a **modular kernel** via an *oracle*:
  `select_wna16_moe_backend(config, kInt4Static)` in
  `fused_moe/oracle/int_wna16.py` picks a `WNA16MoEBackend` from a priority list
  (`_get_priority_backends()`; on ROCm = `[FLASHINFER_TRTLLM, MARLIN,
  BATCHED_MARLIN]`) by the first whose experts-class `is_supported_config` passes.
  On gfx1201 FLASHINFER_TRTLLM is unsupported → it lands on **`MarlinExperts`**
  (ROCm marlin MoE). `backend_to_kernel_cls(backend)` maps backend→experts class.
- Weights are **stacked per-expert**: `w13_qweight (E, hidden, 2*inter//pack)`,
  `w2_qweight (E, inter, hidden//pack)`, plus per-expert group scales + qzeros
  (`is_transposed=True`). `convert_to_wna16_moe_kernel_format` repacks them for the
  chosen backend (marlin layout today).
- The experts class is `mk.FusedMoEExpertsModular`; it implements
  **`workspace_shapes()` + `apply(...)`** and receives the routing tensors from
  `moe_align_block_size`: **`sorted_token_ids`, `expert_ids`,
  `num_tokens_post_padded`** (block→expert map + padded token order). This is
  exactly the "grouped/ragged GEMM" interface.

**Implemented (Task 6) — `moe_experts.py` + `register.py` hook.** Mirrors the dense
hook. What was built, mapped to the plan:
1. `W4A8Fp8WmmaExperts(mk.FusedMoEExpertsModular)` — `apply` COMPOSES the whole gated
   MoE from the one op: `moe_align_block_size(topk_ids, block_m, E,
   pad_sorted_ids=True, ignore_invalid_experts=True)` → `mmq_fp8_moe_gemm`(w13,
   top_k) → vLLM `apply_moe_activation` (silu_and_mul) → `mmq_fp8_moe_gemm`(w2,
   **identity-gather**: top_k=1, gather idx = padded row, sentinel for non-real
   rows) → **fp32 `index_add_` scatter-reduce** weighted by `topk_weights`.
   `finalize_weight_and_reduce_impl()=TopKWeightAndReduceNoOP` (apply does the
   weight+reduce). Surface: `activation_format()=Standard`, `moe_problem_size`
   (N=inter from `w2.size(2)*8`, K=hidden), `workspace_shapes` ((M,K) output;
   scratch is self-allocated), `_supports_*` gates (gfx12x; kInt4Static + asym
   variants; SILU/GELU gated). block_m = v5 tile M via `_choose_block_m` (16/32/64).
2. **Surgical** AWQ hook in `register_moe()` (called from `register()`), NOT a
   global oracle patch — the `int_wna16` oracle is shared by GPTQ (`auto_gptq`),
   AWQ (`awq_marlin`) AND compressed-tensors MoE, all passing `kInt4Static`, so a
   prepended-backend patch would hijack the other two and crash in their own
   un-patched `convert_*`. Instead: wrap `AWQMarlinMoEMethod.__init__` to override
   `wna16_moe_backend`(=sentinel) / `experts_cls`(=ours) ONLY when the config is a
   supported AWQ-asym-4bit MoE (`zero_point`, bits==4, g mult-of-16 ≤128); and patch
   `convert_to_wna16_moe_kernel_format` / `make_wna16_moe_kernel` in the `awq_marlin`
   namespace ONLY (dispatch on the sentinel / class). GPTQ + compressed-tensors stay
   on Marlin. The whole hook is wrapped in try/except so a vLLM-API mismatch can
   never break the (already-registered) dense path. Log signals:
   `[w4a8_fp8_wmma] AWQ MoE hook installed ...` + `[w4a8_fp8_wmma] AWQ MoE ->
   W4A8Fp8WmmaExperts (g=..)` (the stock `Using 'MARLIN' ...` line is the
   pre-override selection).
3. AWQ→our-layout conversion `_awq_moe_to_op_layout` (per-expert loop, memory-safe):
   AWQ unpack (`_REVERSE_AWQ_PACK_ORDER`) + transpose-repack → `(E,N,K//8)` qweight,
   `(E,N//8,K//group)` zeros, `(E,N,K//group)` fp16 scales — the dense-validated
   convention. Returned from the patched `convert_to_wna16_moe_kernel_format`; the
   stock `AWQMarlinMoEMethod.process_weights_after_loading` stores them and
   `get_fused_moe_quant_config` feeds them back as `w1_scale/w1_zp/...`.
4. **TODO — HW validation:** rebuild the .so in kyuz0, run `test_moe_experts.py`
   (conversion + apply-composition vs fp8 ref) then `moe_model_test.py` on a small
   cached AWQ MoE (`cyankiwi/Mellum2-12B-A2.5B-*-AWQ-INT4` ~6GB; 35B too big for
   16GB). Verify coherent text + log lines `[w4a8_fp8_wmma] MoE oracle patched...`
   and `Using 'W4A8_FP8_WMMA' WNA16 MoE backend.`
- **M-ragged edge** (carried from the kernel): `mmq_fp8_moe_gemm` already M/N
  tail-guards via `sorted_token_ids`/`num_tokens_post_padded`. The wiring preserves
  the per-expert block boundary by passing `pad_sorted_ids=True` (so the sorted_ids
  length is a multiple of block_m, which the op binding requires) and masking every
  consumer by `(sorted_ids < M*top_k) & (row < ntp)` to drop the uninitialised
  padding tail. Any future double-K vectorization must keep that invariant.

### 3. Fused operations (latency + bandwidth)
- **Fuse activation fp8 quantization** into the GEMM: today `compute_act_fp8` is a
  separate launch that writes `x_fp8` to global and reads it back. Either fuse the
  per-row max+quantize into the GEMM prologue, or emit fp8 directly from the
  preceding **RMSNorm+quant** (vLLM has fused rmsnorm→quant ops to mirror).
- **Fuse the epilogue**: bias, and for MoE the **SiLU·gate (silu_and_mul)** on the
  gate/up projection, plus expert-weight scaling, into the GEMM output.
- **Fuse expert scatter/gather** with the grouped GEMM (avoid materializing the
  permuted activation buffer).
- Goal: collapse norm→quant→gemm→act→gemm into as few launches/round-trips as
  possible.

### 4. Kernel perf: double-K 128-bit loads
- AMD RDNA4 WMMA guide's fp8 tip: fuse two WMMAs to use 128-bit (`b128`) loads
  (currently `b64`). Needs an interleaved LDS layout so each lane's 2-subtile data
  is contiguous. Likely pushes mid-M (M=512–2048) past Triton, where we currently
  trail (0.66–0.9×). This is what turns "≥ Triton" into "> Triton broadly".

## Verification plan for the goal
1. Get the AWQ MoE model loading on gfx1201 in the container (Triton baseline:
   tokens/s with our plugin OFF / VLLM_ROCM_USE_W4A8_FP8_WMMA=0).
2. Land AWQ + grouped-MoE + fused kernels; re-tune `crossover_cache.json` for the
   model's expert/dense shapes (g32).
3. Compare tokens/s and output quality (perplexity / sample prompts) vs the Triton
   baseline. Target: faster at matched quality.
