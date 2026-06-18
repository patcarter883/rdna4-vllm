# RDNA4 serving engine from mini-SGLang — architecture & phased build

**Date:** 2026-06-17 · **Status:** design, pre-build · **Decision:** Path A (fork mini-SGLang →
native RDNA4 engine). Companion analysis: `docs/MINISGLANG_PATHWAYS.md`.

## Governing principles (locked by the user)

1. **Borrow, don't reinvent, the attention.** Lift the collaborator's tuned RDNA4 `triton_attn`
   (the 3D unified Triton kernel) from the baseline image; extend with GDN. Mine full SGLang +
   vLLM for the GDN design.
2. **Maximise RDNA4 hardware strengths** — native fp8/int4 WMMA, 3D flash-decode, `waves_per_eu`
   tuning, SWMMAC where it ever fits.
3. **Nothing dequants back to F16.** Precise meaning (so it isn't mis-applied): **I/O dtype is
   bf16, compute dtype is fp8 (e4m3fn), accumulation is f32.** Banned = per-element F16 *dequant of
   quantized weights/activations* (the RFP458 anti-pattern). fp32 accumulate and bf16 activations
   are REQUIRED, not violations. Weights int4→fp8→WMMA; KV fp8→(cast once)→bf16→f32-accum; GDN
   state bf16 with f32 scan accumulate.
4. **Clean, tidy, performant, agent+human maintainable.** mini-SGLang's style is the guide:
   small typed modules, Protocol-based backends, `__dict__`-introspection weight loading, no
   special-case spaghetti.

## Success criteria — what makes this worth building (be honest about the 35B)

The combined image **already serves this 35B GDN hybrid in production, on the SAME attention
kernel, SAME FLA-GDN kernels, SAME W4A8 ops** we are lifting. At the kernel level the new engine
**cannot beat it** — so do not chase throughput on the flagship. Two of mini-SGLang's headline
features also *don't apply* to the 35B: (a) AMD requires `no_buffer` scheduling for GDN → **the
overlap scheduler is OFF for the flagship**; (b) GDN state can't be prefix-rolled-back → **the
radix cache helps only the 1-in-4 full-attention layers**. So the bar is per-target:

- **Dense / standard-MoE (7B and friends):** all of mini-SGLang's features apply — overlap
  scheduler, radix prefix cache, clean W4A8. Here a throughput/maintainability win is real and
  unambiguous. **This is the strongest part of the engine.**
- **35B GDN flagship:** the bar is **(1) greedy token-identical to the combined image, (2) no
  latency regression, (3) materially cleaner het-TP + codebase.** NOT "beat throughput." "Parity
  or better" in Phase 4 means *don't regress* + cleaner, not *win on speed we structurally can't*.

## What we're building on (grounded facts from the 2026-06-17 audits)

### Attention — lifts almost wholesale (R1)
- Source in image (vLLM 0.22.69, extracted to `/home/pat/code/scratch/vllm_v1_attn/`):
  `ops/triton_unified_attention.py` (the unified prefill+decode kernel `kernel_unified_attention`
  + `reduce_segments`), `ops/triton_attention_helpers.py` (11 pure `@triton.jit` device fns),
  `ops/triton_reshape_and_cache_flash.py` (fp8 KV write), `backends/triton_attn.py` (metadata
  builder + tuning).
- **Pure-Triton, ~5 stubbable vLLM imports** (`vllm.triton_utils`→`triton`; `envs`→False;
  `logger`→stdlib; `current_platform.fp8_dtype()`→`torch.float8_e4m3fn`;
  `KVQuantMode`→`IntEnum`). The whole compute path lifts as 2 files behind `attention/base.py`.
- **fp8-KV precision (default path):** stored fp8 e4m3 → cast once to bf16 → `tl.dot` bf16×bf16,
  f32 accumulate → scalar k/v scales folded into `score_scale`/`acc` epilogue (exact, per-tensor).
  No F16. ✓ The opt-in fp8-Q WMMA path (`Q_IS_FP8`, gfx12x) is flagged lossy → **default off**.
- **Tuning is NOT `@triton.autotune`.** Launch knobs (`waves_per_eu`, `num_par_softmax_segments`,
  `tile_size_decode`, `num_warps`, `num_stages`) are resolved at startup by `ops/attn_autotune.py`
  (~12 candidates/bucket, sweeps `waves_per_eu∈{1..4}` × segment count) OR read from the offline
  `configs/device_name=AMD_Radeon_R9700,*.json` (gcn_arch gfx1201). **For a fixed-model engine, ship
  a static gfx1201 table** (low risk); the `waves_per_eu` + 3D flash-decode choice is where the win
  is. `block_size % 16 == 0`; decode `tile_size ∈ {16 bf16, ≥32 fp8}`.
  - **Reconsider static-table vs porting the autotuner.** R1 found the autotuner is self-contained
    (torch + the kernel, ~seconds startup) and is the *live* mechanism; the JSONs are for **R9700,
    not the 9070XT**, and aren't read at runtime. Porting `attn_autotune.profile_attn_tuning` may be
    both less curation AND more correct for our card than freezing an R9700 table. Lean autotuner.
  - **Attention matmul is bf16, not fp8.** The fp8-Q WMMA path is lossy → off. "Maximise fp8 WMMA"
    applies to the **W4A8 GEMMs**, not attention; nobody should expect fp8 attention compute.
  - **Vendoring = drift.** The collaborator keeps tuning `triton_attn` upstream; a snapshot won't
    track it. DECISION: track-upstream (periodic re-lift from the latest combined image) vs
    accept-snapshot. Recommend a thin re-lift script + a pinned source commit, so re-syncing is
    mechanical.
- `prepare_metadata` must emit: `cu_seqlens_q` (query_start_loc), `seq_lens`, `block_table`,
  `slot_mapping`, and allocate persistent f32 3D scratch (`segm_output/max/expsum`, sized to
  `max_segments`) with a `seq_threshold_3D` capacity gate. mini-SGLang's radix/page-table already
  produces block tables + seq lens.
- **Do NOT port:** rocm_attn HIP backend, `_aiter_ops` fused RoPE+KV, the TMA/tensor-descriptor
  path (Intel-only, dead on RDNA4).

### GDN — integration, not kernel porting (R2)
- The FLA chunked-scan + recurrent + causal-conv1d Triton kernels are **already AMD-aware and
  already run on gfx1201** (vLLM vendors FLA at `model_executor/layers/fla/ops/`; the combined image
  serves this exact 35B today). Vendor them as-is. CUDA-only paths (FlashInfer/CuteDSL/Blackwell,
  TMA, FLA cudagraph) are cleanly gated and **skipped**.
- **The new subsystem = a per-sequence recurrent state cache** (mini-SGLang has only MHAKVCache).
  Per GDN layer, per seq, two fixed tensors: `conv_state (conv_dim/tp, 3)` and
  `ssm_state (num_v_heads/tp, 128, 128)`, **bf16**, ONE slot per seq, addressed by
  `block_table[:,0]`. Constant memory vs context length. Fork vLLM's `MambaSpec` per-seq-slot model
  (allocator + free-list + prefill/decode gather-scatter by `state_indices`). Defer SGLang's
  `MambaRadixCache` (state can't be prefix-rolled-back).
- 35B GDN dims: 40 layers, 1-in-4 full-attention interleave (`full_attention_interval=4`),
  `linear_num_key_heads=16`, `linear_num_value_heads=32`, `head_k/v_dim=128`, conv kernel 4,
  hidden 2048. GDN projections are **unquantized bf16** (in the checkpoint not-convert list); state
  scan accumulates in **f32**. "No F16 dequant" trivially met — nothing on the GDN path is quantized.
- Kernels to vendor: `fla/ops/chunk.py` (+ `chunk_scaled_dot_kkt`, `solve_tril`, `wy_fast`,
  `chunk_delta_h`, `chunk_o`, `cumsum`, `l2norm`), `fla/ops/fused_recurrent.py`,
  `fla/ops/fused_sigmoid_gating.py`, `mamba/ops/causal_conv1d.py`, `fused_gdn_gating`.
- **Gotchas:** (1) FLA autotune fires on first real batch → needs a **warmup-prefill hook** or it
  OOMs (vLLM's `_warmup_prefill_kernels`). (2) A batch splits into prefill/decode subsets running
  *different* kernels (chunk-scan vs fused-recurrent) — scheduler must supply per-subset
  `query_start_loc` + state indices. (3) cudagraph is **decode-only** for GDN; prefill stays eager.
  (4) **Skip spec-decode/MTP for V1** — it's the dominant complexity source in vLLM's GDN code.

### W4A8 — small clean abstraction + a loader fix (R3)
- mini-SGLang has **zero** quant abstraction (pure bf16 `F.linear`). Integration shape:
  - **Dense:** add a `LinearMethod` Protocol on `_LinearTPImpl` — `apply(layer, x, bias)`; default
    `UnquantizedLinearMethod` = today's `F.linear`; `W4A8LinearMethod` = raw
    `torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm(x, w_q[N,K//8] i32, w_s[N,K//group] f16, zp, ver)`.
    Underscore-prefix `self._method` so `BaseOP` skips it in state_dict.
  - **MoE:** register a `"w4a8"` `BaseMoeBackend` in the existing `SUPPORTED_MOE_BACKENDS` registry
    (near-free) wrapping `mmq_fp8_moe_gemm` (op layout `(E,N,K//8)`); logic ported from
    `w4a8_fp8_wmma/moe_experts.py:_run_grouped_moe`. 35B shapes E=256/hidden=2048/moe_inter=512/
    top_k=8/g32.
  - **Reuse the RAW ops + the layout-conversion helpers** (`_ct_to_op_layout*`, AWQ repack), NOT
    the vLLM `MPLinearKernel`/`AWQMarlinMoEMethod` wrapper classes (vLLM-coupled).
- **No explicit activation-quant step to insert:** the op ingests bf16/f16 at the boundary and
  stages activations to fp8 e4m3 in-register. Routing the linear/MoE through the op IS the
  integration.
- **Weight-loader gaps (the real work):** (a) `engine.py:146` blanket `.to(self.dtype)` corrupts
  packed int32 weights — cast only non-packed tensors. (b) `models/weight.py` handles only
  `.weight`; must route `weight_packed`+`weight_scale`+`weight_zero_point` **as a unit** through
  shard (`_SPLIT_DIM_*`)/merge(`cat dim0`)/stack(`stack dim0`); AWQ needs the bit-order repack.
  (c) the strict `shape==&dtype==` assert in `base.py:45` means quant ctors must pre-declare packed
  buffer shapes/dtypes (needs `group_size`). Add `quantization`+`group_size` to ModelConfig/
  EngineConfig parsed from the checkpoint `quantization_config`.

## Repo & dependency layout (DECISION PENDING — see questions)

- **Recommended:** new sibling repo `/home/pat/code/minisgl-rdna4` (fork of mini-SGLang), which
  **consumes the W4A8 kernel as an installed dependency** from `vllm-gfx1201/w4a8_fp8_wmma/` (the
  single source of truth — NEVER vendor/copy it). Vendor the lifted Triton attention + FLA kernels
  (those ARE ours to carry). Keeps the engine clean and the kernel canonical.
- **Build story (concrete):** engine image `FROM vllm22-w4a8:combined`, then pip-install the engine
  + the W4A8 package so it links the **already-compiled** `_C.so` from that image — **never copy
  csrc**. "Consume as installed dep" = layer on the combined image (which already carries the built
  kernel + ROCm/Triton/torch toolchain + the `.triton-cache-combined` mount).

## Equivalence oracle — the combined image, NOT HF (wire up before Phase 2)

Because we lift the *same* kernels, the right oracle is **greedy token-identical output vs the
combined image**, which is far tighter than "bit-close vs HF" (HF can't reproduce W4A8/GDN
numerics). Reuse the het-TP equivalence harness pattern (`patches/run_het_e2e_combined.sh`): load
the model in both stacks, diff greedy token ids — they must match. Stand this up **before Phase 2**
so every W4A8/GDN validation afterwards is a tight token-diff, not an eyeball.

## Phased build

Each phase ends with a concrete, testable artifact. CPU/build work is unrestricted; GPU validation
needs a `rocm-smi` window (one card suffices through Phase 2; Phase 3 35B wants two).

- **Phase 0 — Fork + bf16 boot.** Create the repo, strip NVIDIA deps (replace flashinfer/sgl_kernel/
  tvm_ffi ops with torch + the lifted Triton), RCCL via `TorchDistributedImpl`. Boot **Qwen2.5-0.5B
  bf16, eager, TP=1**, greedy-equivalent vs HF. *Artifact: correct tokens from a dense model on one
  gfx1201.* (This subsumes the original A0 spike, but targets the tuned attention, not naive SDPA.)
- **Phase 1 — Tuned RDNA4 attention.** Drop `triton_unified_attention.py` +
  `triton_attention_helpers.py` + `reshape_and_cache_flash` behind `attention/base.py`; fp8-KV;
  static gfx1201 tuning table; f32 3D segm scratch in `prepare_metadata`. *Artifact: bit-close vs HF
  on a longer-context prompt; fp8 KV verified no-F16.*
- **Phase 2 — W4A8 dense + MoE.** `LinearMethod` protocol + `"w4a8"` MoE backend; the weight-loader
  fixes (sibling scale/zero routing, no blanket cast, packed-shape ctors); compressed-tensors + AWQ
  loaders. **First verify the on-disk format of each target** — the 35B is named `…AWQ-4bit` but the
  audits read it as compressed-tensors; the 7B is AWQ. Confirm `_ct_to_op_layout` vs
  `_awq_to_op_layout` cover both. Validate on **Qwen2.5-Coder-7B-AWQ** (dense W4A8) via the
  combined-image token-diff oracle. *Artifact: 7B W4A8 token-identical on RDNA4; also where the B2
  W4A8×prefix-cache composition becomes natively testable, with all of mini-SGLang's features active.*
- **★ GATE (end of Phase 2) — re-decide the 35B port.** Phases 0–2 are where mini-SGLang's
  features fully apply and the win is unambiguous; do them regardless. Phase 3 is where the 35B
  rationale narrows to maintainability+het-TP (per Success Criteria) AND ~80% of the risk lives.
  Decide here: port the 35B GDN, or leave the flagship on vLLM and let the clean engine own
  dense/standard-MoE. The engine may be most valuable for everything *except* the GDN flagship.
- **Phase 3 — GDN hybrid (the 35B). UNBUNDLED — the scheduler surgery is the sleeper risk, not the
  kernels (they already run):**
  - **3a — State-cache allocator.** MambaSpec-fork per-seq slot + free-list + gather/scatter by
    `state_indices`. Standalone unit test (alloc/free/reuse), no model. *Lowest risk, do first.*
  - **3b — One GDN layer's numerics.** Vendor FLA kernels; one layer forward token-diff vs the
    combined image on a fixed input. *Pins correctness before integration.*
  - **3c — Scheduler subset-split + warmup hook. ★ THIS is what blows the estimate.** One batch
    splits into prefill/decode subsets running *different* kernels (chunk-scan vs fused-recurrent)
    with per-subset `query_start_loc`+state indices — mini-SGLang's scheduler doesn't model this.
    Plus the FLA first-batch-autotune warmup-OOM hook. Budget the most time here.
  - **3d — Interleave + serve.** qwen3_5 1-in-4 full/linear interleave (full layers reuse Phase-1
    attention); serve the 35B; greedy token-diff vs combined image. No spec-decode/MTP in V1.
  *Artifact: the 35B GDN hybrid token-identical on RDNA4.*
- **Phase 4 — TP + het-TP + cudagraphs + parity.** RCCL TP; het-TP proportional sharding clean in
  `distributed/` — **re-derive the split ratio** (64:56 was fit to vLLM's MoE sharding + the 35B's
  specific bubble; new sharding ⇒ re-measure, don't assume). HIP-graph **decode** capture: target
  the **bf16 attention/GDN-decode path where it helps** — per our own findings the W4A8 GEMM path is
  graph-*invariant* (graphs can make us slower vs stock) and MoE-under-full-compile was
  "capturable but deterministically divergent," so **don't assume the W4A8/MoE path benefits, and
  budget for MoE-divergence resurfacing.** Perf bar per Success Criteria: don't-regress latency +
  cleaner het-TP, NOT beat-throughput on the flagship. *Artifact: no latency regression vs
  `vllm22-w4a8:combined`, het-TP first-class.*

## Explicit non-goals (V1)
rocm_attn HIP backend · FlashInfer/CuteDSL/Blackwell GDN · TMA/tensor-descriptor path · lossy fp8-Q
attention · spec-decode/MTP · MambaRadixCache (prefix-cache over recurrent state) · multi-node.

## Open decisions (for the user)
1. Repo location/name — `minisgl-rdna4` sibling consuming the W4A8 dep (recommended) vs other.
2. Start Phase 0 now (mostly CPU build until the validation run)?
3. Confirm the precision reading in principle #3 (bf16 I/O + fp8 compute + f32 accumulate).
