# Task 6 (MoE vLLM wiring) — continuation handoff

Read this first if you're picking up cold. Deeper context: `ROADMAP.md` (full
status), `REFERENCES.md` (external docs), and the project memory
`w4a8-fp8-wmma-kernel.md`. This doc is the "what's done / what's next / how to not
re-derive everything" for the MoE wiring specifically.

## TL;DR state (2026-06-12)

- **Task 6 = wire the validated grouped MoE op into vLLM so an AWQ-4bit MoE model
  runs its experts on our FP8-WMMA kernel.** The op itself
  (`mmq_fp8_moe_gemm`, in `moe_kernel.hip`) is DONE + HW-validated (20/20 in
  `test_moe_correctness.py`).
- **The Python wiring is WRITTEN but NOT yet run on gfx1201 hardware.** It compiles
  (py_compile) but this dev host has **no GPU and no usable torch** (torch is
  installed but fails to import: missing `libmpi_cxx.so.40`). End-to-end
  validation happens in the container (see below).

### Static-audit pass (2026-06-12, second session) — done, no HW

A full static audit of `moe_experts.py` against the in-repo vLLM source was done.
Outcome:
- **FIXED one hard-crash bug** in `register_moe._patched_make`: its 3rd param was
  named `experts_cls_arg`, but `AWQMarlinMoEMethod._setup_kernel` calls
  `make_wna16_moe_kernel(..., experts_cls=...)` **by keyword** → `TypeError` at
  model-load (and it's OUTSIDE `register.py`'s try/except, so it crashes the run,
  not just the MoE hook). Renamed the param to `experts_cls`; the closure's own
  class is now `our_experts_cls` (avoids shadowing). This would have cost one
  in-container rebuild/debug cycle on the first HW run.
- **Verified all other runtime call signatures line up** (these were the main
  risk): `AWQMarlinMoEMethod.__init__(self, quant_config, moe)` (constructed as
  `AWQMarlinMoEMethod(self, layer.moe_config)`); `convert_to_wna16_moe_kernel_format`
  (called all-keyword, 14-tuple return — our `_patched_convert` matches);
  `make_wna16_moe_kernel`/`maybe_make_prepare_finalize` keywords;
  `moe_align_block_size(..., pad_sorted_ids=, ignore_invalid_experts=)`;
  the `mk.FusedMoEExpertsModular.apply` 15-arg surface; the op wrapper
  `mmq_fp8_moe_gemm`. `convert`/`make` are imported into `awq_marlin` from
  `fused_moe.oracle.int_wna16`, so patching the `awq_marlin` namespace is correct
  and surgical (GPTQ/auto_gptq + compressed-tensors keep their own copies).
- **Verified the output contract**: `MarlinExperts` (the backend we replace) uses
  the identical `workspace_shapes` output `(M,K)` + `finalize_weight_and_reduce_impl
  → TopKWeightAndReduceNoOP()`, and `TopKWeightAndReduceNoOP.apply` asserts
  `output.size()==fused_out.size()` then copies — so our already-reduced `(M,K)`
  `apply` output is exactly what `_finalize` expects. No chunking: `_fused_experts`
  calls `apply` once on the full batch; `hidden_states=a1q` is the full unquantized
  activation with `a1q_scale=None` (int4 w4a16 prepare doesn't quantize).
- **Validated the AWQ→op-layout conversion bit-exactly, torchless** (failure-point
  #3, the most bug-prone piece). `test_moe_conversion_numpy.py` re-implements
  `_awq_moe_to_op_layout`'s integer ops in numpy and checks op-layout dequant ==
  AWQ-source dequant over 5 shapes (incl. g=16/64/128): **max|diff| = 0.0**. So the
  AWQ bit order, transposes, K-pack (weights) / N-pack (zeros), scale transpose and
  `(q−zp)·scale` sign are all correct. The composition math (`apply` mirror) still
  needs the GPU op → `test_moe_experts.py` in-container.

Net: the wiring is API-correct and the weight conversion is proven. Remaining
unknowns are HW-only: kernel numerics at these shapes, dtype edge cases, and the
full `apply` composition. Start in-container straight at `test_moe_experts.py`.
- Everything new lives under `csrc/quantization/w4a8_fp8_wmma/`. No core vLLM files
  were edited (the wiring is runtime monkeypatch only). The two modified core files
  you'll see in `git diff` (`vllm/_aiter_ops.py`, `vllm/platforms/rocm.py`) are
  PRE-EXISTING gfx1201-enablement changes — leave them alone; our code depends on
  `rocm.on_gfx12x()`.

## What "done" looks like (acceptance)

In `kyuz0/vllm-therock-gfx1201`, after rebuilding the `.so`:
1. `python test_moe_experts.py` → ALL PASSED (conversion + apply composition).
2. `python moe_model_test.py <small AWQ MoE>` → coherent text, log shows
   `[w4a8_fp8_wmma] AWQ MoE -> W4A8Fp8WmmaExperts (g=..)`.
Then Task 6 is complete; update ROADMAP/memory to "HW-validated".

## The files I wrote/changed (this session)

- `w4a8_fp8_wmma/moe_experts.py` — **the whole wiring** (experts class + AWQ→op
  conversion + the surgical hook `register_moe()`). ~440 lines, fully commented.
- `w4a8_fp8_wmma/register.py` — now also calls `register_moe()` (wrapped in
  try/except so a failure can't break the dense path).
- `test_moe_experts.py` — NEW. Unit-tests the NEW python code: (1) conversion
  dequant-equality vs AWQ source, (2) full apply composition vs an fp8 reference,
  for v0 & v5. This is the test that catches MY bugs (the op is already validated).
- `moe_model_test.py` — NEW. End-to-end model generation.
- `ROADMAP.md`, memory `w4a8-fp8-wmma-kernel.md`, `MEMORY.md` — status updated.

## How the wiring works (so you don't re-trace vLLM)

An AWQ-4bit MoE model on ROCm routes its expert layers through
`AWQMarlinMoEMethod` (in `vllm/.../quantization/awq_marlin.py`), which:
- `__init__`: `select_wna16_moe_backend(moe, kInt4Static)` → picks
  `(wna16_moe_backend, experts_cls)`; on gfx1201 normally lands on `MarlinExperts`.
- `process_weights_after_loading`: `convert_to_wna16_moe_kernel_format(backend, ...)`
  → repacks weights, then `_setup_kernel` → `make_wna16_moe_kernel(experts_cls=...)`
  builds a modular `FusedMoEKernel`.
- `apply`: `self.moe_kernel.apply(hidden_states=x, w1=layer.w13_qweight,
  w2=layer.w2_qweight, topk_weights, topk_ids, activation, global_num_experts,
  expert_map, apply_router_weight_on_input, ...)`.

The modular kernel eventually calls **`experts.apply(output, hidden_states, w1, w2,
topk_weights, topk_ids, activation, global_num_experts, expert_map, a1q_scale,
a2_scale, workspace13, workspace2, expert_tokens_meta,
apply_router_weight_on_input)`** with:
- `hidden_states` = the FULL `(M,K)` unquantized activation (Standard format; for
  int4-w4a16 the prepare step does NOT quantize → bf16/fp16, `a1q_scale=None`).
- `output` = a `(M,K)` buffer (our `workspace_shapes` output shape) we must FILL.
- With `finalize_weight_and_reduce_impl()=TopKWeightAndReduceNoOP`, **our apply must
  do the topk-weight multiply + reduce itself** and write the final `(M,K)`.

**Our surgical hook (in `register_moe`):** do NOT patch the shared `int_wna16`
oracle — it's used by GPTQ (`auto_gptq`), AWQ AND compressed-tensors MoE, all
passing `kInt4Static`; a global prepend hijacks the other two and crashes in their
un-patched `convert_*`. Instead:
- Wrap `AWQMarlinMoEMethod.__init__`: after the original runs, if the config is a
  supported AWQ-asym-4bit MoE (`zero_point`, `weight_bits==4`, group_size
  mult-of-16 in [16,128]) and we're on gfx12x, override `self.wna16_moe_backend =
  W4A8_FP8_WMMA_BACKEND` (a sentinel) and `self.experts_cls = W4A8Fp8WmmaExperts`.
- Patch `convert_to_wna16_moe_kernel_format` and `make_wna16_moe_kernel` in the
  **`awq_marlin` namespace ONLY** (they imported them by name), dispatching on the
  sentinel / our class. GPTQ + compressed-tensors are untouched → stay on Marlin.

## The MoE math our apply does (the part most likely to have a subtle bug)

`mmq_fp8_moe_gemm(x(T,K) fp16, w_packed(E,N,K/8), scales(E,N,K/g) fp16, w_zeros
(E,N/8,K/g), sorted_token_ids, expert_ids, num_tokens_post_padded, top_k, block_m,
version, w_zeros=)` computes, per padded-sorted row r:
`out[r] = (x[sorted_token_ids[r]//top_k] @ W[expert_ids[r//block_m]]^T)` with
internal per-token fp8 activation quant. Output `(P,N)` in padded-sorted layout.
Padding rows (`sorted_token_ids[r] >= T*top_k`, or block `>= ntp`) are skipped.

apply (gated SiLU MoE), all kept in padded-sorted layout:
1. `moe_align_block_size(topk_ids, block_m, E, expert_map, pad_sorted_ids=True,
   ignore_invalid_experts=True)` → `sorted_ids, expert_ids, ntp`. `block_m` from
   `_choose_block_m` ∈ {16,32,64} (our v5 tile M).
2. gemm1 (w13): `out1 = mmq(x16, w1, w1_s, sorted_ids, expert_ids, ntp, top_k,
   block_m, w_zeros=w1_zp)` → `(P, 2*inter)`.
3. activation: `apply_moe_activation(SILU, buf2, out1)` → `buf2 (P, inter)` =
   `silu(gate)*up` (gate = first inter cols, up = next inter cols).
4. gemm2 (w2) **identity-gather**: `ident[r] = r` if real else sentinel `P`;
   `out2 = mmq(buf2, w2, w2_s, ident, expert_ids, ntp, top_k=1, block_m,
   w_zeros=w2_zp)` → `(P, K)`. (top_k=1 + ident=row makes the op gather buf2[r].)
5. scatter-reduce: `valid = (sorted_ids < M*top_k) & (row_idx < ntp)`;
   `tokens = sorted_ids[valid]//top_k`; fp32 `acc.index_add_(0, tokens,
   out2[valid]*topk_weights.view(-1)[sorted_ids[valid]])`; `output.copy_(acc)`.

### Non-obvious correctness facts (DON'T regress these)
- **`pad_sorted_ids=True` is mandatory**: real vLLM `moe_align` over-allocates
  `sorted_ids` to `M*top_k + E*(block_m-1)` (NOT a multiple of block_m) with an
  **uninitialised tail past `ntp`**. Our op binding asserts `P % block_m == 0` and
  `expert_ids.size(0) == P/block_m`. `pad_sorted_ids=True` rounds P up to a multiple.
- **Mask every consumer by `(sorted_ids < M*top_k) & (row_idx < ntp)`** — the
  `row_idx < ntp` half drops that uninitialised tail (its garbage `sorted_ids` could
  be `< M*top_k`). `ntp` is a `(1,)` device tensor → the comparison broadcasts, no
  host sync.
- **Use the Python wrapper `w4a8_fp8_wmma.mmq_fp8_moe_gemm`** (kwargs `version=`,
  `w_zeros=`), NOT `torch.ops...` directly — the raw op has `w_zeros` as the 4th
  positional arg; the wrapper reorders it.
- **w13 output cols are `[gate(inter); up(inter)]`** so `silu_and_mul` is correct.
- **Activations bf16→fp16** for the op (lossless), output back to bf16.
- **fp32 reduce** in the scatter (Marlin uses `use_fp32_reduce=True`).

## AWQ → our op layout conversion (`_awq_moe_to_op_layout`)

AWQ MoE weights as registered by `create_weights` (RAW, NOT pre-converted — unlike
the dense path there's no `_convert_awq_to_standard_format` before us):
- `w13_qweight (E, hidden, 2*inter//8)` int32, packed-along-OUTPUT, AWQ bit order
  (`_REVERSE_AWQ_PACK_ORDER=[0,4,1,5,2,6,3,7]`); `w2_qweight (E, inter, hidden//8)`.
- `w13_scales (E, hidden//g, 2*inter)`; `w2_scales (E, inter//g, hidden)`.
- `w13_qzeros (E, hidden//g, 2*inter//8)`; `w2_qzeros (E, inter//g, hidden//8)`.

Our op wants `(E, N, K//8)` qweight (packed-along-K, std nibble order),
`(E, N, K//g)` fp16 scales, `(E, N//8, K//g)` zeros. The conversion (per-expert
loop for memory safety): AWQ-unpack + `[..., rev]` → transpose → repack along K
(weights) / along N (zeros), transpose scales. This is the SAME convention the
dense AWQ path was validated against — `test_moe_experts.py` checks the converted
weights dequant-equal the AWQ source.

## Build + run (in the container — the ONLY place this can run)

ABI: the `.so` is bound to the exact torch it's built against. ALWAYS rebuild
in-container. Per prior sessions:
```
rm -rf w4a8_fp8_wmma/_C*.so build              # kill stale-ABI .so from host mount
# build in-container against ITS torch (kyuz0 = torch 2.13, vllm 0.21.1):
pip install /tmp/<pkg> --no-build-isolation --no-deps
# run from a dir that does NOT contain the source w4a8_fp8_wmma/ pkg, or the
# source copy (no fresh _C.so) shadows site-packages (sys.path[0] gotcha).
python test_moe_experts.py
python moe_model_test.py <model_id>
```
Env flags: `VLLM_ROCM_W4A8_FP8_WMMA_MOE=0` (disable just MoE → Marlin baseline),
`VLLM_ROCM_W4A8_FP8_WMMA_MOE_VERSION=0` (scalar golden grouped kernel, for
debugging v5), `VLLM_ROCM_USE_W4A8_FP8_WMMA=0` (master off).

Small AWQ MoE for validation (35B final-goal model ~18GB too big for 16GB):
`cyankiwi/Mellum2-12B-A2.5B-*-AWQ-INT4` (~6GB) or any cached small AWQ MoE
(`Qwen/Qwen1.5-MoE-A2.7B-Chat-AWQ` if available). Confirm the exact cached id.

## Likely failure points + how to debug (I can't pre-test, so triage in order)

1. **vLLM API mismatch** (container's vllm vs what was read). All signatures were
   re-verified against the in-repo vLLM source this session (see the static-audit
   note up top) and the one real mismatch — `_patched_make`'s param name — is fixed.
   Two failure *modes* remain possible if the CONTAINER's vllm differs from the
   in-repo source:
   - **Hook installs but override doesn't engage** (try/except in `register.py`
     swallows an `__init__`/`convert`/`make` signature change → MoE silently stays
     on Marlin). Symptom: no `AWQ MoE -> W4A8Fp8WmmaExperts` log. The interfaces
     relied on: `AWQMarlinMoEMethod.__init__(self, quant_config, moe)`,
     `convert_to_wna16_moe_kernel_format(backend, layer, quant_config, input_dtype,
     w13, w2, w13_g_idx, w2_g_idx, w13_qzeros, w2_qzeros, w13_bias, w2_bias)`
     (14-tuple return), `make_wna16_moe_kernel(moe_quant_config, moe_config,
     experts_cls, is_k_full, w13_g_idx, w2_g_idx, w13_g_idx_sort_indices,
     w2_g_idx_sort_indices, routing_tables)` — **note `experts_cls` is passed by
     keyword**, so our patched param MUST keep that exact name,
     `moe_align_block_size(..., pad_sorted_ids=, ignore_invalid_experts=)`,
     `mk.FusedMoEExpertsModular` 15-arg `apply`. If a signature differs, adapt in
     `moe_experts.py` only.
   - **Override engages, then crashes at load** (a signature our PATCH calls/returns
     no longer matches — this path is NOT in the try/except). Read the traceback;
     most likely `convert`'s return-tuple arity/order or a `make`/`maybe_make_
     prepare_finalize` kwarg.
2. **Binding TORCH_CHECK fires** (shape/dtype): read the exact check in
   `bindings.cpp` `mmq_fp8_moe_gemm_forward`. Most likely `P % block_m`, scales
   fp16, x fp16/contiguous, or `expert_ids.size(0)`.
3. **Wrong output (incoherent text) but no crash**: run `test_moe_experts.py`
   first. NOTE the conversion half is already proven bit-exact off-HW
   (`test_moe_conversion_numpy.py`, numpy) — so if `test_moe_experts.py`'s
   conversion test still fails in-container, suspect a torch-vs-numpy dtype quirk
   (fp16 scale rounding) or a stale `.so`, not the layout logic. If conversion
   passes but composition fails, the bug is in apply (silu halves, identity-gather,
   scatter token/slot indexing, or pad/ntp masking). Set
   `VLLM_ROCM_W4A8_FP8_WMMA_MOE_VERSION=0` to isolate v5-kernel issues from wiring.
4. **`replace_parameter` / dtype**: I store fp16 scales; apply also guards with
   `.to(fp16)`. If the framework asserts scale dtype==model dtype somewhere,
   make the conversion keep model dtype and rely on the apply-time `.to(fp16)`.
5. **EP / multi-GPU**: only single-GPU (`expert_map=None`) is reasoned-through.
   `expert_map` is passed to `moe_align`; the scatter uses `sorted_ids//top_k`.
   Validate single-GPU first.
6. **Override doesn't check activation (latent gap, not a bug for target models)**:
   `_patched_init` overrides on `AWQMarlinConfig + zero_point + 4bit + group_size`
   only — it does NOT consult `moe.activation` / `moe.is_act_and_mul`, unlike the
   stock oracle's `is_supported_config`. `apply` assumes a **gated** activation
   (silu/gelu, `[gate; up]` halves). Every real AWQ MoE is gated SwiGLU, so this is
   fine for the target models, but a non-gated AWQ-4bit MoE would be force-routed to
   us and mis-shape gemm2. Deliberately left out (couldn't HW-test, and a wrong
   `moe.activation` type/None could regress the happy path to Marlin). If you ever
   hit a non-gated AWQ MoE, gate the override inside `_patched_init`'s existing
   try/except with `bool(getattr(moe, "is_act_and_mul", True)) and
   our_experts_cls._supports_activation(moe.activation)`.

## After Task 6 is HW-validated — what's next (toward the final goal)

Final goal: `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` (MoE, AWQ-4bit g32) runs FASTER
than the Marlin/Triton baseline, end-to-end, at matched quality. After wiring works:
1. **Perf**: compare tokens/s vs Marlin (`VLLM_ROCM_W4A8_FP8_WMMA_MOE=0`). Our
   apply currently materialises padded-sorted scratch + does the scatter in torch —
   likely slower than Marlin at decode. Tune `_choose_block_m`; consider fusing the
   scatter; the grouped op's per-token act-quant is a separate launch (Task 3).
2. **Fused ops (Task 3)**: fuse act-fp8-quant into the GEMM prologue; fuse
   silu_and_mul into gemm1 epilogue; fuse the scatter/gather.
3. **Kernel double-K 128-bit loads (Task 4)** for mid-M.
4. Re-tune `crossover_cache.json` for the target model's expert/dense shapes (g32).

## Environment reminders

- Dev host (here): no GPU, no torch — only py_compile / static review possible.
- Build host `blue`: `source activate-build-env.sh` (`.venv-therock714`, torch 2.10,
  GPU_ARCHS=gfx1201) — kernel/correctness only, no vllm.
- Container `kyuz0/vllm-therock-gfx1201`: torch 2.13, vllm 0.21.1 — end-to-end.
  Final destination `rocm/vllm-dev:nightly-therock` may have a different torch →
  build the .so inside whatever image you deploy to.
