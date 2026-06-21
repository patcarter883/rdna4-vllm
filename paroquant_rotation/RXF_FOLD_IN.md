# RXF fold-in: the collaborator shipped the W4A8 + rotation design; what's left

**Date:** 2026-06-20
**Base image:** `tcclaviger/vllm22:dev` @ `sha256:ad046f…` (pushed 2026-06-18; prior local was
`fca67d…` 2026-06-14). Toolchain bumped: torch 2.10, triton 3.6, transformers 5.5.3.
**Read-only source extracted to:** `.rxf-inspect/` in the main checkout (throwaway).

## TL;DR

The updated base image adds **RXF ("Rotated eXtra Fast")** — the collaborator's (tcclaviger)
successor to RFP458. It independently converges on the exact design we sketched in
`feat/rfp458-fp8` + `feat/paroquant-rotation`: a **W4A8 path** and a **per-group rotation**. So
"fold ParoQuant into RFP458→FP8" is reframed:

1. **Re-baseline RFP458 → RXF.** RFP458 is superseded; build against RXF.
2. **Our fp8-activation angle is largely obsoleted for RXF's current models** — RXF's W4A8 is
   **int8×int8** with an *integer* codebook (weight→int8 is *exact*; e4m3 would be lossy for no
   speed gain). fp8 retains only a **narrow, conditional niche**: the non-integer-codebook regime
   the int8 path structurally *cannot run*.
3. **The real surviving ParoQuant lever is the rotation content, not its placement.** RXF uses a
   *fixed, data-blind, 32-wide* FWHT. The rotation is a **standalone pre-pass decoupled from the
   K=32 GEMM**, so a **wider + learned + importance-aware** rotation is a pre-pass-only change with
   the GEMM untouched — and it unblocks the collaborator's own stalled `ACT_AWARE_ENABLED`.

## What RXF actually is (facts from the image)

Files: `…/quantization/rxf.py`, `…/quantization/utils/rxf_kernels.py`,
`…/fused_moe/experts/rxf_moe.py`, `tools/rxf_quant/{quantize_rxf,model_registry}.py`,
`tools/rdna4_config_tuning/tuner_rxf.py`, tuned configs under `…/quantization/utils/configs/`.

- **Format** (`rxf-pack-quantized`, 4.5 bpw): IQ4-NL 4-bit code (2/byte) `weight_packed uint8
  [N,K//2]` + **one plain fp16 scale per group of 32** `weight_scale f16 [N,K//32]`. No exponent,
  no mantissa, no super-block — *simpler than RFP458's block-float* (RFP458 = int8 mantissa/group-16
  + per-channel int8 exponent). Dequant: `NL[idx] * scale`.
- **Group = 32 = "2 chained WMMA K-steps."** Deliberately coarser than RFP458's 16, "paid back by
  FWHT-32 mixing outliers 2× harder than FWHT-16."
- **Two served paths, one config flag** (`act_dtype`, env `RXF_ACT_DTYPE` wins; checkpoint
  identical either way):
  - `fp16` (default, safe): W4A16 — dequant `code*scale`→fp16, wide fp16 `tl.dot`.
  - `int8` (opt-in): **W4A8** — per-token symmetric int8 activation × **int8 codebook**, K=32
    blocked dot, int32 accumulate, per-group fp16 rescale, per-token scale at the end (~2× fp16).
- **The int8 trick:** `_get_nl_table_int8` **raises** unless the 16-point codebook is
  *integer-valued in int8 range* — "the integers ARE the int8 weight operands." So for an integer
  grid the weight side of the dot is **exact**, scale folded as an int32→fp32 epilogue.
- **Rotation** (`rotation="hadamard32"` | None, **mandatory gate**): weights stored **pre-rotated**
  per-32 along K offline; runtime applies the same normalized FWHT-32 to activations so
  `Ĥ·Ĥ = I` cancels. **It is a standalone whole-row pre-pass** (`_rxf_rotate_quant_int8_kernel`:
  one program/token, `BLOCK = next_pow2(K)`, FWHT-per-32 then fuse per-token int8 quant) — **NOT in
  the K-loop.** The int8 GEMM reads the already-rotated, already-int8 activation.
- **Block-wise FP8 attention** (`Fp8BlockLinearMethod`): a checkpoint may mark attention (e.g.
  Step-3.7 fused qkv+o) as e4m3 block-fp8 while MoE/MLP stay RXF — delegated to the stock vLLM
  block-fp8 kernel. (This is *attention* fp8, unrelated to our W4A8 weight fp8.)
- **Protected fp16 experts:** static per-expert format-tag + compact-slot table, one MoE kernel,
  CUDA-graph-safe.
- **Targets** (`model_registry.py`): Step-3.7-Flash (the `LLOYD_TABLE` is fit to its MoE experts),
  Qwen3.5/3.6-MoE, minimax_m2, Llama. New model = one `ArchSpec`, no quantizer edit.

## The two facts that decide the strategy

### 1. The shipped codebooks are integer → int8 covers them; fp8's niche is conditional
`NL_TABLE` and `LLOYD_TABLE` are **both integer-valued**:
```
IQ4-NL : -127 -104 -83 -65 -49 -35 -22 -10  1 13 25 38 53 69 89 113
Lloyd  : -125 -102 -83 -66 -51 -36 -23  -9  4 18 32 46 62 80 100 124   (fit to Step-3.7, ~22% MSE↓)
```
The collaborator **rounded the Lloyd-Max centroids to integers on purpose** so the int8 W4A8 path
stays usable. Consequence for our fp8 work:
- **Integer codebook ⇒ int8 is exact on the weight side and runs at ~2×.** e4m3 (3-bit mantissa,
  spacing 8 near ±112) cannot represent the grid exactly and offers *no* speed advantage over int8
  → **fp8 has no niche here.** Our `feat/rfp458-fp8` fp8-WMMA premise is obsoleted *for RXF's
  current models*.
- **fp8's surviving niche is narrow and conditional:** a model whose accuracy *requires* a
  **non-integer** codebook (integerizing costs too much PPL). There the int8 path **raises** and RXF
  falls back to **W4A16 fp16 (1×)** — and *that* is where `w4a8_fp8_wmma` is the only ~2× path.
  Open empirical question: **how much PPL does integerizing the codebook actually cost?** Step-3.7's
  integer Lloyd already claims 22% MSE↓ vs IQ4-NL, which *weakens* the case. Don't fund fp8 until a
  real model shows integerization is lossy.

### 2. The rotation span is decoupled from the GEMM — ParoQuant's lever survives there
`rxf_linear_int8_kernel` reads a **pre-rotated** int8 activation at `BLOCK_SIZE_K=32`; nothing in
the GEMM assumes the rotation span is 32. The 32 is only the **scale-group granularity of the
rotated weight**. The rotation happens in the pre-pass, which already loads the whole row — so a
**wider rotation (128 / full-K) is nearly free there and leaves the K=32 int8 dot untouched** (you
change only the offline weight rotation + the pre-pass; `R_act·R_weight = I` cancels for any matched
span). RXF's per-32 FWHT is a *performance-driven, deliberately weak* rotation: a 32-wide span
cannot move outlier energy across group boundaries (QuaRot/SpinQuant rotate the full hidden dim for
exactly this reason).

**So ParoQuant lives here as rotation *content*, not placement:**
- **Wider span** than 32 (reach cross-group outliers) — pre-pass-only.
- **Learned + importance-aware** rather than fixed/data-blind. The RXF quantizer explicitly states
  Hadamard and importance are **mutually exclusive** ("importance must be transformed into the
  rotated basis"), and ships `ACT_AWARE_ENABLED = False` *"IN DEVELOPMENT… do not flip on."* **That
  tension is exactly what ParoQuant resolves** — it learns the rotation jointly with calibration.
  This is the genuine, non-redundant edge, and it unblocks work the collaborator has already started
  and stalled on.

(Earlier sketch worry — "rotation must stay block-local, pairs may need a full-K gather" — is moot:
the pre-pass already gathers the full row. Earlier conclusion "ParoQuant evaporates at block-32" was
**wrong**: it evaporates only if you insist on staying at 32, which the GEMM does not require.)

## Operational fold-in

- **Re-baseline:** stop building against RFP458; RXF is the live format. Update the
  `rfp458-fp8-hardware-path` memory accordingly (done).
- **Rebuild combined off the new base** (`docker pull` already done) with a **distinct tag**. The
  new base is **ABI-compatible with the old (confirmed by Pat) — reuse the warm
  `.triton-cache-combined`, no fresh cache dir needed** (despite the torch 2.10 / triton 3.6 bump;
  the CLAUDE.md §3 toolchain caveat does not bite here).
- **Do NOT quote tok/s yet.** Tuned configs are `device_name=AMD_Radeon_R9700` only; `get_rxf_configs`
  keys on device name, so on our RX 9070 XT / 9070 it silently falls back to the **default** config
  (the warning path). Retune (`tuner_rxf.py`) for our device before any throughput A/B.

## Concrete next steps (in order)

1. **Stand RXF up on gfx1201** (a Step-3.7-Flash or Qwen3.x-MoE RXF checkpoint): validate W4A16 vs
   W4A8-int8 **perplexity + tok/s**, rotation on/off. Establishes where RXF actually stands. (Retune
   configs first; lease via `gpu-lease.sh`.)
2. **int8-vs-fp8 A/B, codebook-conditioned** — only meaningful on a **non-integer** codebook. First
   measure the **PPL cost of integerizing** the Step-3.7 Lloyd grid; if ~0, fp8 is dead here and we
   archive `feat/rfp458-fp8`. If material, A/B int8(fp16-fallback) vs our e4m3 kernel on that model.
3. **ParoQuant rotation experiment (offline, pre-pass only):** in `quantize_rxf.py`, swap fixed
   FWHT-32 for (a) a **wider** FWHT/rotation span, and (b) a **learned importance-aware** rotation;
   measure per-group MSE + PPL at iso-bits. This is the one lever with a clear, non-redundant upside
   and it directly advances the collaborator's stalled `ACT_AWARE`. Matching runtime pre-pass change
   is small (the kernel already does any-span FWHT over the resident row).

## Status of the old worktrees
- `feat/rfp458-fp8` — premise (add A8+rotation to RFP458) is **shipped by RXF**; keep only as the
  fp8 microkernel + the conditional non-integer-codebook fallback (step 2).
- `feat/paroquant-rotation` (this worktree) — **promoted**: the rotation-content lever (wider +
  learned + importance-aware) is now the primary contribution, targeting RXF's pre-pass.
