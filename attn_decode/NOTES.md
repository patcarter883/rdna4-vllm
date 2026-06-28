# attn_decode — native flash-DECODE attention for gfx1201 (RDNA4), Triton-free

Phase 2 of the Triton-free attention effort. Companion to `attn_hip/` (prefill, WMMA) and
`gdn_hip/` (GDN). Decode is the serve-loop hot path (one query token per sequence per step).

## Why this is a SEPARATE kernel from prefill (and easier)
Decode is **M=1** — the single query attends over the whole cached KV. That is a
**bandwidth-bound reduction, not a matmul**, so there is **no WMMA here at all**: scores are
warp-reduced dot products, the output is an online-softmax accumulation. Consequence: **none of the
gfx11→gfx12 WMMA fragment-layout risk** that the prefill kernel had to engineer around. The only
things that matter are coalesced K/V loads and a correct cross-warp softmax combine.

## Design (v0)
- One CTA per `(q_head, seq)`. `NWARPS=8` warps split the KV range (each warp strides `j += NWARPS`).
- **Coalesced layout:** lane `l` owns head-dim elements `d = l, l+32, ...` of q/k/v/o — consecutive
  lanes read consecutive `d`, so every K/V row load is coalesced.
- Each warp runs an independent online softmax over its key subset → `(m, l, o)` partial.
- **Cross-warp combine:** partials go to smem; warp 0 merges (global max, reweight each warp's
  `(l, o)` by `exp(m_w - m_g)`), normalizes, writes the output. This is the only synchronization.
- Structure mirrors Atlas `paged_decode_attn.cu` (AGPL — read for structure, not copied).

## Scope / NOT YET
v0: dense (non-paged) bf16 KV, GQA, optional sliding window. `q:[B,Hq,D] k/v:[B,Skv,Hk,D]`. The
single query attends to ALL `Skv` cached keys (causal is implicit — keys are history); SWA masks
keys older than the window. **Roadmap (perf passes, after parity green):**
1. **Paged-KV** — replace the contiguous `k_base + j*stride` addressing with a block-table lookup
   (the only change vs v0; the softmax math is identical). This is what the serve path needs.
2. **fp8 / NVFP4-KV** — load quantized K/V, LUT-dequant in registers/smem (E2M1 nibble + FP8 scale),
   keep the accumulation fp32. See Atlas `paged_decode_attn_turbo{3,4}.cu` for the LUT pattern.
3. **Sparse-V gating** — skip the V load+accumulate when `p` is below threshold (Atlas turbo): saves
   bandwidth on negligible-weight keys at long context.
4. **Split-KV / flash-decoding** across CTAs for very long KV (multiple CTAs per (head,seq),
   second-pass reduce) if a single CTA's `Skv` loop is latency-bound.

## Status
Written, **not yet GPU-validated.** Gate = `attn_decode_parity.py` green under a 1-card lease in
`vllm22-w4a8:combined`. The CPU compile (`GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`)
needs no lease and should be run first to catch build errors. Expected first failures if any: the
cross-warp combine reweighting, the strided-`d` smem indexing in the combine, or GQA `kv_head` map.

## Wiring (later)
Swap the engine's Triton decode-attention for `torch.ops.attn_decode.flash_decode` behind a flag
(mirror `gdn_hip` / `VLLM_GDN_HIP`). Token-diff vs Triton. The `attn_decode::` namespace is separate
from prefill's `attn_hip::` so both .so coexist; fold into one `attn_hip` library when they merge.
