# attn_prefill_paged — native paged/chunked-prefill attention for gfx1201

**The kernel that closes Triton-free SERVING.** Dense `attn_hip` only covers cold prefill; real
serving uses chunked prefill + prefix caching, where prefill reads a PAGED KV prefix. This handles
that: Q = the new tokens (packed varlen), K/V = the paged cache (prefix ⧺ new), causal with a prefix
offset.

## Design
Reuses `attn_hip`'s **validated** gfx12 WMMA core verbatim (rocwmma QK^T/PV + smem-staged online
softmax, fragment-layout-decoupled). Three deltas:
1. **Q packed/varlen** — `cu_seqlens_q[seq]` gives the seq's query offset in the packed `[total_q,
   Hq, D]` tensor; grid is `(q_head, q_tile, seq)`.
2. **K/V from the paged cache** — per-key block-table gather (`block_table` + `kv_block_stride`, as in
   attn_decode); `kv_block_stride=0` = contiguous `[num_blocks,bs,kvh,hd]`, or pass
   `2*bs*kvh*hd` + the unbind views for vLLM's interleaved `[num_blocks,2,...]`.
3. **Prefix-offset causal** — query global pos = `prefix_len + local` where `prefix_len = context_len
   - q_len`; key global pos = its cache index `j`; allowed iff `j <= qpos` (+ optional SWA).

## Interface
`torch.ops.attn_prefill_paged.flash_prefill_paged(q[total_q,Hq,D], k_cache, v_cache,
block_table[S,max_blocks], cu_seqlens_q[S+1] int32, context_lens[S] int32, scale, causal,
sliding_window, max_seqlen_q, kv_block_stride=0)`. Maps to vLLM/sgl extend metadata directly
(cu_seqlens_q, seq_lens=context_lens, block_table, max_query_len).

## Scope / status
v0: bf16, GQA, causal + SWA, head_dim 64/128 (256 gated off — same 64 KB LDS cap as attn_hip).
NOT YET: fp8-KV, the smem-reduction pass for 256. Validate with `attn_prefill_paged_parity.py`
(fp32 SDPA reference over prefix⧺new). With prefix_len=0 it reduces to dense prefill (== attn_hip).

## Wiring
Route the engine's EXTEND/chunked-prefill branch (max_query_len>1 with a non-empty KV cache) here;
dense `attn_hip` for cold prefill; `attn_decode` for decode. Together → Triton-free attention for the
chunked-prefill + prefix-cache serving config.
