# Session prompt — continue the TiDAR serving path for ZAYA1-8B-Diffusion on RDNA4 (gfx1201)

You are continuing the **serving / kernel / cudagraph** side of the TiDAR pivot (Zyphra
ZAYA1-8B-Diffusion). The pivot rationale, algorithm, and architecture are settled — do NOT re-litigate
them or redo the paper reads. READ FIRST, in order:
1. `docs/zaya/tidar-serving-design.md` — the live design note. §9 is the progress checklist (source of
   truth for what's done / next). §1 has the key architectural finding; §3 the mask; §4 the runtime.
2. `docs/zaya/TIDAR_SERVING_PROMPT.md` — the original brief (pivot why, reuse ledger, protocol).
3. Memory `[[tidar-diffusion-pivot]]` — condensed state incl. the GPU-validation results below.

This session owns serving only. AR→TiDAR conversion training is a SEPARATE session
(`docs/zaya/TIDAR_CONVERSION_PROMPT.md`, worktree `feat/tidar-convert`) — do NOT do training/data work.

## Update (2026-06-27 — decode loop + β sampler + triton_attn Route-B gate all DONE; nothing committed)
- **Decode loop + exact-KV/conv evict + β sampler — DONE, CPU-validated** (NEXT items 1 & 2 below):
  `zaya/tidar/tidar_loop.py` + `test_tidar_loop.py` (**6/6 green**, CPU float64, no lease — run with
  the same container-venv command as the mask tests). `StubCCALM` = a random-weight CCA-shaped causal
  LM (causal depthwise conv = `conv_states`, previous-token v term = `prev_hs`). `tidar_forward` = the
  single fused forward over `[prefix | S | R_0..R_{B-1}]` (structured `additive_bias`, per-replica
  **segmented** conv). `beta_verify` = the accept-while-`argmax(β·p_AR+(1−β)·p_diff)` sampler.
  `tidar_decode` = the loop (verify → commit k+bonus → evict B−k → re-draft). Pinned: **β=1 loop ==
  greedy AR exactly** (B∈{1,4,8}×3 seeds); **evict-on-reject == from-scratch recompute** of committed
  KV+conv every step (`IncrementalKVConv`); per-replica segmented conv == recompute; in-forward `R_k`
  == fresh predraft after k accepts (so the one-forward replica shortcut is sound). The
  `_decode_verify_spec` rollback **logic** is modelled standalone (no vLLM runner/weights exist yet);
  folding onto the real `cca.py` path is a later checkpoint-stage step.
- **triton_attn additive-bias path — DONE, GPU-validated (NEXT item 3 below):** the RDNA4 triton
  unified-attention kernel **already has** a query-query additive-bias hook (`unified_attention(
  qq_bias=...)`, `[q_len,q_len]` 0/-inf added post-causal via `load_qq_bias_tile`; used by
  `tree_attn.py`); it takes `tidar_mask.additive_bias` verbatim for the **causal** parts. The one gap
  (`qq_bias` added *after* the causal `seq_mask` → couldn't grant the replica block's **bidirectional**
  attention) is closed by a 19-line `seq_mask` gate in the overlay
  `zaya/tidar/triton_overlay/triton_unified_attention.py`: when `USE_QQ_BIAS`, OR the causal mask back
  to True for query-query keys (`key_rel_pos = seq_offset − context_len ∈ [0, qq_bias_stride_0)`) so
  `qq_bias` alone defines allow/deny there; prefix keys (`key_rel_pos < 0`) stay strictly causal;
  `USE_QQ_BIAS=False` is dead-code-eliminated (stock byte-identical). **GPU SDPA-equivalence gate**
  (`gpu_validate.py` Part D, weight-independent, 1-card lease, `vllm22-w4a8:combined`): patched ==
  boolean-masked fp32 SDPA over `build_allow_matrix`, block_len {4,8,16} × prefix {0,64,512} (cos
  0.999997–0.999999, ≤7 bf16-ULP); **tank-check** the UNPATCHED kernel FAILS every bidirectional case;
  `qq_bias=None` and a causal `qq_bias` are byte-identical patched vs stock. RESULT ALL PASS. Loads the
  kernel by file path (stock site-packages stays pristine for the tank-check). Full writeup: §7.2 + §9.
  **Route B is now the working PAGED structured-mask path.**

## Where things stand (2026-06-26 — all GPU-validated, nothing committed)
The structured-mask attention primitive is **DONE and validated end-to-end on hardware**:
- `zaya/tidar/tidar_mask.py` — backend-neutral TiDAR mask: `build_allow_matrix` / `additive_bias`
  (the `[q_len,kv_len]` form for triton/SDPA) + `build_square_allow_matrix` / `square_additive_bias`
  (the `[L,L]` form for self-attention kernels) + `MaskDescriptor` (compact layout ints for an inline
  kernel predicate) + `select_next_drafts_row_range` (post-accept replica selection).
  Layout (CONFIRMED from the paper, supersedes the brief's `2·block_len`): `q_len = block_len·(1+block_len)`
  = 1 sampling block `S` + `block_len` mask replicas `R_r`, replica `R_r` conditioned on `r` accepted
  drafts (the parallel "pre-draft conditioned on every acceptance length" trick).
- `zaya/tidar/test_tidar_mask.py` — **10/10 CPU tests** (no lease): run via the container venv
  (host torch is broken — missing libmpi). See "How to run" below.
- `attn_hip` (worktree `/home/pat/code/vllm-gfx1201-attn-hip`, branch `feat/attn-hip`) gained an
  **optional** `mask_bias` arg (square `[seq,seq]` fp32 additive bias) — applied in the fp32 `smem_S`
  softmax loop at the existing causal/SWA mask site (`attn_kernels.hip`), schema
  `flash_prefill(..., Tensor? mask_bias=None)` (`bindings.cpp`), fake updated (`op.py`). **null ⇒
  byte-identical** to before; existing callers/parity untouched.
- `zaya/tidar/gpu_validate.py` — 3-part GPU gate, **RESULT: ALL PASS**: (A) attn_hip baseline correct
  cos 0.999999 ≤5 bf16-ULP; (B) TiDAR bias == boolean-masked SDPA exactly (max|Δ|=0); (C) the REAL
  attn_hip kernel + square TiDAR mask == boolean-masked SDPA, cos 0.999999 ≤4 ULP, B∈{4,8}×P∈{0,64,200}.

**Uncommitted, per protocol**, across BOTH worktrees: `feat/tidar-serve` (mask + tests + gpu_validate
+ design note) and `feat/attn-hip` (the `mask_bias` kernel arg + rebuilt `.so`). The attn_hip edit is
on ANOTHER effort's branch — additive/optional and safe, but coordinate before committing it there.

## Gotchas already paid for (don't rediscover)
- The attn_hip parity gate (`attn_hip_parity.py`, flat `max|Δ|≤5e-3`) is **mis-calibrated**, not the
  kernel: one bf16 ULP on a peaked `|out|~2` causal row is ~1.5e-2. Gate with `atol + rtol·|ref|` (a
  few bf16 ULP). Do NOT use a raw per-element ULP ratio — it explodes on near-zero outputs. The
  kernel is correct (cosine 0.999999). It DOES have a real but tiny ~5-ULP ragged/SWA bf16 tail nit
  on 1 element — that's the `feat/attn-hip` owner's to tighten; it is NOT TiDAR-blocking (won't flip
  argmax tokens).
- CCA is a QKV producer feeding a STANDARD vLLM attention; the mask goes in the attention backend, not
  in CCA. Exact-KV evict-on-reject reuses CCA's existing spec rollback (see NEXT).

## NEXT (in order; all weight-independent — no checkpoint exists yet, validate on stubs)
1. ✅ **DONE (2026-06-27).** Decode loop + exact-KV evict stub (§4.2/§4.4) — `zaya/tidar/tidar_loop.py`
   + `test_tidar_loop.py` (6/6). See the 2026-06-27 update block above. (The `_decode_verify_spec`
   reuse is modelled standalone — real-`cca.py` fold is a checkpoint-stage step.)
2. ✅ **DONE (2026-06-27).** β sampler (§4.3) — `beta_verify`; β=1 loop == greedy AR exactly. `p_diff`
   wiring present; β=1 ignores it (lossless). See the update block above.
3. ✅ **DONE (2026-06-27).** triton_attn additive-bias path (§3.2/§7.2) — the `seq_mask` gate landed in
   the overlay `zaya/tidar/triton_overlay/triton_unified_attention.py` (OR the causal mask back to True
   for query-query keys `key_rel_pos ∈ [0, qq_bias_stride_0)`, prefix keys stay causal) and passed the
   GPU SDPA-equivalence gate (`gpu_validate.py` Part D, 1-card lease, `vllm22-w4a8:combined`): patched ==
   boolean-masked fp32 SDPA (cos 0.999997–0.999999, ≤7 bf16-ULP) AND tank-check unpatched FAILS every
   bidirectional case, plus byte-identical regressions for `qq_bias=None` and a causal `qq_bias`. Route B
   is now the working PAGED structured-mask path. See the update block above + design §7.2/§9.
4. **← LEADING PIECE. Confirm the two flagged mask off-by-ones** (§7.1): `replica_offset` (R_r sees r vs r+1 drafts) and
   `sampling_causal` — pin against the paper's Figure 3 / the conversion checkpoint when it lands. Wrong
   → silently breaks LOSSLESSNESS, not just speed.
5. Then: §31g FULL-cudagraph-capture the single forward (reuse `zaya/dflash/` tooling +
   [[profiler-bypasses-cudagraph-replay]]); paged-KV in attn_hip; coherence/throughput once a checkpoint
   exists.

## How to run (host torch is broken — use the container venv)
- CPU mask tests (no lease):
  ```
  docker run --rm -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= \
    -v /home/pat/code/vllm-gfx1201-tidar-serve/zaya/tidar:/work -w /work \
    --entrypoint bash vllm22-w4a8:dflash-rxf -lc \
    'source /app/.venv/bin/activate && python test_tidar_mask.py'
  ```
- GPU validation / rebuild attn_hip (1-card lease, mounts both worktrees + warm Triton cache):
  ```
  scripts/gpu-lease.sh -n 1 -- bash -c 'docker run --rm --device /dev/kfd --device /dev/dri \
    --group-add video --security-opt seccomp=unconfined --security-opt label=disable \
    --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
    -v /home/pat/code/vllm-gfx1201-tidar-serve/zaya/tidar:/tidar \
    -v /home/pat/code/vllm-gfx1201-attn-hip/attn_hip:/pkg/attn_hip \
    -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
    --entrypoint bash vllm22-w4a8:combined -lc "source /app/.venv/bin/activate && \
      export PYTHONPATH=/pkg:/tidar && cd /pkg/attn_hip && \
      GPU_ARCHS=gfx1201 python setup.py build_ext --inplace 2>&1 | tail -6 && \
      python /tidar/gpu_validate.py"'
  ```
  (`attn_hip` is a package → its PARENT must be on PYTHONPATH: mount under `/pkg/attn_hip`, set
  `PYTHONPATH=/pkg`.)

## Protocol (CLAUDE.md — MANDATORY)
- Work in worktree `feat/tidar-serve` (`/home/pat/code/vllm-gfx1201-tidar-serve`, off `feat/zaya-dflash`
  to inherit §31g overlays + the CCA rollback). Edits to `attn_hip` land in `feat/attn-hip` — coordinate.
- EVERY GPU job via `scripts/gpu-lease.sh -n 1 -- …` (TP=1). Never ask a human / never poll rocm-smi.
- Don't commit (uncommitted overlays inherited from `feat/zaya-dflash`; fold at a future M-stage when
  asked). Reference vLLM (read-only): `/home/pat/code/_vllm_ref_combined`.

## DON'T
- Don't redo the paper reads or re-derive the architecture (done — see the design note).
- Don't reopen DFlash AR-spec (confirmed dead end). Don't do conversion training (other session).
- Don't "fix" the attn_hip ragged/SWA tail nit as if it blocks TiDAR — it doesn't; flag it to the
  attn-hip owner. Don't touch the attn_hip op's existing behaviour (keep `mask_bias` optional/null-safe).
