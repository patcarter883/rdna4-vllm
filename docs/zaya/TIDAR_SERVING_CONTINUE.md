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

## Update (2026-06-28 — §31g FULL-cudagraph CAPTURE of the carrier + hooked block forward: DONE, GPU-validated weight-independently at -n 1; nothing committed)
- **§31g capture DONE** (NEXT item 5 below): `gpu_validate.py` gained **Part G**, the FULL-cudagraph
  capture validation of the carrier + hooked block forward. It exercises the one piece the eager Part
  D/E/F path skips and that capture REQUIRES — the in-place, static-ADDRESS active-mask carrier
  (`update_active_tidar_mask_`, an address-preserving `qq_bias.copy_()`) — and gates capture by
  **eager==replay BIT-EQUALITY**, the honest weight-independent signal at `-n 1` (NOT torch.profiler
  launch-count, which bypasses replay — memory `[[profiler-bypasses-cudagraph-replay]]`; a measurement-
  discipline note + a read-only pointer to the §7.6/TP2-gated live-runner `zaya/dflash/{dispatch_probe.sh,
  analyze_launch_count.py}` tooling are inline).
- **G1 (static-address carrier, outside any graph):** after `update_active_tidar_mask_` copies a new
  step's qq_bias INTO the already-active carrier, the hook-wrapped `unified_attention` output == the
  fresh-alloc `build_tidar_mask_meta` path (**max|Δ|=0.0**) AND the carrier `id()`/`data_ptr()` are
  UNCHANGED across two in-place updates (**addr-stable=True**), block_len {4,8} × prefix {0,64,512}.
- **G2 (capture==replay, the core gate):** the carrier+hooked block forward captured under
  `torch.cuda.graph` at fixed `q_len=block_len·(1+block_len)` (static q/KV/out + static-address carrier);
  per fill, copy inputs into the static buffers + `update_active_tidar_mask_` + `g.replay()`, then assert
  replay == the same eager hooked call **bit-equal (max|Δ|=0.0)** for block_len ∈ {4,8,16} × prefix
  {0,64}, ≥2 fills each, carrier `data_ptr()` asserted unmoved every replay — the §5 "k-variability does
  not break capture" property. **Full A–G: ALL PASS** (1-card lease, `vllm22-w4a8:combined`; attn_hip
  rebuilt, NOT touched — the mask rides the standard triton_attn carrier).
- **Evidence level: GPU-validated, weight-independent** (real RDNA4 triton_attn kernel on a controlled
  `_AttnLM` surface; reproduced across two independent runs); NOT the converted checkpoint. The
  live-runner FULL_DECODE_ONLY *dispatch* + real TiDAR throughput vs the **27 tok/s** baseline is the
  §7.6-gated step-6 follow-on (replica `position_ids` + TP=2/>16 GB fit — out of this `-n 1` item's
  scope). The CPU tidar suite (`test_tidar_mask.py`+`test_tidar_loop.py`) stays **16/16** green.
  Recorded: design §9 (this sub-item, flipped to `[x]`) + this block. Nothing committed/pushed.

## Update (2026-06-28 — proposer / model-runner integration: per-step carrier+hook+evict wired into a live β=1 loop; GPU-validated; nothing committed)
- **Proposer / model-runner integration DONE for the per-step wiring** (NEXT item 4b below): the
  step-3 carrier+hook + step-4 evict are wired into a real per-step β=1 TiDAR decode loop via the NEW
  orchestration module `zaya/tidar/tidar_proposer.py` — **no new mask/rollback math**, only routing
  the pinned primitives. `maybe_install_tidar_hook()` (idempotent, null-safe) is wired ADDITIVELY into
  `ZayaForCausalLM.__init__` (overlay `zaya.py`), double-guarded so an import/install failure is a
  silent no-op and an installed-but-inert hook is byte-identical to stock.
  `TidarProposer.run_block(prefix_len)` = the per-step set-before / clear-after carrier boundary
  (fresh-alloc `build_tidar_mask_meta`; clears the carrier even on exception; the in-place
  `update_active_tidar_mask_` is reserved for the §31g capture step). `verify_commit→beta_verify`
  (β=1 lossless), `evict_contract(num_accepted)→num_accepted-1` names the `cca.py:_decode_verify_spec`
  rollback column. Serve-enable flag `VLLM_TIDAR_BLOCK_LEN` (unset ⇒ plain decode, hook inert).
- **CPU gate `test_tidar_proposer.py` 7/7 (full suite 23/23):** env-flag parsing, carrier set/clear
  (incl. clear-on-exception), `verify_commit==beta_verify`, evict-column contract, end-to-end
  β=1==greedy-AR through the proposer surface.
- **GPU gate `gpu_validate.py` Part F (1-card lease, `vllm22-w4a8:combined`): RUNNER-PATH β=1 loop ==
  AR-greedy, token-for-token** through the proposer carrier + the **real RDNA4 triton_attn kernel**
  (B∈{1,4,8}×2 seeds, carrier-clean after every loop) PLUS a null-safety regression (carrier cleared ⇒
  byte-identical to stock, max|Δ|=0). RESULT: **ALL PASS** (A–F). This pins the *wiring* is lossless
  (weights fixed), complementing the standalone `coherence_gate.py` GATE B that pins losslessness on
  the real **weights**.
- **Evidence level:** the per-step proposer/runner wiring is DONE + lossless, GPU-validated against the
  real kernel on a controlled `_AttnLM` surface (NOT the converted checkpoint through the live runner).
  The **live-runner fused single forward on the real checkpoint is §7.6-gated** (needs replica
  `position_ids` for `[S | R_0..R_{B-1}]` + TP=2 / >16 GB fit — out of this `-n 1` item's scope).
  Recorded in design §9 (proposer sub-item `[~]`) + §7.6. Nothing committed/pushed.

## Update (2026-06-28 — STEP 4: decode loop / β sampler / evict-on-reject FOLDED onto the real cca.py rollback; nothing committed)
- **STEP 4 DONE, CPU-validated on the real checkpoint** (NEXT item 4 below): the decode loop / β sampler /
  evict-on-reject are folded onto `cca.py:_decode_verify_spec`'s existing KV+conv-state rollback. The
  `(1+num_spec)` candidate-window conv + per-spec-position rollback IS the TiDAR evict-on-reject path —
  `num_spec` maps to the TiDAR `block_len`; the verify writes the conv window + `prev_hs` ENDING at each
  candidate `j` to slot `state_indices_2d[i, write_col[i,j]]`, and the next step reads column
  `(num_accepted-1)` = the accepted-prefix end. Appending the rejected tail then reading the accepted
  column IS the evict (the `IncrementalKVConv.commit_block` contract; no physical truncation). The
  rollback math was already generic over `1+num_spec`, so the fold required **NO logic change to cca.py**
  — only an additive null-safe `num_spec→block_len` doc anchor (`num_spec==0` ⇒ byte-identical, non-TiDAR
  path untouched).
- **REAL-MODEL evict gate `zaya/tidar/cca_evict_gate.py` (CPU, no lease — 17.7 GB bf16 fits host RAM):**
  drives the checkpoint's actual layer-0 `ZayaCCAProjection` (the conv producer cca.py caches) over
  `[committed | block_len-draft block]`, evicts the rejected tail, asserts committed conv/`prev_hs` ==
  from-scratch recompute of the accepted stream → **PASS, max|Δ|=0.0** for `k_accept` 0..block_len ×
  prefix {4,16}. **Conv-causality confirmed** (max|Δ|=0.0, tail/prefix {(4,4),(8,8),(16,4)}): the
  diffusion FT kept `conv_qk` CAUSAL → §7.5 resolved, **no CCA non-causal branch needed**. Gate is kept
  ISOLATED from the `B*B` mask-replica scratch (§7.5 fusion-contamination): it drives ONLY the conv
  producer; the TiDAR structured mask rides the SEPARATE standard-attention carrier (step 3), not this
  producer.
- **β=1 losslessness re-confirmed:** `coherence_gate.py` GATE B still PASS (β=1 == AR-greedy, all 4
  prompts); GATE A still reports the fully-fused single-forward NOT viable (max|Δlogit|≈1.84) —
  corroborating the isolation the fold satisfies. `test_tidar_loop.py`+`test_tidar_mask.py` **16/16 green**.
- **Evidence level: CPU-only by design** (validates the mathematical premise — real-conv causality ⇒
  evict==recompute — that makes the fold sound). cca.py got no executable change, so there is no new
  GPU code to exercise; the pre-existing rollback math stays pinned by `test_tidar_loop.py` test B
  (`commit_block` contract). No attn_hip change (mask rides the standard-attention carrier). Full writeup:
  design §9 (STEP 4) + §7.5. Nothing committed/pushed.

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
4. ✅ **DONE (2026-06-28).** STEP 4 — decode loop / β sampler / evict-on-reject FOLDED onto the real
   `cca.py:_decode_verify_spec` KV+conv-state rollback. `cca.py`'s `(1+num_spec)` candidate-window conv +
   per-spec-position rollback IS the TiDAR evict-on-reject path (`num_spec` → `block_len`; read accepted
   column == evict == `IncrementalKVConv.commit_block`); the rollback math was already generic over
   `1+num_spec`, so NO logic change — only an additive null-safe `num_spec→block_len` doc anchor
   (`num_spec==0` byte-identical). Real-model gate `cca_evict_gate.py` (CPU, no lease): committed conv/
   `prev_hs` state == from-scratch recompute of the accepted stream, **max|Δ|=0.0** for `k_accept` 0..B ×
   prefix {4,16}; conv-causality max|Δ|=0.0 (the real conv stayed causal → §7.5 resolved, no CCA branch).
   `coherence_gate.py` GATE B still PASS, `test_tidar_loop.py`+`test_tidar_mask.py` 16/16. The two flagged
   mask off-by-ones (`replica_offset`/`sampling_causal`) are already empirically pinned by the β=1
   coherence gate on the real checkpoint (2026-06-28; §9). See the 2026-06-27 update block + design §9.
4b. ✅ **DONE (2026-06-28).** Proposer / model-runner integration — wire the step-3 carrier+hook +
   step-4 evict into a live per-step β=1 TiDAR decode loop. NEW `zaya/tidar/tidar_proposer.py`
   (orchestration only, no new mask/rollback math): `maybe_install_tidar_hook()` (wired additively +
   null-safe into `ZayaForCausalLM.__init__`), `TidarProposer.run_block(prefix_len)` = the per-step
   set-before/clear-after carrier boundary (clears even on exception), `verify_commit→beta_verify`,
   `evict_contract→num_accepted-1` (the cca.py rollback column), env flag `VLLM_TIDAR_BLOCK_LEN`
   (unset ⇒ plain decode, hook inert). CPU gate `test_tidar_proposer.py` 7/7 (suite 23/23). GPU gate
   `gpu_validate.py` **Part F** (1-card lease, `vllm22-w4a8:combined`): RUNNER-PATH β=1 loop ==
   AR-greedy token-for-token through the proposer carrier + the REAL RDNA4 triton_attn kernel
   (B∈{1,4,8}×2 seeds, carrier-clean each loop) + null-safety regression (carrier off ⇒ byte-identical
   to stock). RESULT ALL PASS (A–F). The wiring is lossless; the **live-runner FUSED single forward on
   the converted checkpoint is §7.6-gated** (needs replica `position_ids` + TP=2/>16 GB fit — out of
   this `-n 1` item's scope). See design §9 (this sub-item) + §7.6. Nothing committed.

5. ✅ **DONE (2026-06-28).** §31g FULL-cudagraph-CAPTURE of the carrier + hooked block forward
   (weight-independent, -n 1) — `gpu_validate.py` **Part G**. Exercises the in-place static-ADDRESS
   carrier (`update_active_tidar_mask_`) the eager Part D/E/F path skips, then gates capture by
   eager==replay BIT-EQUALITY (the honest -n 1 signal; NOT profiler launch-count, which bypasses
   replay — [[profiler-bypasses-cudagraph-replay]]). **G1:** in-place carrier copy-in == fresh-alloc
   `build_tidar_mask_meta` (max|Δ|=0.0) + `id()`/`data_ptr()` stable across two updates
   (B∈{4,8}×P∈{0,64,512}). **G2:** capture under `torch.cuda.graph` at fixed
   `q_len=block_len·(1+block_len)` (static q/KV/out + static-address carrier); replay == eager bit-equal
   (max|Δ|=0.0) for B∈{4,8,16}×P∈{0,64}, ≥2 fills, carrier `data_ptr()` asserted unmoved each replay —
   the §5 "k-variability does not break capture" property. Full A–G **ALL PASS** (1-card lease,
   `vllm22-w4a8:combined`; attn_hip rebuilt, NOT touched). See the 2026-06-28 update block + design
   §9 (this sub-item) / §5. Nothing committed.

6. **← LEADING PIECE (gated).** Live-runner FULL_DECODE_ONLY *dispatch* + real TiDAR throughput vs the
   **27 tok/s** baseline (step 5) — needs the §7.6 fused single forward (mask-replica `position_ids`) +
   TP=2/>16 GB on the converted checkpoint (out of the -n 1 capture item's scope). Reference the
   live-runner `zaya/dflash/{dispatch_probe.sh,analyze_launch_count.py}` tooling read-only. Then
   paged-KV in attn_hip (step 7, stays in `feat/attn-hip`, additive/optional/null-default) + β<1 (step 8).

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
