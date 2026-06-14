# CCA 'all'-mode rewrite — bit-lossless speculative decode (implementation plan)

Status: **DESIGN — not yet implemented.** Gated on (a) a numerical-validity
decision (`--mamba-block-size 16`, below) and (b) a GPU window (this is
test-driven kernel work — the current spec path "cost many boot cycles" per
`project-spec-decode-blocker`). Reference impl: mamba2 (`mamba_mixer2.py`),
mapped below.

## Why the current path isn't bit-lossless

The committed CCA spec decode (commit `zaya: CCA ngram speculative decode`) is
coherent + correct (~1.76x) but not bit-identical to greedy after **partial**
draft acceptance. Root cause (proven): in `align`/`none` mamba_cache_mode, vLLM
leaves the cross-step rollback metadata `None`
(`block_idx_last_computed_token`, `block_idx_last_scheduled_token`,
`block_idx_last_scheduled_token_prev_step`, `num_accepted_tokens`,
`query_start_loc_d`), so CCA cannot roll its conv/temporal state to the
last-accepted token and reads slot 0 (after-bonus) instead → divergence. Only
`mamba_cache_mode='all'` populates those fields.

## Prerequisite decisions

1. **`--mamba-block-size 16` numerical validity — RESOLVED: VALID.** 'all' mode
   needs a small mamba block to fit the gfx1201 64 KB LDS (the default 2048
   overflows the attention Triton kernel by ~512 B). `--mamba-block-size 16`
   bypasses the wall and 'all' boots. Is overriding ZAYA's baked-in 2048→16
   numerically safe for CCA?
   - **Analytical: yes.** ZAYA config has `cca_time0=2, cca_time1=2` ⇒ CCA conv
     receptive field = `(2-1)+(2-1) = 2` tokens. CCA caches a *complete
     fixed-width state* (the 2-wide conv window + running `prev_hs`), so
     `mamba_block_size` only sets checkpoint *frequency*, not contents — numerics
     are block-size-invariant as long as one block ≥ the receptive field. `16 ≫
     2` ✓. The 2048 is just a large default.
   - **Empirical proof (GPU): greedy bit-identity test** — 'all' @ block-size 16
     vs the non-'all' greedy baseline (`num_spec=0`), identical prompts; equal
     token ids ⇒ valid in practice. This is the gate to run in the GPU window.
   ⇒ The prior "HW-blocked" verdict is lifted: block-size 16 fits LDS AND is
   numerically sound. Bit-lossless is achievable on gfx1201.
2. **GPU window** for iterative validation (eager 'all' is debuggable; block_idx
   fields become printable).

## Reference: how mamba2 does it (verified file:line)

- `mamba_attn.py:435-457` — in 'all' mode the builder sets
  `state_indices_tensor = block_table_tensor` (full `[num_decodes, max_blocks]`)
  and computes `block_idx_last_computed_token`, `block_idx_first/last_scheduled_token`,
  and (under spec) `block_idx_last_scheduled_token_prev_step`. Non-'all' leaves
  them `None` (426-430). `num_accepted_tokens`/`query_start_loc_d` populated in
  all modes when spec is on (484-486).
- `mamba_mixer2.py:506-510` — `self._decode_state_offsets =
  arange(1+num_spec).unsqueeze(0)`.
- `mamba_mixer2.py:962-984` — decode slot gather:
  - spec (num_spec>0): input = `state_indices.gather(1, blk_last_sched_prev_step
    + _decode_state_offsets)`; output = `gather(1, blk_last_sched + offsets)`.
  - single-token: input = `gather(1, blk_last_computed)`; output =
    `gather(1, blk_last_scheduled)`.
- `mamba_mixer2.py:991-1046` — `causal_conv1d_update(...,
  num_accepted_tokens=, initial_state_idx=blk_last_computed,
  block_idx_last_scheduled_token=blk_last_sched, query_start_loc=qsl_d)` and
  `selective_state_update(..., state_batch_indices=input,
  dst_state_batch_indices=output, num_accepted_tokens=)`. The kernels use
  `num_accepted-1` as the rollback offset into the `[batch, 1+num_spec]` slot
  table (`causal_conv1d.py:836-864`, `mamba_ssm.py:333-352`).

## Changes for CCA (concrete)

CCA has no causal_conv1d/selective_state_update kernel — it does the conv/state
in `cca.py` Python. So the rollback math we get "for free" in mamba2's kernels
must be done explicitly. Two files (both already in the overlay mount list):

### 1. `vllm/v1/attention/backends/cca_attn.py` — NO CHANGE NEEDED ✅
`CCAAttentionMetadata(BaseMambaAttentionMetadata)` and
`CCAAttentionMetadataBuilder(BaseMambaAttentionMetadataBuilder)` are thin
subclasses that inherit `build()` / `build_for_cudagraph_capture()` wholesale.
The 'all'-mode field population (`mamba_attn.py:435-457`,
`_compute_prefix_caching_block_indices`, `num_accepted_tokens`,
`block_idx_last_scheduled_token_prev_step`) is all in the BASE builder, so CCA
already receives every rollback field in 'all' mode. (The blocker memory's "the
CCA backend doesn't populate these" predates this base-builder version / was
observed in align-none where they are correctly None.)

### 2. `vllm/model_executor/layers/mamba/cca.py` — use them
**DRAFT WRITTEN (uncommitted, on top of 77bc19e0f), decode/verify only.** Gated
on `all_avail = blk_scheduled & blk_scheduled_prev & num_accepted all present`
(i.e. 'all' mode); align/none stays byte-identical. In `_decode_verify_spec`:
- read (seed) column = `blk_scheduled_prev + (num_accepted-1)` into the FULL
  block table (no `clamp(max=p-1)` — that compact-layout clamp was the bug that
  corrupted the block column and produced garbage in 'all' mode).
- write column for verify position j = `blk_scheduled + j` (full block table).
Mirrors mamba2 `selective_state_update` init_token_idx = num_accepted-1.

**REMAINING (GPU window):**
- **prefill 'all'-mode slot write** — the prefill path still collapses
  `state_indices_p` to column 0; in 'all' mode it must write each request's
  end-of-prefill conv/prev_hs state to its `block_idx_last_scheduled_token`
  block so the first decode step's seed matches. Risky to write blind; do with
  GPU feedback.
- **single-token decode (`elif has_decode`, num_spec==0)** still reads column 0;
  fine for spec (always goes through verify) but needs block_idx gather if 'all'
  is ever used for plain prefix caching.
- the original `_decode_verify_spec` use-of-them (below) — verify against mamba2
  semantics on hardware.
- `CCA.__init__`: `self._decode_state_offsets = arange(1+num_spec)[None]`.
- **decode (single-token)**: read input state from slot
  `state_indices_2d.gather(1, blk_last_computed[:,None])`, write to
  `gather(1, blk_last_scheduled[:,None])` — replaces the current column-0 read.
- **verify (multi-token)**: input slots =
  `gather(1, blk_last_sched_prev_step[:,None] + _decode_state_offsets)`; output
  slots = `gather(1, blk_last_sched[:,None] + _decode_state_offsets)`; on the
  next step, select the slot at `num_accepted-1` for the committed state. The
  forward-compatible read/write in the current `_decode_verify_spec` (using
  `blk_computed`/`write_col = read_col+1+pos`) is already shaped for this — it's
  inert only because the fields are `None`; once (1) delivers them it activates.
  Verify the `num_accepted-1` indexing matches mamba2's semantics exactly.
- **prefill**: in 'all' mode write each sequence's end state to its
  `block_idx_last_scheduled_token` slot (not the collapsed column-0 base slot).

### 3. Cache spec
`mamba_utils.cca_state_shape` / `zaya.get_mamba_state_shape`: already take
num_spec; in 'all' mode the block count is driven by `num_speculative_blocks`
(set when prefix caching is on) — confirm the CCA `MambaSpec` advertises enough
blocks per sequence.

## Validation (GPU window)
1. `docker-compose.allmode.yml` (ZAYA_SPEC_ALL=1, --enable-prefix-caching
   --mamba-cache-mode all --mamba-block-size 16), **eager first** so block_idx is
   printable.
2. Confirm `mamba_cache_mode='all'` active (not demoted) + fields non-None in
   the CCA layer.
3. Greedy bit-identity gate: same prompts, num_spec=0 vs num_spec=4, assert
   identical token ids (the bar the current path misses on partial acceptance).
4. Then cudagraph + the bit-identity gate again.

## Risk
Large, cudagraph-sensitive, only testable on GPU. If `--mamba-block-size 16` is
numerically invalid (decision 1), 'all' is dead on gfx1201 and the ceiling stays
coherent-not-lossless — do NOT write this rewrite in that case.

## GPU test result (2026-06-13 PM, GPU0, eager) — the real blocker
- ✅ **'all' mode BOOTS** (block-16, KV 4.53 GiB/59292 tok, no 64 KB-LDS
  overflow). HW-unblock confirmed.
- ❌ Output garbage; the decode/verify draft did not fix it.
- **Why (eager probe):** at the CCA layer `num_accepted_tokens` AND
  `block_idx_last_scheduled_token_prev_step` are **None** even in 'all' mode
  (only `block_idx_last_computed/scheduled_token` arrive). So the draft's
  `all_avail` branch never runs, and spec-verify multi-token steps don't even
  reach `_decode_verify_spec` (reorder_batch_threshold collapses to 1 when
  num_accepted is None → verify treated as prefill).
- **Revised remaining work:** §1's "no change needed" was wrong in practice — the
  base builder *has* the code but the **RUNNER doesn't pass num_accepted_tokens /
  prev_last_scheduled_idx to the CCA backend's build()**. Fix = thread those in
  the runner (mirror mamba2's runner path) + add the runner/mamba_attn mounts.
  Core vLLM change, not overlay-only; multi-cycle GPU effort. Until then the
  cca.py 'all'-mode draft is correct-but-inert scaffolding.
