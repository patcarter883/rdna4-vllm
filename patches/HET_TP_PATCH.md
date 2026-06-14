# Heterogeneous proportional TP — integration sketch

Pairs with `patches/het_tp.py` (the apportionment helper + per-rank table). Goal:
split the **FFN and MoE intermediate** dimensions 64:56 (proportional to CU counts)
so the bigger card stops spin-waiting at the all-reduce barrier. Everything else
(attention heads, vocab/lm_head, all collectives) stays on the stock even path.

Line numbers below are against `code/zaya/vllm-therock/vllm/` (vLLM v0.22, therock);
they map 1:1 to the `patcarter883/vllm-gfx1201` wheel. Apply as a build-time source
patch (same mechanism as `patches/moe_wna16.py`).

## 0. The correctness invariant (read first)

`gate_up_proj` output channels and `down_proj` input channels are the **same**
intermediate channels — they MUST be split identically across ranks, or the matmul
pairs the wrong rows. `het_tp.partition_sizes` is deterministic, so calling it with
the **same** `(intermediate, weights, align=group_size)` in both layers guarantees
consistency. Same for MoE `w13` (output) vs `w2` (input). **Always key the split on
the full intermediate size with `align = group_size`** (128 for AWQ-INT4; a multiple
of the int4 pack factor 8 and the WMMA/Triton N-tile, so every shard stays valid).

## 1. The partition table (plumbing)

Already in `het_tp.py`: `get_cu_weights(tp_size)` reads `VLLM_TP_CU_WEIGHTS="64,56"`
(env, deterministic, fail-safe to None=even). Set it in `docker-compose.yml` next to
the TP=2 profile. Optional auto-detect (mirror aiter's `chip_info.py` rocminfo parser)
can fill it, but the env keeps it explicit and reproducible — recommend env.

## 2. Eligibility predicate (scope the blast radius)

Add to `het_tp.py` (or inline): only FFN/MoE layers are eligible; attention + head/
vocab layers are not.

```python
def het_eligible(prefix: str) -> bool:
    # FFN dense ffn + MoE experts; NOT attention qkv/o, NOT lm_head/embed.
    p = prefix.lower()
    if any(k in p for k in ("attn", "lm_head", "embed", "vocab")):
        return False
    return any(k in p for k in ("mlp", "experts", "feed_forward", "ffn", "gate_up", "down_proj"))
```

`prefix` is already threaded into every `*ParallelLinear.__init__` (`linear.py:250,
315, 442, …`) and into the MoE layer.

## 3. linear.py — size the partition + stamp the param

### 3a. ColumnParallelLinear.__init__ (currently line 451)
```python
# BEFORE
self.output_size_per_partition = divide(output_size, self.tp_size)
self.output_partition_sizes = [self.output_size_per_partition]

# AFTER
from vllm_gfx1201.het_tp import het_eligible, het_size, het_align_for
self._het = het_eligible(prefix)
self._het_align = het_align_for(quant_config)   # group_size or safe tile; see §6
if self._het:
    self.output_size_per_partition = het_size(
        output_size, self.tp_rank, self.tp_size, align=self._het_align)
else:
    self.output_size_per_partition = divide(output_size, self.tp_size)
self.output_partition_sizes = [self.output_size_per_partition]
```
Note: `MergedColumnParallelLinear` (gate_up_proj, line ~630) reuses this via
`output_sizes`; loop `het_size` over each sub-output (gate, up) — each is the full
`intermediate`, so both get the identical proportional shard.

### 3b. RowParallelLinear.__init__ (currently line ~1442)
```python
# BEFORE
self.input_size_per_partition = divide(input_size, self.tp_size)
# AFTER
self._het = het_eligible(prefix)
self._het_align = het_align_for(quant_config)
self.input_size_per_partition = (
    het_size(input_size, self.tp_rank, self.tp_size, align=self._het_align)
    if self._het else divide(input_size, self.tp_size))
```

### 3c. Stamp params so the loader knows the global dim + align
After `quant_method.create_weights(...)` in each eligible layer's `__init__`, mark the
sharded params with the info the v2 loader needs:
```python
if self._het:
    for p in self.parameters(recurse=False):
        od, idd = getattr(p, "output_dim", None), getattr(p, "input_dim", None)
        if od is not None:
            p.het = ("col", output_size, self._het_align)   # global col dim
        elif idd is not None:
            p.het = ("row", input_size, self._het_align)     # global row dim
```

## 4. parameter.py — use the proportional offset (the key edits)

The quantized (compressed-tensors / W4A8) path loads via `weight_loader_v2` →
`vLLMParameter.load_{column,row}_parallel_weight`. These hard-code
`self.tp_rank * shard_size`. Replace with the stamped het offset.

**The pinned convention (verified by `patches/test_het_packing.py`).** These loaders
work in **param-axis units**, NOT logical elements: `self.data.shape[dim]` is this
rank's per-partition size in axis units (packed and/or grouped), and
`loaded_weight.shape[dim]` is the **global axis total**. So the het offset is just
`het_offset(loaded_weight.shape[dim], tp_rank, tp_size, align=align_axis)` where
`align_axis = align_logical // units_per_elt` for that axis:

| param / axis | units_per_elt | align_axis (align_logical=128) |
|---|---|---|
| AWQ qweight, N (output, packed) | pack=8 | 16 |
| AWQ qzeros, N (packed) | pack=8 | 16 |
| compressed-tensors w_q, K (input, packed) | pack=8 | 16 |
| scales, K//group (grouped) | group=128 | 1 |
| any unpacked weight axis | 1 | 128 |

Stamp each eligible param at create time with `param.het_align = align_axis` (computed
from `packed_factor`/`packed_dim`/`group_size`, all known in `__init__`); `None` ⇒ even.

### 4a. `_ColumnvLLMParameter.load_column_parallel_weight` (lines 148–151; repeat at 173)
```python
# BEFORE
shard_size = self.data.shape[self.output_dim]
loaded_weight = loaded_weight.narrow(
    self.output_dim, self.tp_rank * shard_size, shard_size)

# AFTER
shard_size = self.data.shape[self.output_dim]
ha = getattr(self, "het_align", None)              # axis-unit align, or None
if ha is not None:
    from vllm_gfx1201.het_tp import het_offset
    start = het_offset(loaded_weight.shape[self.output_dim],
                       self.tp_rank, self.tp_size, align=ha)
else:
    start = self.tp_rank * shard_size
loaded_weight = loaded_weight.narrow(self.output_dim, start, shard_size)
```
Line 173 is the second narrow (packed zeros) on the same param — identical substitution
with that param's own `het_align`.

### 4b. `RowvLLMParameter.load_row_parallel_weight` (line 223)
```python
# BEFORE
loaded_weight = loaded_weight.narrow(
    self.input_dim, self.tp_rank * shard_size, shard_size)
# AFTER
ha = getattr(self, "het_align", None)
start = (het_offset(loaded_weight.shape[self.input_dim], self.tp_rank, self.tp_size,
                    align=ha) if ha is not None else self.tp_rank * shard_size)
loaded_weight = loaded_weight.narrow(self.input_dim, start, shard_size)
```
Because `align_logical = group_size`, the K split lands on whole groups (scales stay
valid) and the N split lands on whole int32 packs (`test_het_packing.py` asserts both).

### 4c. Legacy (non-v2) loaders — `linear.py:556, 719, 844, 1205, 1517`
Only the unquantized fp16 path uses these. Same substitution: where `self._het`,
`start_idx = het_offset(<global dim>, self.tp_rank, self.tp_size, self._het_align)`.
Lower priority (the W4A8 path is all v2); patch if you also want het for fp16 layers.

## 5. fused_moe/config.py — het MoE intermediate (lines 1283–1284)
```python
# BEFORE
assert self.intermediate_size % tp_size == 0
self.intermediate_size_per_partition = self.intermediate_size // tp_size
# AFTER
from vllm_gfx1201.het_tp import get_cu_weights, partition_sizes
w = get_cu_weights(tp_size)
if w is not None:
    align = self.group_size if getattr(self, "group_size", -1) and self.group_size > 0 else 128
    self.intermediate_size_per_partition = partition_sizes(
        self.intermediate_size, w, align=align)[self.moe_parallel_config.tp_rank]
else:
    assert self.intermediate_size % tp_size == 0
    self.intermediate_size_per_partition = self.intermediate_size // tp_size
```
Then the MoE expert weight loaders (the w13/w2 narrows in `routed_experts.py` /
`patches/moe_wna16.py`) must use `partition_offsets(intermediate, w, align)[tp_rank]`
for the start, same pattern as §4. **w13 (col) and w2 (row) must use the same call** —
the §0 invariant.

## 6. `het_align_for(quant_config)`
```python
def het_align_for(quant_config):
    gs = getattr(quant_config, "group_size", None)
    if gs and gs > 0:
        return gs            # 128 for AWQ-INT4: multiple of pack(8) and tile(16/64)
    return 64                # fp16: a safe N-tile multiple
```

## 7. Verification

CPU (no GPU) — both passing today:
1. `python patches/het_tp.py` — apportionment (sum-exact, alignment, bigger-card-≥,
   3-way, **`(1,1)`≡even** safety guarantee). ✅
2. `python patches/test_het_packing.py` — pins the packed/grouped axis-offset
   convention for AWQ (N-packed) and compressed-tensors (K-packed) layouts:
   logical↔axis consistency, group-boundary alignment, reconstruction, and
   even-equivalence. ✅
3. *(after integration, CPU)* load the real model's FFN/MoE param shapes through the
   patched loaders with `VLLM_TP_CU_WEIGHTS="1,1"` and assert byte-identical tensors
   vs the stock even loader — the regression guard.

GPU A/B (your window):
4. Load `Qwen3.6-35B-A3B-AWQ-4bit` TP=2 with `VLLM_TP_CU_WEIGHTS="64,56"`; (a) greedy
   generation must produce **identical tokens** to the even baseline (het sharding is
   math-preserving), then (b) re-profile rank0/rank1 with vLLM's torch profiler — the
   rank1 COMM-spin inflation should shrink and decode tok/s rise toward the ~5% ceiling.

## 8. Risks / limits
- ~5% e2e ceiling (bubble + serial fraction). Do **after** cudagraphs.
- Packed-axis offset is the corruption trap — now **pinned** by §4's table +
  `test_het_packing.py`; the per-axis `align_axis = align_logical // units_per_elt`
  rule is the load-bearing detail.
- Per-sub-output bookkeeping in Merged is the other fiddly bit — QKV stays even
  (attention ineligible), so only `MergedColumnParallelLinear` gate_up needs §9.
- If aiter is ever enabled on this path, give it per-rank `CU_NUM` (rank0=64, rank1=56)
  via per-rank `ROCR_VISIBLE_DEVICES` masking so its grid sizing matches the real card.

## 9. MergedColumnParallelLinear (gate_up_proj) — per-sub-output handling

`gate_up_proj` packs two outputs (gate, up), each the full `intermediate`, concatenated
along `output_dim` in one param: `[ gate_shard | up_shard ]`. The v2 loader calls
`load_merged_column_weight(shard_offset, shard_size, ...)` once per `shard_id`
(`parameter.py:156`). For het, both sub-outputs get the **same** per-rank size
`het_size(intermediate, rank, tp_size, align)` (§0 invariant), so:

- **`__init__`**: build `output_partition_sizes = [het_size(inter, ...)] * 2` (gate, up).
  The merged param's per-rank axis length = `2 * het_axis_size`.
- **`linear.py` MergedColumnParallelLinear.weight_loader_v2**: the *intra-param*
  `shard_offset` for sub `i` is `i * het_axis_size` (cumulative over equal sub-sizes) —
  already correct once `output_partition_sizes` is het. Keep `adjust_shard_indexes_for_packing`
  for the packed output dim (it divides by `packed_factor`, consistent with §4's table).
- **`parameter.py:156 load_merged_column_weight` (lines 172–174)**: the **`loaded_weight`**
  narrow start is the bug for het. The checkpoint holds each full sub-output `[0, inter)`;
  this rank wants `[het_offset(inter,...) , +shard_size)`:
```python
# BEFORE  (line 172-174)
loaded_weight = loaded_weight.narrow(
    self.output_dim, self.tp_rank * shard_size, shard_size)
# AFTER
ha = getattr(self, "het_align", None)
if ha is not None:
    from vllm_gfx1201.het_tp import het_offset
    # each sub-output spans the FULL intermediate in the checkpoint; shard_size is the
    # packing-adjusted per-rank size, so the global axis total = shard_size's source =
    # loaded_weight.shape[output_dim] (one sub-output's full length).
    start = het_offset(loaded_weight.shape[self.output_dim],
                       self.tp_rank, self.tp_size, align=ha)
else:
    start = self.tp_rank * shard_size
loaded_weight = loaded_weight.narrow(self.output_dim, start, shard_size)
```
`param_data`'s narrow (line 171, at `shard_offset`) is unchanged — it already targets
the right sub-output slot because `output_partition_sizes` is het. The result: gate and
up are sharded with the identical 64:56 boundary, matching `down_proj`'s K split.

## 10. Subtleties to test later (when integrating + on the GPU window)
- [x] **Packed-axis offset** — `test_het_packing.py` (helper) + `test_het_loader.py`
  (real `PackedvLLMParameter`, column packed-N) ✅.
- [x] **stamping / per-axis units** — `test_het_loader.py` validates column-packed and
  row-grouped against the real param classes (the `logical//axis_total` derivation
  replaces per-param `packed_factor`/`group_size` lookups). ✅
- [ ] **MergedColumn gate_up (§9)** — `load_merged_column_weight` edited; not isolated-
  tested (test covers `load_column_parallel_weight`). Verify gate+up land on the same
  boundary in the E2E run.
- [ ] **MoE w13/w2 (§5)** — applied (`_het_moe_shard`); verify in E2E that
  `moe_align_block_size`/tile sizes tolerate an uneven `intermediate_size_per_partition`.
- [x] **Marlin/awq_marlin repack** — TRACED, het-safe (see §11).
- [ ] **GPU greedy-equivalence (2 GPUs)** — identical tokens vs even baseline before
  trusting any perf delta.

## 11. awq_marlin trace — het-safe (confirmed)

Question: does `quantization=awq_marlin` re-tile the output (N) dim in a way that breaks
a proportional het N boundary? **No.** Two independent reasons (source:
`quantization/awq_marlin.py`, `quantization/utils/marlin_utils.py`):

1. **On gfx1201 the dense AWQ path delegates to the W4A8 plugin, not a Marlin retile.**
   `AWQMarlinLinearMethod.process_weights_after_loading` (awq_marlin.py:490) runs
   `_convert_awq_to_standard_format(...)` then `self.kernel.process_weights_after_loading`
   — `self.kernel` is the MPLinearKernel selection, and the plugin prepends
   `RocmW4A8Fp8WmmaLinearKernel` to `_POSSIBLE_KERNELS[ROCM]`, so it wins when
   `can_implement` passes. `_convert_awq_to_standard_format` (awq_marlin.py:89) only
   unpacks/reorders/repacks using `N = N_packed * pack_factor` = the **per-rank** N — it
   is per-rank-N-agnostic and shape-preserving, so it carries the het boundary through.
   Flow: het load (AWQ packed-N, align_axis=16) → convert to AutoGPTQ `(K//8, N)` layout
   → plugin repack to op layout. The het boundary is fixed at *load* and never
   redistributed.

2. **Even if Marlin ran, het shards satisfy its tile gates.** `check_marlin_supported`
   (marlin_utils.py:174–193) requires `output_size_per_partition % 64 == 0`,
   `input_size_per_partition % 128 == 0`, and `% group_size == 0`; the MoE check
   (line 251) requires `intermediate_size_per_partition % max(64, group_size) == 0`.
   Our `align = group_size = 128` makes every het shard a multiple of 128 ⇒ all four hold
   (5888, 5120 both %128==0). Marlin's repack permutes **within** the per-rank tensor
   (it does not redistribute across ranks), so it preserves the per-rank het N.

**Net:** `align = group_size` is doing double duty — it keeps scales group-valid AND keeps
both the W4A8 plugin and Marlin tile-valid. The only thing to actually confirm on the box
(it doesn't change correctness, only which kernel runs): that `can_implement` passes for
the FFN/MoE layers so the plugin path is taken; if any layer falls back to Marlin, it's
still het-safe by reason (2). The MoE expert loaders (`patches/moe_wna16.py`) still need
the §5 het-offset edit regardless of backend.

## 12. INTEGRATION STATUS (applied to code/zaya/vllm-therock/vllm/)

**Applied — dense FFN het path (complete, CPU-verified, no-op when env unset):**
- `vllm/distributed/het_tp.py` — NEW (copied from patches/het_tp.py): apportionment +
  `het_align` (lcm(group,128)), `het_eligible`, `het_active`, `het_axis_offset` (the
  logical→axis `//u` converter). Self-tests pass.
- `vllm/model_executor/layers/linear.py` — `ColumnParallelLinear.__init__` and
  `RowParallelLinear.__init__`: proportional `*_per_partition` sizing + post-create
  stamping of v2 params with `param.het = (logical_narrow_dim, align)`. Gated by
  `het_eligible(prefix) and het_active(tp_size) and dim % align == 0`.
- `vllm/model_executor/parameter.py` — `load_column_parallel_weight`,
  `load_merged_column_weight` (gate_up), `load_row_parallel_weight`: use
  `het_axis_offset(...)` when `param.het` is set, else the stock `tp_rank*shard_size`.

**Mechanism implemented (supersedes the §4 `het_align`-stamp sketch):** stamp the
*logical* narrow-dim size + logical `align`; the loader derives per-axis units from
`logical // loaded_weight.shape[dim]`, so one code path covers packed weights, packed
zeros, and grouped scales. The served model is AWQ **g32**, so `align = lcm(32,128) =
128` (NOT 32 — that would break Marlin `min_thread_n/k`).

**Applied — MoE het (config + expert loaders, coordinated, no-op when unset):**
- `vllm/model_executor/layers/fused_moe/config.py:1283` — `intermediate_size_per_partition`
  via `partition_sizes(..., align=HET_ALIGN_FLOOR)`.
- `vllm/model_executor/layers/fused_moe/routed_experts.py` — new `_het_moe_shard`
  helper; `_load_w13` + `_load_w2` use it for the intermediate-axis (start, size) (one
  `replace_all` — the two sites were byte-identical). `_load_combined_w13_weight_scale`
  (ModelOpt-FP4, not AWQ) **raises** under het rather than silently mis-shard.
- Config and loaders share `HET_ALIGN_FLOOR = 128` so per-partition sizes and offsets
  can't disagree (valid for AWQ g16/g32/g64/g128; served model is **g32**).
- Covers the **stock fused_moe (AWQ/wna16)** path that actually runs on the 35B (the
  W4A8 plugin's own MoE loader is being fixed separately).

**Not covered (by design):** legacy/fp16 weight loaders + bias (Qwen3 MoE FFN has no
bias; the AWQ path is all v2). QKV/attention + vocab/lm_head stay even (intended).
ModelOpt-FP4 combined-w13 scales raise under het (loud, not silent).

**Verification done:**
- `py_compile` on all 5 edited/added files ✅.
- CPU math: `het_tp.py` self-tests ✅, `test_het_packing.py` (packed/grouped axis
  convention, AWQ + compressed-tensors, even-equiv) ✅.
- **Runtime against REAL param classes** (`test_het_loader.py`, run in the
  `vllm-gfx1201:latest` container with the edited `parameter.py`+`het_tp.py` mounted over
  site-packages): proportional column packed-N ✅, row grouped-scale ✅, shard
  reconstruction ✅, `VLLM_TP_CU_WEIGHTS` unset ≡ stock even split ✅. `import vllm` with
  the edits is clean.

**Pending (needs 2 free GPUs):** §7.4 — TP=2 load of `Qwen3.6-35B-A3B-AWQ-4bit` with
`VLLM_TP_CU_WEIGHTS="64,56"`, greedy-equivalence vs even baseline, then re-profile for
the bubble/tok-s delta. (Single-GPU het≡even via `VLLM_TP_CU_WEIGHTS="64"` at TP=1 also
exercises every het path — useful if only one GPU is free.)
