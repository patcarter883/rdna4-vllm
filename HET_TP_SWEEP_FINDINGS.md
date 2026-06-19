# Het-TP split tuning — feasibility analysis (2026-06-19)

Two questions were on the table: (1) do the NCCL/RCCL `LL`/`Ring`/`MIN_NCHANNELS=1`
settings let vLLM boot, and (2) plan a 1%-increment sweep of the het-TP split ratio to
drive the rank0/rank1 all-reduce arrival gap to <5 µs.

## 1. NCCL/RCCL LL settings — boot test: **FAIL**

Booting the 35B TP=2 `serve` profile with:
```
NCCL_PROTO=LL  RCCL_PROTO=LL  NCCL_ALGO=Ring  RCCL_ALGO=Ring
NCCL_MIN_NCHANNELS=1  RCCL_MIN_NCHANNELS=1
```
crashes during TP group init (~27 s in, before weights load):
```
pynccl GroupCoordinator.__init__ -> warm-up all_reduce -> ncclAllReduce
RuntimeError: NCCL error: invalid usage (run with NCCL_DEBUG=WARN for details)
```
Log: `nccl-ll-boot-test.log`. RCCL rejects the first collective under this env combo.
These settings are **not** usable as-is. (A `NCCL_DEBUG=WARN` re-run was queued to name
the specific offending knob — see `nccl-ll-warn.log`.)

## 2. The 1%-increment split sweep is **not physically realizable on this model**

### Why — granularity lock
The het-TP knob is `VLLM_TP_CU_WEIGHTS`. The split is quantized to multiples of an
alignment floor:
- **Routed MoE experts** (`fused_moe/layer.py`): `align = HET_ALIGN_FLOOR = 128`.
- **Shared-expert FFN** (`linear.py`, unquantized): `align = lcm(group_size,128) = 128`.

The served model's split dimension is tiny:
`moe_intermediate_size = 512`, `shared_expert_intermediate_size = 512` (hidden=2048,
256 experts, top_k=8, 40 layers, AWQ/compressed-tensors g32).

`512 / 128 = 4 tiles` → only 4 indivisible units to split across 2 ranks. Verified with
the real helper (`partition_sizes`):
```
64:56 -> [256,256]   # == EVEN. The CU ratio (53.3%) rounds to even.
60:60 -> [256,256]
 2:1  -> [384,128]   # the ONLY reachable non-even split = 75/25, far too aggressive
```
**Consequence: `VLLM_TP_CU_WEIGHTS=64,56` is a byte-for-byte no-op on this model** — the
current "het-TP" baseline is the even split. There is no value near the CU ratio that
produces a different split. 1% steps are impossible; the knob is a step function whose
only neighbours are 50% and 75%.

Even lowering the align floor to its finest valid value:
- MoE align=32 (= AWQ group size): steps of 6.25% → 64:56 = [288,224]=56.25%. Reachable
  near-mid points are only {50, 56.25, 62.5}%.
- Shared-FFN align=16 (unquantized): steps of 3.125%.

So the **finest achievable granularity is ~6.25%**, never 1%, and only after a code
change + rebuild.

### Why — the knob can't reach the dominant cost
`het_eligible()` returns False for any prefix containing `attn`, so the **GDN linear
attention** (`linear_attn.*`, the dominant decode compute on this hybrid) is split
**evenly** across the asymmetric cards and the knob cannot touch it. Balancing only the
MoE (~≤10% of decode wall per prior TraceLens) while the GDN imbalance stays even-split
means the knob structurally cannot close the bubble it's aimed at.

### Why — the <5 µs target is below the measurement floor
All-reduce here is coarse PYNCCL over PCIe (custom-AR ruled out on RDNA4). Kineto on ROCm
will not resolve a 5 µs compute-arrival skew; the collective's own latency dwarfs it.
A measurable success metric is *median per-step rank wait as a fraction of step time*.

## 3. What's actually worth doing (cheap → expensive)

1. **Gating measurement on the CURRENT image (no rebuild).** Even split is runnable now;
   `VLLM_TP_CU_WEIGHTS=2,1` gives the 75/25 bracket. Profile a fixed ~50-token decode at
   both, run TraceLens straggler analysis (`profiling/run_tracelens.sh`, `multi/
   straggler_summary.csv`), and attribute the per-step arrival gap to a layer class
   (kernel→parent op). If the gap at even split is already small, or it's GDN/elsewhere
   rather than MoE, the whole exercise is moot — **stop here, build nothing.**
2. **Only if (1) shows MoE compute-arrival is a real, monotone lever and even is
   suboptimal:** make the align floor configurable (e.g. `VLLM_HET_ALIGN`, default 128),
   set 32, rebuild a `:hettune` image, and sweep the *discrete realizable* weights
   {even, 56.25%, 62.5%} — measuring straggler gap per point. Pick the min. This is the
   realizable version of the requested sweep; 1% steps remain impossible.

Bottom line: the sweep as specified (1% increments, <5 µs equilibrium) cannot be run on
this model; the knob is granularity-locked to ~25% steps as shipped (so 64:56 = even),
and it doesn't even address the dominant GDN imbalance.

## 4. Empirical straggler A/B (2026-06-19) — there is no bubble to chase

Ran the offline TP=2 decode profiler (`profiling/run_straggler_sweep.sh`, combined image,
W4A8 on, 16 seqs × 64 decode) at **even** vs **75/25** (`VLLM_TP_CU_WEIGHTS=2,1`), then
TraceLens straggler report (`profiling/tracelens/even/`).

**75/25 does not even load** — two independent failures, either fatal:
- rank0 **HIP OOM**: `Tried to allocate 486 MiB. GPU 0 has 350 MiB free` — putting 75% of
  the MoE/FFN intermediate weight on one 16 GB card overflows it.
- rank1 **het loader bug**: `The expanded size of the tensor (4) must match the existing
  size (8)... Target [1024,4], Tensor [1024,8]` — the grouped scale/zeros axis is
  mis-sharded at the 128-wide shard (128/group32 = 4 groups, 8 supplied). The het MoE
  scale/qzeros offset math (`het_axis_shard`) is wrong for this shard.

So the ONLY reachable non-even split is unusable; **even is the only working config.**

**At even split the ranks are already balanced** (TraceLens straggler_summary):
```
rank 0 (64 CU): mean_wait=142.2µs  computation=1352ms  arrived_last=57.4%  total_nccl=935.9ms
rank 1 (56 CU): mean_wait=134.9µs  computation=1401ms  arrived_last=42.6%  total_nccl=900.5ms
```
- Inter-rank mean-wait gap ≈ **7 µs**; compute-time gap ≈ 3.5%. The "find the <5 µs
  equilibrium" target is essentially already met at even split, with no tuning.
- The ~135 µs mean wait and ~900 ms `total_nccl` are the all-reduce's own **PCIe latency**
  hitting *both* ranks equally (a barrier), NOT a compute imbalance. `vllm::all_reduce`
  is the #1 GPU kernel (909 ms, 13.8%); decode is 65% idle, compute ~21%, exposed comm ~14%.

**Verdict: do NOT build the align-configurable het sweep.** There is no compute-arrival
imbalance to recover on this model; the het-TP knob is granularity-locked, its only
reachable non-even split OOMs+crashes, and the dominant GDN path isn't even eligible. The
real collective cost is all-reduce *latency over PCIe* (~14% exposed comm) — a different
lever entirely (NCCL/RCCL protocol/algo tuning, not split balance). Separately, fix or
guard the het MoE scale/zeros sharding bug exposed by the 75/25 load (it would mis-shard
any non-even split on a grouped checkpoint).
