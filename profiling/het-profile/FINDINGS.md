# Het-TP COMM-bubble A/B — Qwen3.6-35B-A3B-AWQ-4bit, TP=2 (2026-06-14)

Offline torch-profiler A/B (`profiling/run_het_profile_offline.sh` →
`profiling/het_profile_check.py`), image `vllm22-w4a8:hettp` (combined image built
`--build-arg WITH_HET_TP=1`, het patch baked in). W4A8 **off** (stock wna16 MoE — the path the
het edits live in; the bubble is a TP-balance effect independent of W4A8). enforce_eager,
max_model_len 2048, 16 seqs × 128 decode tokens (ignore_eos), gpu_mem 0.90.

even = `VLLM_TP_CU_WEIGHTS` unset (5504/5504). het = `VLLM_TP_CU_WEIGHTS="64,56"`
(intermediate split 5888/5120). Per-rank kineto buckets via `analyze_torch_trace.py`.

## Decode throughput (wall-clock)
| variant | tok/s aggregate |
|---|---|
| even | 200.5 |
| het  | 200.6 |

**Identical.** See "Why tok/s is flat" below.

## Per-rank device-kernel time (the bubble)
| variant / rank | total device | all-reduce/collective | non-collective compute |
|---|---|---|---|
| even rank0 | 3969 ms | 1068 ms (26.9%) | 2901 ms |
| even rank1 | 4375 ms | **1468 ms (33.6%)** | 2907 ms |
| het  rank0 | 4225 ms | 1353 ms (32.0%) | 2872 ms |
| het  rank1 | 4189 ms | 1254 ms (29.9%) | 2935 ms |

- **Even split is imbalanced:** rank1's TP collective is ~400 ms longer than rank0's, and its
  total device time ~406 ms longer — one rank spin-waits at the all-reduce barrier (the
  documented sync bubble; cf. the 2026-06-12 baseline "rank1 COMM" artifact).
- **Het 64:56 balances it:** all-reduce imbalance **399.6 ms → 99.0 ms (~75% smaller)**; total
  device-time imbalance **405.6 ms → 35.8 ms**. The two ranks now reach the barrier in
  near-lockstep. **The proportional sharding works as designed.**
- Non-collective compute is ~equal across all four traces (2872-2935 ms, <2% spread): the cards'
  raw GEMM throughput is close here, so the even-split *compute* was already balanced — the
  bubble lived almost entirely in the collective wait, which het collapses.

## Why tok/s is flat (and what unlocks the win)
This run is **enforce_eager** → decode is **CPU-launch-bound**: non-collective GPU compute is
only ~2.9 s of the ~10.2 s wall (GPU idles between Python kernel launches across 40 layers ×
128 steps). The het split rebalances GPU-side work, but that rebalancing is hidden under launch
latency, so it does not surface in wall-clock tok/s. This is exactly the project's standing
note: **het-TP "do AFTER cudagraphs."**

**Next step to convert the balanced bubble into throughput:** re-run this A/B with **cudagraphs**
(drop enforce_eager — see memory `w4a8-cudagraph-capture-confirmed`). With launch overhead
eliminated, GPU compute becomes the critical path and the ~400 ms even-split barrier imbalance
(and the ~150 ms lower het critical-path device time) should convert toward the ~5% decode
tok/s ceiling. Optionally also tune the ratio (residual het imbalance is ~99 ms on rank0,
hinting the effective split is slightly past balance — try the exact measured CU/throughput
ratio rather than nominal 64:56).

## Correctness (prerequisite, already green)
Greedy-equivalence het≡even is **byte-identical** on both the dense 7B (TP=2) and this 35B MoE
(TP=2) — see `profiling/het-e2e/` and the session log. The het sharding is math-preserving;
this profile only measures the perf balance.
