# ZAYA1-8B FP8 **TP=2** decode profile — gfx1201 (RX 9070 XT dev0 + RX 9070 dev1)

vLLM built-in **torch profiler** (`--profiler-config '{"profiler":"torch",
"torch_profiler_dir":"/profiles","torch_profiler_with_stack":false}'`), TP=2,
`--enforce-eager` (required for per-kernel attribution — FULL decode graphs
collapse into one opaque replay; eager is a profiling-only config, see
`feedback-enforce-eager-profiling-only` memory). Warm cache (a throwaway run
first paid the cold Triton JIT of `fused_moe_kernel` / `kernel_paged_attention_2d`
/ `_fwd_kernel`). Workload: 16 seqs × 128 decode tokens (`ignore_eos`), one
`max-num-seqs` wave, FP8 ZAYA, AITER off, `ZAYA_CCA_HIP=1`.
~187 tok/s aggregate (eager; graphs would be much faster — serving config keeps
graphs on). Driver: `drive_tp2_profile.py`; analyzer: `analyze_torch_trace.py`.
Raw traces: `tp2-torch/dp0_pp0_tp{0,1}_*.pt.trace.json.gz` + vLLM's own
`profiler_out_{0,1}.txt` (key_averages, sorted by Self CUDA).

## Device-kernel time by bucket (per rank)

| bucket | rank0 (64-CU XT) | rank1 (56-CU 9070) |
|---|---:|---:|
| **all-reduce / collective (PYNCCL)** | 373 ms (10.3%) | **5533 ms (64.9%)** |
| CCA conv/state (**replicated**) | 879 ms (24.3%) | 884 ms (10.4%) |
| dense GEMM (rocBLAS/Tensile, skinny decode) | 792 ms (21.9%) | 696 ms |
| MoE expert GEMM (Triton fp8 `fused_moe_kernel`) | 666 ms (18.4%) | 650 ms |
| elementwise/reduce/cat (pointwise) | 378 ms (10.5%) | 308 ms |
| memcpy/copy | 305 ms | 256 ms |
| router/norm/topk | 105 ms | 82 ms |
| paged attention decode | 50 ms | 49 ms |
| fp8 activation quant | 24 ms | 23 ms |
| **total device-kernel time** | **3614 ms** | **8522 ms** |

(594,093 launches/rank, 118 distinct kernels. Both ranks run in lockstep, so the
~12 s combined ≈ the wall; rank1's extra ~4.9 s vs rank0 is almost entirely
all-reduce *wait*.)

## Reading — the TP=2 story

1. **All-reduce dominates and is wildly imbalanced** — 64.9% of rank1 vs 10.3%
   of rank0 (`ncclDevKernel_Generic_4`, 10,660 calls/rank). At decode batch 16
   the per-collective payload is tiny, so it's **latency-bound**, AND the kernel
   duration includes spin-wait at the barrier: rank1 reaches each collective
   first and waits on rank0. Net: the MoE/MoD all-reduce (every layer, every
   step) is the single biggest TP cost (~5.9 s combined). This reproduces the
   prior session's "rank1 COMM ≈ 61%" observation that motivated the (not-yet-
   wired) heterogeneous-TP work in `vllm/distributed/het_tp.py`.
   - Levers: het-TP proportional split to close the sync bubble; QUICK_REDUCE /
     SYMM_MEM all-reduce backend (vLLM listed them available but picked PYNCCL);
     batch more decode work per collective.
   - Caveat: rank0 is the engine **driver** (sampling/detok/coordination), so
     part of "rank0 arrives late" may be host-side driver work, not CU imbalance.
     Worth confirming before attributing it all to the 64:56 asymmetry.

2. **CCA is the #1 compute bucket and pays no TP dividend** — 24% of rank0
   (`cca_decode_qk_kernel` 348 ms alone + conv/state). It's **replicated**: each
   rank runs full-size CCA (warning at zaya.py:793), so its absolute cost equals
   single-GPU while everything else (MoE) halves — making it relatively larger
   under TP. Real CCA sharding (deferred upstream) is the only fix.

3. **MoE + dense GEMM roughly halve per rank** as expected from expert sharding;
   together still ~40% of compute. The skinny Tensile decode GEMMs
   (`Cijk_..._MT16x16x32`, batch-16) remain a TunableOp target.

vs the single-GPU FP8 baseline (`README.md`): MoE drops 25%→18% and CCA rises
3.5%→24% (relative), and an entirely new ~all-reduce bucket appears as the
top cost. **Conclusion: TP=2 decode perf is gated by the small-batch all-reduce
+ replicated-CCA, not by the expert GEMMs.**
