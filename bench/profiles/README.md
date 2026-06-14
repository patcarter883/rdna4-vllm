# ZAYA1-8B FP8 decode profile — gfx1201 (RX 9070 XT), standard kernels

`rocprofv3 --kernel-trace --stats` of an offline single-process decode
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`, graph mode, 16 seqs × 512 tokens,
`vllm-gfx1201:latest`, FP8 ZAYA on the Triton FP8 MoE backend, AITER off).
285 tok/s aggregate; 19.0s GPU-kernel time over 28.8s wall.
Raw: `zaya-fp8-gfx1201-decode-kernel_stats.csv`.

## Where the GPU time goes (grouped)

| % | calls | bucket |
|---:|---:|---|
| 27.0 | 1,418,461 | **elementwise / reduce / cat (pointwise)** |
| 25.4 | 42,320 | **MoE expert GEMM** (Triton fp8 `fused_moe_kernel`, 114 µs avg) |
| 25.1 | 209,877 | dense GEMM (rocBLAS/Tensile: attn qkv/o, router down_proj, lm_head) |
| 7.8 | 111,959 | router/norm fused (Triton softmax+rmsnorm+gather, 55 µs avg) |
| 3.5 | 21,840 | CCA conv/state (`roll_cuda` + a **naive_conv** fallback, 306 µs avg) |
| 3.0 | 77,450 | memcpy/copy |
| 1.8 | 20,800 | paged attention decode |
| 1.1 | 21,164 | MoE top-k select |
| 0.8 | 42,320 | fp8 activation quant |

## Reading

- Decode is **memory-/latency-bound**, consistent with the power measurement
  (100% "use" at ~130 W / 340 W). No single kernel dominates — three ~25%
  buckets.
- **1.4 M tiny pointwise launches** (~2,770 per decode step) is the standout.
  It's 27% of GPU time *and* the main cause of the 19.0s-GPU-vs-28.8s-wall gap
  (~10s of launch/dispatch bubbles). Driven by: torch.compile fusion passes
  mostly OFF in this config (`fuse_norm_quant`/`fuse_act_quant` False,
  `custom_ops:['none']`) and the CCA custom op being a graph-break
  (`splitting_ops` includes `vllm::cca`) so its internal elementwise ops run
  unfused.

## Standard-kernel levers (no custom W4A8 kernel)

1. **Fuse the pointwise tail (biggest, most addressable).** Enable the
   compile fusion passes and/or vLLM fused norm/quant custom ops; reduce CCA
   graph breaks or give CCA a fused state-update kernel. Cuts both memory
   round-trips and launch-bubble wall time.
2. **Tune the Triton `fused_moe` config for gfx1201.** vLLM's
   `get_moe_configs` falls back to heuristic block sizes when no tuned JSON
   exists for the device+shapes; a gfx1201-tuned config (block_m/n/k,
   num_warps, num_stages) targets the 25% MoE bucket.
3. **TunableOp for the dense GEMMs (25%).** They are skinny decode GEMMs at
   batch 16 on small Tensile macro-tiles (16×16×32). `PYTORCH_TUNABLEOP_ENABLED=1`
   auto-selects better rocBLAS/hipBLASLt algos per shape.
4. **CCA `naive_conv` fallback** (306 µs/call) — a tuned/explicit causal-conv
   path would remove an unoptimized MIOpen fallback (small % but wasteful).

Orthogonal (memory-bound root cause): fewer expert-weight bytes/token → the
W4A8 path, tracked separately as the server sessions' kernel.
