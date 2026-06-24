# gdn_hip — native HIP kernels for Gated Delta Net (gfx1201)

A **standalone, framework-agnostic** `torch.ops` extension that replaces the flash-linear-attention
**Triton** GDN kernels with **AOT-compiled HIP** — so the Qwen3-Next / Qwen3.5 / Qwen3.6 GDN
linear-attention path **never JIT/autotunes a Triton kernel again** (compile once, run any shape).
Shared by `minisgl-rdna4` and `vllm-gfx1201` (both route GDN through the same fla recurrence): build
one `.so`, call `torch.ops.gdn_hip.*` from either.

## Ops (replace ~15 Triton kernels)
| op | replaces | notes |
|---|---|---|
| `gdn_decode` | fused_sigmoid_gating decode SSM | 1 token/seq; g,β computed inline; state updated in place |
| `gdn_prefill` | `chunk_gated_delta_rule` | recurrent (correct-first); chunked-HIP optimization is future work |
| `causal_conv1d_update` | mamba conv decode | depthwise causal conv + state roll + SiLU |
| `causal_conv1d_fwd` | mamba conv prefill | varlen depthwise causal conv + state write |
| `rmsnorm_gated` | RMSNormGated | norm-before-gate, SiLU(z) |

Math is lifted verbatim from `fla/ops/fused_recurrent.py` (the recurrence) — the chunked Triton path
is just a throughput optimization of this exact rank-1 update:
`S*=exp(g); v-=S@k; v*=β; S+=outer(v,k); o=S@q` (q,k l2-normed, q scaled 1/√K).

## Build (AOT — no GPU needed to compile)
```bash
# in the combined ROCm image (hipcc + PYTORCH_ROCM_ARCH=gfx1201)
cd gdn_hip && GPU_ARCHS=gfx1201 python setup.py build_ext --inplace
```
Produces `gdn_hip_C*.so`; `import gdn_hip` loads it and registers `torch.ops.gdn_hip.*` (+ fake/meta
impls for torch.compile safety).

## Status
- [x] **Builds AOT for gfx1201** (hipify clean, links torch_hip) — CPU-only build verified.
- [x] **All 6 ops load + register** with valid schemas (CPU).
- [x] **Numeric parity** vs torch reference (`tools/gdn_hip_parity.py`): ALL PASS, max|Δ|~1e-7.
- [x] **Wired into `gdn/layer.py`** (recurrent path) — **serves 4B TP1/TP2 + 35B TP2 coherent** on
      2× gfx1201 (2026-06-24). The Triton GDN compile cliff is gone.
- [x] **Chunked prefill (`gdn_prefill_chunked`)** — numerically correct (max|Δ|~1e-7 vs recurrent),
      but **~4× SLOWER** as a scalar per-row kernel (`tools/gdn_hip_bench.py`: 0.23–0.29× at
      T=256..16384). The serve uses the **recurrent** `gdn_prefill`; the chunked op is kept as a
      validated reference. **The throughput win needs a WMMA/matrix-core formulation** of the
      intra-chunk `KK`/`KQ`/solve matmuls — that is the real "fast path", future work.
- [ ] WMMA chunked prefill (the actual long-context speedup).
- [ ] Delete the (now unused) Triton GDN tree (`gdn/{fla,mamba}`); bf16-native state (v1 = fp32).
