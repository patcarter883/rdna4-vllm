# ParoQuant pairwise-rotation kernel — RDNA4 (gfx1201) port

Faithful HIP port of ParoQuant's inference hot path — the pre-GEMM **scaled pairwise (Givens)
rotation** of the activation — with a self-contained correctness harness. No torch / no vLLM.

Upstream: [`z-lab/paroquant`](https://github.com/z-lab/paroquant)
`paroquant/kernels/cuda/rotation.{cu,cuh}` (© 2025 Haisheng Chen, MIT).

## What the kernel does (recap)

Per linear layer, before a standard INT4 matmul, ParoQuant rotates the **activation** `x`:
load a `GROUP_SIZE` (128/64) channel block into LDS applying a per-channel scale, then apply
`KROT` (default 8) rounds of `GROUP_SIZE/2` disjoint within-group Givens rotations
(`y_i = x_i cosθ + x_j sinθ`, `y_j = −x_i sinθ + x_j cosθ`), `__syncthreads` between rounds.
Rotations are **block-diagonal within each group** → cheap, and input-dim TP-shardable.

Params (per projection, tiny): `theta [KROT, H/2]`, `idx_ij [KROT, H]` int16 (read as packed
int32 → `(i_low, j_high)`, **local** channel indices), `channel_scales [H]`.

## Port (`rotation_hip.hip`)

Mechanical CUDA→HIP: headers (`cuda_*` → `hip/hip_*`), `__nv_bfloat16{,2}` → `__hip_bfloat16{,2}`,
`__floats2bfloat162_rn` → `__float22bfloat162_rn`. The bf16 *packing* intrinsics are kept native;
only the lone scalar `hmul` (scale-load) is routed through float. `__sincosf`, `fmaf`, the half
path, and all index arithmetic are byte-for-byte the upstream logic. `CTA_M=4` (upstream fixes it).

## Correctness methodology

Layout bugs are the failure class a naive test misses: if the reference re-reads the *packed*
buffers with the kernel's own offset expressions, a transposed/off-by-one layout is self-consistent
and passes — and forward∘inverse returns identity even when the layout is consistently wrong. So:

- Data generated in **logical** form: per `[round][group][pair]` a `(i_local, j_local, angle)`,
  per global channel a scale, per `[row][channel]` an x.
- One **packer** encodes the layout contract (`theta [krot,h/2]`, `idx_ij [krot,h]`, `scale [h]`)
  into flat buffers — cross-checked against the upstream vLLM plugin's param shapes.
- The **CPU reference is computed from the logical form**, never from the packed buffers.
- **Test A** (forward kernel vs CPU ref) validates layout+math; **Test B** (forward∘inverse ≈
  identity) validates orthogonality. A catches layout bugs B cannot.
- Within-round pairs are asserted to be a perfect matching (kernel has no intra-round sync).
- Numerics matched to GPU: round-to-dtype between rounds and on the scaled load; angles rounded to
  dtype (kernel stores theta in scalar_t). Covers half/bf16 (packed path) + float (scalar path),
  GS∈{128,64}, KROT∈{1,8} (the only variants upstream compiles), and M not divisible by CTA_M.

## Result (gfx1201, RX 9070) — ALL PASS

| dtype | forward vs CPU ref (max_abs) | forward∘inverse (max_abs) |
|-------|------------------------------|---------------------------|
| float | **2.4e-6**                   | 2.1e-6                    |
| half  | 2.0e-3 (1–2 ulp)             | 4.9e-3                    |
| bf16  | 1.6e-2 (1–2 ulp)             | 3.9e-2                    |

The **float ≈ 2e-6** result is the layout proof: a layout bug would be gross error there, not
float-epsilon. half/bf16 deltas are pure `__sincosf` + mantissa rounding over 8 rounds. (Large
`max_rel` values are Givens outputs near zero — gated on abs error.)

## Build / run

```
/opt/rocm/bin/hipcc --offload-arch=gfx1201 -O3 -I. rotation_hip.hip -o /tmp/rot_test   # CPU, no lease
scripts/gpu-lease.sh -n 1 -- /tmp/rot_test                                             # run under lease
```

## Status & next steps

- [x] Kernel ported, compiles clean on gfx1201, numerically validated (layout + orthogonality).
- [ ] **Real-weights check (deferred):** `z-lab/Qwen3.6-27B-PARO` not in local HF cache (only the
      AWQ-INT4 / DFlash variants are) — load one layer's real `theta`/`pairs`/`channel_scales` and
      run forward∘inverse before wiring a GEMM. Skipped to avoid a download.
- [ ] **GEMM wiring (the actual goal).** Two paths:
  1. *Easy A/B:* rotated activation → existing AWQ-int4 (W4A16) path → ParoQuant's +2.4%-over-AWQ
     accuracy on the served 27B, off the fp8 fast path. Upstream uses `AWQMarlinLinearMethod`
     (NVIDIA Marlin — does **not** exist on RDNA4), so the plugin's GEMM call must be swapped for
     this stack's AWQ-int4 path. Weight format is standard AWQ (`PackedvLLMParameter` qweight/qzeros
     + `GroupQuantScaleParameter`), already consumed here.
  2. *The synergy:* rotation as a front-end to the **W4A8 fp8-WMMA** kernel — keep the fast path and
     add outlier suppression (rotate→fp8) that could lift W4A8 accuracy / open W4A4. Needs measuring.
```
