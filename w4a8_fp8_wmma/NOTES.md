# W4A8-FP8 WMMA kernel — design notes & status

Research kernel: expand packed int4 weights to **fp8 e4m3 in-register** and feed
RDNA4's **FP8 WMMA** units, vs the current int4→f16→f16-WMMA path. See the
approved plan at `~/.claude/plans/while-this-project-is-greedy-stroustrup.md`.

## Status

- **v0 scalar fp8 reference — DONE + VALIDATED on gfx1201 (RX 9070 XT).**
  `test_correctness.py` passes 4 shapes (up to 256×4096×4096), mean abs error
  ~3e-4 vs a same-reduction-order fp8 model. Confirms the core claim: int4→fp8 is
  lossless for symmetric [-8,7]; activation→fp8 is the only loss. ISA check
  confirms `v_cvt_pk_fp8_f32` + `v_cvt_f32_fp8` are emitted (real RDNA4 fp8 HW
  convert path, not emulation). This is the numerical golden the WMMA path must
  match.
- **v1 rocWMMA fp8 16x16x16 — WORKING + VALIDATED on gfx1201.** Passes all
  correctness shapes vs v0 (mean abs ~1e-4). ISA confirms
  **`v_wmma_f32_16x16x16_fp8_fp8`** is emitted (real FP8 matrix hardware) plus the
  `v_cvt_pk_fp8_f32` int4→fp8 in-register expansion. Full concept proven
  end-to-end. rocWMMA `float8_t` fragments handle the gfx1201 lane layout — no
  manual int2 layout needed for correctness.
  - **Build gotcha (fixed):** torch cpp_extension injects
    `-D__HIP_NO_HALF_CONVERSIONS__=1` / `-D__HIP_NO_HALF_OPERATORS__=1`, which
    break rocWMMA's `static_cast<__half>(0.0f)`. Fix: `#undef` both at the very
    top of the .hip before any include.
  - **PERF: NOT YET OPTIMIZED.** v1 is correctness-first: 1 warp/block, 16x16
    tiles, store_matrix→LDS + scale-accumulate every group. Measured ~3 TFLOP/s on
    4096³ vs torch hipBLASLt f16 at ~78–134 TFLOP/s. The gap is occupancy/tiling,
    not the concept. Optimization path = adopt the gfx1151 MMQ structure (64x64
    tiles, 4 warps, keep f32 acc in registers, apply group scale without the LDS
    round-trip, reuse activation tile across N-subtiles).

## Build & test (on the gfx1201 host)

```bash
source /home/pat/code/vllm-rocm714-gfx1250/activate-build-env.sh   # sets GPU_ARCHS=gfx1201, ROCM_PATH
cd /home/pat/code/vllm-gfx1201/w4a8_fp8_wmma                       # the source of truth (in-repo)
python setup.py build_ext --inplace
python test_correctness.py        # exercises v0 on gfx1201
```

Expect first build to JIT/compile for gfx1201 (set `GPU_ARCHS=gfx1201` so the iGPU
gfx1036 isn't enumerated — fp8 cvt fails to compile on it; see memory
`gfx1201-enablement`).

## The FP8 WMMA builtin (confirmed in ROCm 7.14 clang)

```cpp
// gfx1200 / gfx1201, wave32, 16x16x16, FP8(e4m3) x FP8(e4m3) -> F32:
//   A: int2  (2x int32 = 8 fp8 bytes / lane)
//   B: int2
//   C: float8 (8x f32 / lane)   D = A*B + C
float8 d = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(int2 a, int2 b, float8 c);
```
(rocWMMA wraps it: `rocwmma/internal/wmma_impl.hpp:1821`, gated to
`AMDGCN_ARCH_ID_GFX1200/1201`. Also `fp8_bf8`, `bf8_fp8`, `bf8_bf8` variants.)

**The crux:** this packs **int2 = 8 fp8/lane**, a *different* lane layout from the
gfx1151 INT8 reference (`int32x4 = 16 int8/lane`). gfx12 splits the K=16 dim across
lane pairs rather than replicating it RDNA3-style. Getting the lane→element mapping
wrong = silent wrong results.

### v1 strategy (correctness first)

The int4→fp8 expansion happens at **LDS-store time** (when we unpack a nibble and
write its fp8 byte to the LDS tile) — it is *orthogonal* to how the WMMA core reads
fragments. So:

1. First correct version: stage fp8 bytes into an LDS tile, then use **rocWMMA
   fragments** (`rocwmma::fragment<..., rocwmma::float8_t, ...>` +
   `load_matrix_sync` / `mma_sync`) to read them — rocWMMA encodes the correct
   per-dtype gfx1201 layout, so we don't reverse-engineer int2 by hand.
2. Then optimize: replace rocWMMA load+mma with the raw builtin once the layout is
   confirmed by an on-device probe (the gfx1151 ref validated its layout "by
   probe" — do the same: feed identity-ish tiles, read back, infer the mapping).

The fused cheap int4→fp8 expansion is kept in both versions.

### int4 → fp8 e4m3 expansion options (benchmark on-device)

- (a) **hardware convert** (what v0 uses): `(float)(nibble - zp)` →
  `__builtin_amdgcn_cvt_pk_fp8_f32`. Exact for [-8,7]. A few VALU ops, packs 2 at a
  time. Simple, robust.
- (b) **16-entry LUT** via `v_perm_b32` byte-permute (the AMD analogue of NVIDIA
  marlin's `prmt` trick). For symmetric (zp=8 const) the nibble→fp8-byte table is
  compile-time fixed. Potentially fewer ops.
- (c) **precompute at repack time** (Phase A weight repack): store the remapped
  nibble so runtime expansion is minimal. Reuse `marlin_int4_fp8_preprocess`
  (`csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu`) zp-fold logic.

## Findings from reference kernels/blogs (2025) — implications for v1

- **HazyResearch "AMD brr" blog** (CDNA-focused but transferable):
  - AMD matrix-instruction register layouts are "less structured"; **no single
    swizzle works for all layouts** — fp8 and int4 need *separate* swizzle logic.
    Reinforces: validate fp8 fragment layout empirically on gfx1200/1201.
  - **On wave32, wave/warp specialization underperforms** (static register
    allocation). → Do **not** port a producer/consumer warp-specialized design to
    RDNA4. The simpler MMQ structure (single-buffered LDS, like the gfx1151 ref
    which found double-buffering *regressed*) is the right call. Consider the
    **8-wave ping-pong** (alternating memory/compute clusters) if we need overlap.
  - Match LDS swizzle to the actual `ds_read/ds_write` granularity (e.g.
    `ds_read_b128` spans 4 phases / 64 banks).
  - Budget VGPRs: consumer waves can't recoup spilled registers. Good news for us —
    fp8 frags (`int2`) are *half* the register footprint of the int8 path's
    `int32x4`, so register pressure is lower than the int8 baseline.
- **HipKittens** (`github.com/HazyResearch/HipKittens`): tile-primitive C++ DSL,
  but **CDNA3/CDNA4 only** (gfx942/gfx950) as of now — no RDNA4 path. Useful as a
  design reference (tiles sized to matrix units, bank-conflict-free coalesced ops,
  8-wave ping-pong / 4-wave interleave scheduling), not directly usable on gfx1201.
- **Modular `warp_spec_matmul.mojo`**: clean producer/consumer + ring-buffer +
  LDS-pipeline design, MMA 16x16x16, swizzle on smem store, fp8-capable if the tile
  operator implements it. But it's the warp-specialized pattern the blog warns
  against for wave32 — treat as CDNA-oriented inspiration, not a template.

## Open questions to resolve on-device

1. Exact int2 lane→element mapping for `..._fp8_fp8_w32_gfx12` (probe or rocWMMA).
2. Which int4→fp8 expansion (a/b/c) is fewest-VALU without hurting occupancy.
3. Best tile shape for gfx1201 (64 CU, 64 KB LDS, wave32). Start from the gfx1151
   `MMQ_X=64, MMQ_Y=64, NWARPS=4` and retune.
4. Does fp8-WMMA beat int8-WMMA here on *accuracy* at comparable throughput? (Both
   are 2×-fp16 rate on RDNA4 — that's the whole research question.)
5. ISA check: confirm `v_wmma_f32_16x16x16_fp8_fp8` is emitted (`llvm-objdump`).
```
