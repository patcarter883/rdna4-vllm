// tile_config.h — RDNA4 (gfx1201) W-quant-A8 tile primitives + MMA policy.
//
// Step 1 of the tile-framework (TILE_FRAMEWORK_DESIGN.md): factor the shared spine of the
// grouped-MoE kernels (moe_kernel.hip v5/v6/v7) out of the per-version launchers, and put the
// inner matrix-core op behind a *policy* so a second backend (SWMMAC, step 2) slots in without
// touching the staging/epilogue. This header is the single source of the device helpers that are
// currently copy-pasted between w4a8_fp8_wmma_kernel.hip and moe_kernel.hip.
//
// SCOPE: WMMA (dense fp8) only. The SWMMAC policy is declared but intentionally NOT implemented
// here — its sparse-A operand + compression index (RESEARCH_swmmac.md §3, feat/swmmac-microbench)
// change the *B-staging*, not just the inner op, so abstracting it now (before building against
// the real grouped SWMMAC kernel) would be premature. The extension point is marked below.
#pragma once

#include <hip/hip_runtime.h>

namespace w4a8_tile {

// ---- vector aliases (match moe_kernel.hip / w4a8_fp8_wmma_kernel.hip) ----
using v2i_t = int   __attribute__((ext_vector_type(2)));   // 2x int32 = 8 fp8
using v4i_t = int   __attribute__((ext_vector_type(4)));   // 4x int32 = 16 fp8
using v8f_t = float __attribute__((ext_vector_type(8)));   // WMMA/SWMMAC f32 accumulator
using v2f_t = float __attribute__((ext_vector_type(2)));

// ---- quant constants (single source of truth) ----
constexpr int   PACK_FACTOR    = 8;       // int4 nibbles per int32 word
constexpr float E4M3_MAX       = 448.0f;
constexpr int   WMMA_DIM       = 16;      // dense WMMA M/N tile
constexpr int   MAX_GROUP_SIZE = 128;
constexpr int   LDS_PAD        = 8;       // bank-conflict pad on the K stride
constexpr int   MAX_WARPS      = 8;       // block_m <= 128 -> <= 8 warps -> <= 256 threads
constexpr int   MAX_THREADS    = MAX_WARPS * 32;   // __launch_bounds__ (matches v5/v6 = 256)

// ---- fp8 (e4m3) <-> f32, int4 -> fp8 (lower to v_cvt_pk_fp8_f32 / v_cvt_f32_fp8 on gfx12) ----
__device__ __forceinline__ unsigned char f32_to_e4m3(float v) {
    int packed = __builtin_amdgcn_cvt_pk_fp8_f32(v, v, 0, false);
    return static_cast<unsigned char>(packed & 0xFF);
}
__device__ __forceinline__ float e4m3_to_f32(unsigned char b) {
    return __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(b), 0);
}
// (nibble - zp) in [-15,15] is exact in e4m3, so float->e4m3 round-trips the integer exactly.
__device__ __forceinline__ unsigned char int4_signed_to_e4m3(int signed_val) {
    return f32_to_e4m3(static_cast<float>(signed_val));
}

// ============================================================================
// MMA policy concept. A backend provides:
//   using AFrag, BFrag, CFrag;            // per-lane operand / accumulator fragments
//   static constexpr int K_STEP;          // logical K consumed per mma() call
//   static __device__ CFrag zero();
//   static __device__ CFrag mma(AFrag a, BFrag b, CFrag c);   // c += a * b
// The grouped kernel is templated on the policy; staging/scale-fold/epilogue stay shared.
// ============================================================================

// Dense fp8 WMMA, 16x16x16, wave32 (what every grouped-MoE kernel runs today).
struct WmmaFp8 {
    using AFrag = v2i_t;     // 8 fp8 (this lane's K16 slice of one M-row)
    using BFrag = v2i_t;     // 8 fp8 (this lane's K16 slice of one N-col)
    using CFrag = v8f_t;     // 8 f32 partials
    static constexpr int K_STEP = WMMA_DIM;   // 16

    static __device__ __forceinline__ CFrag zero() {
        return CFrag{0, 0, 0, 0, 0, 0, 0, 0};
    }
    static __device__ __forceinline__ CFrag mma(AFrag a, BFrag b, CFrag c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
    }
};

// --- SWMMAC is a SIBLING kernel, NOT a policy on this interface (recipe = RESEARCH_swmmac.md §3).
// Reading the committed swmmac_op.hip showed the dataflow differs from WMMA, so the WmmaFp8-style
// `MMA::mma(a,b,c)` policy CANNOT express it cleanly (see TILE_FRAMEWORK_DESIGN.md §2.3 correction):
//   * operand ROLES swap: the WEIGHT is the sparse-A operand (2:4-compressed fp8 + per-lane idx,
//     read straight from DRAM = half the bytes); the ACTIVATION is the dense-B. WMMA here has
//     A=activation, B=weight.
//   * NO int4->fp8 LDS expansion — the compressed fp8 weight is consumed directly.
//   * the builtin takes an extra per-lane compression index:
//       __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a_sparse, b_dense, c, idx)  // K=32
// So the SWMMAC grouped kernel reuses the SPINE (MoE contract, group-scale fold, epilogue/scatter,
// LDS/tiling helpers, autotuner) but re-authors the GEMM loop. It lives in its own
// moe_gemm_swmmac.h, not as a SwmmacFp8 policy here.

// ============================================================================
// TileConfig — the knobs scattered across moe_kernel.hip's getenv sites
// (VLLM_W4A8_MOE_{BN,GTILE,BLOCK_M,GEMV_*}) collected into one struct. Compile-time fields are
// template params on the kernel; runtime fields are launch-time. One autotune harness sweeps
// these and caches the per-shape winner (folds crossover_cache.json + profile_crossover.py).
// ============================================================================
struct TileConfig {
    int  BN       = 64;    // N-tile width (BN/16 = independent accumulators/warp)
    int  GTILE    = 4;     // groups staged per __syncthreads (1 = v5; >1 = v6 multi-group)
    int  BLOCK_M  = 64;    // moe_align block size (rows/block, mult of 16)
    bool A_IN_LDS = false; // v5 stages A in LDS; v6 gathers A via warp-shuffle (B-only LDS)
};

// LDS K-stride for a given group/tile depth (bank-conflict padded).
__device__ __host__ __forceinline__ int lds_kstride(int group_size) {
    return group_size + LDS_PAD;
}

}  // namespace w4a8_tile
