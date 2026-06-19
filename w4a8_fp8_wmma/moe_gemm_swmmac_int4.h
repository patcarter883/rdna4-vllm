// moe_gemm_swmmac_int4.h — grouped (MoE) 2:4-sparse INT4 + per-group-scale SWMMAC GEMM.
//
// The PRODUCTION 35B rung of the SWMMAC sibling (TILE_FRAMEWORK_DESIGN §2.5 step 3 / "phase 3b").
// It extends moe_gemm_swmmac.h (the fp8/per-channel first rung, self-consistency-validated bit-
// exact) with the two pieces the real 35B MoE needs:
//   * weight = 2:4-compressed *INT4* (E, N, K/4 bytes), not fp8: 8 kept nibbles per int32 word are
//     unpacked to e4m3 with a per-(channel,group) ZERO-POINT (asymmetric `w_zeros`, or symmetric
//     zp=8 when null). ~3 bit/wt vs the fp8 rung's ~5 — the decode-bandwidth win.
//   * per-GROUP weight scale (E, N, K/group_size), folded IN-REGISTER per group exactly like the
//     WMMA tiled spine (moe_gemm_tiled.h), replacing the fp8 rung's single per-channel epilogue
//     scale. group_size must be a multiple of 32 (one SWMMAC K-tile); the 35B is g32 -> one
//     tile == one group.
// Everything else is the SHARED SWMMAC dataflow + grouped-MoE spine of moe_gemm_swmmac.h
// (per-expert weight slabs via expert_ids, routed activation gather via sorted_token_ids,
// per-row act-scale + fp16/scatter epilogue). The operand offsets + per-lane index extraction are
// the VALIDATED recipe of RESEARCH_swmmac.md §3 / feat/swmmac-microbench (do NOT re-derive); the
// int4->fp8 unpack mirrors the VALIDATED bench_swmmac_int4.hip::sparse_int4. Requires N%16==0,
// K%group_size==0, group_size%32==0.
//
// VALIDATION: self-consistency only (no pruned MoE checkpoint exists) — grouped-SWMMAC-int4 vs a
// scalar dense reference on the SAME 2:4-zeroed dequantised weights (bench_swmmac_grouped_int4.hip).
// No served result is possible (35B AWQ experts are structureless under 4:2).
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "tile_config.h"

namespace w4a8_tile {

using v4i_t = int __attribute__((ext_vector_type(4)));   // 16 fp8 (dense-B, one K-tile half)

// Unpack 8 signed-int4 nibbles (one int32 word, given zero-point) -> v2i_t of 8 e4m3 bytes.
// (nibble - zp) in [-15,15] is exact in e4m3, so this round-trips the integer exactly — identical
// to bench_swmmac_int4.hip::i4x8_to_fp8 but with a runtime zp (asymmetric) instead of the fixed 8.
__device__ __forceinline__ v2i_t i4x8_to_e4m3(int packed, int zp) {
    union { v2i_t v; unsigned char b[8]; } o;
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        o.b[j] = int4_signed_to_e4m3(((packed >> (4 * j)) & 0xF) - zp);
    return o.v;
}

// One block computes a 16-wide N-tile (blockIdx.x) for ONE expert's block of `block_m` padded rows
// (blockIdx.y). NWARP = block_m/16 warps, each owning 16 token rows. Identical launch geometry,
// operand offsets, index extraction and output map to moe_gemm_swmmac_kernel — the ONLY changes vs
// the fp8 rung are (a) weight is int4 unpacked with zp, (b) scale folds per group in-register.
template<bool SCATTER, int NWARP>
__global__ void __launch_bounds__(NWARP * 32) moe_gemm_swmmac_int4_kernel(
    const unsigned char* __restrict__ x_fp8,         // (T, K) e4m3 activations
    const float*         __restrict__ act_scales,    // (T,)
    const unsigned char* __restrict__ w_cmp_i4,      // (E, N, K/4) 2:4-compressed int4 weights
    const int*           __restrict__ w_idx,         // (E, N, K/32) compression index
    const __half*        __restrict__ w_scales,      // (E, N, K/group_size) per-group scale
    const int*           __restrict__ w_zeros,       // (E, N/8, K/group_size) packed zp, or null
    const int*           __restrict__ sorted_token_ids,  // (P,)
    const int*           __restrict__ expert_ids,        // (P/block_m,)
    const int*           __restrict__ num_tokens_post_padded,
    __half*              __restrict__ out,           // (P, N) non-scatter
    int T, int N, int K, int group_size, int top_k, int block_m, int num_valid_tokens,
    float*               __restrict__ output_scatter,   // (M, N) fp32 scatter
    const float*         __restrict__ topk_weights,     // (M*top_k,) scatter
    int out_top_k) {

    const int KT = K / 32;                       // SWMMAC K-tiles (16x16x32)
    const int num_groups = K / group_size;
    const int tiles_per_group = group_size / 32; // group_size is a multiple of 32 (g32 -> 1)
    const int block_idx = blockIdx.y;
    const int row0 = block_idx * block_m;
    if (row0 >= num_tokens_post_padded[0]) return;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int e = expert_ids[block_idx];
    const unsigned char* Wc_e = w_cmp_i4 + (size_t)e * N * (K / 4);
    const int*           idx_e = w_idx   + (size_t)e * N * KT;
    const __half*        ws_e  = w_scales + (size_t)e * N * num_groups;
    const int*           wz_e  = (w_zeros != nullptr)
                                  ? w_zeros + (size_t)e * (N / 8) * num_groups : nullptr;

    // m = padded token-row this lane contributes to (load-m == output-m, both use lane&15).
    const int m = row0 + warp * 16 + (lane & 15);
    bool m_valid = false; int src = 0, offs = -1;
    if (m < row0 + block_m) {
        offs = sorted_token_ids[m];
        if (offs < num_valid_tokens) { m_valid = true; src = SCATTER ? m : (offs / top_k); }
    }
    const int src_l = m_valid ? src : 0;   // clamp the activation LOAD row (output guarded below)

    // sparse-A weight row this lane loads (== its output column; load layout uses lane&15).
    const int nrow = blockIdx.x * 16 + (lane & 15);
    const int n    = blockIdx.x * 16 + (lane >> 4) * 8;   // this lane's 8 OUTPUT columns: n..n+7

    v8f_t running = v8f_t{0, 0, 0, 0, 0, 0, 0, 0};
    for (int g = 0; g < num_groups; ++g) {
        // per-(channel,group) zero-point for this lane's weight row (symmetric 8 when no w_zeros).
        int zp = 8;
        if (wz_e != nullptr) {
            const int pk = wz_e[(size_t)(nrow / 8) * num_groups + g];
            zp = (pk >> ((nrow % 8) * 4)) & 0xF;
        }
        v8f_t acc = v8f_t{0, 0, 0, 0, 0, 0, 0, 0};
        for (int tg = 0; tg < tiles_per_group; ++tg) {
            const int t = g * tiles_per_group + tg;
            const int packed = *reinterpret_cast<const int*>(
                &Wc_e[(size_t)nrow * (K / 4) + t * 8 + (lane >> 4) * 4]);   // 8 compressed int4
            v2i_t a = i4x8_to_e4m3(packed, zp);                            // -> 8 e4m3
            v4i_t b = *reinterpret_cast<const v4i_t*>(
                &x_fp8[(size_t)src_l * K + t * 32 + (lane >> 4) * 16]);     // 16 dense fp8
            const int id = (int)(((unsigned)idx_e[(size_t)nrow * KT + t] >> ((lane >> 4) * 16)) & 0xFFFFu);
            acc = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, acc, id);
        }
        // fold this group's per-channel scale: acc[ee] maps to output column n+ee.
        #pragma unroll
        for (int ee = 0; ee < 8; ++ee) {
            if (n + ee >= N) continue;
            running[ee] += acc[ee] * __half2float(ws_e[(size_t)(n + ee) * num_groups + g]);
        }
    }

    // epilogue: per-row act scale, then fp16 store or indirect atomic scatter (the shared spine).
    if (!m_valid) return;
    const float asc = act_scales[src];
    #pragma unroll
    for (int ee = 0; ee < 8; ++ee) {
        if (n + ee >= N) continue;
        const float val = asc * running[ee];
        if constexpr (SCATTER) {
            const int token = offs / out_top_k;
            const float w = topk_weights[offs];
            atomicAdd(&output_scatter[(size_t)token * N + (n + ee)], w * val);
        } else {
            out[(size_t)m * N + (n + ee)] = __float2half(val);
        }
    }
}

}  // namespace w4a8_tile
