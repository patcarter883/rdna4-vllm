// moe_gemm_swmmac.h — grouped (MoE) 2:4-sparse fp8 SWMMAC GEMM. The SWMMAC *sibling* of
// moe_gemm_tiled.h: it shares the grouped-MoE spine (per-expert weight slabs via expert_ids,
// routed activation gather via sorted_token_ids, per-row act-scale + fp16/scatter epilogue) but
// the GEMM loop is the SWMMAC dataflow — NOT a cfg.MMA policy swap (see TILE_FRAMEWORK_DESIGN §2.3):
//   * weight = the SPARSE-A operand: 2:4-compressed e4m3 (N, K/2) + per-lane index (N, K/32),
//     read straight from DRAM (half the bytes) — no int4->fp8 LDS expansion.
//   * activation = the DENSE-B operand (read 16 fp8 / lane / K-tile).
// Construction mirrors the VALIDATED feat/swmmac-microbench/swmmac_op.hip::swmmac_gemm_k
// (operand offsets + index extraction = RESEARCH_swmmac.md §3, cracked 0/256). This is the
// fp8/per-channel-scale first rung; the production int4-sparse + per-group-scale variant is the
// follow-on (see §2.5). Requires N%16==0, K%32==0.
//
// VALIDATION: self-consistency only (no pruned MoE checkpoint exists) — grouped-SWMMAC vs a dense
// reference on the same 2:4-zeroed weight (bench_swmmac_grouped.hip). No served result possible.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "tile_config.h"

namespace w4a8_tile {

using v4i_t = int __attribute__((ext_vector_type(4)));   // 16 fp8 (dense-B, one K-half)

// One block computes a 16-wide N-tile (blockIdx.x) for ONE expert's block of `block_m` padded
// rows (blockIdx.y). NWARP = block_m/16 warps, each owning 16 token rows.
template<bool SCATTER, int NWARP>
__global__ void __launch_bounds__(NWARP * 32) moe_gemm_swmmac_kernel(
    const unsigned char* __restrict__ x_fp8,         // (T, K) e4m3 activations
    const float*         __restrict__ act_scales,    // (T,)
    const unsigned char* __restrict__ w_cmp,         // (E, N, K/2) e4m3 compressed 2:4 weights
    const int*           __restrict__ w_idx,         // (E, N, K/32) compression index
    const float*         __restrict__ w_scales,      // (E, N) per-channel weight scale
    const int*           __restrict__ sorted_token_ids,  // (P,)
    const int*           __restrict__ expert_ids,        // (P/block_m,)
    const int*           __restrict__ num_tokens_post_padded,
    __half*              __restrict__ out,           // (P, N) non-scatter
    int T, int N, int K, int top_k, int block_m, int num_valid_tokens,
    float*               __restrict__ output_scatter,   // (M, N) fp32 scatter
    const float*         __restrict__ topk_weights,     // (M*top_k,) scatter
    int out_top_k) {

    const int KT = K / 32;
    const int block_idx = blockIdx.y;
    const int row0 = block_idx * block_m;
    if (row0 >= num_tokens_post_padded[0]) return;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int e = expert_ids[block_idx];
    const unsigned char* Wc_e = w_cmp + (size_t)e * N * (K / 2);
    const int*           idx_e = w_idx + (size_t)e * N * KT;
    const float*         sw_e  = w_scales + (size_t)e * N;

    // m = padded token-row this lane contributes to (load-m == output-m, both use lane&15).
    const int m = row0 + warp * 16 + (lane & 15);
    bool m_valid = false; int src = 0, offs = -1;
    if (m < row0 + block_m) {
        offs = sorted_token_ids[m];
        if (offs < num_valid_tokens) { m_valid = true; src = SCATTER ? m : (offs / top_k); }
    }
    const int src_l = m_valid ? src : 0;   // clamp the activation LOAD row (output guarded below)

    // sparse-A weight row for this lane (load layout uses lane&15).
    const int nrow = blockIdx.x * 16 + (lane & 15);

    v8f_t acc = v8f_t{0, 0, 0, 0, 0, 0, 0, 0};
    for (int t = 0; t < KT; ++t) {
        v2i_t a = *reinterpret_cast<const v2i_t*>(
            &Wc_e[(size_t)nrow * (K / 2) + t * 16 + (lane >> 4) * 8]);   // 8 compressed fp8
        v4i_t b = *reinterpret_cast<const v4i_t*>(
            &x_fp8[(size_t)src_l * K + t * 32 + (lane >> 4) * 16]);      // 16 dense fp8
        int id = (int)(((unsigned)idx_e[(size_t)nrow * KT + t] >> ((lane >> 4) * 16)) & 0xFFFFu);
        acc = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, acc, id);
    }

    // epilogue: accumulator maps to C[n][m] -> Y[m][n]; output n uses lane>>4 (NOT the load nrow).
    if (!m_valid) return;
    const int n = blockIdx.x * 16 + (lane >> 4) * 8;
    const float asc = act_scales[src];
    #pragma unroll
    for (int ee = 0; ee < 8; ++ee) {
        if (n + ee >= N) continue;
        const float val = asc * sw_e[n + ee] * acc[ee];
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
