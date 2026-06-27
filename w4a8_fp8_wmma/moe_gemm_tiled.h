// moe_gemm_tiled.h — grouped (MoE) W-quant-A8 GEMM, re-expressed over the MMA policy + shared
// tile primitives (tile_config.h). Step 1 of the tile-framework: this reproduces the dense-WMMA
// grouped-v5 kernel (moe_kernel.hip `moe_gemm_v5_kernel`) through the policy abstraction, so a
// second backend (SwmmacFp8) becomes a template argument rather than a copy-pasted kernel.
//
// CORRECTNESS: bit-exact-by-construction vs moe_gemm_v5_kernel — identical staging, identical
// accumulation order, identical epilogue; the only change is `acc = MMA::mma(...)` instead of the
// inline `__builtin_amdgcn_wmma_...`. With MMA=WmmaFp8 that lowers to the same instruction.
// STATUS: compile-verified for gfx1201 (device kernel only, no torch). GPU bit-exact +
// perf-neutrality A/B vs the current v5 launcher is the acceptance gate (TILE_FRAMEWORK_DESIGN
// §2.4) before this replaces the hand-written kernel.
#pragma once

#include <hip/hip_fp16.h>
#include "tile_config.h"

namespace w4a8_tile {

// One block computes a (block_m x BN) tile for ONE expert. block_m/16 warps (M-subtiles).
// Templated on the MMA policy and BN (BN/16 = independent accumulators per warp).
// NB __launch_bounds__(MAX_THREADS=256) MUST match the original (moe_gemm_v5_kernel uses
// __launch_bounds__(MV5_MAX_WARPS*32)). Omitting it targets the default 1024-thread block ->
// a smaller per-thread register budget -> possible spill -> a perf regression that is a refactor
// artifact, not a real difference. Same applies when templating v6/v7 (both carry it).
// WARPS_N warps split the BN columns (each owns NFRAG/WARPS_N frags) on top of the block_m/16
// M-warps, so block_m stays small (minimal MoE padding) while the block runs n_warps_m*WARPS_N
// warps -> fuller WGP occupancy + NFRAG_W (not NFRAG) accumulators/thread (less register pressure).
// Bit-exact to the WARPS_N=1 kernel: each output element keeps the same per-group accumulation.
template<class MMA, bool SCATTER, int BN, int WARPS_N = 1>
__global__ void __launch_bounds__(MAX_THREADS) moe_gemm_tiled_kernel(
    const unsigned char* __restrict__ x_fp8,         // (T, K)
    const float*         __restrict__ act_scales,    // (T,)
    const int*           __restrict__ w_packed,      // (E, N, K/8)
    const __half*        __restrict__ w_scales,      // (E, N, K/group)
    const int*           __restrict__ w_zeros,       // (E, N/8, K/group) or null
    const int*           __restrict__ sorted_token_ids,  // (P,)
    const int*           __restrict__ expert_ids,        // (P/block_m,)
    const int*           __restrict__ num_tokens_post_padded,
    __half*              __restrict__ out,           // (P, N) non-scatter
    int T, int N, int K, int group_size, int top_k, int block_m,
    int num_valid_tokens,
    float*               __restrict__ output_scatter,   // (M, N) fp32 scatter
    const float*         __restrict__ topk_weights,     // (M*top_k,) scatter
    int out_top_k, bool weight_is_e2m1) {

    static_assert(BN % WMMA_DIM == 0, "BN must be a multiple of WMMA_DIM");
    constexpr int NFRAG = BN / WMMA_DIM;             // total N accumulators across the N-warps
    constexpr int NFRAG_W = NFRAG / WARPS_N;         // N accumulators owned by each warp

    const int block_idx = blockIdx.y;               // one moe_align block (one expert)
    const int row0 = block_idx * block_m;
    const int P_valid = num_tokens_post_padded[0];
    if (row0 >= P_valid) return;

    const int block_n = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lrow = lane & 15;
    const int khalf = (lane >> 4) * 8;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;
    const int n_warps_m = block_m / WMMA_DIM;
    const int warp_m = warp_id / WARPS_N;            // which 16-row M-subtile
    const int warp_n = warp_id % WARPS_N;            // which BN/WARPS_N column slab
    if (warp_m >= n_warps_m) return;
    const int nfrag0 = warp_n * NFRAG_W;             // first global N-frag for this warp

    const int e = expert_ids[block_idx];
    const int num_groups = K / group_size;
    const int ppr = K / PACK_FACTOR;
    const int* wq_e = w_packed + (long)e * N * ppr;
    const __half* ws_e = w_scales + (long)e * N * num_groups;
    const int* wz_e = (w_zeros != nullptr)
                      ? w_zeros + (long)e * (N / 8) * num_groups : nullptr;

    const int BK = group_size;
    const int LDSBK = BK + LDS_PAD;
    extern __shared__ unsigned char smem[];
    unsigned char* A_tile = smem;                   // block_m rows
    unsigned char* B_tile = smem + block_m * LDSBK; // BN cols

    float running[NFRAG_W][8];
    #pragma unroll
    for (int f = 0; f < NFRAG_W; ++f)
        #pragma unroll
        for (int ee = 0; ee < 8; ++ee) running[f][ee] = 0.0f;

    const int total_threads = n_warps_m * WARPS_N * 32;
    for (int g = 0; g < num_groups; ++g) {
        const int k0 = g * BK;

        // ---- stage activations: A_tile[r][k] = x_fp8[src(r)][k0+k], guarded ----
        for (int idx = tid * 4; idx < block_m * BK; idx += total_threads * 4) {
            const int r = idx / BK, k = idx % BK;
            unsigned int v = 0u;
            const int offs_token = sorted_token_ids[row0 + r];
            if (offs_token < num_valid_tokens) {
                const int src = SCATTER ? (row0 + r) : (offs_token / top_k);
                v = *reinterpret_cast<const unsigned int*>(&x_fp8[(long)src * K + k0 + k]);
            }
            *reinterpret_cast<unsigned int*>(&A_tile[r * LDSBK + k]) = v;
        }
        // ---- stage weights: expand int4 -> fp8 into B_tile[n][k] ----
        const int njobs = BN * (BK / PACK_FACTOR);
        for (int j = tid; j < njobs; j += total_threads) {
            const int n = j / (BK / PACK_FACTOR), k8 = j % (BK / PACK_FACTOR);
            const int an = block_n + n;
            int word = 0, zp = 8;
            if (an < N) {
                word = wq_e[an * ppr + (k0 / PACK_FACTOR) + k8];
                if (wz_e != nullptr) {
                    const int pk = wz_e[(an / 8) * num_groups + g];
                    zp = (pk >> ((an % 8) * 4)) & 0xF;
                }
            }
            unsigned char* dst = &B_tile[n * LDSBK + k8 * PACK_FACTOR];
            #pragma unroll
            for (int jj = 0; jj < PACK_FACTOR; ++jj)
                dst[jj] = (an < N)
                    ? decode_w4_to_e4m3((word >> (jj * 4)) & 0xF, zp, weight_is_e2m1) : 0;
        }
        __syncthreads();

        typename MMA::CFrag acc[NFRAG_W];
        #pragma unroll
        for (int f = 0; f < NFRAG_W; ++f) acc[f] = MMA::zero();

        const int k_sub = BK / WMMA_DIM;
        const int a_base = (warp_m * WMMA_DIM + lrow) * LDSBK + khalf;
        const int b_base = lrow * LDSBK + khalf;
        for (int kt = 0; kt < k_sub; ++kt) {
            const int ko = kt * WMMA_DIM;
            typename MMA::AFrag a_cur =
                *reinterpret_cast<const typename MMA::AFrag*>(&A_tile[a_base + ko]);
            #pragma unroll
            for (int f = 0; f < NFRAG_W; ++f) {
                const int gf = nfrag0 + f;
                typename MMA::BFrag b_cur = *reinterpret_cast<const typename MMA::BFrag*>(
                    &B_tile[b_base + gf * WMMA_DIM * LDSBK + ko]);
                acc[f] = MMA::mma(a_cur, b_cur, acc[f]);
            }
        }
        #pragma unroll
        for (int f = 0; f < NFRAG_W; ++f) {
            const int abs_n = block_n + (nfrag0 + f) * WMMA_DIM + frag_col;
            const float wsc = (abs_n < N)
                ? __half2float(ws_e[abs_n * num_groups + g]) : 0.0f;
            #pragma unroll
            for (int ee = 0; ee < 8; ++ee) running[f][ee] += acc[f][ee] * wsc;
        }
        __syncthreads();
    }

    // ---- epilogue: per-row act scale, then non-scatter store or indirect atomic scatter ----
    #pragma unroll
    for (int ee = 0; ee < 8; ++ee) {
        const int r = warp_m * WMMA_DIM + frag_row0 + ee;
        if (r >= block_m) continue;
        const int row_pad = row0 + r;
        const int offs_token = sorted_token_ids[row_pad];
        if (offs_token >= num_valid_tokens) continue;
        const int src = SCATTER ? row_pad : (offs_token / top_k);
        const float asc = act_scales[src];
        if constexpr (SCATTER) {
            const int token = offs_token / out_top_k;
            const float w = topk_weights[offs_token];
            #pragma unroll
            for (int f = 0; f < NFRAG_W; ++f) {
                const int abs_n = block_n + (nfrag0 + f) * WMMA_DIM + frag_col;
                if (abs_n < N)
                    atomicAdd(&output_scatter[(long)token * N + abs_n],
                              w * running[f][ee] * asc);
            }
        } else {
            #pragma unroll
            for (int f = 0; f < NFRAG_W; ++f) {
                const int abs_n = block_n + (nfrag0 + f) * WMMA_DIM + frag_col;
                if (abs_n < N)
                    out[(long)row_pad * N + abs_n] = __float2half(running[f][ee] * asc);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// v6 variant: A gathered out of LDS via warp-shuffle (B-only LDS), gtile groups staged per
// __syncthreads. Reproduces moe_gemm_v6_kernel. NOTE the A-shuffle index math
// (a_src_lane = 2*(lane&15)+(lane>>4)) is WMMA-fragment-layout-specific — for the SWMMAC backend
// (K=32, different fragment layout) this gather must change, which is why the SWMMAC port starts
// from the v5 (A-in-LDS) backbone, not this one. Templated on the MMA policy for uniformity.
template<class MMA, bool SCATTER, int BN, int WARPS_N = 1>
__global__ void __launch_bounds__(MAX_THREADS) moe_gemm_tiled_v6_kernel(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const int*           __restrict__ w_packed,
    const __half*        __restrict__ w_scales,
    const int*           __restrict__ w_zeros,
    const int*           __restrict__ sorted_token_ids,
    const int*           __restrict__ expert_ids,
    const int*           __restrict__ num_tokens_post_padded,
    __half*              __restrict__ out,
    int T, int N, int K, int group_size, int top_k, int block_m,
    int num_valid_tokens,
    float*               __restrict__ output_scatter,
    const float*         __restrict__ topk_weights,
    int out_top_k, int gtile, bool weight_is_e2m1) {

    static_assert(BN % WMMA_DIM == 0, "BN must be a multiple of WMMA_DIM");
    constexpr int NFRAG = BN / WMMA_DIM;
    constexpr int NFRAG_W = NFRAG / WARPS_N;
    const int block_idx = blockIdx.y;
    const int row0 = block_idx * block_m;
    if (row0 >= num_tokens_post_padded[0]) return;

    const int block_n = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lrow = lane & 15;
    const int khalf = (lane >> 4) * 8;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;
    const int n_warps_m = block_m / WMMA_DIM;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    if (warp_m >= n_warps_m) return;
    const int nfrag0 = warp_n * NFRAG_W;

    const int e = expert_ids[block_idx];
    const int num_groups = K / group_size;
    const int ppr = K / PACK_FACTOR;
    const int* wq_e = w_packed + (long)e * N * ppr;
    const __half* ws_e = w_scales + (long)e * N * num_groups;
    const int* wz_e = (w_zeros != nullptr)
                      ? w_zeros + (long)e * (N / 8) * num_groups : nullptr;

    const int BK = group_size;
    const int LDSBK = BK + LDS_PAD;
    extern __shared__ unsigned char smem[];
    unsigned char* B_tile = smem;                 // B only -- no A in LDS

    // hoisted per-lane activation gather (constant across the K loop). WMMA-layout-specific.
    // Indexed by warp_m: the WARPS_N warps that share an M-subtile each gather the same A.
    const int a_my_row = row0 + warp_m * WMMA_DIM + (lane >> 1);
    const int a_offs = sorted_token_ids[a_my_row];
    const bool a_valid = a_offs < num_valid_tokens;
    const int a_src = SCATTER ? a_my_row : (a_offs / top_k);
    const int a_kh = (lane & 1) * 8;
    const int a_src_lane = 2 * (lane & 15) + (lane >> 4);

    float running[NFRAG_W][8];
    #pragma unroll
    for (int f = 0; f < NFRAG_W; ++f)
        #pragma unroll
        for (int ee = 0; ee < 8; ++ee) running[f][ee] = 0.0f;

    const int total_threads = n_warps_m * WARPS_N * 32;
    const int k_sub = BK / WMMA_DIM;
    const int b_base = lrow * LDSBK + khalf;
    const int njobs = BN * (BK / PACK_FACTOR);
    for (int gt = 0; gt < num_groups; gt += gtile) {
        const int gend = (gt + gtile < num_groups) ? (gt + gtile) : num_groups;
        for (int gi = gt; gi < gend; ++gi) {
            const int k0 = gi * BK;
            unsigned char* Bg = B_tile + (long)(gi - gt) * BN * LDSBK;
            for (int j = tid; j < njobs; j += total_threads) {
                const int n = j / (BK / PACK_FACTOR), k8 = j % (BK / PACK_FACTOR);
                const int an = block_n + n;
                int word = 0, zp = 8;
                if (an < N) {
                    word = wq_e[an * ppr + (k0 / PACK_FACTOR) + k8];
                    if (wz_e != nullptr) {
                        const int pk = wz_e[(an / 8) * num_groups + gi];
                        zp = (pk >> ((an % 8) * 4)) & 0xF;
                    }
                }
                unsigned char* dst = &Bg[n * LDSBK + k8 * PACK_FACTOR];
                #pragma unroll
                for (int jj = 0; jj < PACK_FACTOR; ++jj)
                    dst[jj] = (an < N)
                        ? decode_w4_to_e4m3((word >> (jj * 4)) & 0xF, zp, weight_is_e2m1) : 0;
            }
        }
        __syncthreads();

        for (int gi = gt; gi < gend; ++gi) {
            const int k0 = gi * BK;
            unsigned char* Bg = B_tile + (long)(gi - gt) * BN * LDSBK;
            typename MMA::CFrag acc[NFRAG_W];
            #pragma unroll
            for (int f = 0; f < NFRAG_W; ++f) acc[f] = MMA::zero();

            for (int kt = 0; kt < k_sub; ++kt) {
                const int gk = k0 + kt * WMMA_DIM + a_kh;
                v2i_t av = v2i_t{0, 0};
                if (a_valid)
                    av = *reinterpret_cast<const v2i_t*>(&x_fp8[(long)a_src * K + gk]);
                typename MMA::AFrag a_cur;
                a_cur[0] = __shfl(av[0], a_src_lane);
                a_cur[1] = __shfl(av[1], a_src_lane);
                #pragma unroll
                for (int f = 0; f < NFRAG_W; ++f) {
                    const int gf = nfrag0 + f;
                    typename MMA::BFrag b_cur = *reinterpret_cast<const typename MMA::BFrag*>(
                        &Bg[b_base + gf * WMMA_DIM * LDSBK + kt * WMMA_DIM]);
                    acc[f] = MMA::mma(a_cur, b_cur, acc[f]);
                }
            }
            #pragma unroll
            for (int f = 0; f < NFRAG_W; ++f) {
                const int abs_n = block_n + (nfrag0 + f) * WMMA_DIM + frag_col;
                const float wsc = (abs_n < N)
                    ? __half2float(ws_e[abs_n * num_groups + gi]) : 0.0f;
                #pragma unroll
                for (int ee = 0; ee < 8; ++ee) running[f][ee] += acc[f][ee] * wsc;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int ee = 0; ee < 8; ++ee) {
        const int r = warp_m * WMMA_DIM + frag_row0 + ee;
        if (r >= block_m) continue;
        const int row_pad = row0 + r;
        const int offs_token = sorted_token_ids[row_pad];
        if (offs_token >= num_valid_tokens) continue;
        const int src = SCATTER ? row_pad : (offs_token / top_k);
        const float asc = act_scales[src];
        if constexpr (SCATTER) {
            const int token = offs_token / out_top_k;
            const float w = topk_weights[offs_token];
            #pragma unroll
            for (int f = 0; f < NFRAG_W; ++f) {
                const int abs_n = block_n + (nfrag0 + f) * WMMA_DIM + frag_col;
                if (abs_n < N)
                    atomicAdd(&output_scatter[(long)token * N + abs_n],
                              w * running[f][ee] * asc);
            }
        } else {
            #pragma unroll
            for (int f = 0; f < NFRAG_W; ++f) {
                const int abs_n = block_n + (nfrag0 + f) * WMMA_DIM + frag_col;
                if (abs_n < N)
                    out[(long)row_pad * N + abs_n] = __float2half(running[f][ee] * asc);
            }
        }
    }
}

}  // namespace w4a8_tile
