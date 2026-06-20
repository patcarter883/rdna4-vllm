// gemm_tiled.h — DENSE W4A8-FP8 prefill GEMM, re-expressed over the MMA policy
// (tile_config.h), the dense sibling of moe_gemm_tiled.h. Two templated kernels:
//
//   gemm_tiled_kernel<MMA,BM,BN>                 ← re-expresses mmq_fp8_gemm_kernel_v5
//                                                  (A staged in LDS; 2-deep operand prefetch)
//   gemm_tiled_ashuffle_kernel<MMA,BM,BN,NWARPS,BKT,DB>
//                                                ← re-expresses mmq_fp8_gemm_kernel_v10
//                                                  (A via warp-shuffle, B-only LDS; DB double-buffer)
//
// CORRECTNESS: bit-exact-by-construction vs the v5/v10 launchers — IDENTICAL staging,
// accumulation order, and epilogue (incl. the fused per-row act-quant prologue); the ONLY
// change is `acc = MMA::mma(a,b,c)` in place of the inline
// `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c)`. With MMA=DenseWmmaFp8 that
// lowers to the exact same instruction, so the bytes are identical. PROVEN: the torch.equal
// A/B (test_dense_tiled_bitexact.py) is 32/32 max|diff|==0, and perf is neutral
// (test_dense_tiled_perf.py). STATUS: a VALIDATED, ready alternative behind
// VLLM_W4A8_DENSE_TILED — NOT the served default. Rationale: bit-exact == zero runtime upside,
// and dense (unlike grouped-MoE, which flipped to the tiled path for the SWMMAC second backend)
// has no second backend, so flipping production for no gain isn't justified; v5/v10 stay default.
//
// __launch_bounds__ MUST match the originals (v5/v10 both use (#warps*32)); omitting it
// targets the default 1024-thread block -> smaller per-thread register budget -> spill ->
// a perf regression that is a refactor artifact, not a real difference.
//
// SELF-CONTAINED within the dense TU: the dense .hip has its OWN v2i_t/v8f_t/WMMA_DIM/
// int4_signed_to_e4m3 (NOT shared with the grouped-MoE w4a8_tile world), and dense has no
// second backend, so we define a LOCAL DenseWmmaFp8 policy here rather than bridging to
// tile_config.h's WmmaFp8 (which would drag in duplicate primitives across two namespaces).
// The kernels still fuse the act-quant inline, so this header must be #included by
// w4a8_fp8_wmma_kernel.hip AFTER the helpers it uses are defined (compute_block_act_scales,
// stage_act_word, stage_act_v2i, int4_signed_to_e4m3, v2i_t, v8f_t, WMMA_DIM, PACK_FACTOR,
// MAX_GROUP_SIZE — all in namespace w4a8_fp8_wmma).
#pragma once

#include <hip/hip_fp16.h>

namespace w4a8_fp8_wmma {

// Local dense fp8 WMMA policy (16x16x16, wave32). Identical in content to
// w4a8_tile::WmmaFp8 but over the dense TU's own v2i_t/v8f_t — kept here so the
// tiled kernels stay templated on a policy (a future dense backend slots in as a
// template arg) without coupling the dense .hip to the grouped-MoE tile_config.h.
struct DenseWmmaFp8 {
    using AFrag = v2i_t;     // 8 fp8 (this lane's K16 slice of one M-row)
    using BFrag = v2i_t;     // 8 fp8 (this lane's K16 slice of one N-col)
    using CFrag = v8f_t;     // 8 f32 partials
    static __device__ __forceinline__ CFrag zero() {
        return CFrag{0, 0, 0, 0, 0, 0, 0, 0};
    }
    static __device__ __forceinline__ CFrag mma(AFrag a, BFrag b, CFrag c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
    }
};

// ============================================================
// v5 analogue: A+B in LDS, raw fp8 WMMA, 2-deep operand prefetch.
// Templated on the MMA policy + tile dims (BM rows, BN cols; NWARPS=BM/16,
// NFRAG=BN/16). Reproduces mmq_fp8_gemm_kernel_v5 line-for-line (only MMA::mma
// replaces the inline builtin). Static LDS sized for MAX_GROUP_SIZE+LDS_PAD.
// ============================================================
template <class MMA, int BM, int BN>
__global__ void __launch_bounds__((BM / 16) * 32) gemm_tiled_kernel(
    const __half*        __restrict__ x,            // (M, K) fp16 activations
    const int*           __restrict__ w_packed,
    const __half*        __restrict__ w_scales,
    const int*           __restrict__ w_zeros_packed,
    __half*              __restrict__ out,
    int M, int N, int K, int group_size) {

    constexpr int NWARPS = BM / 16;
    constexpr int NFRAG  = BN / WMMA_DIM;

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lrow = lane & 15;
    const int khalf = (lane >> 4) * 8;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;

    const int BK = group_size;
    const int num_k_groups = K / BK;
    const int packed_per_row = K / PACK_FACTOR;
    const int k_sub = BK / WMMA_DIM;
    // Pad the LDS row stride by 8 bytes so consecutive rows fall on different banks
    // (matches v5's local LDS_PAD; the dense .hip's own primitives, NOT w4a8_tile's).
    constexpr int LDS_PAD = 8;
    const int LDSBK = BK + LDS_PAD;

    __shared__ unsigned char A_tile[BM * (MAX_GROUP_SIZE + LDS_PAD)];
    __shared__ unsigned char B_tile[BN * (MAX_GROUP_SIZE + LDS_PAD)];
    __shared__ float act_scale_s[BM];
    compute_block_act_scales(x, M, K, block_m, BM, tid, NWARPS * 32, act_scale_s);

    float running[NFRAG][8];
    #pragma unroll
    for (int f = 0; f < NFRAG; ++f)
        #pragma unroll
        for (int e = 0; e < 8; ++e) running[f][e] = 0.0f;

    for (int g = 0; g < num_k_groups; ++g) {
        const int k0 = g * BK;

        for (int idx = tid * 4; idx < BM * BK; idx += (NWARPS * 32) * 4) {
            const int r = idx / BK, k = idx % BK, am = block_m + r;
            const float inv_scale = 1.0f / act_scale_s[r];
            unsigned int v = stage_act_word(x, M, K, am, k0 + k, inv_scale);
            *reinterpret_cast<unsigned int*>(&A_tile[r * LDSBK + k]) = v;
        }
        const int njobs = BN * (BK / PACK_FACTOR);
        for (int j = tid; j < njobs; j += NWARPS * 32) {
            const int n = j / (BK / PACK_FACTOR), k8 = j % (BK / PACK_FACTOR);
            const int an = block_n + n;
            int word = 0, zp = 8;
            if (an < N) {
                word = w_packed[an * packed_per_row + (k0 / PACK_FACTOR) + k8];
                if (w_zeros_packed != nullptr) {
                    const int pk = w_zeros_packed[(an / 8) * num_k_groups + g];
                    zp = (pk >> ((an % 8) * 4)) & 0xF;
                }
            }
            unsigned char* dst = &B_tile[n * LDSBK + k8 * PACK_FACTOR];
            #pragma unroll
            for (int jj = 0; jj < PACK_FACTOR; ++jj)
                dst[jj] = (an < N)
                    ? int4_signed_to_e4m3(((word >> (jj * 4)) & 0xF) - zp) : 0;
        }
        __syncthreads();

        typename MMA::CFrag acc[NFRAG];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) acc[f] = MMA::zero();

        const int a_base = (warp_id * WMMA_DIM + lrow) * LDSBK + khalf;
        const int b_base = lrow * LDSBK + khalf;
        // 2-deep pipeline: prefetch next kt's operands while the current kt's MMAs run.
        typename MMA::AFrag a_cur =
            *reinterpret_cast<const typename MMA::AFrag*>(&A_tile[a_base]);
        typename MMA::BFrag b_cur[NFRAG];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f)
            b_cur[f] = *reinterpret_cast<const typename MMA::BFrag*>(
                &B_tile[b_base + f * WMMA_DIM * LDSBK]);

        for (int kt = 0; kt < k_sub; ++kt) {
            typename MMA::AFrag a_nx;
            typename MMA::BFrag b_nx[NFRAG];
            if (kt + 1 < k_sub) {
                const int ko = (kt + 1) * WMMA_DIM;
                a_nx = *reinterpret_cast<const typename MMA::AFrag*>(&A_tile[a_base + ko]);
                #pragma unroll
                for (int f = 0; f < NFRAG; ++f)
                    b_nx[f] = *reinterpret_cast<const typename MMA::BFrag*>(
                        &B_tile[b_base + f * WMMA_DIM * LDSBK + ko]);
            }
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f)
                acc[f] = MMA::mma(a_cur, b_cur[f], acc[f]);
            a_cur = a_nx;
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f) b_cur[f] = b_nx[f];
        }

        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            const float wsc = (abs_n < N)
                ? __half2float(w_scales[abs_n * num_k_groups + g]) : 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; ++e) running[f][e] += acc[f][e] * wsc;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int lrow_m = warp_id * WMMA_DIM + frag_row0 + e;   // row within block
        const int abs_m = block_m + lrow_m;
        if (abs_m >= M) continue;
        const float asc = act_scale_s[lrow_m];   // fused per-row act scale
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            if (abs_n < N) out[abs_m * N + abs_n] = __float2half(running[f][e] * asc);
        }
    }
}

// ============================================================
// v10 analogue: A via warp-shuffle (no A-LDS), B-only LDS, optional double-buffer
// (DB). Templated on the MMA policy + <BM,BN,NWARPS,BKT,DB>. Reproduces
// mmq_fp8_gemm_kernel_v10 line-for-line (only MMA::mma replaces the inline
// builtin). Dynamic LDS (extern, 16B-aligned). PRESERVES the DB template flag.
// ============================================================
template <class MMA, int BM, int BN, int NWARPS, int BKT, bool DB>
__global__ void __launch_bounds__(NWARPS * 32) gemm_tiled_ashuffle_kernel(
    const __half*        __restrict__ x,            // (M, K) fp16 activations
    const int*           __restrict__ w_packed,
    const __half*        __restrict__ w_scales,
    const int*           __restrict__ w_zeros_packed,
    __half*              __restrict__ out,
    int M, int N, int K, int group_size, int swiz) {

    constexpr int NFRAG = BN / WMMA_DIM;
    constexpr int LDS_PAD = 8;
    const int block_m = (swiz ? blockIdx.x : blockIdx.y) * BM;
    const int block_n = (swiz ? blockIdx.y : blockIdx.x) * BN;
    const int tid = threadIdx.x;
    const int nthreads = NWARPS * 32;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lrow = lane & 15;
    const int khalf = (lane >> 4) * 8;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;

    const int BK = BKT ? BKT : group_size;
    const int num_k_groups = K / BK;
    const int packed_per_row = K / PACK_FACTOR;
    const int k_sub = BK / WMMA_DIM;
    const int LDSBK = BK + LDS_PAD;

    extern __shared__ __attribute__((aligned(16))) unsigned char gt_smem[];
    unsigned char* B_buf[2];
    B_buf[0] = gt_smem;
    B_buf[1] = gt_smem + (DB ? BN * LDSBK : 0);
    __shared__ float gt_wsc[2][BN];
    __shared__ float gt_act_scale[BM];
    compute_block_act_scales(x, M, K, block_m, BM, tid, nthreads, gt_act_scale);

    const int a_lrow = warp_id * WMMA_DIM + (lane >> 1);   // row within block
    const int a_row = block_m + a_lrow;
    const int a_kh = (lane & 1) * 8;
    const int a_shfl_src = 2 * (lane & 15) + (lane >> 4);
    const float a_inv_scale = 1.0f / gt_act_scale[a_lrow];
    auto load_A = [&](int k0, int kt) -> typename MMA::AFrag {
        const int gk = k0 + kt * WMMA_DIM + a_kh;
        v2i_t v = stage_act_v2i(x, M, K, a_row, gk, a_inv_scale);
        typename MMA::AFrag w;
        w[0] = __shfl(v[0], a_shfl_src);
        w[1] = __shfl(v[1], a_shfl_src);
        return w;
    };

    auto stage_B = [&](int g, int buf) {
        const int k0 = g * BK;
        unsigned char* Bt = B_buf[buf];
        const int njobs = BN * (BK / PACK_FACTOR);
        for (int j = tid; j < njobs; j += nthreads) {
            const int n = j / (BK / PACK_FACTOR), k8 = j % (BK / PACK_FACTOR);
            const int an = block_n + n;
            int word = 0, zp = 8;
            if (an < N) {
                word = w_packed[an * packed_per_row + (k0 / PACK_FACTOR) + k8];
                if (w_zeros_packed != nullptr) {
                    const int pk = w_zeros_packed[(an / 8) * num_k_groups + g];
                    zp = (pk >> ((an % 8) * 4)) & 0xF;
                }
            }
            unsigned char* dst = &Bt[n * LDSBK + k8 * PACK_FACTOR];
            #pragma unroll
            for (int jj = 0; jj < PACK_FACTOR; ++jj)
                dst[jj] = (an < N)
                    ? int4_signed_to_e4m3(((word >> (jj * 4)) & 0xF) - zp) : 0;
        }
        for (int i = tid; i < BN; i += nthreads) {
            const int an = block_n + i;
            gt_wsc[buf][i] = (an < N)
                ? __half2float(w_scales[an * num_k_groups + g]) : 0.0f;
        }
    };

    float running[NFRAG][8];
    #pragma unroll
    for (int f = 0; f < NFRAG; ++f)
        #pragma unroll
        for (int e = 0; e < 8; ++e) running[f][e] = 0.0f;

    const int b_base = lrow * LDSBK + khalf;

    stage_B(0, 0);
    __syncthreads();

    for (int g = 0; g < num_k_groups; ++g) {
        const int cur = DB ? (g & 1) : 0;
        if (DB && g + 1 < num_k_groups) stage_B(g + 1, (g + 1) & 1);
        const int k0 = g * BK;
        const unsigned char* Bt = B_buf[cur];

        typename MMA::CFrag acc[NFRAG];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) acc[f] = MMA::zero();

        typename MMA::AFrag a_cur = load_A(k0, 0);
        for (int kt = 0; kt < k_sub; ++kt) {
            typename MMA::AFrag a_nx;
            if (kt + 1 < k_sub) a_nx = load_A(k0, kt + 1);
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f) {
                typename MMA::BFrag b = *reinterpret_cast<const typename MMA::BFrag*>(
                    &Bt[b_base + f * WMMA_DIM * LDSBK + kt * WMMA_DIM]);
                acc[f] = MMA::mma(a_cur, b, acc[f]);
            }
            a_cur = a_nx;
        }

        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
            const float wsc = gt_wsc[cur][f * WMMA_DIM + frag_col];
            #pragma unroll
            for (int e = 0; e < 8; ++e) running[f][e] += acc[f][e] * wsc;
        }
        if (!DB) {
            __syncthreads();
            if (g + 1 < num_k_groups) { stage_B(g + 1, 0); __syncthreads(); }
        } else {
            __syncthreads();
        }
    }

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int lrow_m = warp_id * WMMA_DIM + frag_row0 + e;   // row within block
        const int abs_m = block_m + lrow_m;
        if (abs_m >= M) continue;
        const float asc = gt_act_scale[lrow_m];   // fused per-row act scale
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            if (abs_n < N) out[abs_m * N + abs_n] = __float2half(running[f][e] * asc);
        }
    }
}

}  // namespace w4a8_fp8_wmma
