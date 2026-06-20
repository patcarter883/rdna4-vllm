// W4A8-FP8 MMQ kernel for AMD RDNA4 (gfx1201).
//
// Research goal: accelerate 4-bit quantized GEMMs by expanding packed int4
// weights directly to FP8 (e4m3) *in registers* and feeding RDNA4's FP8 WMMA
// units, instead of expanding int4 -> f16/bf16 and using the half-rate f16 path.
//
// FP8 analogue of the gfx1151 INT8 MMQ kernel
// (reference/vllm-awq4-qwen/csrc/awq_mmq_gfx1151). Substitutions vs that kernel:
//   int8 LDS tiles            -> fp8 e4m3 LDS tiles
//   __builtin..._iu8_w32      -> wmma_f32_16x16x16_fp8_fp8 (via rocWMMA)
//   i32 accumulate            -> f32 accumulate
//   act-quant max_abs/127     -> max_abs/448 (e4m3 max)
//   int4 nibble -> int8 byte  -> int4 nibble -> fp8 e4m3 byte (in-register expand)
//
// A symmetric int4 value in [-8,7] is exactly representable in fp8 e4m3, so
// int4 -> fp8 is lossless on the weight side; only activation -> fp8 loses.
//
// Kernels:
//   v0: scalar reference (one thread per output element). Exact fp8 round-trip;
//       the numerical golden. Any M/N/K. Slow.
//   v1: rocWMMA fp8 16x16x16, one wave32 warp per 16x16 output tile. The int4
//       -> fp8 expansion is fused into the LDS store. Handles M/N tails. Any
//       group_size that is a multiple of 16 (32/64/128), derived at runtime.
//
// Tensor convention (matches vLLM compressed_tensors_wNa16 / the gfx1151 ref):
//   x         (M, K)            fp16    activations
//   w_packed  (N, K/8)          int32   weights, 8 uint4b8 per int32, low nibble first
//   scales    (N, K/group_size) fp16    per-group weight scales  (group_size = K / scales.size(1))
//   w_zeros   (N/8, K/group_size) int32 optional packed per-group zero points (AWQ)
//   out       (M, N)            fp16    output

// torch's cpp_extension injects -D__HIP_NO_HALF_CONVERSIONS__=1 /
// -D__HIP_NO_HALF_OPERATORS__=1, which disable the float<->__half conversions
// rocWMMA's vector headers rely on (static_cast<__half>(0.0f)). Undefine before
// any include so __half gets its conversions and rocWMMA compiles.
#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__

#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <rocwmma/rocwmma.hpp>
#include "kernel_names.h"   // w4a8::DenseKernel / dense_kernel_valid (opaque-int ABI tags)

namespace w4a8_fp8_wmma {

constexpr int PACK_FACTOR = 8;       // 8 uint4 per int32
constexpr float E4M3_MAX = 448.0f;   // OCP e4m3 max magnitude
constexpr int WMMA_DIM = 16;         // M=N=K tile for the fp8 WMMA
constexpr int MAX_GROUP_SIZE = 128;  // LDS staging sized for the largest group

// ============================================================
// FP8 e4m3 device conversion helpers (OCP e4m3 on gfx12).
// Lower to v_cvt_pk_fp8_f32 / v_cvt_f32_fp8 on RDNA4.
// ============================================================

__device__ __forceinline__ unsigned char f32_to_e4m3(float v) {
    int packed = __builtin_amdgcn_cvt_pk_fp8_f32(v, v, 0, false);
    return static_cast<unsigned char>(packed & 0xFF);
}

__device__ __forceinline__ float e4m3_to_f32(unsigned char b) {
    return __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(b), 0);
}

// Expand a signed-small-int weight value (nibble minus zp, in [-8,7] for
// symmetric) to its exact fp8 e4m3 byte. The "bit-shift magic before the WMMA
// units", via the hardware f32->fp8 convert; exact for [-8,7].
__device__ __forceinline__ unsigned char int4_signed_to_e4m3(int signed_val) {
    return f32_to_e4m3(static_cast<float>(signed_val));
}

// ============================================================
// v0: scalar reference (one thread per output element)
// ============================================================
constexpr int V0_BLOCK_M = 16;
constexpr int V0_BLOCK_N = 16;

__global__ void mmq_fp8_gemm_kernel_v0(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const int*           __restrict__ w_packed,
    const __half*        __restrict__ w_scales,
    const int*           __restrict__ w_zeros_packed,  // (N/8, K/gs) or nullptr
    __half*              __restrict__ out,
    int M, int N, int K, int group_size) {

    const int row = blockIdx.y * V0_BLOCK_M + threadIdx.y;
    const int col = blockIdx.x * V0_BLOCK_N + threadIdx.x;
    if (row >= M || col >= N) return;

    const int num_k_groups = K / group_size;
    const int packed_per_row = K / PACK_FACTOR;
    const float a_scale = act_scales[row];

    float acc = 0.0f;
    for (int g = 0; g < num_k_groups; ++g) {
        const float w_scale = __half2float(w_scales[col * num_k_groups + g]);
        int zp = 8;
        if (w_zeros_packed != nullptr) {
            const int packed = w_zeros_packed[(col / 8) * num_k_groups + g];
            zp = (packed >> ((col % 8) * 4)) & 0xF;
        }
        float group_acc = 0.0f;
        const int k_base = g * group_size;
        for (int kk = 0; kk < group_size; ++kk) {
            const int k = k_base + kk;
            const int word = w_packed[col * packed_per_row + (k / PACK_FACTOR)];
            const int w_signed = ((word >> ((k % PACK_FACTOR) * 4)) & 0xF) - zp;
            const float w_f = e4m3_to_f32(int4_signed_to_e4m3(w_signed));
            const float a_f = e4m3_to_f32(x_fp8[row * K + k]);
            group_acc += w_f * a_f;
        }
        acc += group_acc * w_scale;
    }
    out[row * N + col] = __float2half(acc * a_scale);
}

// ============================================================
// Activation quantization pre-kernel: fp16 -> e4m3 per row.
// One block per row; per-row scale = max(|x|) / 448.
// ============================================================
constexpr int ACTQUANT_BLOCK = 256;

__global__ void compute_act_fp8_and_scales_kernel(
    const __half*  __restrict__ x,
    unsigned char* __restrict__ x_fp8,
    float*         __restrict__ act_scales,
    int M, int K) {

    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const __half* x_row = x + row * K;
    unsigned char* out_row = x_fp8 + row * K;

    float local_max = 0.0f;
    for (int k = tid; k < K; k += ACTQUANT_BLOCK) {
        float v = fabsf(__half2float(x_row[k]));
        if (v > local_max) local_max = v;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor(local_max, offset);
        if (other > local_max) local_max = other;
    }
    __shared__ float warp_maxes[ACTQUANT_BLOCK / 32];
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) warp_maxes[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < ACTQUANT_BLOCK / 32) ? warp_maxes[lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            float other = __shfl_xor(v, offset);
            if (other > v) v = other;
        }
        if (lane == 0) {
            float scale = v / E4M3_MAX;
            if (scale < 1e-8f) scale = 1e-8f;
            warp_maxes[0] = scale;
            act_scales[row] = scale;
        }
    }
    __syncthreads();
    const float inv_scale = 1.0f / warp_maxes[0];
    for (int k = tid; k < K; k += ACTQUANT_BLOCK) {
        out_row[k] = f32_to_e4m3(__half2float(x_row[k]) * inv_scale);
    }
}

// 2x int32 (= 8 fp8 bytes / lane) vector — the fp8 WMMA A/B fragment unit; also
// used by the fused act-quant staging below (defined here so the helpers can
// return it; the GEMM kernels reuse this same alias).
using v2i_t = int __attribute__((ext_vector_type(2)));

// ============================================================
// Fused activation-quant prologue (replaces the separate pre-kernel +
// (M,K) x_fp8 HBM round-trip for the dispatched WMMA path, v5/v10).
//
// compute_block_act_scales: every thread cooperatively computes the per-row
// act scale for this block's `nrows` rows (block_m .. block_m+nrows) directly
// from the fp16 activations in global memory, writing scale_s[0..nrows) in LDS.
// The K-loop then quantizes x_fp16 -> e4m3 IN-LINE via stage_act_word (so the
// fp8 operand is materialized in LDS/registers, never in HBM).
//
// BIT-EXACTNESS: the per-row scale is max(|x|)/448 clamped to >=1e-8, reduced in
// fp32 over the full K row -- identical to compute_act_fp8_and_scales_kernel
// (max is associative+commutative+exact, so the threads' tiling is irrelevant).
// stage_act_word emits f32_to_e4m3(x*inv_scale) byte-for-byte as the old
// pre-kernel did, so the fp8 fed to the WMMA -- and the epilogue scale -- match
// the validated x_fp8 path exactly. (nthreads must be a multiple of 32.)
// ============================================================
__device__ __forceinline__ void compute_block_act_scales(
    const __half* __restrict__ x, int M, int K,
    int block_m, int nrows, int tid, int nthreads,
    float* scale_s) {            // LDS out, >= nrows floats
    // Each warp fully OWNS a strided set of rows (no cross-warp reduction, so
    // just ONE __syncthreads() for the whole prologue). Within a warp the 32
    // lanes split K and warp-shuffle the row max; lane 0 writes the scale.
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int nwarps = nthreads >> 5;
    for (int r = warp_id; r < nrows; r += nwarps) {
        const int am = block_m + r;
        float local_max = 0.0f;
        if (am < M) {
            const __half* x_row = x + (long)am * K;
            for (int k = lane; k < K; k += 32) {
                float v = fabsf(__half2float(x_row[k]));
                if (v > local_max) local_max = v;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_xor(local_max, offset);
            if (other > local_max) local_max = other;
        }
        if (lane == 0) {
            float scale = local_max / E4M3_MAX;
            if (scale < 1e-8f) scale = 1e-8f;
            scale_s[r] = scale;     // final per-row scale for this block row
        }
    }
    __syncthreads();
}

// Quantize 4 consecutive fp16 activations at x[am][k..k+4) to a packed uint32 of
// 4 e4m3 bytes using the precomputed per-row inv_scale. Out-of-range rows -> 0,
// matching the old staging's tail guard. Byte-identical to the pre-kernel.
__device__ __forceinline__ unsigned int stage_act_word(
    const __half* __restrict__ x, int M, int K, int am, int k, float inv_scale) {
    if (am >= M) return 0u;
    const __half* p = x + (long)am * K + k;
    unsigned int w = 0u;
    unsigned char* b = reinterpret_cast<unsigned char*>(&w);
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        b[j] = f32_to_e4m3(__half2float(p[j]) * inv_scale);
    return w;
}

// v10's load_A reads 8 fp8 (a v2i_t = 2 int32) per lane before the warp shuffle.
// Fuse: quantize the 8 fp16 at x[am][k..k+8) -> 8 e4m3 bytes (byte-identical to
// the pre-kernel's x_fp8). Out-of-range -> 0, matching the old v2i_t{0,0} guard.
__device__ __forceinline__ v2i_t stage_act_v2i(
    const __half* __restrict__ x, int M, int K, int am, int k, float inv_scale) {
    if (am >= M) return v2i_t{0, 0};
    const __half* p = x + (long)am * K + k;
    v2i_t v;
    unsigned char* b = reinterpret_cast<unsigned char*>(&v);
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        b[j] = f32_to_e4m3(__half2float(p[j]) * inv_scale);
    return v;
}

// ============================================================
// v1: rocWMMA fp8 16x16x16. One wave32 warp per 16x16 output tile.
// The int4 -> fp8 expansion is fused into the LDS store (STAGE 2), so weights
// stay packed int4 in global memory and become fp8 only in LDS, right before the
// WMMA units. Per-K-group weight scaling via store-to-LDS + scaled accumulate;
// per-row activation scale applied in the epilogue. M/N tails handled by bounds
// checks (out-of-range staged as fp8 zero).
// ============================================================

using FragA = rocwmma::fragment<rocwmma::matrix_a, WMMA_DIM, WMMA_DIM, WMMA_DIM,
                                rocwmma::float8_t, rocwmma::row_major>;
using FragB = rocwmma::fragment<rocwmma::matrix_b, WMMA_DIM, WMMA_DIM, WMMA_DIM,
                                rocwmma::float8_t, rocwmma::col_major>;
using FragAcc = rocwmma::fragment<rocwmma::accumulator, WMMA_DIM, WMMA_DIM,
                                  WMMA_DIM, float>;

__global__ void mmq_fp8_gemm_kernel_v1(
    const unsigned char* __restrict__ x_fp8,        // (M, K) e4m3
    const float*         __restrict__ act_scales,   // (M,)
    const int*           __restrict__ w_packed,     // (N, K/8) int32
    const __half*        __restrict__ w_scales,     // (N, K/gs) fp16
    const int*           __restrict__ w_zeros_packed,// (N/8, K/gs) int32 or nullptr
    __half*              __restrict__ out,          // (M, N) fp16
    int M, int N, int K, int group_size) {

    const int block_m = blockIdx.y * WMMA_DIM;
    const int block_n = blockIdx.x * WMMA_DIM;
    const int lane = threadIdx.x;  // [0, 32)

    const int num_k_groups = K / group_size;
    const int packed_per_row = K / PACK_FACTOR;
    const int k_subtiles = group_size / WMMA_DIM;  // WMMA steps per group

    __shared__ unsigned char a_tile[WMMA_DIM * MAX_GROUP_SIZE];
    __shared__ unsigned char b_tile[WMMA_DIM * MAX_GROUP_SIZE];
    __shared__ float tmp_lds[WMMA_DIM * WMMA_DIM];
    __shared__ float out_acc[WMMA_DIM * WMMA_DIM];
    __shared__ float wscale_lds[WMMA_DIM];

    #pragma unroll
    for (int i = lane; i < WMMA_DIM * WMMA_DIM; i += 32) out_acc[i] = 0.0f;

    const int tile_elems = WMMA_DIM * group_size;  // <= WMMA_DIM*MAX_GROUP_SIZE

    for (int g = 0; g < num_k_groups; ++g) {
        const int k_base = g * group_size;

        // STAGE 1: per-col weight scale for this group (N tail -> 0).
        if (lane < WMMA_DIM) {
            const int abs_n = block_n + lane;
            wscale_lds[lane] = (abs_n < N)
                ? __half2float(w_scales[abs_n * num_k_groups + g]) : 0.0f;
        }

        // STAGE 2: stage activations (fp8 copy) + weights (int4 -> fp8 expand).
        for (int idx = lane; idx < tile_elems; idx += 32) {
            const int r = idx / group_size;   // row in tile [0,16)
            const int kk = idx % group_size;  // k within group
            const int k = k_base + kk;
            const int abs_m = block_m + r;
            const int abs_n = block_n + r;
            a_tile[idx] = (abs_m < M) ? x_fp8[abs_m * K + k] : 0;
            if (abs_n < N) {
                int zp = 8;
                if (w_zeros_packed != nullptr) {
                    const int pk = w_zeros_packed[(abs_n / 8) * num_k_groups + g];
                    zp = (pk >> ((abs_n % 8) * 4)) & 0xF;
                }
                const int word = w_packed[abs_n * packed_per_row + (k / PACK_FACTOR)];
                const int w_signed = ((word >> ((k % PACK_FACTOR) * 4)) & 0xF) - zp;
                b_tile[idx] = int4_signed_to_e4m3(w_signed);
            } else {
                b_tile[idx] = 0;
            }
        }
        __syncthreads();

        FragAcc acc_g;
        rocwmma::fill_fragment(acc_g, 0.0f);
        for (int kt = 0; kt < k_subtiles; ++kt) {
            FragA a_frag;
            FragB b_frag;
            rocwmma::load_matrix_sync(
                a_frag,
                reinterpret_cast<const rocwmma::float8_t*>(a_tile + kt * WMMA_DIM),
                group_size);
            rocwmma::load_matrix_sync(
                b_frag,
                reinterpret_cast<const rocwmma::float8_t*>(b_tile + kt * WMMA_DIM),
                group_size);
            rocwmma::mma_sync(acc_g, a_frag, b_frag, acc_g);
        }

        rocwmma::store_matrix_sync(tmp_lds, acc_g, WMMA_DIM, rocwmma::mem_row_major);
        __syncthreads();
        #pragma unroll
        for (int i = lane; i < WMMA_DIM * WMMA_DIM; i += 32) {
            const int col = i % WMMA_DIM;
            out_acc[i] += tmp_lds[i] * wscale_lds[col];
        }
        __syncthreads();
    }

    // Epilogue: per-row activation scale, cast fp16, write (M/N tail guarded).
    #pragma unroll
    for (int i = lane; i < WMMA_DIM * WMMA_DIM; i += 32) {
        const int r = i / WMMA_DIM;
        const int c = i % WMMA_DIM;
        const int abs_m = block_m + r;
        const int abs_n = block_n + c;
        if (abs_m < M && abs_n < N) {
            const float a_scale = act_scales[abs_m];
            out[abs_m * N + abs_n] = __float2half(out_acc[i] * a_scale);
        }
    }
}

// ============================================================
// v2: optimized tiled fp8 WMMA GEMM.
//   - block computes a BM x BN output tile with NWARPS=4 warps (wave32);
//     warp w owns rows [w*16, w*16+16), all BN cols (BN/16 N-fragments).
//   - f32 accumulation kept in REGISTERS across K-groups (running[]).
//   - per-group weight scale applied IN-REGISTER using the probed accumulator
//     layout: for accumulator element e in lane L, col = L & 15 (constant per
//     lane), row = (L>>4)*8 + e. So one scale read per lane per N-fragment.
//   - staging: coalesced uint32 (4 fp8) loads for activations, coalesced int32
//     loads + int4->fp8 expand for weights, both M/N tail-guarded.
// Constraints: K % group_size == 0, group_size in {32,64,128}. Any M, N (tails
// handled). One group staged per main-loop iteration (BK = group_size).
// ============================================================
constexpr int V2_BM = 256;
constexpr int V2_BN = 64;
constexpr int V2_NWARPS = V2_BM / 16;   // one warp per 16 output rows
constexpr int V2_NFRAG = V2_BN / 16;    // N-fragments per warp

__global__ void __launch_bounds__(V2_NWARPS * 32) mmq_fp8_gemm_kernel_v2(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const int*           __restrict__ w_packed,
    const __half*        __restrict__ w_scales,
    const int*           __restrict__ w_zeros_packed,
    __half*              __restrict__ out,
    int M, int N, int K, int group_size) {

    const int block_m = blockIdx.y * V2_BM;
    const int block_n = blockIdx.x * V2_BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;     // 0..3
    const int lane = tid & 31;
    const int frag_col = lane & 15;        // column within a 16x16 fragment
    const int frag_row0 = (lane >> 4) * 8; // first of 8 rows this lane holds

    const int BK = group_size;
    const int num_k_groups = K / BK;
    const int packed_per_row = K / PACK_FACTOR;

    __shared__ unsigned char A_tile[V2_BM * MAX_GROUP_SIZE];
    __shared__ unsigned char B_tile[V2_BN * MAX_GROUP_SIZE];

    float running[V2_NFRAG][8];
    #pragma unroll
    for (int f = 0; f < V2_NFRAG; ++f)
        #pragma unroll
        for (int e = 0; e < 8; ++e) running[f][e] = 0.0f;

    for (int g = 0; g < num_k_groups; ++g) {
        const int k0 = g * BK;

        // ---- stage activations: A_tile[r][k] = x_fp8[block_m+r][k0+k] ----
        // uint32 (4 fp8) coalesced loads; row tail-guarded.
        for (int idx = tid * 4; idx < V2_BM * BK; idx += (V2_NWARPS * 32) * 4) {
            const int r = idx / BK;
            const int k = idx % BK;
            const int am = block_m + r;
            unsigned int v = 0u;
            if (am < M)
                v = *reinterpret_cast<const unsigned int*>(&x_fp8[am * K + k0 + k]);
            *reinterpret_cast<unsigned int*>(&A_tile[r * BK + k]) = v;
        }

        // ---- stage weights: expand int4 -> fp8 into B_tile[n][k] ----
        const int njobs = V2_BN * (BK / PACK_FACTOR);
        for (int j = tid; j < njobs; j += V2_NWARPS * 32) {
            const int n = j / (BK / PACK_FACTOR);
            const int k8 = j % (BK / PACK_FACTOR);
            const int an = block_n + n;
            int word = 0, zp = 8;
            if (an < N) {
                word = w_packed[an * packed_per_row + (k0 / PACK_FACTOR) + k8];
                if (w_zeros_packed != nullptr) {
                    const int pk = w_zeros_packed[(an / 8) * num_k_groups + g];
                    zp = (pk >> ((an % 8) * 4)) & 0xF;
                }
            }
            unsigned char* dst = &B_tile[n * BK + k8 * PACK_FACTOR];
            #pragma unroll
            for (int jj = 0; jj < PACK_FACTOR; ++jj) {
                dst[jj] = (an < N) ? int4_signed_to_e4m3(((word >> (jj * 4)) & 0xF) - zp)
                                   : 0;
            }
        }
        __syncthreads();

        // ---- WMMA over the BK/16 k-subtiles of this group ----
        FragAcc group_acc[V2_NFRAG];
        #pragma unroll
        for (int f = 0; f < V2_NFRAG; ++f) rocwmma::fill_fragment(group_acc[f], 0.0f);

        const int k_sub = BK / WMMA_DIM;
        for (int kt = 0; kt < k_sub; ++kt) {
            FragA a_frag;
            rocwmma::load_matrix_sync(
                a_frag,
                reinterpret_cast<const rocwmma::float8_t*>(
                    &A_tile[warp_id * WMMA_DIM * BK + kt * WMMA_DIM]),
                BK);
            #pragma unroll
            for (int f = 0; f < V2_NFRAG; ++f) {
                FragB b_frag;
                rocwmma::load_matrix_sync(
                    b_frag,
                    reinterpret_cast<const rocwmma::float8_t*>(
                        &B_tile[f * WMMA_DIM * BK + kt * WMMA_DIM]),
                    BK);
                rocwmma::mma_sync(group_acc[f], a_frag, b_frag, group_acc[f]);
            }
        }

        // ---- in-register per-group scale: col = frag_col (constant per lane) ----
        #pragma unroll
        for (int f = 0; f < V2_NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            const float wsc = (abs_n < N)
                ? __half2float(w_scales[abs_n * num_k_groups + g]) : 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; ++e)
                running[f][e] += group_acc[f].x[e] * wsc;
        }
        __syncthreads();  // before next iteration overwrites A_tile/B_tile
    }

    // ---- epilogue: apply per-row activation scale, write out (tail-guarded) ----
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int abs_m = block_m + warp_id * WMMA_DIM + frag_row0 + e;
        if (abs_m >= M) continue;
        const float asc = act_scales[abs_m];
        #pragma unroll
        for (int f = 0; f < V2_NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            if (abs_n < N)
                out[abs_m * N + abs_n] = __float2half(running[f][e] * asc);
        }
    }
}

// ============================================================
// v4: v2 single-buffer staging + INNER-LOOP (kt) fragment pipelining. The next
// k-subtile's A and B fragments are loaded from LDS while the current subtile's
// MMAs execute, hiding LDS->register latency (the thing keeping v2 at ~5% of the
// 370 TFLOP/s fp8 WMMA peak). rocWMMA mma kept; only the scheduling changes.
// ============================================================
__global__ void __launch_bounds__(V2_NWARPS * 32) mmq_fp8_gemm_kernel_v4(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const int*           __restrict__ w_packed,
    const __half*        __restrict__ w_scales,
    const int*           __restrict__ w_zeros_packed,
    __half*              __restrict__ out,
    int M, int N, int K, int group_size) {

    const int block_m = blockIdx.y * V2_BM;
    const int block_n = blockIdx.x * V2_BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;

    const int BK = group_size;
    const int num_k_groups = K / BK;
    const int packed_per_row = K / PACK_FACTOR;
    const int k_sub = BK / WMMA_DIM;

    __shared__ unsigned char A_tile[V2_BM * MAX_GROUP_SIZE];
    __shared__ unsigned char B_tile[V2_BN * MAX_GROUP_SIZE];

    float running[V2_NFRAG][8];
    #pragma unroll
    for (int f = 0; f < V2_NFRAG; ++f)
        #pragma unroll
        for (int e = 0; e < 8; ++e) running[f][e] = 0.0f;

    auto loadA = [&](FragA& a, int kt) {
        rocwmma::load_matrix_sync(
            a, reinterpret_cast<const rocwmma::float8_t*>(
                   &A_tile[warp_id * WMMA_DIM * BK + kt * WMMA_DIM]), BK);
    };
    auto loadB = [&](FragB& b, int f, int kt) {
        rocwmma::load_matrix_sync(
            b, reinterpret_cast<const rocwmma::float8_t*>(
                   &B_tile[f * WMMA_DIM * BK + kt * WMMA_DIM]), BK);
    };

    for (int g = 0; g < num_k_groups; ++g) {
        const int k0 = g * BK;

        for (int idx = tid * 4; idx < V2_BM * BK; idx += (V2_NWARPS * 32) * 4) {
            const int r = idx / BK, k = idx % BK, am = block_m + r;
            unsigned int v = 0u;
            if (am < M)
                v = *reinterpret_cast<const unsigned int*>(&x_fp8[am * K + k0 + k]);
            *reinterpret_cast<unsigned int*>(&A_tile[r * BK + k]) = v;
        }
        const int njobs = V2_BN * (BK / PACK_FACTOR);
        for (int j = tid; j < njobs; j += V2_NWARPS * 32) {
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
            unsigned char* dst = &B_tile[n * BK + k8 * PACK_FACTOR];
            #pragma unroll
            for (int jj = 0; jj < PACK_FACTOR; ++jj)
                dst[jj] = (an < N)
                    ? int4_signed_to_e4m3(((word >> (jj * 4)) & 0xF) - zp) : 0;
        }
        __syncthreads();

        FragAcc acc[V2_NFRAG];
        #pragma unroll
        for (int f = 0; f < V2_NFRAG; ++f) rocwmma::fill_fragment(acc[f], 0.0f);

        // kt-pipelined inner loop: load kt+1 fragments while MMA-ing kt.
        FragA a[2];
        FragB b[2][V2_NFRAG];
        loadA(a[0], 0);
        #pragma unroll
        for (int f = 0; f < V2_NFRAG; ++f) loadB(b[0][f], f, 0);

        for (int kt = 0; kt < k_sub; ++kt) {
            const int c = kt & 1, nx = (kt + 1) & 1;
            if (kt + 1 < k_sub) {
                loadA(a[nx], kt + 1);
                #pragma unroll
                for (int f = 0; f < V2_NFRAG; ++f) loadB(b[nx][f], f, kt + 1);
            }
            #pragma unroll
            for (int f = 0; f < V2_NFRAG; ++f)
                rocwmma::mma_sync(acc[f], a[c], b[c][f], acc[f]);
        }

        #pragma unroll
        for (int f = 0; f < V2_NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            const float wsc = (abs_n < N)
                ? __half2float(w_scales[abs_n * num_k_groups + g]) : 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; ++e) running[f][e] += acc[f].x[e] * wsc;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int abs_m = block_m + warp_id * WMMA_DIM + frag_row0 + e;
        if (abs_m >= M) continue;
        const float asc = act_scales[abs_m];
        #pragma unroll
        for (int f = 0; f < V2_NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            if (abs_n < N) out[abs_m * N + abs_n] = __float2half(running[f][e] * asc);
        }
    }
}

// ============================================================
// v5: RAW fp8 WMMA builtin + direct K-major int2 loads (no rocWMMA). Each lane
// loads its 8-fp8 operand with a single contiguous b64 load from K-major LDS
// (layout probed: A operand lane L = A[(L&15)][(L>>4)*8], same for B^T). Replaces
// rocWMMA's mixed ds_load_b32/b64 fragment loads (the ~20 TFLOP/s ceiling).
// ============================================================
// (v2i_t defined above near the fused act-quant helpers.)
using v8f_t = float __attribute__((ext_vector_type(8)));
using v8h_t = _Float16 __attribute__((ext_vector_type(8)));   // f16 WMMA operand

__global__ void __launch_bounds__(V2_NWARPS * 32) mmq_fp8_gemm_kernel_v5(
    const __half*        __restrict__ x,            // (M, K) fp16 activations
    const int*           __restrict__ w_packed,
    const __half*        __restrict__ w_scales,
    const int*           __restrict__ w_zeros_packed,
    __half*              __restrict__ out,
    int M, int N, int K, int group_size) {

    const int block_m = blockIdx.y * V2_BM;
    const int block_n = blockIdx.x * V2_BN;
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
    // Pad the LDS row stride by 8 bytes so consecutive rows fall on different
    // banks: at stride BK (=128 for g128 = 32 banks) all 16 lanes of a b64 read
    // hit the same bank (16-way conflict). Stride BK+8 spreads them.
    constexpr int LDS_PAD = 8;
    const int LDSBK = BK + LDS_PAD;

    __shared__ unsigned char A_tile[V2_BM * (MAX_GROUP_SIZE + LDS_PAD)];
    __shared__ unsigned char B_tile[V2_BN * (MAX_GROUP_SIZE + LDS_PAD)];
    // Fused act-quant: per-row e4m3 scale for this block's V2_BM rows. Replaces
    // the separate compute_act_fp8_and_scales_kernel launch and the (M,K) x_fp8
    // HBM round-trip (activations stay fp16 in HBM, become e4m3 in A_tile during
    // staging). Bit-exact with the old path.
    __shared__ float act_scale_s[V2_BM];
    compute_block_act_scales(x, M, K, block_m, V2_BM, tid, V2_NWARPS * 32,
                             act_scale_s);

    float running[V2_NFRAG][8];
    #pragma unroll
    for (int f = 0; f < V2_NFRAG; ++f)
        #pragma unroll
        for (int e = 0; e < 8; ++e) running[f][e] = 0.0f;

    for (int g = 0; g < num_k_groups; ++g) {
        const int k0 = g * BK;

        for (int idx = tid * 4; idx < V2_BM * BK; idx += (V2_NWARPS * 32) * 4) {
            const int r = idx / BK, k = idx % BK, am = block_m + r;
            const float inv_scale = 1.0f / act_scale_s[r];
            unsigned int v = stage_act_word(x, M, K, am, k0 + k, inv_scale);
            *reinterpret_cast<unsigned int*>(&A_tile[r * LDSBK + k]) = v;
        }
        const int njobs = V2_BN * (BK / PACK_FACTOR);
        for (int j = tid; j < njobs; j += V2_NWARPS * 32) {
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

        v8f_t acc[V2_NFRAG];
        #pragma unroll
        for (int f = 0; f < V2_NFRAG; ++f) acc[f] = v8f_t{0, 0, 0, 0, 0, 0, 0, 0};

        const int a_base = (warp_id * WMMA_DIM + lrow) * LDSBK + khalf;
        const int b_base = lrow * LDSBK + khalf;
        // 2-deep pipeline: prefetch next kt's operands (plain b64 loads) into
        // registers while the current kt's MMAs run, so the scheduler overlaps
        // LDS-read latency with compute.
        v2i_t a_cur = *reinterpret_cast<const v2i_t*>(&A_tile[a_base]);
        v2i_t b_cur[V2_NFRAG];
        #pragma unroll
        for (int f = 0; f < V2_NFRAG; ++f)
            b_cur[f] = *reinterpret_cast<const v2i_t*>(&B_tile[b_base + f * WMMA_DIM * LDSBK]);

        for (int kt = 0; kt < k_sub; ++kt) {
            v2i_t a_nx;
            v2i_t b_nx[V2_NFRAG];
            if (kt + 1 < k_sub) {
                const int ko = (kt + 1) * WMMA_DIM;
                a_nx = *reinterpret_cast<const v2i_t*>(&A_tile[a_base + ko]);
                #pragma unroll
                for (int f = 0; f < V2_NFRAG; ++f)
                    b_nx[f] = *reinterpret_cast<const v2i_t*>(
                        &B_tile[b_base + f * WMMA_DIM * LDSBK + ko]);
            }
            #pragma unroll
            for (int f = 0; f < V2_NFRAG; ++f)
                acc[f] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    a_cur, b_cur[f], acc[f]);
            a_cur = a_nx;
            #pragma unroll
            for (int f = 0; f < V2_NFRAG; ++f) b_cur[f] = b_nx[f];
        }

        #pragma unroll
        for (int f = 0; f < V2_NFRAG; ++f) {
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
        for (int f = 0; f < V2_NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            if (abs_n < N) out[abs_m * N + abs_n] = __float2half(running[f][e] * asc);
        }
    }
}

// ============================================================
// v6: K-EXTENSION (AMD RDNA4 WMMA guide part 2, §3 "Extend the K dimension").
// Each lane loads its operands for TWO consecutive K-subtiles with a SINGLE
// b128 (int4 / v4i_t = 16 fp8) LDS read, then issues TWO back-to-back WMMAs.
// Halves the number of LDS load instructions per WMMA vs v5's b64-per-subtile,
// and hands the scheduler two independent MMAs per load for ILP.
//
// The catch: a single b128 must contain THIS lane's 8 fp8 for subtile 2*kp AND
// its 8 fp8 for subtile 2*kp+1 contiguously. v5's plain K-major layout puts a
// full subtile (all 16 k of two lanes) in 16 contiguous bytes, which is the
// WRONG packing. So v6 stores each 32-wide K block under a permutation:
//   perm(k) within a 32-block: p = (kb&8 ? 16:0) + (kb>>4)*8 + (kb&7),  kb=k&31
// which lays the block out as chunks [k0:8][k16:24][k8:16][k24:32]. Then a
// khalf=0 lane's first 16 bytes = {subtile0 lo, subtile1 lo} and a khalf=8
// lane's = {subtile0 hi, subtile1 hi} -- exactly the two operands it needs.
// The loaded bytes are BIT-IDENTICAL to what v5 reads, so numerics match v5.
//
// Dynamic LDS (extern __shared__, 16B-aligned) sized for the actual (BK+pad) so
// the b128 row stride LDSBK is a multiple of 16 (required for ds_read_b128) and
// the pad is sweepable at launch (env VLLM_W4A8_V6_PAD, must be a multiple of
// 16). Requires group_size % 32 == 0 (k_sub even); else launcher falls to v5.
// ============================================================
using v4i_t = int __attribute__((ext_vector_type(4)));
using v2f_t = float __attribute__((ext_vector_type(2)));

__device__ __forceinline__ int v6_permk(int k) {
    const int blk = k >> 5;      // which 32-wide K block
    const int kb = k & 31;       // k within the block
    return blk * 32 + ((kb & 8) ? 16 : 0) + ((kb >> 4) << 3) + (kb & 7);
}

__global__ void __launch_bounds__(V2_NWARPS * 32) mmq_fp8_gemm_kernel_v6(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const int*           __restrict__ w_packed,
    const __half*        __restrict__ w_scales,
    const int*           __restrict__ w_zeros_packed,
    __half*              __restrict__ out,
    int M, int N, int K, int group_size, int lds_pad) {

    const int block_m = blockIdx.y * V2_BM;
    const int block_n = blockIdx.x * V2_BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lrow = lane & 15;
    const int hflag = lane >> 4;          // 0 or 1 (which K-half this lane owns)
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;

    const int BK = group_size;
    const int num_k_groups = K / BK;
    const int packed_per_row = K / PACK_FACTOR;
    const int k_sub = BK / WMMA_DIM;      // even (BK % 32 == 0)
    const int npair = k_sub >> 1;         // b128 loads per group
    const int LDSBK = BK + lds_pad;       // multiple of 16

    extern __shared__ __attribute__((aligned(16))) unsigned char v6_smem[];
    unsigned char* A_tile = v6_smem;
    unsigned char* B_tile = v6_smem + V2_BM * LDSBK;

    float running[V2_NFRAG][8];
    #pragma unroll
    for (int f = 0; f < V2_NFRAG; ++f)
        #pragma unroll
        for (int e = 0; e < 8; ++e) running[f][e] = 0.0f;

    for (int g = 0; g < num_k_groups; ++g) {
        const int k0 = g * BK;

        // ---- stage activations: uint32 (4 fp8) coalesced loads, permuted dst ----
        for (int idx = tid * 4; idx < V2_BM * BK; idx += (V2_NWARPS * 32) * 4) {
            const int r = idx / BK, k = idx % BK, am = block_m + r;
            unsigned int v = 0u;
            if (am < M)
                v = *reinterpret_cast<const unsigned int*>(&x_fp8[am * K + k0 + k]);
            *reinterpret_cast<unsigned int*>(&A_tile[r * LDSBK + v6_permk(k)]) = v;
        }
        // ---- stage weights: expand int4 -> fp8, 8 per chunk, permuted dst ----
        const int njobs = V2_BN * (BK / PACK_FACTOR);
        for (int j = tid; j < njobs; j += V2_NWARPS * 32) {
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
            unsigned char* dst = &B_tile[n * LDSBK + v6_permk(k8 * PACK_FACTOR)];
            #pragma unroll
            for (int jj = 0; jj < PACK_FACTOR; ++jj)
                dst[jj] = (an < N)
                    ? int4_signed_to_e4m3(((word >> (jj * 4)) & 0xF) - zp) : 0;
        }
        __syncthreads();

        v8f_t acc[V2_NFRAG];
        #pragma unroll
        for (int f = 0; f < V2_NFRAG; ++f) acc[f] = v8f_t{0, 0, 0, 0, 0, 0, 0, 0};

        // hflag selects bytes [0,16) (khalf=0) or [16,32) (khalf=8) of each block.
        const int a_base = (warp_id * WMMA_DIM + lrow) * LDSBK + hflag * 16;
        const int b_base = lrow * LDSBK + hflag * 16;

        // 2-deep pipeline at the K-PAIR granularity: prefetch the next pair's
        // b128 operands while the current pair's two WMMAs run.
        v4i_t a_cur = *reinterpret_cast<const v4i_t*>(&A_tile[a_base]);
        v4i_t b_cur[V2_NFRAG];
        #pragma unroll
        for (int f = 0; f < V2_NFRAG; ++f)
            b_cur[f] = *reinterpret_cast<const v4i_t*>(
                &B_tile[b_base + f * WMMA_DIM * LDSBK]);

        for (int kp = 0; kp < npair; ++kp) {
            v4i_t a_nx;
            v4i_t b_nx[V2_NFRAG];
            if (kp + 1 < npair) {
                const int off = (kp + 1) * 32;
                a_nx = *reinterpret_cast<const v4i_t*>(&A_tile[a_base + off]);
                #pragma unroll
                for (int f = 0; f < V2_NFRAG; ++f)
                    b_nx[f] = *reinterpret_cast<const v4i_t*>(
                        &B_tile[b_base + f * WMMA_DIM * LDSBK + off]);
            }
            // split each b128 into its two K-subtile (b64) operands.
            const v2i_t a_lo = v2i_t{a_cur.x, a_cur.y};
            const v2i_t a_hi = v2i_t{a_cur.z, a_cur.w};
            #pragma unroll
            for (int f = 0; f < V2_NFRAG; ++f) {
                const v2i_t b_lo = v2i_t{b_cur[f].x, b_cur[f].y};
                const v2i_t b_hi = v2i_t{b_cur[f].z, b_cur[f].w};
                acc[f] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    a_lo, b_lo, acc[f]);
                acc[f] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    a_hi, b_hi, acc[f]);
            }
            a_cur = a_nx;
            #pragma unroll
            for (int f = 0; f < V2_NFRAG; ++f) b_cur[f] = b_nx[f];
        }

        #pragma unroll
        for (int f = 0; f < V2_NFRAG; ++f) {
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
        const int abs_m = block_m + warp_id * WMMA_DIM + frag_row0 + e;
        if (abs_m >= M) continue;
        const float asc = act_scales[abs_m];
        #pragma unroll
        for (int f = 0; f < V2_NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            if (abs_n < N) out[abs_m * N + abs_n] = __float2half(running[f][e] * asc);
        }
    }
}

// ============================================================
// v7: v5's algorithm (raw b64 K-major loads, kt-pipelined) but with a
// TEMPLATED, SMALLER tile + dynamic LDS to raise occupancy. v5 runs BM=256/4
// warps/~46KB LDS = 1 block/CU = 4 waves/CU (~12% occupancy, ~= our 12%-of-peak
// throughput). A smaller tile shrinks LDS so multiple blocks co-reside per CU,
// giving the scheduler more waves to hide the __syncthreads + LDS-read latency
// that v6 proved is the real ceiling (load width was not). Tile chosen at launch
// via VLLM_W4A8_V7_BM (64/128/256); BN fixed at 64.
// ============================================================
template <int BM, int BN, int NWARPS, int BKT>
__global__ void __launch_bounds__(NWARPS * 32) mmq_fp8_gemm_kernel_v7(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const int*           __restrict__ w_packed,
    const __half*        __restrict__ w_scales,
    const int*           __restrict__ w_zeros_packed,
    __half*              __restrict__ out,
    int M, int N, int K, int group_size, int diag, int swiz) {

    // diag (env VLLM_W4A8_V7_DIAG, perf breakdown only; 0 = normal/correct):
    //   bit0(1)=skip WMMA, bit1(2)=skip global loads, bit2(4)=skip LDS writes.
    const bool DIAG_NO_WMMA  = diag & 1;
    const bool DIAG_NO_GLOBAL = diag & 2;
    const bool DIAG_NO_STORE = diag & 4;
    const bool DIAG_NO_LDSREAD = diag & 8;  // skip per-kt LDS operand reloads
    const bool DIAG_NO_SCALE = diag & 16;   // skip per-group w_scale global read
    constexpr int NFRAG = BN / WMMA_DIM;
    constexpr int LDS_PAD = 8;
    // swiz: map M to the fast (x) grid dim so consecutively-scheduled blocks
    // share the same block_n weight slab -> it stays resident in L2/MALL across
    // the M-blocks instead of being re-streamed from VRAM (cuts redundant weight
    // traffic at large K). swiz=0 keeps N on x (original).
    const int block_m = (swiz ? blockIdx.x : blockIdx.y) * BM;
    const int block_n = (swiz ? blockIdx.y : blockIdx.x) * BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lrow = lane & 15;
    const int khalf = (lane >> 4) * 8;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;

    // BKT>0 makes the group size a compile-time constant so idx/BK & idx%BK in
    // the staging loops lower to shift/mask instead of runtime integer div/mod.
    const int BK = BKT ? BKT : group_size;
    const int num_k_groups = K / BK;
    const int packed_per_row = K / PACK_FACTOR;
    const int k_sub = BK / WMMA_DIM;
    const int LDSBK = BK + LDS_PAD;

    extern __shared__ __attribute__((aligned(16))) unsigned char v7_smem[];
    unsigned char* A_tile = v7_smem;
    unsigned char* B_tile = v7_smem + BM * LDSBK;
    // per-group weight scales staged once (cooperatively) into LDS, instead of
    // each of the NWARPS warps re-reading the same uncoalesced fp16 scales.
    __shared__ float v7_wsc[BN];

    float running[NFRAG][8];
    #pragma unroll
    for (int f = 0; f < NFRAG; ++f)
        #pragma unroll
        for (int e = 0; e < 8; ++e) running[f][e] = 0.0f;

    for (int g = 0; g < num_k_groups; ++g) {
        const int k0 = g * BK;

        for (int idx = tid * 4; idx < BM * BK; idx += (NWARPS * 32) * 4) {
            const int r = idx / BK, k = idx % BK, am = block_m + r;
            unsigned int v = 0u;
            if (am < M && !DIAG_NO_GLOBAL)
                v = *reinterpret_cast<const unsigned int*>(&x_fp8[am * K + k0 + k]);
            if (!DIAG_NO_STORE)
                *reinterpret_cast<unsigned int*>(&A_tile[r * LDSBK + k]) = v;
        }
        const int njobs = BN * (BK / PACK_FACTOR);
        for (int j = tid; j < njobs; j += NWARPS * 32) {
            const int n = j / (BK / PACK_FACTOR), k8 = j % (BK / PACK_FACTOR);
            const int an = block_n + n;
            int word = 0, zp = 8;
            if (an < N && !DIAG_NO_GLOBAL) {
                word = w_packed[an * packed_per_row + (k0 / PACK_FACTOR) + k8];
                if (w_zeros_packed != nullptr) {
                    const int pk = w_zeros_packed[(an / 8) * num_k_groups + g];
                    zp = (pk >> ((an % 8) * 4)) & 0xF;
                }
            }
            if (!DIAG_NO_STORE) {
                unsigned char* dst = &B_tile[n * LDSBK + k8 * PACK_FACTOR];
                #pragma unroll
                for (int jj = 0; jj < PACK_FACTOR; ++jj)
                    dst[jj] = (an < N)
                        ? int4_signed_to_e4m3(((word >> (jj * 4)) & 0xF) - zp) : 0;
            }
        }
        // stage this group's BN weight scales into LDS once (cooperative).
        if (!DIAG_NO_SCALE)
            for (int i = tid; i < BN; i += NWARPS * 32) {
                const int an = block_n + i;
                v7_wsc[i] = (an < N)
                    ? __half2float(w_scales[an * num_k_groups + g]) : 0.0f;
            }
        __syncthreads();

        v8f_t acc[NFRAG];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) acc[f] = v8f_t{0, 0, 0, 0, 0, 0, 0, 0};

        const int a_base = (warp_id * WMMA_DIM + lrow) * LDSBK + khalf;
        const int b_base = lrow * LDSBK + khalf;
        v2i_t a_cur = *reinterpret_cast<const v2i_t*>(&A_tile[a_base]);
        v2i_t b_cur[NFRAG];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f)
            b_cur[f] = *reinterpret_cast<const v2i_t*>(&B_tile[b_base + f * WMMA_DIM * LDSBK]);

        for (int kt = 0; kt < k_sub; ++kt) {
            v2i_t a_nx;
            v2i_t b_nx[NFRAG];
            if (!DIAG_NO_LDSREAD && kt + 1 < k_sub) {
                const int ko = (kt + 1) * WMMA_DIM;
                a_nx = *reinterpret_cast<const v2i_t*>(&A_tile[a_base + ko]);
                #pragma unroll
                for (int f = 0; f < NFRAG; ++f)
                    b_nx[f] = *reinterpret_cast<const v2i_t*>(
                        &B_tile[b_base + f * WMMA_DIM * LDSBK + ko]);
            }
            if (!DIAG_NO_WMMA) {
                #pragma unroll
                for (int f = 0; f < NFRAG; ++f)
                    acc[f] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        a_cur, b_cur[f], acc[f]);
            }
            if (!DIAG_NO_LDSREAD) {
                a_cur = a_nx;
                #pragma unroll
                for (int f = 0; f < NFRAG; ++f) b_cur[f] = b_nx[f];
            }
        }

        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
            const float wsc = DIAG_NO_SCALE ? 1.0f : v7_wsc[f * WMMA_DIM + frag_col];
            #pragma unroll
            for (int e = 0; e < 8; ++e) running[f][e] += acc[f][e] * wsc;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int abs_m = block_m + warp_id * WMMA_DIM + frag_row0 + e;
        if (abs_m >= M) continue;
        const float asc = act_scales[abs_m];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            if (abs_n < N) out[abs_m * N + abs_n] = __float2half(running[f][e] * asc);
        }
    }
}

// ============================================================
// v8: v7 tile (BN=128 => 8-acc ILP) + LDS DOUBLE-BUFFERING. The diag breakdown
// showed WMMA compute is fully hidden and ~50% of time is the serial
// stage->barrier->compute->barrier structure (+ ~29% global weight reads at
// large K). v8 ping-pongs two LDS buffers so the next K-group's global load +
// int4->fp8 expand overlaps the current group's compute, and uses ONE barrier
// per group instead of two. Needs 2x LDS, so BM is capped (BM<=96 at BN=128,
// G=128 to fit 64KB). Templated <BM,BN,NWARPS>; CFG via VLLM_W4A8_V8_CFG.
// ============================================================
template <int BM, int BN, int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32) mmq_fp8_gemm_kernel_v8(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const int*           __restrict__ w_packed,
    const __half*        __restrict__ w_scales,
    const int*           __restrict__ w_zeros_packed,
    __half*              __restrict__ out,
    int M, int N, int K, int group_size) {

    constexpr int NFRAG = BN / WMMA_DIM;
    constexpr int LDS_PAD = 8;
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int nthreads = NWARPS * 32;
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
    const int LDSBK = BK + LDS_PAD;

    extern __shared__ __attribute__((aligned(16))) unsigned char v8_smem[];
    unsigned char* A_buf[2];
    unsigned char* B_buf[2];
    A_buf[0] = v8_smem;
    A_buf[1] = A_buf[0] + BM * LDSBK;
    B_buf[0] = A_buf[1] + BM * LDSBK;
    B_buf[1] = B_buf[0] + BN * LDSBK;

    // stage group g into buffer `buf`: A copy (uint32) + B int4->fp8 expand.
    auto stage = [&](int g, int buf) {
        const int k0 = g * BK;
        unsigned char* At = A_buf[buf];
        unsigned char* Bt = B_buf[buf];
        for (int idx = tid * 4; idx < BM * BK; idx += nthreads * 4) {
            const int r = idx / BK, k = idx % BK, am = block_m + r;
            unsigned int v = 0u;
            if (am < M)
                v = *reinterpret_cast<const unsigned int*>(&x_fp8[am * K + k0 + k]);
            *reinterpret_cast<unsigned int*>(&At[r * LDSBK + k]) = v;
        }
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
    };

    float running[NFRAG][8];
    #pragma unroll
    for (int f = 0; f < NFRAG; ++f)
        #pragma unroll
        for (int e = 0; e < 8; ++e) running[f][e] = 0.0f;

    stage(0, 0);
    __syncthreads();

    const int a_base = (warp_id * WMMA_DIM + lrow) * LDSBK + khalf;
    const int b_base = lrow * LDSBK + khalf;

    for (int g = 0; g < num_k_groups; ++g) {
        const int cur = g & 1;
        // prefetch next group into the OTHER buffer; overlaps the compute below
        // (different buffer, no barrier between) so global latency hides.
        if (g + 1 < num_k_groups) stage(g + 1, (g + 1) & 1);

        const unsigned char* At = A_buf[cur];
        const unsigned char* Bt = B_buf[cur];
        v8f_t acc[NFRAG];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) acc[f] = v8f_t{0, 0, 0, 0, 0, 0, 0, 0};

        v2i_t a_cur = *reinterpret_cast<const v2i_t*>(&At[a_base]);
        v2i_t b_cur[NFRAG];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f)
            b_cur[f] = *reinterpret_cast<const v2i_t*>(&Bt[b_base + f * WMMA_DIM * LDSBK]);

        for (int kt = 0; kt < k_sub; ++kt) {
            v2i_t a_nx;
            v2i_t b_nx[NFRAG];
            if (kt + 1 < k_sub) {
                const int ko = (kt + 1) * WMMA_DIM;
                a_nx = *reinterpret_cast<const v2i_t*>(&At[a_base + ko]);
                #pragma unroll
                for (int f = 0; f < NFRAG; ++f)
                    b_nx[f] = *reinterpret_cast<const v2i_t*>(
                        &Bt[b_base + f * WMMA_DIM * LDSBK + ko]);
            }
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f)
                acc[f] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    a_cur, b_cur[f], acc[f]);
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
        __syncthreads();  // stage(g+1) done + compute(g) reads done before reuse
    }

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int abs_m = block_m + warp_id * WMMA_DIM + frag_row0 + e;
        if (abs_m >= M) continue;
        const float asc = act_scales[abs_m];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            if (abs_n < N) out[abs_m * N + abs_n] = __float2half(running[f][e] * asc);
        }
    }
}

// ============================================================
// v9: KEEP the winning BM=256/BN=128 tile (8-acc ILP) AND double-buffer, by
// DECOUPLING the LDS staging depth from the quant group size. We stage BK_LDS
// (=64 for g128) -deep chunks and ping-pong two buffers, so 2x LDS fits in 64KB
// (vs the full-128-group double-buffer that didn't). This overlaps the next
// chunk's global load + int4->fp8 expand with the current chunk's compute and
// uses one barrier per chunk. The fp32 accumulator persists across the NCHUNK
// chunks of a quant group; the per-group scale is applied at the group boundary.
// Requires a compile-time group size (BKT in {32,128}); compile-time BK also
// folds the staging div/mod to shifts. Templated <BM,BN,NWARPS,BKT>.
// ============================================================
template <int BM, int BN, int NWARPS, int BKT>
__global__ void __launch_bounds__(NWARPS * 32) mmq_fp8_gemm_kernel_v9(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const int*           __restrict__ w_packed,
    const __half*        __restrict__ w_scales,
    const int*           __restrict__ w_zeros_packed,
    __half*              __restrict__ out,
    int M, int N, int K, int group_size) {

    constexpr int NFRAG = BN / WMMA_DIM;
    constexpr int LDS_PAD = 8;
    constexpr int BK_LDS = (BKT >= 128) ? 64 : BKT;  // chunk depth
    constexpr int NCHUNK = BKT / BK_LDS;             // chunks per quant group
    constexpr int CK_SUB = BK_LDS / WMMA_DIM;        // WMMA subtiles per chunk
    constexpr int LDSBK = BK_LDS + LDS_PAD;

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int nthreads = NWARPS * 32;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lrow = lane & 15;
    const int khalf = (lane >> 4) * 8;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;

    const int num_k_groups = K / BKT;
    const int packed_per_row = K / PACK_FACTOR;
    const int total_chunks = K / BK_LDS;

    extern __shared__ __attribute__((aligned(16))) unsigned char v9_smem[];
    unsigned char* A_buf[2];
    unsigned char* B_buf[2];
    A_buf[0] = v9_smem;
    A_buf[1] = A_buf[0] + BM * LDSBK;
    B_buf[0] = A_buf[1] + BM * LDSBK;
    B_buf[1] = B_buf[0] + BN * LDSBK;

    // stage chunk c (BK_LDS columns at k0=c*BK_LDS) into buffer `buf`.
    auto stage = [&](int c, int buf) {
        const int k0 = c * BK_LDS;
        const int g = c / NCHUNK;     // quant group (for zp)
        unsigned char* At = A_buf[buf];
        unsigned char* Bt = B_buf[buf];
        for (int idx = tid * 4; idx < BM * BK_LDS; idx += nthreads * 4) {
            const int r = idx / BK_LDS, k = idx % BK_LDS, am = block_m + r;
            unsigned int v = 0u;
            if (am < M)
                v = *reinterpret_cast<const unsigned int*>(&x_fp8[am * K + k0 + k]);
            *reinterpret_cast<unsigned int*>(&At[r * LDSBK + k]) = v;
        }
        const int njobs = BN * (BK_LDS / PACK_FACTOR);
        for (int j = tid; j < njobs; j += nthreads) {
            const int n = j / (BK_LDS / PACK_FACTOR), k8 = j % (BK_LDS / PACK_FACTOR);
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
    };

    float running[NFRAG][8];
    #pragma unroll
    for (int f = 0; f < NFRAG; ++f)
        #pragma unroll
        for (int e = 0; e < 8; ++e) running[f][e] = 0.0f;

    v8f_t acc[NFRAG];
    #pragma unroll
    for (int f = 0; f < NFRAG; ++f) acc[f] = v8f_t{0, 0, 0, 0, 0, 0, 0, 0};

    const int a_base = (warp_id * WMMA_DIM + lrow) * LDSBK + khalf;
    const int b_base = lrow * LDSBK + khalf;

    stage(0, 0);
    __syncthreads();

    for (int c = 0; c < total_chunks; ++c) {
        const int cur = c & 1;
        if (c + 1 < total_chunks) stage(c + 1, (c + 1) & 1);

        const unsigned char* At = A_buf[cur];
        const unsigned char* Bt = B_buf[cur];
        v2i_t a_cur = *reinterpret_cast<const v2i_t*>(&At[a_base]);
        v2i_t b_cur[NFRAG];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f)
            b_cur[f] = *reinterpret_cast<const v2i_t*>(&Bt[b_base + f * WMMA_DIM * LDSBK]);

        #pragma unroll
        for (int kt = 0; kt < CK_SUB; ++kt) {
            v2i_t a_nx;
            v2i_t b_nx[NFRAG];
            if (kt + 1 < CK_SUB) {
                const int ko = (kt + 1) * WMMA_DIM;
                a_nx = *reinterpret_cast<const v2i_t*>(&At[a_base + ko]);
                #pragma unroll
                for (int f = 0; f < NFRAG; ++f)
                    b_nx[f] = *reinterpret_cast<const v2i_t*>(
                        &Bt[b_base + f * WMMA_DIM * LDSBK + ko]);
            }
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f)
                acc[f] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    a_cur, b_cur[f], acc[f]);
            a_cur = a_nx;
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f) b_cur[f] = b_nx[f];
        }

        // quant-group boundary: apply per-group weight scale, reset accumulator.
        if ((c + 1) % NCHUNK == 0) {
            const int g = c / NCHUNK;
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f) {
                const int abs_n = block_n + f * WMMA_DIM + frag_col;
                const float wsc = (abs_n < N)
                    ? __half2float(w_scales[abs_n * num_k_groups + g]) : 0.0f;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    running[f][e] += acc[f][e] * wsc;
                    acc[f][e] = 0.0f;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int abs_m = block_m + warp_id * WMMA_DIM + frag_row0 + e;
        if (abs_m >= M) continue;
        const float asc = act_scales[abs_m];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            if (abs_n < N) out[abs_m * N + abs_n] = __float2half(running[f][e] * asc);
        }
    }
}

// ============================================================
// v10: get the (warp-private) activation A OUT of LDS via a warp-shuffle
// transpose, so LDS holds B (weights) ONLY -> frees ~34KB, the prerequisite to
// double-buffer B at the winning BM=256/BN=128 tile and kill the per-group
// barrier serialization (the ~35% floor). A is loaded coalesced from global
// (lane L -> row L/2, k-half L&1) then shuffled to the WMMA layout
// (lane L wants row L&15, k-half (L>>4)*8): src lane = 2*(L&15) + (L>>4).
// DB toggles B double-buffering. Step 1 (DB=false) validates the A-shuffle.
// ============================================================
template <int BM, int BN, int NWARPS, int BKT, bool DB>
__global__ void __launch_bounds__(NWARPS * 32) mmq_fp8_gemm_kernel_v10(
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

    extern __shared__ __attribute__((aligned(16))) unsigned char v10_smem[];
    unsigned char* B_buf[2];
    B_buf[0] = v10_smem;
    B_buf[1] = v10_smem + (DB ? BN * LDSBK : 0);
    __shared__ float v10_wsc[2][BN];
    // Fused act-quant: per-row e4m3 scale for this block's BM rows. Replaces the
    // separate compute_act_fp8_and_scales_kernel launch and the (M,K) x_fp8 HBM
    // round-trip (load_A quantizes fp16->e4m3 in-register right before the
    // warp-shuffle). Bit-exact with the old path.
    __shared__ float v10_act_scale[BM];
    compute_block_act_scales(x, M, K, block_m, BM, tid, nthreads, v10_act_scale);

    // coalesced global A load for this warp's 16 rows at K-offset k0+kt*16,
    // quantize fp16->e4m3 in-register (per-row scale), then transpose to the WMMA
    // fragment layout via warp shuffle.
    // (lane L -> row L/2, k-half L&1; shuffle src = 2*(L&15)+(L>>4)).
    const int a_lrow = warp_id * WMMA_DIM + (lane >> 1);   // row within block
    const int a_row = block_m + a_lrow;
    const int a_kh = (lane & 1) * 8;
    const int a_shfl_src = 2 * (lane & 15) + (lane >> 4);
    const float a_inv_scale = 1.0f / v10_act_scale[a_lrow];
    auto load_A = [&](int k0, int kt) -> v2i_t {
        const int gk = k0 + kt * WMMA_DIM + a_kh;
        v2i_t v = stage_act_v2i(x, M, K, a_row, gk, a_inv_scale);
        v2i_t w;
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
            v10_wsc[buf][i] = (an < N)
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

        v8f_t acc[NFRAG];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) acc[f] = v8f_t{0, 0, 0, 0, 0, 0, 0, 0};

        v2i_t a_cur = load_A(k0, 0);
        for (int kt = 0; kt < k_sub; ++kt) {
            v2i_t a_nx;
            if (kt + 1 < k_sub) a_nx = load_A(k0, kt + 1);
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f) {
                v2i_t b = *reinterpret_cast<const v2i_t*>(
                    &Bt[b_base + f * WMMA_DIM * LDSBK + kt * WMMA_DIM]);
                acc[f] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    a_cur, b, acc[f]);
            }
            a_cur = a_nx;
        }

        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
            const float wsc = v10_wsc[cur][f * WMMA_DIM + frag_col];
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
        const float asc = v10_act_scale[lrow_m];   // fused per-row act scale
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            if (abs_n < N) out[abs_m * N + abs_n] = __float2half(running[f][e] * asc);
        }
    }
}

// ============================================================
// v11: DECODE-specialised W4A8 GEMV for small M. No WMMA, no B-LDS, no barriers
// -- it just STREAMS the weights coalesced and does the dot product, so it hits
// memory bandwidth instead of paying the prefill tile machinery. One warp per
// output column n: the 32 lanes stream w_packed[n] (its K/8 int32 row) in b128
// chunks (v4i = 4 int32 = 32 fp8/lane), expand int4->fp8 in-register, multiply by
// the staged activations, accumulate with the per-group weight scale. Activations
// (M,K) fp8 are staged once into LDS (needs M*K <= ~64KB, i.e. small M). M
// accumulators per lane; warp-reduce -> out[m][n]. Requires K % 1024 == 0 (warp
// step = 32 lanes * 32 k) and group_size % 32 == 0.
// ============================================================
template <int NWARPS, int MMAX>
__global__ void __launch_bounds__(NWARPS * 32) mmq_fp8_gemv_kernel_v11(
    const unsigned char* __restrict__ x_fp8,    // (M, K)
    const float*         __restrict__ act_scales,
    const int*           __restrict__ w_packed, // (N, K/8)
    const __half*        __restrict__ w_scales, // (N, K/g)
    const int*           __restrict__ w_zeros,  // (N/8, K/g) or null
    __half*              __restrict__ out,       // (M, N)
    int M, int N, int K, int group_size, int BK) {  // BK: K-chunk staged in LDS

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int n = blockIdx.x * NWARPS + warp_id;   // output column for this warp

    const int num_groups = K / group_size;

    // K-TILED: stage only (M, BK) fp8 into LDS per chunk and accumulate across
    // chunks in registers, so LDS = M*BK (<=64KB) is independent of total K. This
    // lifts the old M*K<=65536 cap that locked v11 out of large-K decode layers.
    extern __shared__ __attribute__((aligned(16))) unsigned char v11_a[];

    // weight row + zero-point indexing for this output column (guarded for n>=N:
    // those warps still cooperate in the staging loop + barriers, but skip compute).
    const int* wq = (n < N) ? w_packed + (long)n * (K / PACK_FACTOR) : nullptr;
    const int zrow = (n / 8) * num_groups;
    const int zsh = (n % 8) * 4;

    float acc[MMAX];
    #pragma unroll
    for (int m = 0; m < MMAX; ++m) acc[m] = 0.0f;

    for (int kc = 0; kc < K; kc += BK) {
        const int ck = (K - kc < BK) ? (K - kc) : BK;    // this chunk's k (mult of 1024)
        // stage activation slice (M, ck) cooperatively: v11_a[m*ck+kk] = x_fp8[m*K+kc+kk]
        const int tile = M * ck;
        for (int i = tid; i < tile; i += NWARPS * 32) {
            const int m = i / ck, kk = i - m * ck;
            v11_a[i] = x_fp8[(long)m * K + kc + kk];
        }
        __syncthreads();
        if (n < N) {
            const int* wqc = wq + kc / PACK_FACTOR;       // weights for this K-chunk
            const int ppr_chunk = ck / PACK_FACTOR;
            // each lane streams int32 indices lane*4 in steps of 32*4 (b128/lane).
            for (int base = lane * 4; base < ppr_chunk; base += 32 * 4) {
                const v4i_t wv = *reinterpret_cast<const v4i_t*>(&wqc[base]);
                const int k0 = base * PACK_FACTOR;        // first k of these 32, within chunk
                const int g = (kc + k0) / group_size;     // global group (32 k stay in one)
                const float wsc = __half2float(w_scales[(long)n * num_groups + g]);
                int zp = 8;
                if (w_zeros != nullptr) zp = (w_zeros[zrow + g] >> zsh) & 0xF;
                // weight -> f32 ONCE (nibble-zp in [-15,15] is exact in e4m3, so the
                // old int4->fp8->f32 round-trip == (float)(nibble-zp)); reuse across the
                // M rows. Activations packed-converted 2/instruction. Same recipe that
                // took the MoE GEMV from 1.x to <1.0 vs stock.
                float wf[32];
                #pragma unroll
                for (int t = 0; t < 32; ++t)
                    wf[t] = (float)(((wv[t >> 3] >> ((t & 7) * 4)) & 0xF) - zp);
                for (int m = 0; m < M; ++m) {
                    const unsigned int* ar4 =
                        reinterpret_cast<const unsigned int*>(v11_a + m * ck + k0);
                    float p = 0.0f;
                    #pragma unroll
                    for (int q = 0; q < 8; ++q) {
                        const unsigned int a4 = ar4[q];
                        const v2f_t lo = __builtin_amdgcn_cvt_pk_f32_fp8(a4, false);
                        const v2f_t hi = __builtin_amdgcn_cvt_pk_f32_fp8(a4, true);
                        const int b = q * 4;
                        p += lo[0] * wf[b] + lo[1] * wf[b + 1]
                           + hi[0] * wf[b + 2] + hi[1] * wf[b + 3];
                    }
                    acc[m] += p * wsc;
                }
            }
        }
        __syncthreads();   // before the next chunk overwrites the LDS tile
    }

    if (n >= N) return;
    // warp-reduce each m's partial across the 32 lanes.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        for (int m = 0; m < M; ++m)
            acc[m] += __shfl_xor(acc[m], off);

    if (lane == 0)
        for (int m = 0; m < M; ++m)
            out[(long)m * N + n] = __float2half(acc[m] * act_scales[m]);
}

// ============================================================
// v12: SMALL-M split-K WMMA. The small-batch (M=4-64) regime is occupancy-bound,
// not FLOP-bound: a single full-K tile launches only N/BN blocks (e.g. 32 at
// M<=16, N=4096), starving the 64-CU GPU -> ~108 GB/s weight read, ~1.8x behind
// Triton's register-direct fp16 path. v12 SPLITS K across `splitk` blocks
// (grid.z), each accumulating its K-range's partial and atomicAdd-ing into an
// fp32 (M,N) accumulator -> splitk x more blocks in flight, same HBM traffic. The
// per-row act scale distributes over the split sum, so each block applies it to
// its partial before the atomic. Structure == v5 (raw fp8 WMMA, bank-padded LDS)
// over a group sub-range. Output is fp32; a convert kernel casts to fp16.
// ============================================================
template <int BM, int BN>
__global__ void __launch_bounds__((BM / 16) * 32) mmq_fp8_gemm_kernel_v12(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const int*           __restrict__ w_packed,
    const __half*        __restrict__ w_scales,
    const int*           __restrict__ w_zeros_packed,
    float*               __restrict__ out_acc,   // (M, N) fp32, pre-zeroed (atomic)
    int M, int N, int K, int group_size, int splitk) {

    constexpr int NWARPS = BM / 16;
    constexpr int NFRAG = BN / 16;
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
    constexpr int LDS_PAD = 8;
    const int LDSBK = BK + LDS_PAD;

    // this block's contiguous K-group sub-range [g0, g1).
    const int gps = (num_k_groups + splitk - 1) / splitk;
    const int g0 = blockIdx.z * gps;
    int g1 = g0 + gps;
    if (g1 > num_k_groups) g1 = num_k_groups;
    if (g0 >= g1) return;

    __shared__ unsigned char A_tile[BM * (MAX_GROUP_SIZE + LDS_PAD)];
    __shared__ unsigned char B_tile[BN * (MAX_GROUP_SIZE + LDS_PAD)];

    float running[NFRAG][8];
    #pragma unroll
    for (int f = 0; f < NFRAG; ++f)
        #pragma unroll
        for (int e = 0; e < 8; ++e) running[f][e] = 0.0f;

    for (int g = g0; g < g1; ++g) {
        const int k0 = g * BK;
        for (int idx = tid * 4; idx < BM * BK; idx += (NWARPS * 32) * 4) {
            const int r = idx / BK, k = idx % BK, am = block_m + r;
            unsigned int v = 0u;
            if (am < M)
                v = *reinterpret_cast<const unsigned int*>(&x_fp8[am * K + k0 + k]);
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
            if (an < N) {
                // pack 2 nibbles per cvt_pk_fp8_f32 (was 1 via cvt_pk(v,v)) -> halves
                // the expansion converts, which cap this LDS-staged path's throughput.
                #pragma unroll
                for (int jj = 0; jj < PACK_FACTOR; jj += 2) {
                    const float a = (float)(((word >> (jj * 4)) & 0xF) - zp);
                    const float b = (float)(((word >> ((jj + 1) * 4)) & 0xF) - zp);
                    const int p = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
                    dst[jj] = p & 0xFF;
                    dst[jj + 1] = (p >> 8) & 0xFF;
                }
            } else {
                #pragma unroll
                for (int jj = 0; jj < PACK_FACTOR; ++jj) dst[jj] = 0;
            }
        }
        __syncthreads();

        v8f_t acc[NFRAG];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) acc[f] = v8f_t{0, 0, 0, 0, 0, 0, 0, 0};

        const int a_base = (warp_id * WMMA_DIM + lrow) * LDSBK + khalf;
        const int b_base = lrow * LDSBK + khalf;
        v2i_t a_cur = *reinterpret_cast<const v2i_t*>(&A_tile[a_base]);
        v2i_t b_cur[NFRAG];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f)
            b_cur[f] = *reinterpret_cast<const v2i_t*>(&B_tile[b_base + f * WMMA_DIM * LDSBK]);

        for (int kt = 0; kt < k_sub; ++kt) {
            v2i_t a_nx;
            v2i_t b_nx[NFRAG];
            if (kt + 1 < k_sub) {
                const int ko = (kt + 1) * WMMA_DIM;
                a_nx = *reinterpret_cast<const v2i_t*>(&A_tile[a_base + ko]);
                #pragma unroll
                for (int f = 0; f < NFRAG; ++f)
                    b_nx[f] = *reinterpret_cast<const v2i_t*>(
                        &B_tile[b_base + f * WMMA_DIM * LDSBK + ko]);
            }
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f)
                acc[f] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    a_cur, b_cur[f], acc[f]);
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

    // epilogue: per-row act scale, atomic-accumulate the K-split partial.
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int abs_m = block_m + warp_id * WMMA_DIM + frag_row0 + e;
        if (abs_m >= M) continue;
        const float asc = act_scales[abs_m];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            if (abs_n < N) atomicAdd(&out_acc[abs_m * N + abs_n], running[f][e] * asc);
        }
    }
}

__global__ void cvt_f32_f16_kernel(const float* __restrict__ in,
                                   __half* __restrict__ out, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

// ============================================================
// v13: SMALL-M REGISTER-DIRECT WMMA + split-K. The small-batch (M=4-64) loss is
// the int4->fp8 LDS round-trip: v5/v10 stage expanded weights in LDS (108 GB/s)
// where Triton dequants register-direct (190). v13 feeds the fp8 WMMA B operand
// straight from registers: load int4 weights COALESCED (lane L -> col L/2,
// k-half L&1: 2-lane 8B bursts, the same pattern v10's A-shuffle hits 273 GB/s
// with), warp-SHUFFLE to the B-fragment layout (src = 2*(L&15)+(L>>4)), expand
// int4->fp8 in-register -- no B LDS, no barrier. A is register-direct too (v10's
// shuffle). Produces the EXACT v5 fragment values, so bit-identical. split-K
// (grid.z) restores occupancy at small M (few N-tiles otherwise). fp32 atomic out.
// ============================================================
template <int BM, int BN>
__global__ void __launch_bounds__((BM / 16) * 32) mmq_fp8_gemm_kernel_v13(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const int*           __restrict__ w_packed,
    const __half*        __restrict__ w_scales,
    const int*           __restrict__ w_zeros_packed,
    float*               __restrict__ out_acc,   // (M, N) fp32, pre-zeroed (atomic)
    int M, int N, int K, int group_size, int splitk) {

    constexpr int NWARPS = BM / 16;
    constexpr int NFRAG = BN / 16;
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;

    const int BK = group_size;
    const int num_k_groups = K / BK;
    const int ppr = K / PACK_FACTOR;
    const int k_sub = BK / WMMA_DIM;

    const int gps = (num_k_groups + splitk - 1) / splitk;
    const int g0 = blockIdx.z * gps;
    int g1 = g0 + gps;
    if (g1 > num_k_groups) g1 = num_k_groups;
    if (g0 >= g1) return;

    // A register-direct (coalesced load + shuffle to fragment), like v10.
    const int a_row = block_m + warp_id * WMMA_DIM + (lane >> 1);
    const int a_kh = (lane & 1) * 8;
    const int shfl_src = 2 * (lane & 15) + (lane >> 4);
    // B register-direct: this lane loads column (L/2) of each N-frag, k-half (L&1).
    const int b_col_lane = lane >> 1;   // 0..15 -> which of the frag's 16 cols
    const int b_whalf = lane & 1;       // which 8-k word within the 16-k subtile

    float running[NFRAG][8];
    #pragma unroll
    for (int f = 0; f < NFRAG; ++f)
        #pragma unroll
        for (int e = 0; e < 8; ++e) running[f][e] = 0.0f;

    for (int g = g0; g < g1; ++g) {
        const int k0 = g * BK;
        v8f_t acc[NFRAG];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) acc[f] = v8f_t{0, 0, 0, 0, 0, 0, 0, 0};

        for (int kt = 0; kt < k_sub; ++kt) {
            const int k_abs = k0 + kt * WMMA_DIM;
            // ---- A: coalesced global load + warp-shuffle to fragment ----
            v2i_t av = v2i_t{0, 0};
            if (a_row < M)
                av = *reinterpret_cast<const v2i_t*>(&x_fp8[(long)a_row * K + k_abs + a_kh]);
            v2i_t a_cur;
            a_cur[0] = __shfl(av[0], shfl_src);
            a_cur[1] = __shfl(av[1], shfl_src);
            // ---- B: per frag, coalesced int4 load + shuffle + in-register expand ----
            #pragma unroll
            for (int f = 0; f < NFRAG; ++f) {
                const int load_col = block_n + f * WMMA_DIM + b_col_lane;
                int wword = (load_col < N)
                    ? w_packed[(long)load_col * ppr + k_abs / PACK_FACTOR + b_whalf] : 0;
                const int b_word = __shfl(wword, shfl_src);  // -> col (L&15), k-half (L>>4)
                const int nf = block_n + f * WMMA_DIM + (lane & 15);
                int zp = 8;
                if (w_zeros_packed != nullptr && nf < N)
                    zp = (w_zeros_packed[(nf / 8) * num_k_groups + g] >> ((nf % 8) * 4)) & 0xF;
                unsigned char bf[8];
                #pragma unroll
                for (int jj = 0; jj < 8; ++jj)
                    bf[jj] = int4_signed_to_e4m3(((b_word >> (jj * 4)) & 0xF) - zp);
                const v2i_t b_cur = *reinterpret_cast<const v2i_t*>(bf);
                acc[f] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    a_cur, b_cur, acc[f]);
            }
        }
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            const float wsc = (abs_n < N)
                ? __half2float(w_scales[abs_n * num_k_groups + g]) : 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; ++e) running[f][e] += acc[f][e] * wsc;
        }
    }

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int abs_m = block_m + warp_id * WMMA_DIM + frag_row0 + e;
        if (abs_m >= M) continue;
        const float asc = act_scales[abs_m];
        #pragma unroll
        for (int f = 0; f < NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            if (abs_n < N) atomicAdd(&out_acc[abs_m * N + abs_n], running[f][e] * asc);
        }
    }
}

// ============================================================
// v14: SMALL-M WMMA with N-SPLIT warps. v10/v5 use BM=256 -> at M=8 the WMMA
// computes 256 rows for 8 real (32x wasted WMMA throughput). v14 fixes BM=16 (one
// M-subtile, <=16 real -> only 2x waste) but keeps NWARPS warps by splitting the N
// dimension instead of M: all warps share the 16 rows (one A staging), each warp
// owns WFRAG = (BN/16)/NWARPS of the N-fragments and cooperates on the B staging.
// So small BM (low WMMA waste) AND many staging threads (the thing BM=16 in v10
// lacked). split-K (grid.z) for occupancy; fp32 atomic out -> fp16.
// ============================================================
template <int BN, int NWARPS, int GTILE>
__global__ void __launch_bounds__(NWARPS * 32) mmq_fp8_gemm_kernel_v14(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const int*           __restrict__ w_packed,
    const __half*        __restrict__ w_scales,
    const int*           __restrict__ w_zeros_packed,
    float*               __restrict__ out_acc,   // (M,N) fp32 pre-zeroed (atomic)
    int M, int N, int K, int group_size, int splitk) {

    constexpr int NFRAG = BN / 16;
    constexpr int WFRAG = NFRAG / NWARPS;   // N-fragments per warp
    const int block_m = blockIdx.y * 16;    // BM = 16
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
    constexpr int LDS_PAD = 8;
    const int LDSBK = BK + LDS_PAD;

    const int gps = (num_k_groups + splitk - 1) / splitk;
    const int g0 = blockIdx.z * gps;
    int g1 = g0 + gps; if (g1 > num_k_groups) g1 = num_k_groups;
    if (g0 >= g1) return;

    // GTILE groups staged per __syncthreads pair (the v6 multi-group sync-reduction
    // win, applied to the dense N-split WMMA). Static LDS sized for GTILE groups.
    __shared__ unsigned char A_tile[GTILE * 16 * (MAX_GROUP_SIZE + LDS_PAD)];
    __shared__ unsigned char B_tile[GTILE * BN * (MAX_GROUP_SIZE + LDS_PAD)];

    float running[WFRAG][8];
    #pragma unroll
    for (int f = 0; f < WFRAG; ++f)
        #pragma unroll
        for (int e = 0; e < 8; ++e) running[f][e] = 0.0f;

    const int njobs = BN * (BK / PACK_FACTOR);
    for (int gt = g0; gt < g1; gt += GTILE) {
        const int gend = (gt + GTILE < g1) ? (gt + GTILE) : g1;
        for (int gi = gt; gi < gend; ++gi) {
            const int k0 = gi * BK;
            unsigned char* Ag = A_tile + (gi - gt) * 16 * LDSBK;
            unsigned char* Bg = B_tile + (gi - gt) * BN * LDSBK;
            for (int idx = tid * 4; idx < 16 * BK; idx += (NWARPS * 32) * 4) {
                const int r = idx / BK, k = idx % BK, am = block_m + r;
                unsigned int v = 0u;
                if (am < M)
                    v = *reinterpret_cast<const unsigned int*>(&x_fp8[am * K + k0 + k]);
                *reinterpret_cast<unsigned int*>(&Ag[r * LDSBK + k]) = v;
            }
            for (int j = tid; j < njobs; j += NWARPS * 32) {
                const int n = j / (BK / PACK_FACTOR), k8 = j % (BK / PACK_FACTOR);
                const int an = block_n + n;
                int word = 0, zp = 8;
                if (an < N) {
                    word = w_packed[an * packed_per_row + (k0 / PACK_FACTOR) + k8];
                    if (w_zeros_packed != nullptr) {
                        const int pk = w_zeros_packed[(an / 8) * num_k_groups + gi];
                        zp = (pk >> ((an % 8) * 4)) & 0xF;
                    }
                }
                unsigned char* dst = &Bg[n * LDSBK + k8 * PACK_FACTOR];
                if (an < N) {
                    #pragma unroll
                    for (int jj = 0; jj < PACK_FACTOR; jj += 2) {
                        const float a = (float)(((word >> (jj * 4)) & 0xF) - zp);
                        const float b = (float)(((word >> ((jj + 1) * 4)) & 0xF) - zp);
                        const int p = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
                        dst[jj] = p & 0xFF; dst[jj + 1] = (p >> 8) & 0xFF;
                    }
                } else {
                    #pragma unroll
                    for (int jj = 0; jj < PACK_FACTOR; ++jj) dst[jj] = 0;
                }
            }
        }
        __syncthreads();

        for (int gi = gt; gi < gend; ++gi) {
            const unsigned char* Ag = A_tile + (gi - gt) * 16 * LDSBK;
            const unsigned char* Bg = B_tile + (gi - gt) * BN * LDSBK;
            v8f_t acc[WFRAG];
            #pragma unroll
            for (int f = 0; f < WFRAG; ++f) acc[f] = v8f_t{0, 0, 0, 0, 0, 0, 0, 0};
            const int a_base = lrow * LDSBK + khalf;       // all warps share the 16 rows
            for (int kt = 0; kt < k_sub; ++kt) {
                const v2i_t a_cur = *reinterpret_cast<const v2i_t*>(&Ag[a_base + kt * WMMA_DIM]);
                #pragma unroll
                for (int wf = 0; wf < WFRAG; ++wf) {
                    const int f = warp_id * WFRAG + wf;     // global N-fragment
                    const v2i_t b_cur = *reinterpret_cast<const v2i_t*>(
                        &Bg[(f * WMMA_DIM + lrow) * LDSBK + khalf + kt * WMMA_DIM]);
                    acc[wf] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        a_cur, b_cur, acc[wf]);
                }
            }
            #pragma unroll
            for (int wf = 0; wf < WFRAG; ++wf) {
                const int f = warp_id * WFRAG + wf;
                const int abs_n = block_n + f * WMMA_DIM + frag_col;
                const float wsc = (abs_n < N)
                    ? __half2float(w_scales[abs_n * num_k_groups + gi]) : 0.0f;
                #pragma unroll
                for (int e = 0; e < 8; ++e) running[wf][e] += acc[wf][e] * wsc;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int abs_m = block_m + frag_row0 + e;     // 16 rows, no warp offset
        if (abs_m >= M) continue;
        const float asc = act_scales[abs_m];
        #pragma unroll
        for (int wf = 0; wf < WFRAG; ++wf) {
            const int f = warp_id * WFRAG + wf;
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            if (abs_n < N) atomicAdd(&out_acc[abs_m * N + abs_n], running[wf][e] * asc);
        }
    }
}

// ============================================================
// v15: MARLIN-style REGISTER-DIRECT WMMA. Weights are pre-repacked offline into
// WMMA-B-fragment lane-order -- w_rep[n_tile][k_tile][lane] holds exactly the 8
// int4 nibbles lane `lane` needs for the (col=16*n_tile+(lane&15),
// k=16*k_tile+(lane>>4)*8 .. +8) fragment slot. So a single coalesced 128-byte
// load per warp (32 consecutive int32) fills the B fragment with NO LDS staging
// and NO warp shuffle -- the thing that capped the LDS path at ~108 GB/s while
// Triton's register-resident tl.dot hit ~195. int4->fp8 expand stays in-register.
// A is staged to LDS (16 rows, small). N-split warps (BM=16), split-K, fp32 atomic.
// ============================================================
template <int BN, int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32) mmq_fp8_gemm_kernel_v15(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const int*           __restrict__ w_rep,       // (N/16, K/16, 32) int32 lane-order
    const __half*        __restrict__ w_scales,    // (N, K/group) fp16
    const int*           __restrict__ w_zeros_packed,
    float*               __restrict__ out_acc,     // (M, N) fp32 pre-zeroed (atomic)
    int M, int N, int K, int group_size, int splitk) {

    constexpr int NFRAG = BN / 16;
    constexpr int WFRAG = NFRAG / NWARPS;
    const int block_m = blockIdx.y * 16;
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
    const int k_sub = BK / WMMA_DIM;
    const int ktiles = K / WMMA_DIM;     // total 16-wide k-tiles
    constexpr int LDS_PAD = 8;
    const int LDSBK = BK + LDS_PAD;

    const int gps = (num_k_groups + splitk - 1) / splitk;
    const int g0 = blockIdx.z * gps;
    int g1 = g0 + gps; if (g1 > num_k_groups) g1 = num_k_groups;
    if (g0 >= g1) return;

    // FULLY register-direct: A via v10 warp-shuffle, B via repacked coalesced load.
    // NO LDS, NO __syncthreads -- directly attacks the latency-bound small-M wall
    // (the per-group A-staging barriers were the last sync left in the WMMA path).
    (void)LDSBK; (void)frag_row0; (void)lrow; (void)khalf;
    float running[WFRAG][8];
    #pragma unroll
    for (int f = 0; f < WFRAG; ++f)
        #pragma unroll
        for (int e = 0; e < 8; ++e) running[f][e] = 0.0f;

    const int* rep_w[WFRAG];
    #pragma unroll
    for (int wf = 0; wf < WFRAG; ++wf) {
        const int f = warp_id * WFRAG + wf;
        const int n_tile = block_n / 16 + f;
        rep_w[wf] = w_rep + (long)n_tile * ktiles * 32;
    }
    const int a_row = block_m + (lane >> 1);   // all warps share the 16 rows
    const int a_kh = (lane & 1) * 8;
    const int a_shfl = 2 * (lane & 15) + (lane >> 4);

    for (int g = g0; g < g1; ++g) {
        int zp_w[WFRAG];
        #pragma unroll
        for (int wf = 0; wf < WFRAG; ++wf) {
            const int f = warp_id * WFRAG + wf;
            const int nc = block_n + f * 16 + (lane & 15);
            zp_w[wf] = (w_zeros_packed != nullptr && nc < N)
                ? ((w_zeros_packed[(nc / 8) * num_k_groups + g] >> ((nc % 8) * 4)) & 0xF) : 8;
        }
        v8f_t acc[WFRAG];
        #pragma unroll
        for (int f = 0; f < WFRAG; ++f) acc[f] = v8f_t{0, 0, 0, 0, 0, 0, 0, 0};
        for (int kt = 0; kt < k_sub; ++kt) {
            const int gk = g * BK + kt * WMMA_DIM + a_kh;
            v2i_t av = v2i_t{0, 0};
            if (a_row < M)
                av = *reinterpret_cast<const v2i_t*>(&x_fp8[(long)a_row * K + gk]);
            v2i_t a_cur;
            a_cur[0] = __shfl(av[0], a_shfl);
            a_cur[1] = __shfl(av[1], a_shfl);
            const int kabs = g * k_sub + kt;
            #pragma unroll
            for (int wf = 0; wf < WFRAG; ++wf) {
                const int bword = rep_w[wf][kabs * 32 + lane];
                const int zp = zp_w[wf];
                unsigned char bf[8];
                #pragma unroll
                for (int jj = 0; jj < 8; ++jj)
                    bf[jj] = int4_signed_to_e4m3(((bword >> (jj * 4)) & 0xF) - zp);
                const v2i_t b_cur = *reinterpret_cast<const v2i_t*>(bf);
                acc[wf] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    a_cur, b_cur, acc[wf]);
            }
        }
        #pragma unroll
        for (int wf = 0; wf < WFRAG; ++wf) {
            const int f = warp_id * WFRAG + wf;
            const int abs_n = block_n + f * 16 + frag_col;
            const float wsc = (abs_n < N)
                ? __half2float(w_scales[abs_n * num_k_groups + g]) : 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; ++e) running[wf][e] += acc[wf][e] * wsc;
        }
    }   // no __syncthreads in the K loop -- fully register-direct

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int abs_m = block_m + frag_row0 + e;
        if (abs_m >= M) continue;
        const float asc = act_scales[abs_m];
        #pragma unroll
        for (int wf = 0; wf < WFRAG; ++wf) {
            const int f = warp_id * WFRAG + wf;
            const int abs_n = block_n + f * 16 + frag_col;
            if (abs_n < N) atomicAdd(&out_acc[abs_m * N + abs_n], running[wf][e] * asc);
        }
    }
}

// ============================================================
// v16: the ONE structural axis untested across the 20 fp8 variants -- f16 WMMA.
// Triton wins dense M=16-32 with an fp16 tl.dot (int4->fp16 register-direct);
// every hand-written variant so far used the fp8 WMMA. v16 is v15's exact
// zero-LDS / zero-sync structure (same A-shuffle, same w_rep B-layout) but feeds
// __builtin_amdgcn_wmma_f32_16x16x16_f16 -- A = fp8->f16 (raw value, scale at
// epilogue), B = (nibble-zp) as f16. Tests whether the fp8-WMMA instruction's
// scheduling (not the data path) was the ~108 GB/s small-GEMM wall.
// ============================================================
template <int BN, int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32) mmq_fp8_gemm_kernel_v16(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const int*           __restrict__ w_rep,       // (N/16, K/16, 32) int32 lane-order
    const __half*        __restrict__ w_scales,    // (N, K/group) fp16
    const int*           __restrict__ w_zeros_packed,
    float*               __restrict__ out_acc,     // (M, N) fp32 pre-zeroed (atomic)
    int M, int N, int K, int group_size, int splitk) {

    constexpr int NFRAG = BN / 16;
    constexpr int WFRAG = NFRAG / NWARPS;
    const int block_m = blockIdx.y * 16;
    const int block_n = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;

    const int BK = group_size;
    const int num_k_groups = K / BK;
    const int k_sub = BK / WMMA_DIM;
    const int ktiles = K / WMMA_DIM;

    const int gps = (num_k_groups + splitk - 1) / splitk;
    const int g0 = blockIdx.z * gps;
    int g1 = g0 + gps; if (g1 > num_k_groups) g1 = num_k_groups;
    if (g0 >= g1) return;

    float running[WFRAG][8];
    #pragma unroll
    for (int f = 0; f < WFRAG; ++f)
        #pragma unroll
        for (int e = 0; e < 8; ++e) running[f][e] = 0.0f;

    const int* rep_w[WFRAG];
    #pragma unroll
    for (int wf = 0; wf < WFRAG; ++wf) {
        const int f = warp_id * WFRAG + wf;
        const int n_tile = block_n / 16 + f;
        rep_w[wf] = w_rep + (long)n_tile * ktiles * 32;
    }
    const int a_row = block_m + (lane >> 1);   // all warps share the 16 rows
    const int a_kh = (lane & 1) * 8;
    const int a_shfl = 2 * (lane & 15) + (lane >> 4);

    for (int g = g0; g < g1; ++g) {
        int zp_w[WFRAG];
        #pragma unroll
        for (int wf = 0; wf < WFRAG; ++wf) {
            const int f = warp_id * WFRAG + wf;
            const int nc = block_n + f * 16 + (lane & 15);
            zp_w[wf] = (w_zeros_packed != nullptr && nc < N)
                ? ((w_zeros_packed[(nc / 8) * num_k_groups + g] >> ((nc % 8) * 4)) & 0xF) : 8;
        }
        v8f_t acc[WFRAG];
        #pragma unroll
        for (int f = 0; f < WFRAG; ++f) acc[f] = v8f_t{0, 0, 0, 0, 0, 0, 0, 0};
        for (int kt = 0; kt < k_sub; ++kt) {
            const int gk = g * BK + kt * WMMA_DIM + a_kh;
            v2i_t av = v2i_t{0, 0};
            if (a_row < M)
                av = *reinterpret_cast<const v2i_t*>(&x_fp8[(long)a_row * K + gk]);
            v2i_t a_cur;
            a_cur[0] = __shfl(av[0], a_shfl);
            a_cur[1] = __shfl(av[1], a_shfl);
            // fp8 A fragment -> f16 (raw e4m3 value; per-row act_scale at epilogue)
            const unsigned char* ab = reinterpret_cast<const unsigned char*>(&a_cur);
            v8h_t a_h;
            #pragma unroll
            for (int jj = 0; jj < 8; ++jj) a_h[jj] = (_Float16)e4m3_to_f32(ab[jj]);
            const int kabs = g * k_sub + kt;
            #pragma unroll
            for (int wf = 0; wf < WFRAG; ++wf) {
                const int bword = rep_w[wf][kabs * 32 + lane];
                const int zp = zp_w[wf];
                v8h_t b_h;
                #pragma unroll
                for (int jj = 0; jj < 8; ++jj)
                    b_h[jj] = (_Float16)(((bword >> (jj * 4)) & 0xF) - zp);
                acc[wf] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    a_h, b_h, acc[wf]);
            }
        }
        #pragma unroll
        for (int wf = 0; wf < WFRAG; ++wf) {
            const int f = warp_id * WFRAG + wf;
            const int abs_n = block_n + f * 16 + frag_col;
            const float wsc = (abs_n < N)
                ? __half2float(w_scales[abs_n * num_k_groups + g]) : 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; ++e) running[wf][e] += acc[wf][e] * wsc;
        }
    }

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int abs_m = block_m + frag_row0 + e;
        if (abs_m >= M) continue;
        const float asc = act_scales[abs_m];
        #pragma unroll
        for (int wf = 0; wf < WFRAG; ++wf) {
            const int f = warp_id * WFRAG + wf;
            const int abs_n = block_n + f * 16 + frag_col;
            if (abs_n < N) atomicAdd(&out_acc[abs_m * N + abs_n], running[wf][e] * asc);
        }
    }
}

// ============================================================
// v17: true W4A16 (the real analogue of Triton's path). v16 showed f16 WMMA only
// marginally beats fp8 WMMA -- but v16 STILL fp8-quantizes activations (an extra
// act-quant kernel launch + per-row max + fp8->f16 round-trip), pure overhead
// Triton's W4A16 never pays. v17 feeds fp16 activations DIRECTLY to the f16 WMMA
// (no act-quant, no act_scale): isolates whether removing the W4A8 act-quant
// overhead lets the hand-written kernel reach Triton. splitk==1 writes f16 out
// directly (no out_acc / no cvt epilogue -- kills 2 more launches).
// ============================================================
template <int BN, int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32) mmq_w4a16_gemm_kernel_v17(
    const __half*        __restrict__ x_h,         // (M, K) fp16 -- DIRECT, no quant
    const int*           __restrict__ w_rep,       // (N/16, K/16, 32) int32 lane-order
    const __half*        __restrict__ w_scales,    // (N, K/group) fp16
    const int*           __restrict__ w_zeros_packed,
    float*               __restrict__ out_acc,     // (M, N) fp32 (atomic; used if splitk>1)
    __half*              __restrict__ out_h,        // (M, N) fp16 (direct; used if splitk==1)
    int M, int N, int K, int group_size, int splitk) {

    constexpr int NFRAG = BN / 16;
    constexpr int WFRAG = NFRAG / NWARPS;
    const int block_m = blockIdx.y * 16;
    const int block_n = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;

    const int BK = group_size;
    const int num_k_groups = K / BK;
    const int k_sub = BK / WMMA_DIM;
    const int ktiles = K / WMMA_DIM;

    const int gps = (num_k_groups + splitk - 1) / splitk;
    const int g0 = blockIdx.z * gps;
    int g1 = g0 + gps; if (g1 > num_k_groups) g1 = num_k_groups;
    if (g0 >= g1) return;

    float running[WFRAG][8];
    #pragma unroll
    for (int f = 0; f < WFRAG; ++f)
        #pragma unroll
        for (int e = 0; e < 8; ++e) running[f][e] = 0.0f;

    const int* rep_w[WFRAG];
    #pragma unroll
    for (int wf = 0; wf < WFRAG; ++wf) {
        const int f = warp_id * WFRAG + wf;
        const int n_tile = block_n / 16 + f;
        rep_w[wf] = w_rep + (long)n_tile * ktiles * 32;
    }
    const int a_row = block_m + (lane >> 1);
    const int a_kh = (lane & 1) * 8;
    const int a_shfl = 2 * (lane & 15) + (lane >> 4);

    for (int g = g0; g < g1; ++g) {
        int zp_w[WFRAG];
        #pragma unroll
        for (int wf = 0; wf < WFRAG; ++wf) {
            const int f = warp_id * WFRAG + wf;
            const int nc = block_n + f * 16 + (lane & 15);
            zp_w[wf] = (w_zeros_packed != nullptr && nc < N)
                ? ((w_zeros_packed[(nc / 8) * num_k_groups + g] >> ((nc % 8) * 4)) & 0xF) : 8;
        }
        v8f_t acc[WFRAG];
        #pragma unroll
        for (int f = 0; f < WFRAG; ++f) acc[f] = v8f_t{0, 0, 0, 0, 0, 0, 0, 0};
        for (int kt = 0; kt < k_sub; ++kt) {
            const int gk = g * BK + kt * WMMA_DIM + a_kh;
            v8h_t av = v8h_t{0, 0, 0, 0, 0, 0, 0, 0};
            if (a_row < M)
                av = *reinterpret_cast<const v8h_t*>(&x_h[(long)a_row * K + gk]);
            // shuffle the 8 fp16 (4x32-bit) to A-fragment lane order (same a_shfl as v15)
            v8h_t a_h;
            int* avi = reinterpret_cast<int*>(&av);
            int* ahi = reinterpret_cast<int*>(&a_h);
            #pragma unroll
            for (int i = 0; i < 4; ++i) ahi[i] = __shfl(avi[i], a_shfl);
            const int kabs = g * k_sub + kt;
            #pragma unroll
            for (int wf = 0; wf < WFRAG; ++wf) {
                const int bword = rep_w[wf][kabs * 32 + lane];
                const int zp = zp_w[wf];
                v8h_t b_h;
                #pragma unroll
                for (int jj = 0; jj < 8; ++jj)
                    b_h[jj] = (_Float16)(((bword >> (jj * 4)) & 0xF) - zp);
                acc[wf] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    a_h, b_h, acc[wf]);
            }
        }
        #pragma unroll
        for (int wf = 0; wf < WFRAG; ++wf) {
            const int f = warp_id * WFRAG + wf;
            const int abs_n = block_n + f * 16 + frag_col;
            const float wsc = (abs_n < N)
                ? __half2float(w_scales[abs_n * num_k_groups + g]) : 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; ++e) running[wf][e] += acc[wf][e] * wsc;
        }
    }

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int abs_m = block_m + frag_row0 + e;
        if (abs_m >= M) continue;
        #pragma unroll
        for (int wf = 0; wf < WFRAG; ++wf) {
            const int f = warp_id * WFRAG + wf;
            const int abs_n = block_n + f * 16 + frag_col;
            if (abs_n >= N) continue;
            if (splitk == 1) out_h[abs_m * N + abs_n] = (__half)running[wf][e];   // direct, no cvt
            else atomicAdd(&out_acc[abs_m * N + abs_n], running[wf][e]);
        }
    }
}

}  // namespace w4a8_fp8_wmma

// =================== Host launcher ===================
void launch_mmq_fp8_gemm_gfx1201(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const at::Tensor& w_zeros,
    at::Tensor& out,
    int64_t kernel) {

    using namespace w4a8_fp8_wmma;
    using DK = w4a8::DenseKernel;
    const DK dk = static_cast<DK>(kernel);   // opaque int -> scoped tag (values == old version ints)

    const int M = x.size(0);
    const int K = x.size(1);
    const int N = w_packed.size(0);
    const int group_size = K / scales.size(1);

    TORCH_CHECK(K % group_size == 0, "K not divisible by derived group_size");
    TORCH_CHECK(group_size % WMMA_DIM == 0 && group_size <= MAX_GROUP_SIZE,
                "group_size must be a multiple of 16 and <= 128; got ", group_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // The dispatched WMMA path (prefill_wmma / prefill_wmma_ashuffle) FUSES the
    // activation fp8-quant into the GEMM prologue (per-row amax + e4m3 expand
    // in-kernel), so it consumes the fp16 activations directly -- no separate
    // launch, no (M,K) x_fp8 HBM round-trip. Every other (non-dispatched) kernel
    // still uses the standalone pre-kernel + x_fp8 buffer below.
    const __half* x_h = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    const bool fused_actquant = (dk == DK::PrefillWmma || dk == DK::PrefillWmmaAshuffle);
    at::Tensor x_fp8, act_scales;
    if (!fused_actquant) {
        x_fp8 = at::empty({M, K}, x.options().dtype(at::kByte));
        act_scales = at::empty({M}, x.options().dtype(at::kFloat));
        compute_act_fp8_and_scales_kernel<<<dim3(M), dim3(ACTQUANT_BLOCK), 0, stream>>>(
            x_h, x_fp8.data_ptr<unsigned char>(), act_scales.data_ptr<float>(), M, K);
    }

    const int* w_zeros_ptr = (w_zeros.defined() && w_zeros.numel() > 0)
                             ? w_zeros.data_ptr<int>() : nullptr;

    if (dk == DK::RegdirectShuffle) {  // v13
        // small-M register-direct WMMA (no B-LDS) + split-K. CFG VLLM_W4A8_V13_CFG
        // (BMxBN, def 16x64), VLLM_W4A8_V13_SPLITK (def 8). fp32 atomic -> fp16.
        int bm = 16, bn = 64, splitk = 8;
        if (const char* e = std::getenv("VLLM_W4A8_V13_CFG")) {
            const char* p = e; bm = std::atoi(p);
            while (*p && *p != 'x' && *p != 'X') ++p;
            if (*p) bn = std::atoi(p + 1);
        }
        if (const char* e = std::getenv("VLLM_W4A8_V13_SPLITK")) splitk = std::atoi(e);
        const int nkg = K / group_size;
        if (splitk > nkg) splitk = nkg;
        if (splitk < 1) splitk = 1;
        auto out_acc = at::zeros({M, N}, x.options().dtype(at::kFloat));
        dim3 grid((N + bn - 1) / bn, (M + bm - 1) / bm, splitk);
        #define V13_L(BM, BN) mmq_fp8_gemm_kernel_v13<BM, BN><<<grid, dim3((BM/16)*32), 0, stream>>>( \
            x_fp8.data_ptr<unsigned char>(), act_scales.data_ptr<float>(), \
            w_packed.data_ptr<int>(), \
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()), w_zeros_ptr, \
            out_acc.data_ptr<float>(), M, N, K, group_size, splitk)
        if (bm == 16 && bn == 64) V13_L(16, 64);
        else if (bm == 16 && bn == 32) V13_L(16, 32);
        else if (bm == 16 && bn == 128) V13_L(16, 128);
        else if (bm == 32 && bn == 64) V13_L(32, 64);
        else if (bm == 32 && bn == 128) V13_L(32, 128);
        else if (bm == 64 && bn == 64) V13_L(64, 64);
        else TORCH_CHECK(false, "unsupported V13_CFG ", bm, "x", bn);
        #undef V13_L
        const long ne = (long)M * N;
        cvt_f32_f16_kernel<<<dim3((ne + 255) / 256), dim3(256), 0, stream>>>(
            out_acc.data_ptr<float>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), ne);
    } else if (dk == DK::NsplitSmallm) {  // v14
        // small-M WMMA, N-split warps + split-K. CFG VLLM_W4A8_V14_CFG (BNxNWARPS,
        // def 128x8), VLLM_W4A8_V14_SPLITK (def 4). fp32 atomic -> fp16.
        TORCH_CHECK(group_size == 128 || group_size == 32,
                    "v14 needs group_size 32 or 128; got ", group_size);
        int bn = 128, nw = 8, splitk = 4;
        if (const char* e = std::getenv("VLLM_W4A8_V14_CFG")) {
            const char* p = e; bn = std::atoi(p);
            while (*p && *p != 'x' && *p != 'X') ++p;
            if (*p) nw = std::atoi(p + 1);
        }
        if (const char* e = std::getenv("VLLM_W4A8_V14_SPLITK")) splitk = std::atoi(e);
        int gt = 4;  // multi-group staging (the v6 sync-reduction win)
        if (const char* e = std::getenv("VLLM_W4A8_V14_GTILE")) gt = std::atoi(e);
        if (gt != 1 && gt != 2 && gt != 4) gt = 1;
        const int nkg = K / group_size;
        if (splitk > nkg) splitk = nkg;
        if (splitk < 1) splitk = 1;
        auto out_acc = at::zeros({M, N}, x.options().dtype(at::kFloat));
        dim3 grid((N + bn - 1) / bn, (M + 15) / 16, splitk);
        #define V14_L(BN, NW, GT) mmq_fp8_gemm_kernel_v14<BN, NW, GT><<<grid, dim3(NW*32), 0, stream>>>( \
            x_fp8.data_ptr<unsigned char>(), act_scales.data_ptr<float>(), \
            w_packed.data_ptr<int>(), \
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()), w_zeros_ptr, \
            out_acc.data_ptr<float>(), M, N, K, group_size, splitk)
        // GTILE x BN static LDS must stay <=64KB (array sized for MAX_GROUP_SIZE):
        // BN<=64 -> up to GTILE 4; BN=128 -> up to 2; BN=256 -> 1.
        #define V14_G4(BN, NW) do { if (gt>=4) V14_L(BN,NW,4); else if (gt>=2) V14_L(BN,NW,2); else V14_L(BN,NW,1); } while(0)
        #define V14_G2(BN, NW) do { if (gt>=2) V14_L(BN,NW,2); else V14_L(BN,NW,1); } while(0)
        if (bn == 128 && nw == 8) V14_G2(128, 8);
        else if (bn == 128 && nw == 4) V14_G2(128, 4);
        else if (bn == 64 && nw == 4) V14_G4(64, 4);
        else if (bn == 64 && nw == 2) V14_G4(64, 2);
        else if (bn == 64 && nw == 1) V14_G4(64, 1);
        else if (bn == 32 && nw == 2) V14_G4(32, 2);
        else if (bn == 32 && nw == 1) V14_G4(32, 1);
        else if (bn == 256 && nw == 8) V14_L(256, 8, 1);
        else TORCH_CHECK(false, "unsupported V14_CFG ", bn, "x", nw);
        #undef V14_G2
        #undef V14_G4
        #undef V14_L
        const long ne = (long)M * N;
        cvt_f32_f16_kernel<<<dim3((ne + 255) / 256), dim3(256), 0, stream>>>(
            out_acc.data_ptr<float>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), ne);
    } else if (dk == DK::SplitkSmallm) {  // v12
        // small-M split-K WMMA. CFG VLLM_W4A8_V12_CFG (BMxBN, def 64x64),
        // VLLM_W4A8_V12_SPLITK (def 8). fp32 atomic accumulate -> cast to fp16.
        TORCH_CHECK(group_size == 128 || group_size == 32,
                    "v12 needs group_size 32 or 128; got ", group_size);
        int bm = 64, bn = 64, splitk = 8;
        if (const char* e = std::getenv("VLLM_W4A8_V12_CFG")) {
            const char* p = e; bm = std::atoi(p);
            while (*p && *p != 'x' && *p != 'X') ++p;
            if (*p) bn = std::atoi(p + 1);
        }
        if (const char* e = std::getenv("VLLM_W4A8_V12_SPLITK")) splitk = std::atoi(e);
        const int num_k_groups = K / group_size;
        if (splitk > num_k_groups) splitk = num_k_groups;
        if (splitk < 1) splitk = 1;
        auto out_acc = at::zeros({M, N}, x.options().dtype(at::kFloat));
        dim3 grid((N + bn - 1) / bn, (M + bm - 1) / bm, splitk);
        #define V12_L(BM, BN) mmq_fp8_gemm_kernel_v12<BM, BN><<<grid, dim3((BM/16)*32), 0, stream>>>( \
            x_fp8.data_ptr<unsigned char>(), act_scales.data_ptr<float>(), \
            w_packed.data_ptr<int>(), \
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()), w_zeros_ptr, \
            out_acc.data_ptr<float>(), M, N, K, group_size, splitk)
        if (bm == 16 && bn == 64) V12_L(16, 64);
        else if (bm == 32 && bn == 64) V12_L(32, 64);
        else if (bm == 64 && bn == 64) V12_L(64, 64);
        else if (bm == 32 && bn == 128) V12_L(32, 128);
        else if (bm == 64 && bn == 128) V12_L(64, 128);
        else if (bm == 16 && bn == 128) V12_L(16, 128);
        else TORCH_CHECK(false, "unsupported V12_CFG ", bm, "x", bn);
        #undef V12_L
        const long ne = (long)M * N;
        cvt_f32_f16_kernel<<<dim3((ne + 255) / 256), dim3(256), 0, stream>>>(
            out_acc.data_ptr<float>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), ne);
    } else if (dk == DK::DecodeGemv) {  // v11
        // v11: decode GEMV, K-TILED. one warp per output column; streams weights.
        TORCH_CHECK(K % 1024 == 0, "v11 needs K % 1024 == 0; got ", K);
        TORCH_CHECK(group_size % 32 == 0, "v11 needs group_size % 32; got ", group_size);
        TORCH_CHECK(M <= 16, "v11 is decode-only (M <= 16); got M=", M);
        int nw = 16;
        if (const char* e = std::getenv("VLLM_W4A8_V11_NW")) nw = std::atoi(e);
        // Pick BK = largest multiple of 1024 with M*BK <= 64KB LDS (and <= K). The
        // kernel clamps the final chunk, so BK need not divide K. M<=16 => BK>=4096.
        int BK = K;
        const int budget = 65536;
        if ((size_t)M * K > (size_t)budget) {
            int allow = (budget / M) & ~1023;      // round down to multiple of 1024
            if (allow < 1024) allow = 1024;        // M<=16 guarantees M*1024 <= 16384
            BK = (allow < K) ? allow : K;
        }
        if (const char* e = std::getenv("VLLM_W4A8_V11_BK")) { int b = std::atoi(e); if (b > 0) BK = b; }
        const size_t shmem = (size_t)M * BK;
        dim3 block(nw * 32);
        dim3 grid((N + nw - 1) / nw);
        auto L = [&](auto kern) { kern<<<grid, block, shmem, stream>>>(
            x_fp8.data_ptr<unsigned char>(), act_scales.data_ptr<float>(),
            w_packed.data_ptr<int>(),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()), w_zeros_ptr,
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), M, N, K, group_size, BK); };
        if (nw == 8) L(mmq_fp8_gemv_kernel_v11<8, 16>);
        else if (nw == 16) L(mmq_fp8_gemv_kernel_v11<16, 16>);
        else if (nw == 32) L(mmq_fp8_gemv_kernel_v11<32, 16>);
        else TORCH_CHECK(false, "V11_NW must be 8/16/32");
    } else if (dk == DK::PrefillWmmaAshuffle) {  // v10
        // v10: A out of LDS (warp-shuffle transpose) -> B-only LDS -> double-buffer
        // B at the full BM=256/BN=128 tile. CFG VLLM_W4A8_V10_CFG (def 256x128),
        // DB via VLLM_W4A8_V10_DB (def 1), swiz via VLLM_W4A8_V7_SWIZ.
        TORCH_CHECK(group_size == 128 || group_size == 32,
                    "v10 needs group_size 32 or 128; got ", group_size);
        int bm = 256, bn = 128, db = 1, swiz = 1;
        if (const char* e = std::getenv("VLLM_W4A8_V10_CFG")) {
            const char* x = e; bm = std::atoi(x);
            while (*x && *x != 'x' && *x != 'X') ++x;
            if (*x) bn = std::atoi(x + 1);
        }
        if (const char* e = std::getenv("VLLM_W4A8_V10_DB")) db = std::atoi(e);
        if (const char* e = std::getenv("VLLM_W4A8_V7_SWIZ")) swiz = std::atoi(e);
        const int LDSBK = group_size + 8;
        const size_t shmem = static_cast<size_t>(db ? 2 : 1) * bn * LDSBK;
        dim3 grid = swiz ? dim3((M + bm - 1) / bm, (N + bn - 1) / bn)
                         : dim3((N + bn - 1) / bn, (M + bm - 1) / bm);
        auto launch = [&](auto kern, int nwarps) {
            dim3 block(nwarps * 32);
            kern<<<grid, block, shmem, stream>>>(
                x_h,                       // fused act-quant: fp16 activations
                w_packed.data_ptr<int>(),
                reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
                w_zeros_ptr,
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                M, N, K, group_size, swiz);
        };
        #define V10_L(BM, BN, G) do { \
            if (db) launch(mmq_fp8_gemm_kernel_v10<BM, BN, BM/16, G, true>, BM/16); \
            else    launch(mmq_fp8_gemm_kernel_v10<BM, BN, BM/16, G, false>, BM/16); \
        } while (0)
        #define V10_BKT(BM, BN) do { \
            if (group_size == 128) V10_L(BM, BN, 128); else V10_L(BM, BN, 32); \
        } while (0)
        if (bm == 16 && bn == 128) V10_BKT(16, 128);
        else if (bm == 32 && bn == 128) V10_BKT(32, 128);
        else if (bm == 64 && bn == 128) V10_BKT(64, 128);
        else if (bm == 128 && bn == 128) V10_BKT(128, 128);
        else if (bm == 256 && bn == 128) V10_BKT(256, 128);
        else if (bm == 256 && bn == 256) V10_BKT(256, 256);
        else if (bm == 384 && bn == 128) V10_BKT(384, 128);
        else if (bm == 512 && bn == 128) V10_BKT(512, 128);
        else TORCH_CHECK(false, "unsupported V10_CFG ", bm, "x", bn);
        #undef V10_BKT
        #undef V10_L
    } else if (dk == DK::WmmaDbuf2) {  // v9
        // v9: decoupled-depth double-buffer; keeps BM=256/BN=128 8-acc tile.
        // Needs a compile-time group size (32 or 128). CFG VLLM_W4A8_V9_CFG.
        TORCH_CHECK(group_size == 128 || group_size == 32,
                    "v9 needs group_size 32 or 128; got ", group_size);
        int bm = 256, bn = 128;
        if (const char* e = std::getenv("VLLM_W4A8_V9_CFG")) {
            const char* x = e; bm = std::atoi(x);
            while (*x && *x != 'x' && *x != 'X') ++x;
            if (*x) bn = std::atoi(x + 1);
        }
        const int bk_lds = (group_size >= 128) ? 64 : group_size;
        const size_t shmem = static_cast<size_t>(2) * (bm + bn) * (bk_lds + 8);
        dim3 grid((N + bn - 1) / bn, (M + bm - 1) / bm);
        auto launch = [&](auto kern, int nwarps) {
            dim3 block(nwarps * 32);
            kern<<<grid, block, shmem, stream>>>(
                x_fp8.data_ptr<unsigned char>(),
                act_scales.data_ptr<float>(),
                w_packed.data_ptr<int>(),
                reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
                w_zeros_ptr,
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                M, N, K, group_size);
        };
        #define V9_BKT(BM, BN) do { \
            if (group_size == 128) launch(mmq_fp8_gemm_kernel_v9<BM, BN, BM/16, 128>, BM/16); \
            else launch(mmq_fp8_gemm_kernel_v9<BM, BN, BM/16, 32>, BM/16); \
        } while (0)
        if (bm == 128 && bn == 128) V9_BKT(128, 128);
        else if (bm == 192 && bn == 128) V9_BKT(192, 128);
        else if (bm == 256 && bn == 128) V9_BKT(256, 128);
        else if (bm == 256 && bn == 64) V9_BKT(256, 64);
        else TORCH_CHECK(false, "unsupported V9_CFG ", bm, "x", bn);
        #undef V9_BKT
    } else if (dk == DK::WmmaDbuf) {  // v8
        // v8: double-buffered (overlap staging with compute, 1 barrier/group).
        // CFG via VLLM_W4A8_V8_CFG="BMxBN" (default 96x128 — the largest BM that
        // fits 2x LDS at BN=128/G=128 in 64KB while keeping 8-acc ILP).
        int bm = 96, bn = 128;
        if (const char* e = std::getenv("VLLM_W4A8_V8_CFG")) {
            const char* x = e; bm = std::atoi(x);
            while (*x && *x != 'x' && *x != 'X') ++x;
            if (*x) bn = std::atoi(x + 1);
        }
        const int LDSBK = group_size + 8;
        const size_t shmem = static_cast<size_t>(2) * (bm + bn) * LDSBK;
        dim3 grid((N + bn - 1) / bn, (M + bm - 1) / bm);
        auto launch = [&](auto kern, int nwarps) {
            dim3 block(nwarps * 32);
            kern<<<grid, block, shmem, stream>>>(
                x_fp8.data_ptr<unsigned char>(),
                act_scales.data_ptr<float>(),
                w_packed.data_ptr<int>(),
                reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
                w_zeros_ptr,
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                M, N, K, group_size);
        };
        #define V8_LAUNCH(BM, BN) \
            launch(mmq_fp8_gemm_kernel_v8<BM, BN, BM / 16>, BM / 16)
        if (bm == 64 && bn == 128) V8_LAUNCH(64, 128);
        else if (bm == 80 && bn == 128) V8_LAUNCH(80, 128);
        else if (bm == 96 && bn == 128) V8_LAUNCH(96, 128);
        else if (bm == 112 && bn == 128) V8_LAUNCH(112, 128);
        else if (bm == 128 && bn == 64) V8_LAUNCH(128, 64);
        else if (bm == 256 && bn == 128) V8_LAUNCH(256, 128);  // only fits small G
        else TORCH_CHECK(false, "unsupported V8_CFG ", bm, "x", bn);
        #undef V8_LAUNCH
    } else if (dk == DK::WmmaTiledTuned) {  // v7
        // v7: v5 algorithm, templated (BM x BN) tile + dynamic LDS, to sweep
        // occupancy (tile size) AND WMMA-ILP depth (BN/16 = independent accs per
        // warp; peak needs ~8). Config via VLLM_W4A8_V7_CFG = "BMxBN"
        // (default "256x128", the swept optimum: BN=128 => 8 independent WMMA
        // accumulators/warp = peak's NACC, BM=256 amortizes staging; ~1.3x v5).
        // One warp per 16 rows (NWARPS = BM/16).
        int bm = 256, bn = 128;
        if (const char* e = std::getenv("VLLM_W4A8_V7_CFG")) {
            const char* x = e;
            bm = std::atoi(x);
            while (*x && *x != 'x' && *x != 'X') ++x;
            if (*x) bn = std::atoi(x + 1);
        }
        int diag = 0;
        if (const char* e = std::getenv("VLLM_W4A8_V7_DIAG")) diag = std::atoi(e);
        int swiz = 1;  // default on: weight-slab L2/MALL reuse
        if (const char* e = std::getenv("VLLM_W4A8_V7_SWIZ")) swiz = std::atoi(e);
        const int LDSBK = group_size + 8;
        const size_t shmem = static_cast<size_t>(bm + bn) * LDSBK;
        dim3 grid = swiz ? dim3((M + bm - 1) / bm, (N + bn - 1) / bn)
                         : dim3((N + bn - 1) / bn, (M + bm - 1) / bm);
        auto launch = [&](auto kern, int nwarps) {
            dim3 block(nwarps * 32);
            kern<<<grid, block, shmem, stream>>>(
                x_fp8.data_ptr<unsigned char>(),
                act_scales.data_ptr<float>(),
                w_packed.data_ptr<int>(),
                reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
                w_zeros_ptr,
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                M, N, K, group_size, diag, swiz);
        };
        // Pick a compile-time-BK instantiation for the common group sizes
        // (128/32) so the staging div/mod fold to shifts; else BKT=0 (runtime).
        #define V7_BKT(BM, BN) do { \
            if (group_size == 128) launch(mmq_fp8_gemm_kernel_v7<BM, BN, BM/16, 128>, BM/16); \
            else if (group_size == 32) launch(mmq_fp8_gemm_kernel_v7<BM, BN, BM/16, 32>, BM/16); \
            else launch(mmq_fp8_gemm_kernel_v7<BM, BN, BM/16, 0>, BM/16); \
        } while (0)
        if (bm == 64 && bn == 64) V7_BKT(64, 64);
        else if (bm == 64 && bn == 128) V7_BKT(64, 128);
        else if (bm == 80 && bn == 128) V7_BKT(80, 128);
        else if (bm == 96 && bn == 128) V7_BKT(96, 128);
        else if (bm == 112 && bn == 128) V7_BKT(112, 128);
        else if (bm == 128 && bn == 64) V7_BKT(128, 64);
        else if (bm == 128 && bn == 128) V7_BKT(128, 128);
        else if (bm == 128 && bn == 256) V7_BKT(128, 256);
        else if (bm == 192 && bn == 128) V7_BKT(192, 128);
        else if (bm == 256 && bn == 64) V7_BKT(256, 64);
        else if (bm == 256 && bn == 128) V7_BKT(256, 128);
        else if (bm == 256 && bn == 192) V7_BKT(256, 192);
        else TORCH_CHECK(false, "unsupported V7_CFG ", bm, "x", bn);
        #undef V7_BKT
    } else if (dk == DK::PrefillWmmaB128) {  // v6
        // v6: K-extension -- b128 dual-subtile loads + 2 WMMAs per load.
        // Requires group_size % 32 == 0; pad must be a multiple of 16 for the
        // b128 alignment / row stride. Sweepable via VLLM_W4A8_V6_PAD (def 16).
        TORCH_CHECK(group_size % 32 == 0,
                    "v6 needs group_size % 32 == 0; got ", group_size);
        int lds_pad = 16;
        if (const char* e = std::getenv("VLLM_W4A8_V6_PAD")) {
            lds_pad = std::atoi(e);
            TORCH_CHECK(lds_pad >= 0 && lds_pad % 16 == 0,
                        "VLLM_W4A8_V6_PAD must be a non-negative multiple of 16");
        }
        const int LDSBK = group_size + lds_pad;
        const size_t shmem = static_cast<size_t>(V2_BM + V2_BN) * LDSBK;
        dim3 block(V2_NWARPS * 32);
        dim3 grid((N + V2_BN - 1) / V2_BN, (M + V2_BM - 1) / V2_BM);
        mmq_fp8_gemm_kernel_v6<<<grid, block, shmem, stream>>>(
            x_fp8.data_ptr<unsigned char>(),
            act_scales.data_ptr<float>(),
            w_packed.data_ptr<int>(),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            w_zeros_ptr,
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            M, N, K, group_size, lds_pad);
    } else if (dk == DK::PrefillWmma) {  // v5
        // v5: raw fp8 WMMA builtin + direct K-major int2 loads.
        // Fused act-quant (fp16 in, e4m3 in A_tile during staging).
        dim3 block(V2_NWARPS * 32);
        dim3 grid((N + V2_BN - 1) / V2_BN, (M + V2_BM - 1) / V2_BM);
        mmq_fp8_gemm_kernel_v5<<<grid, block, 0, stream>>>(
            x_h,
            w_packed.data_ptr<int>(),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            w_zeros_ptr,
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            M, N, K, group_size);
    } else if (dk == DK::RocwmmaPipe) {  // v4
        // v4: single-buffer staging + kt-level fragment pipelining.
        dim3 block(V2_NWARPS * 32);
        dim3 grid((N + V2_BN - 1) / V2_BN, (M + V2_BM - 1) / V2_BM);
        mmq_fp8_gemm_kernel_v4<<<grid, block, 0, stream>>>(
            x_fp8.data_ptr<unsigned char>(),
            act_scales.data_ptr<float>(),
            w_packed.data_ptr<int>(),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            w_zeros_ptr,
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            M, N, K, group_size);
    } else if (dk == DK::RocwmmaTiled) {  // v2
        // v2: optimized tiled fp8 WMMA (BM x BN tile, 4 warps, register accum).
        dim3 block(V2_NWARPS * 32);
        dim3 grid((N + V2_BN - 1) / V2_BN, (M + V2_BM - 1) / V2_BM);
        mmq_fp8_gemm_kernel_v2<<<grid, block, 0, stream>>>(
            x_fp8.data_ptr<unsigned char>(),
            act_scales.data_ptr<float>(),
            w_packed.data_ptr<int>(),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            w_zeros_ptr,
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            M, N, K, group_size);
    } else if (dk == DK::RocwmmaV1) {  // v1
        // v1: rocWMMA fp8 16x16x16, one wave32 warp per 16x16 output tile.
        dim3 block(32);
        dim3 grid((N + WMMA_DIM - 1) / WMMA_DIM, (M + WMMA_DIM - 1) / WMMA_DIM);
        mmq_fp8_gemm_kernel_v1<<<grid, block, 0, stream>>>(
            x_fp8.data_ptr<unsigned char>(),
            act_scales.data_ptr<float>(),
            w_packed.data_ptr<int>(),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            w_zeros_ptr,
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            M, N, K, group_size);
    } else {
        dim3 block(V0_BLOCK_N, V0_BLOCK_M);
        dim3 grid((N + V0_BLOCK_N - 1) / V0_BLOCK_N,
                  (M + V0_BLOCK_M - 1) / V0_BLOCK_M);
        mmq_fp8_gemm_kernel_v0<<<grid, block, 0, stream>>>(
            x_fp8.data_ptr<unsigned char>(),
            act_scales.data_ptr<float>(),
            w_packed.data_ptr<int>(),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            w_zeros_ptr,
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            M, N, K, group_size);
    }
}

// =================== v15 launcher (register-direct, repacked weights) ===================
void launch_mmq_fp8_gemm_v15_gfx1201(
    const at::Tensor& x,           // (M, K) fp16
    const at::Tensor& w_rep,       // (N/16, K/16, 32) int32 lane-order
    const at::Tensor& scales,      // (N, K/group) fp16
    const at::Tensor& w_zeros,     // (N/8, K/group) int32 or empty
    at::Tensor& out,               // (M, N) fp16
    int64_t N) {
    using namespace w4a8_fp8_wmma;
    const int M = x.size(0);
    const int K = x.size(1);
    const int group_size = K / scales.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto x_fp8 = at::empty({M, K}, x.options().dtype(at::kByte));
    auto act_scales = at::empty({M}, x.options().dtype(at::kFloat));
    compute_act_fp8_and_scales_kernel<<<dim3(M), dim3(ACTQUANT_BLOCK), 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        x_fp8.data_ptr<unsigned char>(), act_scales.data_ptr<float>(), M, K);
    const int* wz = (w_zeros.defined() && w_zeros.numel() > 0) ? w_zeros.data_ptr<int>() : nullptr;
    int bn = 64, nw = 4, splitk = 2;
    if (const char* e = std::getenv("VLLM_W4A8_V15_CFG")) {
        const char* p = e; bn = std::atoi(p);
        while (*p && *p != 'x' && *p != 'X') ++p;
        if (*p) nw = std::atoi(p + 1);
    }
    if (const char* e = std::getenv("VLLM_W4A8_V15_SPLITK")) splitk = std::atoi(e);
    const int nkg = K / group_size;
    if (splitk > nkg) splitk = nkg; if (splitk < 1) splitk = 1;
    auto out_acc = at::zeros({M, (int)N}, x.options().dtype(at::kFloat));
    dim3 grid(((int)N + bn - 1) / bn, (M + 15) / 16, splitk);
    #define V15_L(BN, NW) mmq_fp8_gemm_kernel_v15<BN, NW><<<grid, dim3(NW*32), 0, stream>>>( \
        x_fp8.data_ptr<unsigned char>(), act_scales.data_ptr<float>(), w_rep.data_ptr<int>(), \
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()), wz, \
        out_acc.data_ptr<float>(), M, (int)N, K, group_size, splitk)
    if (bn == 64 && nw == 4) V15_L(64, 4);
    else if (bn == 64 && nw == 2) V15_L(64, 2);
    else if (bn == 128 && nw == 8) V15_L(128, 8);
    else if (bn == 128 && nw == 4) V15_L(128, 4);
    else if (bn == 32 && nw == 2) V15_L(32, 2);
    else if (bn == 256 && nw == 8) V15_L(256, 8);
    else TORCH_CHECK(false, "unsupported V15_CFG ", bn, "x", nw);
    #undef V15_L
    const long ne = (long)M * N;
    w4a8_fp8_wmma::cvt_f32_f16_kernel<<<dim3((ne + 255) / 256), dim3(256), 0, stream>>>(
        out_acc.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()), ne);
}

// v16: f16-WMMA twin of v15 (same w_rep contract). Reuses v15's repacked weights.
void launch_mmq_fp8_gemm_v16_gfx1201(
    const at::Tensor& x,           // (M, K) fp16
    const at::Tensor& w_rep,       // (N/16, K/16, 32) int32 lane-order
    const at::Tensor& scales,      // (N, K/group) fp16
    const at::Tensor& w_zeros,     // (N/8, K/group) int32 or empty
    at::Tensor& out,               // (M, N) fp16
    int64_t N) {
    using namespace w4a8_fp8_wmma;
    const int M = x.size(0);
    const int K = x.size(1);
    const int group_size = K / scales.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto x_fp8 = at::empty({M, K}, x.options().dtype(at::kByte));
    auto act_scales = at::empty({M}, x.options().dtype(at::kFloat));
    compute_act_fp8_and_scales_kernel<<<dim3(M), dim3(ACTQUANT_BLOCK), 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        x_fp8.data_ptr<unsigned char>(), act_scales.data_ptr<float>(), M, K);
    const int* wz = (w_zeros.defined() && w_zeros.numel() > 0) ? w_zeros.data_ptr<int>() : nullptr;
    int bn = 64, nw = 4, splitk = 2;
    if (const char* e = std::getenv("VLLM_W4A8_V16_CFG")) {
        const char* p = e; bn = std::atoi(p);
        while (*p && *p != 'x' && *p != 'X') ++p;
        if (*p) nw = std::atoi(p + 1);
    }
    if (const char* e = std::getenv("VLLM_W4A8_V16_SPLITK")) splitk = std::atoi(e);
    const int nkg = K / group_size;
    if (splitk > nkg) splitk = nkg; if (splitk < 1) splitk = 1;
    auto out_acc = at::zeros({M, (int)N}, x.options().dtype(at::kFloat));
    dim3 grid(((int)N + bn - 1) / bn, (M + 15) / 16, splitk);
    #define V16_L(BN, NW) mmq_fp8_gemm_kernel_v16<BN, NW><<<grid, dim3(NW*32), 0, stream>>>( \
        x_fp8.data_ptr<unsigned char>(), act_scales.data_ptr<float>(), w_rep.data_ptr<int>(), \
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()), wz, \
        out_acc.data_ptr<float>(), M, (int)N, K, group_size, splitk)
    if (bn == 64 && nw == 4) V16_L(64, 4);
    else if (bn == 64 && nw == 2) V16_L(64, 2);
    else if (bn == 128 && nw == 8) V16_L(128, 8);
    else if (bn == 128 && nw == 4) V16_L(128, 4);
    else if (bn == 32 && nw == 2) V16_L(32, 2);
    else if (bn == 32 && nw == 1) V16_L(32, 1);
    else if (bn == 256 && nw == 8) V16_L(256, 8);
    else TORCH_CHECK(false, "unsupported V16_CFG ", bn, "x", nw);
    #undef V16_L
    const long ne = (long)M * N;
    w4a8_fp8_wmma::cvt_f32_f16_kernel<<<dim3((ne + 255) / 256), dim3(256), 0, stream>>>(
        out_acc.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()), ne);
}

// v17: true W4A16 -- fp16 activations DIRECT (no act-quant), same w_rep contract.
void launch_mmq_w4a16_gemm_v17_gfx1201(
    const at::Tensor& x,           // (M, K) fp16
    const at::Tensor& w_rep,       // (N/16, K/16, 32) int32 lane-order
    const at::Tensor& scales,      // (N, K/group) fp16
    const at::Tensor& w_zeros,     // (N/8, K/group) int32 or empty
    at::Tensor& out,               // (M, N) fp16
    int64_t N) {
    using namespace w4a8_fp8_wmma;
    const int M = x.size(0);
    const int K = x.size(1);
    const int group_size = K / scales.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int* wz = (w_zeros.defined() && w_zeros.numel() > 0) ? w_zeros.data_ptr<int>() : nullptr;
    int bn = 64, nw = 4, splitk = 1;
    if (const char* e = std::getenv("VLLM_W4A8_V17_CFG")) {
        const char* p = e; bn = std::atoi(p);
        while (*p && *p != 'x' && *p != 'X') ++p;
        if (*p) nw = std::atoi(p + 1);
    }
    if (const char* e = std::getenv("VLLM_W4A8_V17_SPLITK")) splitk = std::atoi(e);
    const int nkg = K / group_size;
    if (splitk > nkg) splitk = nkg; if (splitk < 1) splitk = 1;
    // splitk==1 -> direct f16 write (no out_acc, no cvt). splitk>1 -> f32 atomic + cvt.
    at::Tensor out_acc;
    float* oacc = nullptr;
    if (splitk > 1) {
        out_acc = at::zeros({M, (int)N}, x.options().dtype(at::kFloat));
        oacc = out_acc.data_ptr<float>();
    }
    const __half* xh = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    const __half* sh = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
    __half* oh = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    dim3 grid(((int)N + bn - 1) / bn, (M + 15) / 16, splitk);
    #define V17_L(BN, NW) mmq_w4a16_gemm_kernel_v17<BN, NW><<<grid, dim3(NW*32), 0, stream>>>( \
        xh, w_rep.data_ptr<int>(), sh, wz, oacc, oh, M, (int)N, K, group_size, splitk)
    if (bn == 64 && nw == 4) V17_L(64, 4);
    else if (bn == 64 && nw == 2) V17_L(64, 2);
    else if (bn == 128 && nw == 8) V17_L(128, 8);
    else if (bn == 128 && nw == 4) V17_L(128, 4);
    else if (bn == 32 && nw == 2) V17_L(32, 2);
    else if (bn == 32 && nw == 1) V17_L(32, 1);
    else if (bn == 256 && nw == 8) V17_L(256, 8);
    else TORCH_CHECK(false, "unsupported V17_CFG ", bn, "x", nw);
    #undef V17_L
    if (splitk > 1) {
        const long ne = (long)M * N;
        w4a8_fp8_wmma::cvt_f32_f16_kernel<<<dim3((ne + 255) / 256), dim3(256), 0, stream>>>(
            oacc, oh, ne);
    }
}
