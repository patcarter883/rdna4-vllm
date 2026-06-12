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

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <rocwmma/rocwmma.hpp>

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
using v2i_t = int __attribute__((ext_vector_type(2)));
using v8f_t = float __attribute__((ext_vector_type(8)));

__global__ void __launch_bounds__(V2_NWARPS * 32) mmq_fp8_gemm_kernel_v5(
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

    float running[V2_NFRAG][8];
    #pragma unroll
    for (int f = 0; f < V2_NFRAG; ++f)
        #pragma unroll
        for (int e = 0; e < 8; ++e) running[f][e] = 0.0f;

    for (int g = 0; g < num_k_groups; ++g) {
        const int k0 = g * BK;

        for (int idx = tid * 4; idx < V2_BM * BK; idx += (V2_NWARPS * 32) * 4) {
            const int r = idx / BK, k = idx % BK, am = block_m + r;
            unsigned int v = 0u;
            if (am < M)
                v = *reinterpret_cast<const unsigned int*>(&x_fp8[am * K + k0 + k]);
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

}  // namespace w4a8_fp8_wmma

// =================== Host launcher ===================
void launch_mmq_fp8_gemm_gfx1201(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const at::Tensor& w_zeros,
    at::Tensor& out,
    int64_t version) {

    using namespace w4a8_fp8_wmma;

    const int M = x.size(0);
    const int K = x.size(1);
    const int N = w_packed.size(0);
    const int group_size = K / scales.size(1);

    TORCH_CHECK(K % group_size == 0, "K not divisible by derived group_size");
    TORCH_CHECK(group_size % WMMA_DIM == 0 && group_size <= MAX_GROUP_SIZE,
                "group_size must be a multiple of 16 and <= 128; got ", group_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto x_fp8 = at::empty({M, K}, x.options().dtype(at::kByte));
    auto act_scales = at::empty({M}, x.options().dtype(at::kFloat));
    compute_act_fp8_and_scales_kernel<<<dim3(M), dim3(ACTQUANT_BLOCK), 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        x_fp8.data_ptr<unsigned char>(),
        act_scales.data_ptr<float>(),
        M, K);

    const int* w_zeros_ptr = (w_zeros.defined() && w_zeros.numel() > 0)
                             ? w_zeros.data_ptr<int>() : nullptr;

    if (version == 5) {
        // v5: raw fp8 WMMA builtin + direct K-major int2 loads.
        dim3 block(V2_NWARPS * 32);
        dim3 grid((N + V2_BN - 1) / V2_BN, (M + V2_BM - 1) / V2_BM);
        mmq_fp8_gemm_kernel_v5<<<grid, block, 0, stream>>>(
            x_fp8.data_ptr<unsigned char>(),
            act_scales.data_ptr<float>(),
            w_packed.data_ptr<int>(),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            w_zeros_ptr,
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            M, N, K, group_size);
    } else if (version == 4) {
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
    } else if (version == 2) {
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
    } else if (version == 1) {
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
