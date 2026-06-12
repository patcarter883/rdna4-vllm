// Grouped (MoE) W4A8-FP8 GEMM for AMD RDNA4 (gfx1201).
//
// The MoE analogue of w4a8_fp8_wmma_kernel.hip: instead of one (M,K)x(K,N) GEMM,
// it computes a *grouped* GEMM over routed tokens, matching vLLM's fused_moe
// contract (moe_align_block_size + a stacked per-expert weight tensor).
//
// Contract (mirrors vllm/model_executor/layers/fused_moe/fused_moe.py):
//   x          (T, K)            fp16   token activations (T = num tokens)
//   w_packed   (E, N, K/8)       int32  stacked per-expert weights (uint4b8/AWQ)
//   scales     (E, N, K/group)   fp16   per-expert per-group weight scales
//   w_zeros    (E, N/8, K/group) int32  per-expert packed zeros (AWQ) or empty
//   sorted_token_ids (P,)        int32  padded, expert-sorted token slots
//   expert_ids (P/block_m,)      int32  expert for each block of block_m rows
//   num_tokens_post_padded (1,)  int32  valid padded length
//   top_k                        int    activation row = offs_token / top_k
//   block_m                      int    moe_align block size (mult. of 16)
//   out        (P, N)            fp16   one row per (token, expert-slot)
//
// Each block of `block_m` sorted rows belongs to ONE expert (expert_ids[block]),
// so a WMMA tile of block_m rows reads a single expert's weight slab — that's why
// the tile's M dimension is block_m (16/32/64), not the dense kernel's BM=256.
// Padding rows (offs_token >= num_valid) are guarded: their A is staged as fp8
// zero and their output is not written. The topk-weight scaling + scatter-reduce
// over the P rows is a separate epilogue (done by the caller / a fused step).

#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace w4a8_fp8_moe {

constexpr int PACK_FACTOR = 8;
constexpr float E4M3_MAX = 448.0f;
constexpr int WMMA_DIM = 16;
constexpr int MAX_GROUP_SIZE = 128;

__device__ __forceinline__ unsigned char f32_to_e4m3(float v) {
    int packed = __builtin_amdgcn_cvt_pk_fp8_f32(v, v, 0, false);
    return static_cast<unsigned char>(packed & 0xFF);
}
__device__ __forceinline__ float e4m3_to_f32(unsigned char b) {
    return __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(b), 0);
}
__device__ __forceinline__ unsigned char int4_signed_to_e4m3(int signed_val) {
    return f32_to_e4m3(static_cast<float>(signed_val));
}

// ---- per-token activation fp8 quant (one block per token row, T rows) ----
constexpr int ACTQUANT_BLOCK = 256;
__global__ void moe_compute_act_fp8_kernel(
    const __half* __restrict__ x, unsigned char* __restrict__ x_fp8,
    float* __restrict__ act_scales, int T, int K) {
    const int row = blockIdx.x;
    if (row >= T) return;
    const int tid = threadIdx.x;
    const __half* xr = x + row * K;
    unsigned char* outr = x_fp8 + row * K;
    float local_max = 0.0f;
    for (int k = tid; k < K; k += ACTQUANT_BLOCK) {
        float v = fabsf(__half2float(xr[k]));
        if (v > local_max) local_max = v;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float other = __shfl_xor(local_max, o);
        if (other > local_max) local_max = other;
    }
    __shared__ float wm[ACTQUANT_BLOCK / 32];
    const int lane = tid & 31, wid = tid >> 5;
    if (lane == 0) wm[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < ACTQUANT_BLOCK / 32) ? wm[lane] : 0.0f;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            float other = __shfl_xor(v, o);
            if (other > v) v = other;
        }
        if (lane == 0) {
            float s = v / E4M3_MAX;
            if (s < 1e-8f) s = 1e-8f;
            wm[0] = s;
            act_scales[row] = s;
        }
    }
    __syncthreads();
    const float inv = 1.0f / wm[0];
    for (int k = tid; k < K; k += ACTQUANT_BLOCK)
        outr[k] = f32_to_e4m3(__half2float(xr[k]) * inv);
}

// ============================================================
// grouped v0: scalar golden. One thread per (padded_row, col).
// ============================================================
constexpr int MV0_BM = 16, MV0_BN = 16;
__global__ void moe_gemm_v0_kernel(
    const unsigned char* __restrict__ x_fp8,    // (T, K)
    const float*         __restrict__ act_scales,// (T,)
    const int*           __restrict__ w_packed, // (E, N, K/8)
    const __half*        __restrict__ w_scales, // (E, N, K/group)
    const int*           __restrict__ w_zeros,  // (E, N/8, K/group) or null
    const int*           __restrict__ sorted_token_ids, // (P,)
    const int*           __restrict__ expert_ids,       // (P/block_m,)
    const int*           __restrict__ num_tokens_post_padded,
    __half*              __restrict__ out,       // (P, N)
    int T, int N, int K, int group_size, int top_k, int block_m,
    int num_valid_tokens) {

    const int row_pad = blockIdx.y * MV0_BM + threadIdx.y;
    const int col = blockIdx.x * MV0_BN + threadIdx.x;
    if (col >= N) return;
    const int P_valid = num_tokens_post_padded[0];
    if (row_pad >= P_valid) return;

    const int offs_token = sorted_token_ids[row_pad];
    if (offs_token >= num_valid_tokens) return;  // padding slot
    const int src = offs_token / top_k;          // activation row
    const int e = expert_ids[row_pad / block_m];

    const int num_groups = K / group_size;
    const int ppr = K / PACK_FACTOR;
    const int* wq_e = w_packed + (long)e * N * ppr;
    const __half* ws_e = w_scales + (long)e * N * num_groups;
    const int* wz_e = (w_zeros != nullptr)
                      ? w_zeros + (long)e * (N / 8) * num_groups : nullptr;
    const float a_scale = act_scales[src];

    float acc = 0.0f;
    for (int g = 0; g < num_groups; ++g) {
        const float wsc = __half2float(ws_e[col * num_groups + g]);
        int zp = 8;
        if (wz_e != nullptr) {
            const int pk = wz_e[(col / 8) * num_groups + g];
            zp = (pk >> ((col % 8) * 4)) & 0xF;
        }
        float gacc = 0.0f;
        const int kb = g * group_size;
        for (int kk = 0; kk < group_size; ++kk) {
            const int k = kb + kk;
            const int word = wq_e[col * ppr + (k / PACK_FACTOR)];
            const int ws = ((word >> ((k % PACK_FACTOR) * 4)) & 0xF) - zp;
            const float wf = e4m3_to_f32(int4_signed_to_e4m3(ws));
            const float af = e4m3_to_f32(x_fp8[src * K + k]);
            gacc += wf * af;
        }
        acc += gacc * wsc;
    }
    out[(long)row_pad * N + col] = __float2half(acc * a_scale);
}

// ============================================================
// grouped v5: raw fp8 WMMA builtin, K-major int2 loads, bank-padded LDS.
// One block computes a (block_m x MV5_BN) tile for ONE expert. block_m is the
// moe_align block size (16/32/64) -> block_m/16 M-subtiles (warps).
// ============================================================
using v2i_t = int __attribute__((ext_vector_type(2)));
using v8f_t = float __attribute__((ext_vector_type(8)));
constexpr int MV5_BN = 64;
constexpr int MV5_NFRAG = MV5_BN / 16;     // N-fragments per warp
constexpr int MV5_MAX_WARPS = 4;           // block_m up to 64

__global__ void __launch_bounds__(MV5_MAX_WARPS * 32) moe_gemm_v5_kernel(
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
    int num_valid_tokens) {

    const int block_idx = blockIdx.y;          // one moe_align block (one expert)
    const int row0 = block_idx * block_m;      // first padded row
    const int P_valid = num_tokens_post_padded[0];
    if (row0 >= P_valid) return;

    const int block_n = blockIdx.x * MV5_BN;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;              // which 16-row M-subtile
    const int lane = tid & 31;
    const int lrow = lane & 15;
    const int khalf = (lane >> 4) * 8;
    const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8;
    const int n_warps = block_m / WMMA_DIM;    // active warps
    if (warp_id >= n_warps) return;

    const int e = expert_ids[block_idx];
    const int num_groups = K / group_size;
    const int ppr = K / PACK_FACTOR;
    const int* wq_e = w_packed + (long)e * N * ppr;
    const __half* ws_e = w_scales + (long)e * N * num_groups;
    const int* wz_e = (w_zeros != nullptr)
                      ? w_zeros + (long)e * (N / 8) * num_groups : nullptr;

    const int BK = group_size;
    constexpr int LDS_PAD = 8;
    const int LDSBK = BK + LDS_PAD;
    // A_tile: block_m rows; B_tile: MV5_BN cols. Sized for max block_m=64.
    __shared__ unsigned char A_tile[64 * (MAX_GROUP_SIZE + LDS_PAD)];
    __shared__ unsigned char B_tile[MV5_BN * (MAX_GROUP_SIZE + LDS_PAD)];
    // Per-warp source token row for this M-subtile's 16 rows (gathered once/group
    // is wasteful; gather once up front into LDS-free regs is hard with 16 rows
    // per warp split across lanes — instead recompute per stage from sorted ids).

    float running[MV5_NFRAG][8];
    #pragma unroll
    for (int f = 0; f < MV5_NFRAG; ++f)
        #pragma unroll
        for (int ee = 0; ee < 8; ++ee) running[f][ee] = 0.0f;

    const int total_threads = n_warps * 32;
    for (int g = 0; g < num_groups; ++g) {
        const int k0 = g * BK;

        // ---- stage activations: A_tile[r][k] = x_fp8[src(r)][k0+k], guarded ----
        // uint32 (4 fp8) coalesced loads over block_m rows.
        for (int idx = tid * 4; idx < block_m * BK; idx += total_threads * 4) {
            const int r = idx / BK, k = idx % BK;
            unsigned int v = 0u;
            const int offs_token = sorted_token_ids[row0 + r];
            if (offs_token < num_valid_tokens) {
                const int src = offs_token / top_k;
                v = *reinterpret_cast<const unsigned int*>(&x_fp8[(long)src * K + k0 + k]);
            }
            *reinterpret_cast<unsigned int*>(&A_tile[r * LDSBK + k]) = v;
        }
        // ---- stage weights: expand int4 -> fp8 into B_tile[n][k] ----
        const int njobs = MV5_BN * (BK / PACK_FACTOR);
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
                    ? int4_signed_to_e4m3(((word >> (jj * 4)) & 0xF) - zp) : 0;
        }
        __syncthreads();

        v8f_t acc[MV5_NFRAG];
        #pragma unroll
        for (int f = 0; f < MV5_NFRAG; ++f) acc[f] = v8f_t{0,0,0,0,0,0,0,0};

        const int k_sub = BK / WMMA_DIM;
        const int a_base = (warp_id * WMMA_DIM + lrow) * LDSBK + khalf;
        const int b_base = lrow * LDSBK + khalf;
        for (int kt = 0; kt < k_sub; ++kt) {
            const int ko = kt * WMMA_DIM;
            v2i_t a_cur = *reinterpret_cast<const v2i_t*>(&A_tile[a_base + ko]);
            #pragma unroll
            for (int f = 0; f < MV5_NFRAG; ++f) {
                v2i_t b_cur = *reinterpret_cast<const v2i_t*>(
                    &B_tile[b_base + f * WMMA_DIM * LDSBK + ko]);
                acc[f] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    a_cur, b_cur, acc[f]);
            }
        }
        #pragma unroll
        for (int f = 0; f < MV5_NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            const float wsc = (abs_n < N)
                ? __half2float(ws_e[abs_n * num_groups + g]) : 0.0f;
            #pragma unroll
            for (int ee = 0; ee < 8; ++ee) running[f][ee] += acc[f][ee] * wsc;
        }
        __syncthreads();
    }

    // ---- epilogue: per-row act scale, write out at padded rows (guarded) ----
    #pragma unroll
    for (int ee = 0; ee < 8; ++ee) {
        const int r = warp_id * WMMA_DIM + frag_row0 + ee;
        if (r >= block_m) continue;
        const int row_pad = row0 + r;
        const int offs_token = sorted_token_ids[row_pad];
        if (offs_token >= num_valid_tokens) continue;
        const int src = offs_token / top_k;
        const float asc = act_scales[src];
        #pragma unroll
        for (int f = 0; f < MV5_NFRAG; ++f) {
            const int abs_n = block_n + f * WMMA_DIM + frag_col;
            if (abs_n < N)
                out[(long)row_pad * N + abs_n] = __float2half(running[f][ee] * asc);
        }
    }
}

}  // namespace w4a8_fp8_moe

// =================== Host launcher ===================
void launch_mmq_fp8_moe_gemm_gfx1201(
    const at::Tensor& x,                 // (T, K) fp16
    const at::Tensor& w_packed,          // (E, N, K/8) int32
    const at::Tensor& scales,            // (E, N, K/group) fp16
    const at::Tensor& w_zeros,           // (E, N/8, K/group) int32 or empty
    const at::Tensor& sorted_token_ids,  // (P,) int32
    const at::Tensor& expert_ids,        // (P/block_m,) int32
    const at::Tensor& num_tokens_post_padded,  // (1,) int32
    at::Tensor& out,                     // (P, N) fp16
    int64_t top_k, int64_t block_m, int64_t version) {

    using namespace w4a8_fp8_moe;
    const int T = x.size(0);
    const int K = x.size(1);
    const int N = w_packed.size(1);
    const int group_size = K / scales.size(2);
    const int P = sorted_token_ids.size(0);
    const int num_valid_tokens = T * (int)top_k;

    TORCH_CHECK(group_size % WMMA_DIM == 0 && group_size <= MAX_GROUP_SIZE,
                "group_size must be mult of 16 and <=128; got ", group_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto x_fp8 = at::empty({T, K}, x.options().dtype(at::kByte));
    auto act_scales = at::empty({T}, x.options().dtype(at::kFloat));
    moe_compute_act_fp8_kernel<<<dim3(T), dim3(ACTQUANT_BLOCK), 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        x_fp8.data_ptr<unsigned char>(), act_scales.data_ptr<float>(), T, K);

    const int* wz = (w_zeros.defined() && w_zeros.numel() > 0)
                    ? w_zeros.data_ptr<int>() : nullptr;

    if (version == 5) {
        TORCH_CHECK(block_m % WMMA_DIM == 0 && block_m <= 64,
                    "v5 block_m must be mult of 16 and <=64; got ", block_m);
        dim3 block(MV5_MAX_WARPS * 32);
        dim3 grid((N + MV5_BN - 1) / MV5_BN, P / (int)block_m);
        moe_gemm_v5_kernel<<<grid, block, 0, stream>>>(
            x_fp8.data_ptr<unsigned char>(), act_scales.data_ptr<float>(),
            w_packed.data_ptr<int>(),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()), wz,
            sorted_token_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),
            num_tokens_post_padded.data_ptr<int>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            T, N, K, group_size, (int)top_k, (int)block_m, num_valid_tokens);
    } else {
        dim3 block(MV0_BN, MV0_BM);
        dim3 grid((N + MV0_BN - 1) / MV0_BN, (P + MV0_BM - 1) / MV0_BM);
        moe_gemm_v0_kernel<<<grid, block, 0, stream>>>(
            x_fp8.data_ptr<unsigned char>(), act_scales.data_ptr<float>(),
            w_packed.data_ptr<int>(),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()), wz,
            sorted_token_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),
            num_tokens_post_padded.data_ptr<int>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            T, N, K, group_size, (int)top_k, (int)block_m, num_valid_tokens);
    }
}
