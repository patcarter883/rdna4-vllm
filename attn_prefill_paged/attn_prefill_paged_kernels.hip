// Native PAGED / chunked-prefill (extend) flash-attention for gfx1201 (RDNA4) — rocwmma, Triton-free.
//
// This is the kernel that makes PREFILL Triton-free for the real serving config (chunked prefill +
// prefix caching). dense attn_hip only covers cold prefill; here Q is the NEW tokens and K/V come
// from the PAGED cache (prefix ⧺ new), so radix/APC prefix-cache hits and chunked prefill are
// handled natively.
//
// It reuses attn_hip's VALIDATED gfx12 WMMA core verbatim (rocwmma QK^T/PV + the smem-staged online
// softmax that decouples the math from the fragment layout). Three things change vs attn_hip:
//   1. Q is PACKED / varlen across sequences (indexed via cu_seqlens_q), not a single contiguous seq.
//   2. K/V load from the PAGED cache via block_table + kv_block_stride (as in attn_decode), not a
//      contiguous [seq,kv_heads,hd] tensor.
//   3. Causal mask carries a PREFIX OFFSET: query global position = prefix_len + local_q_idx, where
//      prefix_len = context_len - q_len; a key (global pos = its cache index j) is allowed iff
//      j <= query_global_pos.
//
// SCOPE: bf16 + fp8-KV (e4m3, per-tensor descale), GQA, causal + optional sliding window, head_dim
// 64/128/256. head_dim 256 drops the flash tile to BR=BC=16 (1 warp) so the static LDS stays under
// the gfx1201 64 KB cap — same WMMA math, smaller tile (see tile_rows() below). KV cache
// [num_blocks, block_size, kv_heads, hd] (kv_block_stride=0) OR vLLM's interleaved
// [num_blocks,2,...] (pass the doubled stride + unbind views).
//
// STATUS: GPU-validated on gfx1201 — attn_prefill_paged_parity.py ALL PASS (22 cases: bf16 + fp8,
// head_dim 64/128/256, incl. cold/extend/ragged/batch/SWA/GQA, cos≈0.999998, 0 bf16-ULP violations).

#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rocwmma/rocwmma.hpp>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {

using bf16_t = rocwmma::bfloat16_t;

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
// Tile (query rows / key cols) is HEAD_DIM-dependent so the static LDS stays under the gfx1201
// 64 KB cap. 64/128 keep the VALIDATED BR=BC=32 (2 warps / 64 threads, ~30-49 KB LDS). HEAD_DIM=256
// would need ~120 KB at BR=32 (smem_Q/K/V + the fp32 O/PV accumulators all double), so it drops to
// BR=BC=16 (1 warp / 32 threads, ~58 KB) — same WMMA math, smaller flash tile.
constexpr int tile_rows(int head_dim) { return head_dim >= 256 ? 16 : 32; }
constexpr int nthreads(int head_dim) { return tile_rows(head_dim) / WMMA_M * 32; }

using FragQ  = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16_t, rocwmma::row_major>;
using FragKt = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16_t, rocwmma::col_major>;
using FragP  = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16_t, rocwmma::row_major>;
using FragV  = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16_t, rocwmma::row_major>;
using FragAcc = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

template <int HEAD_DIM>
__global__ __launch_bounds__(nthreads(HEAD_DIM)) void flash_prefill_paged_kernel(
    const __hip_bfloat16* __restrict__ Q,        // [total_q_tokens, num_q_heads, HEAD_DIM] (packed)
    const __hip_bfloat16* __restrict__ K_cache,  // [num_blocks, block_size, num_kv_heads, HEAD_DIM]
    const __hip_bfloat16* __restrict__ V_cache,
    const int* __restrict__ block_table,         // [num_seqs, max_blocks]
    const int* __restrict__ cu_seqlens_q,        // [num_seqs+1] : query offsets in packed Q
    const int* __restrict__ context_lens,        // [num_seqs] : total KV length per seq
    __hip_bfloat16* __restrict__ O,              // [total_q_tokens, num_q_heads, HEAD_DIM]
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int block_size,
    const unsigned int max_blocks,
    const long kv_block_stride,
    const float inv_sqrt_d,
    const unsigned int causal,
    const unsigned int sliding_window) {
    constexpr int BR = tile_rows(HEAD_DIM), BC = BR;
    constexpr int M_TILES = BR / WMMA_M;
    constexpr int N_TILES = BC / WMMA_N;
    constexpr int WARPS = M_TILES;
    constexpr int THREADS = WARPS * 32;
    constexpr int KSTEPS = HEAD_DIM / WMMA_K;
    constexpr int D_TILES = HEAD_DIM / WMMA_N;
    constexpr int PAD = 8;
    constexpr int LDK = HEAD_DIM + PAD;
    constexpr int VEC = 8, CH_PER_ROW = HEAD_DIM / VEC;

    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int seq = blockIdx.z;
    if (q_head >= num_q_heads) return;

    const unsigned int q_base = cu_seqlens_q[seq];          // packed offset of this seq's queries
    const unsigned int q_len = cu_seqlens_q[seq + 1] - q_base;
    const unsigned int q_start = q_block * BR;              // local query row base
    if (q_start >= q_len) return;
    const unsigned int q_rows = min(BR, q_len - q_start);

    const unsigned int context_len = (unsigned int)context_lens[seq];
    const unsigned int prefix_len = context_len - q_len;   // cached prefix before the new tokens

    const unsigned int tid = threadIdx.x, warp_id = tid >> 5;
    const unsigned int gqa = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa;
    const unsigned int q_stride = num_q_heads * HEAD_DIM;
    const unsigned int kv_row_stride = num_kv_heads * HEAD_DIM;
    const int* bt = block_table + (size_t)seq * max_blocks;

    __shared__ bf16_t smem_Q[BR][LDK];
    __shared__ bf16_t smem_K[BC][LDK];
    __shared__ bf16_t smem_V[BC][LDK];
    __shared__ bf16_t smem_P[BR][BC];
    __shared__ float  smem_S[BR][BC];
    __shared__ float  smem_PV[BR][HEAD_DIM];
    __shared__ float  smem_O[BR][HEAD_DIM];
    __shared__ float  smem_m[BR];
    __shared__ float  smem_l[BR];
    __shared__ float  smem_resc[BR];

    for (unsigned int r = tid; r < BR; r += THREADS) { smem_m[r] = -1e30f; smem_l[r] = 0.0f; }
    for (unsigned int i = tid; i < BR * HEAD_DIM; i += THREADS) ((float*)smem_O)[i] = 0.0f;

    // ---- load Q tile (packed varlen) ----
    for (unsigned int idx = tid; idx < BR * CH_PER_ROW; idx += THREADS) {
        unsigned int row = idx / CH_PER_ROW, col = (idx % CH_PER_ROW) * VEC;
        if (row < q_rows) {
            unsigned int gq = q_base + q_start + row;       // packed query token index
            *(uint4*)&smem_Q[row][col] = *(const uint4*)&Q[gq * q_stride + q_head * HEAD_DIM + col];
        } else {
            *(uint4*)&smem_Q[row][col] = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    // causal cap: the last query in this tile has global pos prefix_len + q_start + q_rows - 1,
    // so no key beyond that index can be attended.
    unsigned int kv_limit = context_len;
    if (causal) kv_limit = min(context_len, prefix_len + q_start + q_rows);
    const unsigned int num_kv_blocks = (kv_limit + BC - 1) / BC;

    for (unsigned int kb = 0; kb < num_kv_blocks; kb++) {
        const unsigned int kv_start = kb * BC;
        const unsigned int kv_len = min(BC, context_len - kv_start);

        // ---- load K/V tile from the PAGED cache (per-key block-table gather) ----
        for (unsigned int idx = tid; idx < BC * CH_PER_ROW; idx += THREADS) {
            unsigned int row = idx / CH_PER_ROW, col = (idx % CH_PER_ROW) * VEC;
            unsigned int j = kv_start + row;                // key global index in this seq's cache
            if (j < context_len) {
                size_t base = (size_t)(unsigned int)bt[j / block_size] * kv_block_stride
                              + (size_t)(j % block_size) * kv_row_stride + kv_head * HEAD_DIM;
                *(uint4*)&smem_K[row][col] = *(const uint4*)&K_cache[base + col];
                *(uint4*)&smem_V[row][col] = *(const uint4*)&V_cache[base + col];
            } else {
                *(uint4*)&smem_K[row][col] = make_uint4(0, 0, 0, 0);
                *(uint4*)&smem_V[row][col] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // ---- QK^T (warp owns query m-tile) ----
        {
            const int mt = warp_id;
            for (int nt = 0; nt < N_TILES; nt++) {
                FragAcc acc; rocwmma::fill_fragment(acc, 0.0f);
                for (int ks = 0; ks < KSTEPS; ks++) {
                    FragQ qf; FragKt kf;
                    rocwmma::load_matrix_sync(qf, &smem_Q[mt * WMMA_M][ks * WMMA_K], LDK);
                    rocwmma::load_matrix_sync(kf, &smem_K[nt * WMMA_N][ks * WMMA_K], LDK);
                    rocwmma::mma_sync(acc, qf, kf, acc);
                }
                rocwmma::store_matrix_sync(&smem_S[mt * WMMA_M][nt * WMMA_N], acc, BC,
                                           rocwmma::mem_row_major);
            }
        }
        __syncthreads();

        // ---- online softmax in smem, one thread per query row, prefix-offset causal ----
        if (tid < BR) {
            const unsigned int r = tid;
            const bool row_valid = (r < q_rows);
            const unsigned int qpos = prefix_len + q_start + r;   // query GLOBAL position
            float rmax = -1e30f;
            for (unsigned int c = 0; c < BC; c++) {
                float s = smem_S[r][c] * inv_sqrt_d;
                const unsigned int kpos = kv_start + c;           // key GLOBAL position
                bool masked = (c >= kv_len) || !row_valid;
                if (causal && kpos > qpos) masked = true;
                if (causal && sliding_window > 0 && kpos <= qpos && (qpos - kpos) >= sliding_window)
                    masked = true;
                if (masked) s = -1e30f;
                smem_S[r][c] = s;
                rmax = fmaxf(rmax, s);
            }
            const float m_old = smem_m[r];
            const float m_new = fmaxf(m_old, rmax);
            const float resc = __expf(m_old - m_new);
            float sum = 0.0f;
            for (unsigned int c = 0; c < BC; c++) {
                bf16_t pb = (bf16_t)__expf(smem_S[r][c] - m_new);  // round P->bf16 before summing l
                smem_P[r][c] = pb;
                sum += (float)pb;
            }
            smem_m[r] = m_new;
            smem_l[r] = smem_l[r] * resc + sum;
            smem_resc[r] = resc;
        }
        __syncthreads();

        // ---- PV ----
        {
            const int mt = warp_id;
            for (int dt = 0; dt < D_TILES; dt++) {
                FragAcc acc; rocwmma::fill_fragment(acc, 0.0f);
                for (int nt = 0; nt < N_TILES; nt++) {
                    FragP pf; FragV vf;
                    rocwmma::load_matrix_sync(pf, &smem_P[mt * WMMA_M][nt * WMMA_N], BC);
                    rocwmma::load_matrix_sync(vf, &smem_V[nt * WMMA_N][dt * WMMA_N], LDK);
                    rocwmma::mma_sync(acc, pf, vf, acc);
                }
                rocwmma::store_matrix_sync(&smem_PV[mt * WMMA_M][dt * WMMA_N], acc, HEAD_DIM,
                                           rocwmma::mem_row_major);
            }
        }
        __syncthreads();

        for (unsigned int i = tid; i < BR * HEAD_DIM; i += THREADS) {
            unsigned int r = i / HEAD_DIM, d = i % HEAD_DIM;
            smem_O[r][d] = smem_O[r][d] * smem_resc[r] + smem_PV[r][d];
        }
        __syncthreads();
    }

    // ---- normalize + store (packed) ----
    for (unsigned int i = tid; i < BR * HEAD_DIM; i += THREADS) {
        unsigned int r = i / HEAD_DIM, d = i % HEAD_DIM;
        if (r < q_rows) {
            unsigned int gq = q_base + q_start + r;
            float l = smem_l[r];
            float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
            O[gq * q_stride + q_head * HEAD_DIM + d] = (__hip_bfloat16)(smem_O[r][d] * inv_l);
        }
    }
}

}  // namespace

// q:[total_q,Hq,D]  k/v_cache:[num_blocks,bs,Hk,D]  block_table:[S,max_blocks]
// cu_seqlens_q:[S+1] int32  context_lens:[S] int32
void launch_flash_prefill_paged(const at::Tensor& q, const at::Tensor& k_cache,
                                const at::Tensor& v_cache, const at::Tensor& block_table,
                                const at::Tensor& cu_seqlens_q, const at::Tensor& context_lens,
                                at::Tensor& out, double scale, int64_t causal,
                                int64_t sliding_window, int64_t kv_block_stride,
                                int64_t max_seqlen_q) {
    const unsigned int num_q_heads = q.size(1);
    const unsigned int head_dim = q.size(2);
    const unsigned int block_size = k_cache.size(1);
    const unsigned int num_kv_heads = k_cache.size(2);
    const unsigned int num_seqs = context_lens.size(0);
    const unsigned int max_blocks = block_table.size(1);
    const float inv = (scale > 0) ? (float)scale : 1.0f / sqrtf((float)head_dim);
    const long kvbs = (kv_block_stride > 0) ? (long)kv_block_stride
                                            : (long)block_size * num_kv_heads * head_dim;

    const int br = tile_rows(head_dim);
    dim3 grid(num_q_heads, (max_seqlen_q + br - 1) / br, num_seqs);
    dim3 block(nthreads(head_dim), 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto* qp = reinterpret_cast<const __hip_bfloat16*>(q.data_ptr());
    const auto* kp = reinterpret_cast<const __hip_bfloat16*>(k_cache.data_ptr());
    const auto* vp = reinterpret_cast<const __hip_bfloat16*>(v_cache.data_ptr());
    const int* btp = block_table.data_ptr<int>();
    const int* cqp = cu_seqlens_q.data_ptr<int>();
    const int* clp = context_lens.data_ptr<int>();
    auto* op = reinterpret_cast<__hip_bfloat16*>(out.data_ptr());

#define LAUNCH(HD)                                                                          \
    flash_prefill_paged_kernel<HD><<<grid, block, 0, stream>>>(                             \
        qp, kp, vp, btp, cqp, clp, op, num_q_heads, num_kv_heads, block_size, max_blocks,   \
        kvbs, inv, (unsigned)causal, (unsigned)sliding_window);                             \
    break

    switch (head_dim) {
        case 64:  LAUNCH(64);
        case 128: LAUNCH(128);
        case 256: LAUNCH(256);
        default:
            TORCH_CHECK(false, "attn_prefill_paged: head_dim ", head_dim, " unsupported (64/128/256)");
    }
#undef LAUNCH
}

// ============================================================================
// fp8-KV variant — OCP e4m3 K/V cache (torch float8_e4m3fn), per-tensor descale (matches
// vLLM/minisgl FP8_PER_TENSOR + attn_decode's flash_decode_paged_fp8). Identical to the bf16 kernel
// except: K/V are 1-byte e4m3, dequant'd to bf16 on smem load (e4m3 ⊂ bf16, lossless) so the WMMA is
// unchanged; k_descale folds into the score (via inv_sqrt_d in the launcher), v_descale into the
// output (post-accumulation). Decode uses the hardware v_cvt_f32_fp8 (exact OCP e4m3), as in W4A8.
// ============================================================================
namespace {

__device__ __forceinline__ float e4m3_to_f32(unsigned char b) {
    return __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(b), 0);  // OCP e4m3, gfx12
}

template <int HEAD_DIM>
__global__ __launch_bounds__(nthreads(HEAD_DIM)) void flash_prefill_paged_fp8_kernel(
    const __hip_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ K_cache,   // [num_blocks, block_size, num_kv_heads, HEAD_DIM] e4m3
    const unsigned char* __restrict__ V_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ context_lens,
    __hip_bfloat16* __restrict__ O,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int block_size,
    const unsigned int max_blocks,
    const long kv_block_stride,
    const float inv_sqrt_d,                      // already * k_descale (folded in launcher)
    const float v_descale,
    const unsigned int causal,
    const unsigned int sliding_window) {
    constexpr int BR = tile_rows(HEAD_DIM), BC = BR;
    constexpr int M_TILES = BR / WMMA_M;
    constexpr int N_TILES = BC / WMMA_N;
    constexpr int WARPS = M_TILES;
    constexpr int THREADS = WARPS * 32;
    constexpr int KSTEPS = HEAD_DIM / WMMA_K;
    constexpr int D_TILES = HEAD_DIM / WMMA_N;
    constexpr int PAD = 8;
    constexpr int LDK = HEAD_DIM + PAD;
    constexpr int VEC = 8, CH_PER_ROW = HEAD_DIM / VEC;

    const unsigned int q_head = blockIdx.x, q_block = blockIdx.y, seq = blockIdx.z;
    if (q_head >= num_q_heads) return;
    const unsigned int q_base = cu_seqlens_q[seq];
    const unsigned int q_len = cu_seqlens_q[seq + 1] - q_base;
    const unsigned int q_start = q_block * BR;
    if (q_start >= q_len) return;
    const unsigned int q_rows = min(BR, q_len - q_start);
    const unsigned int context_len = (unsigned int)context_lens[seq];
    const unsigned int prefix_len = context_len - q_len;

    const unsigned int tid = threadIdx.x, warp_id = tid >> 5;
    const unsigned int gqa = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa;
    const unsigned int q_stride = num_q_heads * HEAD_DIM;
    const unsigned int kv_row_stride = num_kv_heads * HEAD_DIM;
    const int* bt = block_table + (size_t)seq * max_blocks;

    __shared__ bf16_t smem_Q[BR][LDK], smem_K[BC][LDK], smem_V[BC][LDK], smem_P[BR][BC];
    __shared__ float smem_S[BR][BC], smem_PV[BR][HEAD_DIM], smem_O[BR][HEAD_DIM];
    __shared__ float smem_m[BR], smem_l[BR], smem_resc[BR];

    for (unsigned int r = tid; r < BR; r += THREADS) { smem_m[r] = -1e30f; smem_l[r] = 0.0f; }
    for (unsigned int i = tid; i < BR * HEAD_DIM; i += THREADS) ((float*)smem_O)[i] = 0.0f;

    for (unsigned int idx = tid; idx < BR * CH_PER_ROW; idx += THREADS) {
        unsigned int row = idx / CH_PER_ROW, col = (idx % CH_PER_ROW) * VEC;
        if (row < q_rows) {
            unsigned int gq = q_base + q_start + row;
            *(uint4*)&smem_Q[row][col] = *(const uint4*)&Q[gq * q_stride + q_head * HEAD_DIM + col];
        } else {
            *(uint4*)&smem_Q[row][col] = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    unsigned int kv_limit = context_len;
    if (causal) kv_limit = min(context_len, prefix_len + q_start + q_rows);
    const unsigned int num_kv_blocks = (kv_limit + BC - 1) / BC;

    for (unsigned int kb = 0; kb < num_kv_blocks; kb++) {
        const unsigned int kv_start = kb * BC;
        const unsigned int kv_len = min(BC, context_len - kv_start);

        for (unsigned int idx = tid; idx < BC * CH_PER_ROW; idx += THREADS) {
            unsigned int row = idx / CH_PER_ROW, col = (idx % CH_PER_ROW) * VEC;
            unsigned int j = kv_start + row;
            if (j < context_len) {
                size_t base = (size_t)(unsigned int)bt[j / block_size] * kv_block_stride
                              + (size_t)(j % block_size) * kv_row_stride + kv_head * HEAD_DIM;
                const unsigned char* kpb = &K_cache[base + col];
                const unsigned char* vpb = &V_cache[base + col];
                #pragma unroll
                for (int e = 0; e < VEC; e++) {
                    smem_K[row][col + e] = (bf16_t)e4m3_to_f32(kpb[e]);
                    smem_V[row][col + e] = (bf16_t)e4m3_to_f32(vpb[e]);
                }
            } else {
                #pragma unroll
                for (int e = 0; e < VEC; e++) { smem_K[row][col + e] = (bf16_t)0.0f; smem_V[row][col + e] = (bf16_t)0.0f; }
            }
        }
        __syncthreads();

        {
            const int mt = warp_id;
            for (int nt = 0; nt < N_TILES; nt++) {
                FragAcc acc; rocwmma::fill_fragment(acc, 0.0f);
                for (int ks = 0; ks < KSTEPS; ks++) {
                    FragQ qf; FragKt kf;
                    rocwmma::load_matrix_sync(qf, &smem_Q[mt * WMMA_M][ks * WMMA_K], LDK);
                    rocwmma::load_matrix_sync(kf, &smem_K[nt * WMMA_N][ks * WMMA_K], LDK);
                    rocwmma::mma_sync(acc, qf, kf, acc);
                }
                rocwmma::store_matrix_sync(&smem_S[mt * WMMA_M][nt * WMMA_N], acc, BC, rocwmma::mem_row_major);
            }
        }
        __syncthreads();

        if (tid < BR) {
            const unsigned int r = tid;
            const bool row_valid = (r < q_rows);
            const unsigned int qpos = prefix_len + q_start + r;
            float rmax = -1e30f;
            for (unsigned int c = 0; c < BC; c++) {
                float s = smem_S[r][c] * inv_sqrt_d;           // inv_sqrt_d already * k_descale
                const unsigned int kpos = kv_start + c;
                bool masked = (c >= kv_len) || !row_valid;
                if (causal && kpos > qpos) masked = true;
                if (causal && sliding_window > 0 && kpos <= qpos && (qpos - kpos) >= sliding_window)
                    masked = true;
                if (masked) s = -1e30f;
                smem_S[r][c] = s;
                rmax = fmaxf(rmax, s);
            }
            const float m_old = smem_m[r];
            const float m_new = fmaxf(m_old, rmax);
            const float resc = __expf(m_old - m_new);
            float sum = 0.0f;
            for (unsigned int c = 0; c < BC; c++) {
                bf16_t pb = (bf16_t)__expf(smem_S[r][c] - m_new);
                smem_P[r][c] = pb;
                sum += (float)pb;
            }
            smem_m[r] = m_new;
            smem_l[r] = smem_l[r] * resc + sum;
            smem_resc[r] = resc;
        }
        __syncthreads();

        {
            const int mt = warp_id;
            for (int dt = 0; dt < D_TILES; dt++) {
                FragAcc acc; rocwmma::fill_fragment(acc, 0.0f);
                for (int nt = 0; nt < N_TILES; nt++) {
                    FragP pf; FragV vf;
                    rocwmma::load_matrix_sync(pf, &smem_P[mt * WMMA_M][nt * WMMA_N], BC);
                    rocwmma::load_matrix_sync(vf, &smem_V[nt * WMMA_N][dt * WMMA_N], LDK);
                    rocwmma::mma_sync(acc, pf, vf, acc);
                }
                rocwmma::store_matrix_sync(&smem_PV[mt * WMMA_M][dt * WMMA_N], acc, HEAD_DIM, rocwmma::mem_row_major);
            }
        }
        __syncthreads();

        for (unsigned int i = tid; i < BR * HEAD_DIM; i += THREADS) {
            unsigned int r = i / HEAD_DIM, d = i % HEAD_DIM;
            smem_O[r][d] = smem_O[r][d] * smem_resc[r] + smem_PV[r][d];
        }
        __syncthreads();
    }

    for (unsigned int i = tid; i < BR * HEAD_DIM; i += THREADS) {
        unsigned int r = i / HEAD_DIM, d = i % HEAD_DIM;
        if (r < q_rows) {
            unsigned int gq = q_base + q_start + r;
            float l = smem_l[r];
            float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
            O[gq * q_stride + q_head * HEAD_DIM + d] = (__hip_bfloat16)(smem_O[r][d] * inv_l * v_descale);
        }
    }
}

}  // namespace

void launch_flash_prefill_paged_fp8(const at::Tensor& q, const at::Tensor& k_cache,
                                    const at::Tensor& v_cache, const at::Tensor& block_table,
                                    const at::Tensor& cu_seqlens_q, const at::Tensor& context_lens,
                                    at::Tensor& out, double scale, double k_descale, double v_descale,
                                    int64_t causal, int64_t sliding_window, int64_t kv_block_stride,
                                    int64_t max_seqlen_q) {
    const unsigned int num_q_heads = q.size(1);
    const unsigned int head_dim = q.size(2);
    const unsigned int block_size = k_cache.size(1);
    const unsigned int num_kv_heads = k_cache.size(2);
    const unsigned int num_seqs = context_lens.size(0);
    const unsigned int max_blocks = block_table.size(1);
    const float inv = ((scale > 0) ? (float)scale : 1.0f / sqrtf((float)head_dim)) * (float)k_descale;
    const long kvbs = (kv_block_stride > 0) ? (long)kv_block_stride
                                            : (long)block_size * num_kv_heads * head_dim;

    const int br = tile_rows(head_dim);
    dim3 grid(num_q_heads, (max_seqlen_q + br - 1) / br, num_seqs);
    dim3 block(nthreads(head_dim), 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto* qp = reinterpret_cast<const __hip_bfloat16*>(q.data_ptr());
    const auto* kp = reinterpret_cast<const unsigned char*>(k_cache.data_ptr());
    const auto* vp = reinterpret_cast<const unsigned char*>(v_cache.data_ptr());
    const int* btp = block_table.data_ptr<int>();
    const int* cqp = cu_seqlens_q.data_ptr<int>();
    const int* clp = context_lens.data_ptr<int>();
    auto* op = reinterpret_cast<__hip_bfloat16*>(out.data_ptr());

#define LAUNCHF(HD)                                                                            \
    flash_prefill_paged_fp8_kernel<HD><<<grid, block, 0, stream>>>(                            \
        qp, kp, vp, btp, cqp, clp, op, num_q_heads, num_kv_heads, block_size, max_blocks,      \
        kvbs, inv, (float)v_descale, (unsigned)causal, (unsigned)sliding_window);              \
    break

    switch (head_dim) {
        case 64:  LAUNCHF(64);
        case 128: LAUNCHF(128);
        case 256: LAUNCHF(256);
        default:
            TORCH_CHECK(false, "attn_prefill_paged fp8: head_dim ", head_dim, " unsupported (64/128/256)");
    }
#undef LAUNCHF
}
