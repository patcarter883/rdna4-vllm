// Native flash-DECODE attention for gfx1201 (RDNA4) — Triton-free, no WMMA.
//
// Companion to the prefill kernel (attn_hip). Decode is M=1 (one query token per
// sequence) attending over the cached KV — it is bandwidth-bound, NOT matmul-bound,
// so there is NO WMMA here at all: scores are warp-reduced dot products and the
// output is an online-softmax accumulation. That means none of the gfx11->gfx12
// fragment-layout risk that the prefill kernel had to design around; the only thing
// that matters is coalesced K/V loads and a correct cross-warp softmax combine.
//
// Structure (mirrors Atlas paged_decode_attn.cu, AGPL — read for structure, not
// copied): one CTA per (q_head, seq). NWARPS warps split the KV range; each warp
// runs an independent online softmax over its strided key subset, then a cross-warp
// combine merges the per-warp (m, l, o) partials. Coalesced layout: lane `l` owns
// head-dim elements d = l, l+32, l+64, ... (D_PER_LANE of them) of q / k / v / o, so
// consecutive lanes read consecutive d (fully coalesced global loads).
//
// SCOPE (v0, correctness-first): dense (non-paged) bf16 KV, GQA, optional sliding
//   window. q:[B,Hq,D]  k/v:[B,Skv,Hk,D]  out:[B,Hq,D]. Pure decode: the single
//   query attends to ALL Skv cached keys (causal is implicit — keys are history),
//   sliding_window masks keys older than the window. NOT YET: paged-KV block tables,
//   fp8/NVFP4-KV + LUT dequant + sparse-V gating (the "turbo" path). Those are perf
//   passes on a kernel that must first match torch SDPA in fp32 accumulation.
//
// STATUS: compiles-pending / numerics-pending GPU. Validate with attn_decode_parity.py
//   under a 1-card lease before trusting.

#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {

constexpr int WARP = 32;
constexpr int NWARPS = 8;
constexpr int THREADS = WARP * NWARPS;   // 256

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor(v, o);
    return v;
}

// One CTA per (q_head, seq). NWARPS warps split the key range; cross-warp combine at end.
template <int HEAD_DIM>
__global__ __launch_bounds__(THREADS) void flash_decode_kernel(
    const __hip_bfloat16* __restrict__ Q,   // [B, num_q_heads, HEAD_DIM]
    const __hip_bfloat16* __restrict__ K,   // [B, Skv, num_kv_heads, HEAD_DIM]
    const __hip_bfloat16* __restrict__ V,   // [B, Skv, num_kv_heads, HEAD_DIM]
    __hip_bfloat16* __restrict__ O,         // [B, num_q_heads, HEAD_DIM]
    const unsigned int Skv,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const float scale,
    const unsigned int sliding_window) {     // 0 = unlimited
    constexpr int D_PER_LANE = HEAD_DIM / WARP;     // each lane owns this many d's

    const unsigned int q_head = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (q_head >= num_q_heads) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int gqa = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa;

    const __hip_bfloat16* q_ptr = Q + (b * num_q_heads + q_head) * HEAD_DIM;
    const __hip_bfloat16* k_base = K + (size_t)b * Skv * num_kv_heads * HEAD_DIM + kv_head * HEAD_DIM;
    const __hip_bfloat16* v_base = V + (size_t)b * Skv * num_kv_heads * HEAD_DIM + kv_head * HEAD_DIM;
    const unsigned int kv_row_stride = num_kv_heads * HEAD_DIM;

    // This lane's q elements (d = lane, lane+32, ...): load once.
    float q_reg[D_PER_LANE];
    #pragma unroll
    for (int i = 0; i < D_PER_LANE; i++)
        q_reg[i] = (float)q_ptr[i * WARP + lane];

    // Per-warp online-softmax state.
    float m = -1e30f, l = 0.0f;
    float o_reg[D_PER_LANE];
    #pragma unroll
    for (int i = 0; i < D_PER_LANE; i++) o_reg[i] = 0.0f;

    // Each warp strides over keys: j = warp_id, warp_id+NWARPS, ...
    for (unsigned int j = warp_id; j < Skv; j += NWARPS) {
        if (sliding_window > 0 && (Skv - 1 - j) >= sliding_window) continue;
        const __hip_bfloat16* k_ptr = k_base + (size_t)j * kv_row_stride;
        // score = scale * dot(q, k_j)
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; i++)
            partial += q_reg[i] * (float)k_ptr[i * WARP + lane];
        float score = warp_reduce_sum(partial) * scale;   // same on all lanes

        // online softmax update (whole warp shares m/l/score)
        const float m_new = fmaxf(m, score);
        const float correct = __expf(m - m_new);
        const float p = __expf(score - m_new);
        const __hip_bfloat16* v_ptr = v_base + (size_t)j * kv_row_stride;
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; i++)
            o_reg[i] = o_reg[i] * correct + p * (float)v_ptr[i * WARP + lane];
        l = l * correct + p;
        m = m_new;
    }

    // ---- cross-warp combine ----
    __shared__ float s_m[NWARPS];
    __shared__ float s_l[NWARPS];
    __shared__ float s_o[NWARPS][HEAD_DIM];
    if (lane == 0) { s_m[warp_id] = m; s_l[warp_id] = l; }
    #pragma unroll
    for (int i = 0; i < D_PER_LANE; i++)
        s_o[warp_id][i * WARP + lane] = o_reg[i];
    __syncthreads();

    // Warp 0 merges the NWARPS partials and writes the output (coalesced).
    if (warp_id == 0) {
        float m_g = -1e30f;
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) m_g = fmaxf(m_g, s_m[w]);
        float l_g = 0.0f;
        float w_corr[NWARPS];
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) { w_corr[w] = __expf(s_m[w] - m_g); l_g += s_l[w] * w_corr[w]; }
        const float inv_l = (l_g > 0.0f) ? 1.0f / l_g : 0.0f;
        __hip_bfloat16* o_ptr = O + (b * num_q_heads + q_head) * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; i++) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < NWARPS; w++) acc += s_o[w][i * WARP + lane] * w_corr[w];
            o_ptr[i * WARP + lane] = (__hip_bfloat16)(acc * inv_l);
        }
    }
}

}  // namespace

// ----------------------------------------------------------------------------
// Launcher. q:[B,Hq,D]  k/v:[B,Skv,Hk,D]  -> out:[B,Hq,D]  (dense bf16 decode)
// ----------------------------------------------------------------------------
void launch_flash_decode(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                         at::Tensor& out, double scale, int64_t sliding_window) {
    const unsigned int B = q.size(0);
    const unsigned int num_q_heads = q.size(1);
    const unsigned int head_dim = q.size(2);
    const unsigned int Skv = k.size(1);
    const unsigned int num_kv_heads = k.size(2);
    const float sc = (scale > 0) ? (float)scale : 1.0f / sqrtf((float)head_dim);

    dim3 grid(num_q_heads, B, 1);
    dim3 block(THREADS, 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto* qp = reinterpret_cast<const __hip_bfloat16*>(q.data_ptr());
    const auto* kp = reinterpret_cast<const __hip_bfloat16*>(k.data_ptr());
    const auto* vp = reinterpret_cast<const __hip_bfloat16*>(v.data_ptr());
    auto* op = reinterpret_cast<__hip_bfloat16*>(out.data_ptr());

#define LAUNCH(HD)                                                                  \
    flash_decode_kernel<HD><<<grid, block, 0, stream>>>(                            \
        qp, kp, vp, op, Skv, num_q_heads, num_kv_heads, sc, (unsigned)sliding_window); \
    break

    switch (head_dim) {
        case 64:  LAUNCH(64);
        case 128: LAUNCH(128);
        case 256: LAUNCH(256);
        default:
            TORCH_CHECK(false, "attn_decode: unsupported head_dim ", head_dim,
                        " (add a template instantiation; must be a multiple of 32)");
    }
#undef LAUNCH
}

// ============================================================================
// PAGED variant — serve-ready. Identical online-softmax math to the dense kernel;
// the ONLY difference is K/V addressing goes through a block table, and each
// sequence has its own context length. KV cache layout (vLLM/sgl NHD):
//   k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
//   block_table:     [B, max_blocks_per_seq] int32  (logical block -> physical block)
//   context_lens:    [B] int32                       (valid keys per sequence)
// Logical key j -> physical row = block_table[b][j / block_size] * block_size
//                                 + (j % block_size).
// ============================================================================
namespace {

template <int HEAD_DIM>
__global__ __launch_bounds__(THREADS) void flash_decode_paged_kernel(
    const __hip_bfloat16* __restrict__ Q,        // [B, num_q_heads, HEAD_DIM]
    const __hip_bfloat16* __restrict__ K_cache,  // [num_blocks, block_size, num_kv_heads, HEAD_DIM]
    const __hip_bfloat16* __restrict__ V_cache,
    const int* __restrict__ block_table,         // [B, max_blocks]
    const int* __restrict__ context_lens,        // [B]
    __hip_bfloat16* __restrict__ O,              // [B, num_q_heads, HEAD_DIM]
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int block_size,
    const unsigned int max_blocks,
    const float scale,
    const unsigned int sliding_window) {
    constexpr int D_PER_LANE = HEAD_DIM / WARP;

    const unsigned int q_head = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (q_head >= num_q_heads) return;
    const unsigned int ctx_len = (unsigned int)context_lens[b];
    if (ctx_len == 0) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int gqa = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa;
    const unsigned int kv_row_stride = num_kv_heads * HEAD_DIM;     // within a block
    const int* bt = block_table + (size_t)b * max_blocks;

    const __hip_bfloat16* q_ptr = Q + (b * num_q_heads + q_head) * HEAD_DIM;
    float q_reg[D_PER_LANE];
    #pragma unroll
    for (int i = 0; i < D_PER_LANE; i++) q_reg[i] = (float)q_ptr[i * WARP + lane];

    float m = -1e30f, l = 0.0f;
    float o_reg[D_PER_LANE];
    #pragma unroll
    for (int i = 0; i < D_PER_LANE; i++) o_reg[i] = 0.0f;

    for (unsigned int j = warp_id; j < ctx_len; j += NWARPS) {
        if (sliding_window > 0 && (ctx_len - 1 - j) >= sliding_window) continue;
        const unsigned int phys = (unsigned int)bt[j / block_size] * block_size + (j % block_size);
        const __hip_bfloat16* k_ptr = K_cache + (size_t)phys * kv_row_stride + kv_head * HEAD_DIM;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; i++)
            partial += q_reg[i] * (float)k_ptr[i * WARP + lane];
        float score = warp_reduce_sum(partial) * scale;

        const float m_new = fmaxf(m, score);
        const float correct = __expf(m - m_new);
        const float p = __expf(score - m_new);
        const __hip_bfloat16* v_ptr = V_cache + (size_t)phys * kv_row_stride + kv_head * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; i++)
            o_reg[i] = o_reg[i] * correct + p * (float)v_ptr[i * WARP + lane];
        l = l * correct + p;
        m = m_new;
    }

    __shared__ float s_m[NWARPS];
    __shared__ float s_l[NWARPS];
    __shared__ float s_o[NWARPS][HEAD_DIM];
    if (lane == 0) { s_m[warp_id] = m; s_l[warp_id] = l; }
    #pragma unroll
    for (int i = 0; i < D_PER_LANE; i++) s_o[warp_id][i * WARP + lane] = o_reg[i];
    __syncthreads();

    if (warp_id == 0) {
        float m_g = -1e30f;
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) m_g = fmaxf(m_g, s_m[w]);
        float l_g = 0.0f;
        float w_corr[NWARPS];
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) { w_corr[w] = __expf(s_m[w] - m_g); l_g += s_l[w] * w_corr[w]; }
        const float inv_l = (l_g > 0.0f) ? 1.0f / l_g : 0.0f;
        __hip_bfloat16* o_ptr = O + (b * num_q_heads + q_head) * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; i++) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < NWARPS; w++) acc += s_o[w][i * WARP + lane] * w_corr[w];
            o_ptr[i * WARP + lane] = (__hip_bfloat16)(acc * inv_l);
        }
    }
}

}  // namespace

// q:[B,Hq,D]  k/v_cache:[num_blocks,block_size,Hk,D]  block_table:[B,max_blocks]  ctx_lens:[B]
void launch_flash_decode_paged(const at::Tensor& q, const at::Tensor& k_cache,
                               const at::Tensor& v_cache, const at::Tensor& block_table,
                               const at::Tensor& context_lens, at::Tensor& out, double scale,
                               int64_t sliding_window) {
    const unsigned int B = q.size(0);
    const unsigned int num_q_heads = q.size(1);
    const unsigned int head_dim = q.size(2);
    const unsigned int block_size = k_cache.size(1);
    const unsigned int num_kv_heads = k_cache.size(2);
    const unsigned int max_blocks = block_table.size(1);
    const float sc = (scale > 0) ? (float)scale : 1.0f / sqrtf((float)head_dim);

    dim3 grid(num_q_heads, B, 1);
    dim3 block(THREADS, 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto* qp = reinterpret_cast<const __hip_bfloat16*>(q.data_ptr());
    const auto* kp = reinterpret_cast<const __hip_bfloat16*>(k_cache.data_ptr());
    const auto* vp = reinterpret_cast<const __hip_bfloat16*>(v_cache.data_ptr());
    const int* bt = block_table.data_ptr<int>();
    const int* cl = context_lens.data_ptr<int>();
    auto* op = reinterpret_cast<__hip_bfloat16*>(out.data_ptr());

#define LAUNCHP(HD)                                                                          \
    flash_decode_paged_kernel<HD><<<grid, block, 0, stream>>>(                               \
        qp, kp, vp, bt, cl, op, num_q_heads, num_kv_heads, block_size, max_blocks, sc,       \
        (unsigned)sliding_window);                                                           \
    break

    switch (head_dim) {
        case 64:  LAUNCHP(64);
        case 128: LAUNCHP(128);
        case 256: LAUNCHP(256);
        default:
            TORCH_CHECK(false, "attn_decode: unsupported head_dim ", head_dim);
    }
#undef LAUNCHP
}

// ============================================================================
// PAGED fp8-KV variant — the "turbo" path. KV cache is OCP e4m3 (torch
// float8_e4m3fn), one byte per element, same paged layout. Per-tensor descale
// (matches minisgl FP8_PER_TENSOR): k_descale folds into the SCORE (pre-softmax),
// v_descale folds into the OUTPUT (post-accumulation, linear). Decode uses the
// hardware v_cvt_f32_fp8 (exact OCP e4m3), same as w4a8_fp8_wmma_kernel.hip.
// ============================================================================
namespace {

__device__ __forceinline__ float e4m3_to_f32(unsigned char b) {
    return __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(b), 0);  // OCP e4m3, gfx12
}

template <int HEAD_DIM>
__global__ __launch_bounds__(THREADS) void flash_decode_paged_fp8_kernel(
    const __hip_bfloat16* __restrict__ Q,        // [B, num_q_heads, HEAD_DIM] (bf16)
    const unsigned char* __restrict__ K_cache,   // [num_blocks, block_size, num_kv_heads, HEAD_DIM] e4m3
    const unsigned char* __restrict__ V_cache,
    const int* __restrict__ block_table,         // [B, max_blocks]
    const int* __restrict__ context_lens,        // [B]
    __hip_bfloat16* __restrict__ O,              // [B, num_q_heads, HEAD_DIM]
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int block_size,
    const unsigned int max_blocks,
    const float scale,
    const float k_descale,
    const float v_descale,
    const unsigned int sliding_window) {
    constexpr int D_PER_LANE = HEAD_DIM / WARP;

    const unsigned int q_head = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (q_head >= num_q_heads) return;
    const unsigned int ctx_len = (unsigned int)context_lens[b];
    if (ctx_len == 0) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int gqa = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa;
    const unsigned int kv_row_stride = num_kv_heads * HEAD_DIM;
    const int* bt = block_table + (size_t)b * max_blocks;
    const float score_scale = scale * k_descale;     // k descale folds into pre-softmax score

    const __hip_bfloat16* q_ptr = Q + (b * num_q_heads + q_head) * HEAD_DIM;
    float q_reg[D_PER_LANE];
    #pragma unroll
    for (int i = 0; i < D_PER_LANE; i++) q_reg[i] = (float)q_ptr[i * WARP + lane];

    float m = -1e30f, l = 0.0f;
    float o_reg[D_PER_LANE];
    #pragma unroll
    for (int i = 0; i < D_PER_LANE; i++) o_reg[i] = 0.0f;

    for (unsigned int j = warp_id; j < ctx_len; j += NWARPS) {
        if (sliding_window > 0 && (ctx_len - 1 - j) >= sliding_window) continue;
        const unsigned int phys = (unsigned int)bt[j / block_size] * block_size + (j % block_size);
        const unsigned char* k_ptr = K_cache + (size_t)phys * kv_row_stride + kv_head * HEAD_DIM;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; i++)
            partial += q_reg[i] * e4m3_to_f32(k_ptr[i * WARP + lane]);
        float score = warp_reduce_sum(partial) * score_scale;

        const float m_new = fmaxf(m, score);
        const float correct = __expf(m - m_new);
        const float p = __expf(score - m_new);
        const unsigned char* v_ptr = V_cache + (size_t)phys * kv_row_stride + kv_head * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; i++)
            o_reg[i] = o_reg[i] * correct + p * e4m3_to_f32(v_ptr[i * WARP + lane]);
        l = l * correct + p;
        m = m_new;
    }

    __shared__ float s_m[NWARPS];
    __shared__ float s_l[NWARPS];
    __shared__ float s_o[NWARPS][HEAD_DIM];
    if (lane == 0) { s_m[warp_id] = m; s_l[warp_id] = l; }
    #pragma unroll
    for (int i = 0; i < D_PER_LANE; i++) s_o[warp_id][i * WARP + lane] = o_reg[i];
    __syncthreads();

    if (warp_id == 0) {
        float m_g = -1e30f;
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) m_g = fmaxf(m_g, s_m[w]);
        float l_g = 0.0f;
        float w_corr[NWARPS];
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) { w_corr[w] = __expf(s_m[w] - m_g); l_g += s_l[w] * w_corr[w]; }
        // v descale folds linearly into the output; combine with 1/l.
        const float out_scale = (l_g > 0.0f) ? (v_descale / l_g) : 0.0f;
        __hip_bfloat16* o_ptr = O + (b * num_q_heads + q_head) * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < D_PER_LANE; i++) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < NWARPS; w++) acc += s_o[w][i * WARP + lane] * w_corr[w];
            o_ptr[i * WARP + lane] = (__hip_bfloat16)(acc * out_scale);
        }
    }
}

}  // namespace

// q:[B,Hq,D] bf16  k/v_cache:[num_blocks,block_size,Hk,D] e4m3  block_table:[B,max_blocks]  ctx_lens:[B]
void launch_flash_decode_paged_fp8(const at::Tensor& q, const at::Tensor& k_cache,
                                   const at::Tensor& v_cache, const at::Tensor& block_table,
                                   const at::Tensor& context_lens, at::Tensor& out, double scale,
                                   double k_descale, double v_descale, int64_t sliding_window) {
    const unsigned int B = q.size(0);
    const unsigned int num_q_heads = q.size(1);
    const unsigned int head_dim = q.size(2);
    const unsigned int block_size = k_cache.size(1);
    const unsigned int num_kv_heads = k_cache.size(2);
    const unsigned int max_blocks = block_table.size(1);
    const float sc = (scale > 0) ? (float)scale : 1.0f / sqrtf((float)head_dim);

    dim3 grid(num_q_heads, B, 1);
    dim3 block(THREADS, 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto* qp = reinterpret_cast<const __hip_bfloat16*>(q.data_ptr());
    const auto* kp = reinterpret_cast<const unsigned char*>(k_cache.data_ptr());
    const auto* vp = reinterpret_cast<const unsigned char*>(v_cache.data_ptr());
    const int* bt = block_table.data_ptr<int>();
    const int* cl = context_lens.data_ptr<int>();
    auto* op = reinterpret_cast<__hip_bfloat16*>(out.data_ptr());

#define LAUNCHF(HD)                                                                              \
    flash_decode_paged_fp8_kernel<HD><<<grid, block, 0, stream>>>(                               \
        qp, kp, vp, bt, cl, op, num_q_heads, num_kv_heads, block_size, max_blocks, sc,           \
        (float)k_descale, (float)v_descale, (unsigned)sliding_window);                           \
    break

    switch (head_dim) {
        case 64:  LAUNCHF(64);
        case 128: LAUNCHF(128);
        case 256: LAUNCHF(256);
        default:
            TORCH_CHECK(false, "attn_decode: unsupported head_dim ", head_dim);
    }
#undef LAUNCHF
}
