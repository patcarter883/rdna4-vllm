// Native HIP kernels for the Qwen3-Next / Qwen3.5 Gated Delta Net (GDN) linear-attention path,
// for AMD RDNA4 (gfx1201). AOT-compiled ONCE (no per-shape Triton JIT/autotune) and exposed as
// framework-agnostic torch.ops.gdn_hip.* — consumable from minisgl AND vLLM (both route GDN through
// the same flash-linear-attention recurrence). Correctness-first v1: faithful to the fla reference
// (fused_recurrent_gated_delta_rule), fp32 throughout; the chunked throughput optimization is later.
//
// Math (per sequence, per value-head; state S is [V x K], V=head_v_dim, K=head_k_dim), per token:
//   q,k l2-normalized over K; q *= scale (=1/sqrt(K))
//   g    = -exp(A_log) * softplus(a + dt_bias)      (decay)         beta = sigmoid(b)   (write gate)
//   S   *= exp(g)                                   (gated decay)
//   v   -= S @ k                                    (delta correction)
//   v   *= beta
//   S   += outer(v, k)                              (rank-1 write)
//   o    = S @ q                                    (readout)
// Lifted verbatim from fused_recurrent_gated_delta_rule_packed_decode_kernel / _fwd_kernel.
//
// Layout: one thread block per (sequence, value-head); blockDim.x == V; thread r owns state row
// S[r, 0..K-1] in registers (K<=GDN_MAX_K) across the whole token loop. q/k for the current token
// live in LDS (shared across the V rows). No cross-row dependency -> no block reduction in the step.

// WMMA (matrix-core) prefill path needs float16_t ops; the torch cpp_extension passes
// -D__HIP_NO_HALF_*=1, which breaks rocwmma::float16_t arithmetic/conversions — undo it before any
// half use (same gotcha as the W4A8 kernel). Must precede the rocwmma include.
#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef GDN_MAX_K
#define GDN_MAX_K 128  // head_k_dim (Qwen3.5/3.6 GDN = 128); state row held in registers
#endif

__device__ __forceinline__ float gdn_softplus(float x) {
  // matches the Triton ref: threshold 20 to avoid overflow in log1p(exp(x))
  return x <= 20.0f ? log1pf(__expf(x)) : x;
}

// Cooperative L2 norm of a length-K vector held in LDS `vec` (blockDim.x threads, V>=?).
// Returns 1/sqrt(sum(vec^2)+eps). Result broadcast via LDS scalar slot.
__device__ __forceinline__ float gdn_inv_l2(const float* __restrict__ vec, int K,
                                            float* __restrict__ red, int tid, int nthreads) {
  float partial = 0.0f;
  for (int j = tid; j < K; j += nthreads) partial += vec[j] * vec[j];
  red[tid] = partial;
  __syncthreads();
  // simple tree reduction over nthreads (power-of-two not required; serial tail by thread 0)
  for (int s = nthreads >> 1; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] + 1e-6f);
  __syncthreads();
  return inv;
}

// One recurrence step for thread r's state row s_row[0..K-1] (registers). q/k post-l2norm in LDS.
__device__ __forceinline__ float gdn_step(float* s_row, const float* q_s, const float* k_s,
                                          float v_r, float g, float beta, int K) {
  const float eg = __expf(g);
  // S *= exp(g);  delta = v - S@k
  float t = 0.0f;
#pragma unroll 4
  for (int j = 0; j < K; ++j) { s_row[j] *= eg; t += s_row[j] * k_s[j]; }
  float vr = (v_r - t) * beta;
  // S += outer(v,k);  o = S@q
  float o = 0.0f;
#pragma unroll 4
  for (int j = 0; j < K; ++j) { s_row[j] += vr * k_s[j]; o += s_row[j] * q_s[j]; }
  return o;
}

// ---------------------------------------------------------------------------------------------
// DECODE: one token per sequence. Computes g/beta inline from a,b,A_log,dt_bias. Updates state
// in place at slot = state_indices[seq]. Skips (writes zero) when slot <= 0 (NULL block).
//   q,k:        [B, H,  K]   v: [B, HV, V]   a,b: [B, HV]   A_log,dt_bias: [HV]
//   ssm_state:  [num_slots, HV, V, K]   state_indices: [B]   out: [B, HV, V]
//   GQA: head h = hv / (HV / H).
// grid = (B * HV), block = V threads.
__global__ void gdn_decode_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    float* __restrict__ ssm_state, const long* __restrict__ state_indices, float* __restrict__ out,
    int B, int H, int HV, int K, int V, float scale, int use_l2norm) {
  const int blk = blockIdx.x, r = threadIdx.x;
  const int bi = blk / HV, hv = blk % HV, hq = hv / (HV / H);

  extern __shared__ float smem[];
  float* q_s = smem;            // [K]
  float* k_s = smem + K;        // [K]
  float* red = smem + 2 * K;    // [blockDim.x] reduction scratch

  // load q,k for (bi,hq) into LDS
  for (int j = r; j < K; j += blockDim.x) {
    q_s[j] = q[(bi * H + hq) * K + j];
    k_s[j] = k[(bi * H + hq) * K + j];
  }
  __syncthreads();
  if (use_l2norm) {
    float invq = gdn_inv_l2(q_s, K, red, r, blockDim.x);
    float invk = gdn_inv_l2(k_s, K, red, r, blockDim.x);
    for (int j = r; j < K; j += blockDim.x) { q_s[j] *= invq; k_s[j] *= invk; }
    __syncthreads();
  }
  for (int j = r; j < K; j += blockDim.x) q_s[j] *= scale;
  __syncthreads();

  const long slot = state_indices[bi];
  if (r < V) {
    float* o_p = out + (bi * HV + hv) * V + r;
    if (slot <= 0) { *o_p = 0.0f; return; }
    const float x = a[bi * HV + hv] + dt_bias[hv];
    const float g = -__expf(A_log[hv]) * gdn_softplus(x);
    const float beta = 1.0f / (1.0f + __expf(-b[bi * HV + hv]));
    float* s_row = ssm_state + ((slot * HV + hv) * V + r) * K;
    float s_reg[GDN_MAX_K];
#pragma unroll 4
    for (int j = 0; j < K; ++j) s_reg[j] = s_row[j];
    float o = gdn_step(s_reg, q_s, k_s, v[(bi * HV + hv) * V + r], g, beta, K);
#pragma unroll 4
    for (int j = 0; j < K; ++j) s_row[j] = s_reg[j];
    *o_p = o;
  }
}

// ---------------------------------------------------------------------------------------------
// PREFILL: varlen, one block per (sequence, value-head); loops over the sequence's tokens carrying
// the state row in registers, writes o per token, persists the final state to slot. g,beta are
// computed inline per token from a,b (per-token) and A_log,dt_bias (per-head).
//   q,k:  [T, H, K]   v: [T, HV, V]   a,b: [T, HV]   (T = total tokens across the batch)
//   cu_seqlens: [N+1]   state_indices: [N]   has_initial_state: [N] (uint8)
//   ssm_state: [num_slots, HV, V, K]   out: [T, HV, V]
// grid = (N * HV), block = V threads.
__global__ void gdn_prefill_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    const int* __restrict__ cu_seqlens, const long* __restrict__ state_indices,
    const unsigned char* __restrict__ has_initial_state,
    float* __restrict__ ssm_state, float* __restrict__ out,
    int N, int H, int HV, int K, int V, float scale, int use_l2norm) {
  const int blk = blockIdx.x, r = threadIdx.x;
  const int n = blk / HV, hv = blk % HV, hq = hv / (HV / H);
  const int bos = cu_seqlens[n], eos = cu_seqlens[n + 1];
  const long slot = state_indices[n];
  if (slot <= 0 || bos >= eos) return;

  extern __shared__ float smem[];
  float* q_s = smem;             // [K]
  float* k_s = smem + K;         // [K]
  float* red = smem + 2 * K;     // [blockDim.x]

  // state row in registers; init from slot if has_initial_state, else zero
  float s_reg[GDN_MAX_K];
  if (r < V) {
    float* s_row = ssm_state + ((slot * HV + hv) * V + r) * K;
    const bool use_init = has_initial_state[n] != 0;
#pragma unroll 4
    for (int j = 0; j < K; ++j) s_reg[j] = use_init ? s_row[j] : 0.0f;
  }

  const float Alog = A_log[hv], dtb = dt_bias[hv];
  for (int t = bos; t < eos; ++t) {
    for (int j = r; j < K; j += blockDim.x) {
      q_s[j] = q[(t * H + hq) * K + j];
      k_s[j] = k[(t * H + hq) * K + j];
    }
    __syncthreads();
    if (use_l2norm) {
      float invq = gdn_inv_l2(q_s, K, red, r, blockDim.x);
      float invk = gdn_inv_l2(k_s, K, red, r, blockDim.x);
      for (int j = r; j < K; j += blockDim.x) { q_s[j] *= invq; k_s[j] *= invk; }
      __syncthreads();
    }
    for (int j = r; j < K; j += blockDim.x) q_s[j] *= scale;
    __syncthreads();

    if (r < V) {
      const float x = a[t * HV + hv] + dtb;
      const float g = -__expf(Alog) * gdn_softplus(x);
      const float beta = 1.0f / (1.0f + __expf(-b[t * HV + hv]));
      float o = gdn_step(s_reg, q_s, k_s, v[(t * HV + hv) * V + r], g, beta, K);
      out[(t * HV + hv) * V + r] = o;
    }
    __syncthreads();  // q_s/k_s reused next token
  }
  if (r < V) {
    float* s_row = ssm_state + ((slot * HV + hv) * V + r) * K;
#pragma unroll 4
    for (int j = 0; j < K; ++j) s_row[j] = s_reg[j];
  }
}

// ---------------------------------------------------------------------------------------------
// CHUNKED prefill (fast path) — same result as gdn_prefill, but processes the sequence in chunks of
// GDN_CHUNK tokens with intra-chunk parallelism instead of a per-token recurrence. Validated against
// gdn_prefill (the recurrent oracle). Derivation: within a chunk (carried state S0), the gated delta
// rule's written values u_j satisfy a lower-triangular system
//     u_j + beta_j * sum_{i<j} (gamma_j/gamma_i)(k_i.k_j) u_i = beta_j (v_j - gamma_j S0 k_j)
// (gamma_j = cumulative decay = prod_{i<=j} exp(g_i)); solve by forward substitution, then
//     o_t  = gamma_t (S0 q_t) + sum_{j<=t} (gamma_t/gamma_j)(k_j.q_t) u_j
//     S_C  = gamma_C S0 + sum_j (gamma_C/gamma_j) u_j k_j^T.
// Crucially each v-ROW is independent (the k.k / k.q couplings are shared across rows), so we keep
// the one-thread-per-v-row mapping: thread r owns S row (registers) + its U column (registers); the
// per-chunk shared matrices (l2-normed k,q; KK=k_i.k_j; KQ=k_j.q_t; gamma; beta) live in LDS.
#ifndef GDN_CHUNK
#define GDN_CHUNK 32  // chunk length; LDS = 2*C*K + 2*C*C floats (C=32,K=128 -> ~40 KB)
#endif

__global__ void gdn_prefill_chunked_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    const int* __restrict__ cu_seqlens, const long* __restrict__ state_indices,
    const unsigned char* __restrict__ has_initial_state,
    float* __restrict__ ssm_state, float* __restrict__ out,
    int N, int H, int HV, int K, int V, float scale, int use_l2norm) {
  const int blk = blockIdx.x, r = threadIdx.x;
  const int n = blk / HV, hv = blk % HV, hq = hv / (HV / H);
  const int bos = cu_seqlens[n], eos = cu_seqlens[n + 1];
  const long slot = state_indices[n];
  if (slot <= 0 || bos >= eos) return;

  __shared__ float ks[GDN_CHUNK * GDN_MAX_K];  // k (C x K), l2-normed
  __shared__ float qs[GDN_CHUNK * GDN_MAX_K];  // q (C x K), l2-normed * scale
  __shared__ float KK[GDN_CHUNK * GDN_CHUNK];  // k_i . k_j
  __shared__ float KQ[GDN_CHUNK * GDN_CHUNK];  // k_j . q_t  (row j, col t)
  __shared__ float gam[GDN_CHUNK];             // cumulative decay within chunk
  __shared__ float bet[GDN_CHUNK];

  const bool active = (r < V);
  float* S = ssm_state + ((slot * HV + hv) * V + r) * K;
  float s_row[GDN_MAX_K];
  if (active) {
    const bool ui = has_initial_state[n] != 0;
    for (int j = 0; j < K; ++j) s_row[j] = ui ? S[j] : 0.0f;
  }

  for (int c0 = bos; c0 < eos; c0 += GDN_CHUNK) {
    const int L = min(GDN_CHUNK, eos - c0);
    for (int idx = r; idx < L * K; idx += blockDim.x) {
      const int j = idx / K, kk = idx % K;
      ks[j * K + kk] = k[((c0 + j) * H + hq) * K + kk];
      qs[j * K + kk] = q[((c0 + j) * H + hq) * K + kk];
    }
    __syncthreads();
    for (int j = r; j < L; j += blockDim.x) {  // one token per thread: l2norm(k,q), scale q
      float nk = 0.0f, nq = 0.0f;
      for (int kk = 0; kk < K; ++kk) { nk += ks[j * K + kk] * ks[j * K + kk];
                                       nq += qs[j * K + kk] * qs[j * K + kk]; }
      const float ik = use_l2norm ? rsqrtf(nk + 1e-6f) : 1.0f;
      const float iq = (use_l2norm ? rsqrtf(nq + 1e-6f) : 1.0f) * scale;
      for (int kk = 0; kk < K; ++kk) { ks[j * K + kk] *= ik; qs[j * K + kk] *= iq; }
    }
    if (r == 0) {  // cumulative decay + write gate
      float acc = 0.0f;
      for (int j = 0; j < L; ++j) {
        const float x = a[(c0 + j) * HV + hv] + dt_bias[hv];
        acc += -__expf(A_log[hv]) * gdn_softplus(x);
        gam[j] = __expf(acc);
        bet[j] = 1.0f / (1.0f + __expf(-b[(c0 + j) * HV + hv]));
      }
    }
    __syncthreads();
    for (int idx = r; idx < L * L; idx += blockDim.x) {  // KK, KQ
      const int i = idx / L, j = idx % L;
      float dk = 0.0f, dq = 0.0f;
      for (int kk = 0; kk < K; ++kk) { dk += ks[i * K + kk] * ks[j * K + kk];
                                       dq += ks[i * K + kk] * qs[j * K + kk]; }
      KK[i * GDN_CHUNK + j] = dk;
      KQ[i * GDN_CHUNK + j] = dq;  // row i = k, col j = q
    }
    __syncthreads();

    if (active) {
      float U[GDN_CHUNK];
      for (int j = 0; j < L; ++j) {  // B[j] then forward-substitution
        float Sk = 0.0f;
        for (int kk = 0; kk < K; ++kk) Sk += s_row[kk] * ks[j * K + kk];
        float uj = bet[j] * (v[((c0 + j) * HV + hv) * V + r] - gam[j] * Sk);
        for (int i = 0; i < j; ++i)
          uj -= bet[j] * (gam[j] / gam[i]) * KK[i * GDN_CHUNK + j] * U[i];
        U[j] = uj;
      }
      for (int t = 0; t < L; ++t) {  // outputs
        float Sq = 0.0f;
        for (int kk = 0; kk < K; ++kk) Sq += s_row[kk] * qs[t * K + kk];
        float o = gam[t] * Sq;
        for (int j = 0; j <= t; ++j) o += (gam[t] / gam[j]) * KQ[j * GDN_CHUNK + t] * U[j];
        out[((c0 + t) * HV + hv) * V + r] = o;
      }
      const float gL = gam[L - 1];  // carry state to next chunk
      for (int kk = 0; kk < K; ++kk) {
        float acc = gL * s_row[kk];
        for (int j = 0; j < L; ++j) acc += (gL / gam[j]) * U[j] * ks[j * K + kk];
        s_row[kk] = acc;
      }
    }
    __syncthreads();  // ks/qs/KK/KQ reused next chunk
  }
  if (active) for (int j = 0; j < K; ++j) S[j] = s_row[j];
}

// ---------------------------------------------------------------------------------------------
// WMMA (matrix-core) chunked prefill — same math as gdn_prefill_chunked, reformulated as rocWMMA
// 16x16 fp16 GEMMs (4.8-5.9x faster than the scalar recurrent prefill on long context). Decay is
// NOT absorbed into the stored fp16 operands (that overflows fp16 when cumulative decay underflows);
// instead k/q are kept raw l2-normed (~O(1)) and decay enters only as bounded (<=1) fp32 post-scales
// in log space, so no operand ever overflows. fp16 matmul / fp32 accumulate; ported verbatim from
// minisgl gdn_hip/gdn_kernels.hip. Requires K==V==128 (Qwen3.5/3.6 GDN). Validated token-parity vs
// the recurrent oracle (gdn_prefill); dispatch is gated so the recurrent path remains the fallback.
#ifndef GDN_WC
#define GDN_WC 16  // WMMA chunk length == one 16x16 tile dim
#endif
#define GDN_WNWARP 8       // warps per block (8*32 = 256 threads)
#define GDN_WTHREADS 256

namespace gdnw {
using namespace rocwmma;
using FragAccF = fragment<accumulator, 16, 16, 16, float>;
using f16 = float16_t;

// out[16x16] (fp32 LDS, ld=ldout, at outRow/outCol) = A[*,K] @ B[*,K]^T, contract K (step 16).
// A row-major ld=lda from aRow; B[N,K] row-major ld=ldb from bCol read col_major => B^T. (NT)
__device__ __forceinline__ void nt_tile(const f16* A, int lda, int aRow, const f16* B, int ldb,
                                        int bCol, int K, float* out, int ldout, int oR, int oC) {
  FragAccF acc;
  fill_fragment(acc, 0.0f);
  for (int i = 0; i < K; i += 16) {
    fragment<matrix_a, 16, 16, 16, f16, row_major> af;
    fragment<matrix_b, 16, 16, 16, f16, col_major> bf;
    load_matrix_sync(af, A + aRow * lda + i, lda);
    load_matrix_sync(bf, B + bCol * ldb + i, ldb);
    mma_sync(acc, af, bf, acc);
  }
  store_matrix_sync(out + oR * ldout + oC, acc, ldout, mem_row_major);
}

// out[16x16] (fp32 LDS) = A[*,K] @ B[K,*], contract K (step 16). A,B both row-major. (NN)
__device__ __forceinline__ void nn_tile(const f16* A, int lda, int aRow, const f16* B, int ldb,
                                        int bCol, int K, float* out, int ldout, int oR, int oC) {
  FragAccF acc;
  fill_fragment(acc, 0.0f);
  for (int i = 0; i < K; i += 16) {
    fragment<matrix_a, 16, 16, 16, f16, row_major> af;
    fragment<matrix_b, 16, 16, 16, f16, row_major> bf;
    load_matrix_sync(af, A + aRow * lda + i, lda);
    load_matrix_sync(bf, B + i * ldb + bCol, ldb);
    mma_sync(acc, af, bf, acc);
  }
  store_matrix_sync(out + oR * ldout + oC, acc, ldout, mem_row_major);
}

// out[16x16] (fp32 LDS, ld=16) = A[K,M]^T @ B[K,N] tile at (aRowM over M, bColN over N), contract
// K (step 16). A row-major ld=lda(=M) read col_major => A^T; B row-major ld=ldb(=N). (TN)
__device__ __forceinline__ void tn_tile(const f16* A, int lda, int aRowM, const f16* B, int ldb,
                                        int bColN, int K, float* out) {
  FragAccF acc;
  fill_fragment(acc, 0.0f);
  for (int i = 0; i < K; i += 16) {
    fragment<matrix_a, 16, 16, 16, f16, col_major> af;
    fragment<matrix_b, 16, 16, 16, f16, row_major> bf;
    load_matrix_sync(af, A + i * lda + aRowM, lda);
    load_matrix_sync(bf, B + i * ldb + bColN, ldb);
    mma_sync(acc, af, bf, acc);
  }
  store_matrix_sync(out, acc, 16, mem_row_major);
}
}  // namespace gdnw

template <typename scalar_t, typename state_t>
__global__ void gdn_prefill_wmma_kernel(
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k, const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ a, const scalar_t* __restrict__ b,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    const int* __restrict__ cu_seqlens, const long* __restrict__ state_indices,
    const unsigned char* __restrict__ has_initial_state,
    state_t* __restrict__ ssm_state, scalar_t* __restrict__ out,
    int N, int H, int HV, int K, int V, float scale, int use_l2norm) {
  using gdnw::f16;
  const int blk = blockIdx.x, tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int n = blk / HV, hv = blk % HV, hq = hv / (HV / H);
  const int bos = cu_seqlens[n], eos = cu_seqlens[n + 1];
  const long slot = state_indices[n];
  if (slot <= 0 || bos >= eos) return;
  const int Dk = K, Dv = V;             // both 128 for Qwen3.5/3.6
  const int NTv = Dv / 16;              // v-dim tiles (8)

  // ---- LDS arena (manual reuse; peak ~62 KB). Each Rn is 8 KB = [16 x 128] fp32. ----
  // Decay is NOT absorbed into the stored operands (that needs k/gamma, which overflows fp16 when the
  // cumulative decay gamma underflows). Instead kk/qq hold the RAW l2-normed k, scaled q (bounded
  // ~O(1)); decay enters only as bounded (<=1) fp32 post-scales computed in log space from clog[]
  // (clog[j] = cumulative log-decay <= 0; gamma_j = exp(clog[j])), so no operand ever overflows.
  __shared__ f16 Sh[GDN_MAX_K * GDN_MAX_K];     // 32 KB  state S [Dv x Dk] row-major (n-major), fp16
  __shared__ f16 kk[GDN_WC * GDN_MAX_K];        //  4 KB  k l2-normed (raw)        (live thru carry)
  __shared__ float R1[GDN_WC * GDN_MAX_K];      //  8 KB  qq=q*scale (f16, phase A) | U16(f16, ph C,D)
  __shared__ float R2[GDN_WC * GDN_MAX_K];      //  8 KB  KS0->B (f32) | then O-accum (f32)
  __shared__ float R3[GDN_WC * GDN_MAX_K];      //  8 KB  QS0 (f32, thru phase C) | then carry scratch
  __shared__ float Mf[GDN_WC * GDN_WC];         //  1 KB  G=k.k (f32) -> L (f32) -> Atril(f16) overlay
  __shared__ float Af[GDN_WC * GDN_WC];         //  1 KB  Aqk=q.k (f32, thru phase C)
  __shared__ float clog[GDN_WC], bet[GDN_WC], invk[GDN_WC], invq[GDN_WC];
  __shared__ float gamC_s;
  f16* qq = reinterpret_cast<f16*>(R1);    // [16x128] f16 during phase A (q l2-normed * scale)
  f16* U16 = reinterpret_cast<f16*>(R1);   // R1 free after phase A -> holds U as fp16 (phases C,D)

  // ---- load S from slot (or zero) ----
  const bool use_init = has_initial_state[n] != 0;
  {
    const state_t* Sg = ssm_state + (slot * HV + hv) * (long)Dv * Dk;
    for (int idx = tid; idx < Dv * Dk; idx += GDN_WTHREADS)
      Sh[idx] = use_init ? (f16)(float)Sg[idx] : (f16)0.0f;
  }
  __syncthreads();

  const float Alog = A_log[hv], dtb = dt_bias[hv];
  for (int c0 = bos; c0 < eos; c0 += GDN_WC) {
    const int L = min(GDN_WC, eos - c0);
    // raw k,q for the chunk -> R2(=rawk), R3(=rawq) as fp32; rows >= L zeroed
    float* rawk = R2;
    float* rawq = R3;
    for (int idx = tid; idx < GDN_WC * Dk; idx += GDN_WTHREADS) {
      const int j = idx / Dk, kk = idx % Dk;
      if (j < L) {
        rawk[idx] = (float)k[((c0 + j) * H + hq) * Dk + kk];
        rawq[idx] = (float)q[((c0 + j) * H + hq) * Dk + kk];
      } else { rawk[idx] = 0.0f; rawq[idx] = 0.0f; }
    }
    __syncthreads();
    // per-row l2norm (thread j<C) into invk/invq (invq folds the q scale); cumulative decay/gate (t0)
    if (tid < GDN_WC) {
      const int j = tid;
      if (j < L) {
        float nk = 0.0f, nq = 0.0f;
        for (int kk = 0; kk < Dk; ++kk) { nk += rawk[j * Dk + kk] * rawk[j * Dk + kk];
                                          nq += rawq[j * Dk + kk] * rawq[j * Dk + kk]; }
        invk[j] = use_l2norm ? rsqrtf(nk + 1e-6f) : 1.0f;
        invq[j] = (use_l2norm ? rsqrtf(nq + 1e-6f) : 1.0f) * scale;
      } else { invk[j] = 0.0f; invq[j] = 0.0f; }
    }
    if (tid == 0) {
      float acc = 0.0f;  // cumulative log-decay within the chunk (<= 0, monotone decreasing)
      for (int j = 0; j < L; ++j) {
        const float x = (float)a[(c0 + j) * HV + hv] + dtb;
        acc += -__expf(Alog) * gdn_softplus(x);
        clog[j] = acc;
        bet[j] = 1.0f / (1.0f + __expf(-(float)b[(c0 + j) * HV + hv]));
      }
      for (int j = L; j < GDN_WC; ++j) { clog[j] = (L > 0 ? acc : 0.0f); bet[j] = 0.0f; }
      gamC_s = __expf(L > 0 ? clog[L - 1] : 0.0f);  // gamma_C = exp(cumulative log-decay over chunk)
    }
    __syncthreads();
    // build RAW fp16 operands: kk = l2norm(k), qq = l2norm(q) * scale (NO decay absorption — decay is
    // applied later as bounded fp32 scalings, so these stay O(1) and never overflow fp16).
    for (int idx = tid; idx < GDN_WC * Dk; idx += GDN_WTHREADS) {
      const int j = idx / Dk;
      kk[idx] = (f16)(rawk[idx] * invk[j]);
      qq[idx] = (f16)(rawq[idx] * invq[j]);   // invq already folds the q scale
    }
    __syncthreads();

    // ---- phase A (decay-free matmuls): G=k.k, Aqk=q.k (single tile each); KS0=k.S0^T, QS0=q.S0^T
    // (8 tiles each). rawk/rawq (R2/R3) freed. ----
    {
      const int w = warp;
      gdnw::nt_tile(kk, Dk, 0, Sh, Dk, w * 16, Dk, R2, Dv, 0, w * 16);   // KS0 tile w -> R2
      gdnw::nt_tile(qq, Dk, 0, Sh, Dk, w * 16, Dk, R3, Dv, 0, w * 16);   // QS0 tile w -> R3
      if (w == 0) gdnw::nt_tile(kk, Dk, 0, kk, Dk, 0, Dk, Mf, GDN_WC, 0, 0);   // G  = k.k  -> Mf
      if (w == 1) gdnw::nt_tile(qq, Dk, 0, kk, Dk, 0, Dk, Af, GDN_WC, 0, 0);   // Aqk = q.k  -> Af
    }
    __syncthreads();
    float* KS0 = R2;   // [16x128]
    float* QS0 = R3;   // [16x128]

    // ---- phase B: B = beta*(V - gamma_j*KS0) in-place into R2; L = tril(beta*(gamma_j/gamma_i)*G,-1)
    // into Mf; solve U. Decay ratios via exp(clog diff) (<=1, log-space => stable). ----
    for (int idx = tid; idx < GDN_WC * Dv; idx += GDN_WTHREADS) {
      const int j = idx / Dv, nn = idx % Dv;
      if (j < L) KS0[idx] = bet[j] * ((float)v[((c0 + j) * HV + hv) * Dv + nn] - __expf(clog[j]) * KS0[idx]);
      else KS0[idx] = 0.0f;
    }
    for (int idx = tid; idx < GDN_WC * GDN_WC; idx += GDN_WTHREADS) {
      const int j = idx / GDN_WC, i = idx % GDN_WC;       // L[j][i] = beta_j (gamma_j/gamma_i) G[j][i]
      Mf[idx] = (i < j) ? bet[j] * __expf(clog[j] - clog[i]) * Mf[idx] : 0.0f;
    }
    __syncthreads();
    // forward-substitution solve: thread nn<Dv owns column nn of U; result -> U16 (fp16, in R1).
    // U is consumed only by the two fp16 matmuls below (O and the carry), so fp16 storage suffices.
    if (tid < Dv) {
      const int nn = tid;
      float u[GDN_WC];
      for (int j = 0; j < GDN_WC; ++j) {
        float s = KS0[j * Dv + nn];             // B[j][nn]
        for (int i = 0; i < j; ++i) s -= Mf[j * GDN_WC + i] * u[i];
        u[j] = s;
        U16[j * Dv + nn] = (f16)s;
      }
    }
    __syncthreads();

    // ---- phase C: Atril[t][j] = (j<=t)(gamma_t/gamma_j) Aqk[t][j] (fp16, in Mf); O = gamma_t*QS0 +
    // tril(Atil)@U -> out ----
    f16* Atril = reinterpret_cast<f16*>(Mf);   // 16x16 f16 (512 B) fits in Mf (1 KB)
    for (int idx = tid; idx < GDN_WC * GDN_WC; idx += GDN_WTHREADS) {
      const int t = idx / GDN_WC, j = idx % GDN_WC;
      Atril[idx] = (j <= t) ? (f16)(__expf(clog[t] - clog[j]) * Af[idx]) : (f16)0.0f;
    }
    __syncthreads();
    float* Ofull = R2;   // KS0/B no longer needed -> reuse R2 for O accumulation
    gdnw::nn_tile(Atril, GDN_WC, 0, U16, Dv, warp * 16, GDN_WC, Ofull, Dv, 0, warp * 16);
    __syncthreads();
    for (int idx = tid; idx < L * Dv; idx += GDN_WTHREADS) {
      const int j = idx / Dv, nn = idx % Dv;
      out[((c0 + j) * HV + hv) * Dv + nn] = (scalar_t)(__expf(clog[j]) * QS0[idx] + Ofull[idx]);
    }
    __syncthreads();

    // ---- phase D carry: S' = gamma_C*S0 + U'^T @ k, with U'[j] = (gamma_C/gamma_j) U[j] folded into
    // U16 (bounded <=U, stays fp16-safe). 64 tiles over (Dv x Dk), 8 per warp. ----
    const float gamC = gamC_s;
    const float clogC = clog[L > 0 ? L - 1 : 0];
    for (int idx = tid; idx < GDN_WC * Dv; idx += GDN_WTHREADS) {   // U16[j] *= gamma_C/gamma_j
      const int j = idx / Dv;
      U16[idx] = (f16)(__expf(clogC - clog[j]) * (float)U16[idx]);
    }
    __syncthreads();
    float* scr = R3 + warp * 256;   // per-warp 16x16 fp32 scratch (R3 = 8 KB = 8*256*4); QS0 dead
    for (int ti = warp; ti < NTv * (Dk / 16); ti += GDN_WNWARP) {
      const int rt = ti / (Dk / 16), ct = ti % (Dk / 16);   // rt over Dv, ct over Dk
      gdnw::tn_tile(U16, Dv, rt * 16, kk, Dk, ct * 16, GDN_WC, scr);
      for (int e = lane; e < 256; e += 32) {
        const int rr = e >> 4, cc = e & 15;
        const int nn = rt * 16 + rr, kc = ct * 16 + cc;
        Sh[nn * Dk + kc] = (f16)(gamC * (float)Sh[nn * Dk + kc] + scr[e]);
      }
    }
    __syncthreads();
  }

  // ---- persist final state to slot ----
  {
    state_t* Sg = ssm_state + (slot * HV + hv) * (long)Dv * Dk;
    for (int idx = tid; idx < Dv * Dk; idx += GDN_WTHREADS) Sg[idx] = (state_t)(float)Sh[idx];
  }
}

// ---------------------------------------------------------------------------------------------
// Depthwise causal conv1d, DECODE update (one token/seq). conv_state holds the last (W-1) inputs
// per channel; we append the new token, dot with the kernel, optional SiLU, and roll the state.
//   x:       [B, C]   weight: [C, W]   bias: [C] or null   conv_state: [num_slots, C, W-1]
//   state_indices: [B]   out: [B, C]   (silu applied iff activation!=0)
// grid = (B), block = min(C, 1024); each thread strides over channels.
__global__ void causal_conv1d_update_kernel(
    const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ conv_state, const long* __restrict__ state_indices, float* __restrict__ out,
    int B, int C, int W, int activation) {
  const int bi = blockIdx.x;
  const long slot = state_indices[bi];
  const int Wm1 = W - 1;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float acc = bias ? bias[c] : 0.0f;
    const float* wc = weight + c * W;
    float* st = (slot > 0) ? conv_state + (slot * C + c) * Wm1 : nullptr;
    // window = [state(W-1) , x_new]; conv = sum_{i<W} window[i]*w[i]
    float xnew = x[bi * C + c];
    if (st) {
      for (int i = 0; i < Wm1; ++i) acc += st[i] * wc[i];
    }
    acc += xnew * wc[Wm1];
    // roll state left, append new token at tail
    if (st) {
      for (int i = 0; i < Wm1 - 1; ++i) st[i] = st[i + 1];
      st[Wm1 - 1] = xnew;
    }
    if (activation) acc = acc / (1.0f + __expf(-acc));  // SiLU
    out[bi * C + c] = acc;
  }
}

// Depthwise causal conv1d, PREFILL (varlen). For each sequence, run the causal conv over its tokens
// using the optional initial conv_state as left padding, write outputs, and store the trailing
// (W-1) inputs back to the slot for the subsequent decode.
//   x:    [T, C]  (token-major)   weight: [C, W]   bias: [C] or null
//   cu_seqlens: [N+1]   state_indices: [N]   has_initial_state: [N]   conv_state: [num_slots,C,W-1]
//   out:  [T, C]
// grid = (N, ceil(C/CT)); block = CT threads over channels. Simple per-channel sliding window.
__global__ void causal_conv1d_fwd_kernel(
    const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias,
    const int* __restrict__ cu_seqlens, const long* __restrict__ state_indices,
    const unsigned char* __restrict__ has_initial_state, float* __restrict__ conv_state,
    float* __restrict__ out, int N, int C, int W, int activation) {
  const int n = blockIdx.x;
  const int bos = cu_seqlens[n], eos = cu_seqlens[n + 1];
  const long slot = state_indices[n];
  const int Wm1 = W - 1;
  const bool use_init = has_initial_state && has_initial_state[n] != 0;
  for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < C; c += gridDim.y * blockDim.x) {
    const float* wc = weight + c * W;
    const float bc = bias ? bias[c] : 0.0f;
    float* st = (slot > 0) ? conv_state + (slot * C + c) * Wm1 : nullptr;
    // local history ring of last W-1 inputs, seeded from conv_state (or zero)
    float hist[16];  // W-1 <= 15 (GDN W=4); generous bound
    for (int i = 0; i < Wm1; ++i) hist[i] = (use_init && st) ? st[i] : 0.0f;
    for (int t = bos; t < eos; ++t) {
      float xnew = x[t * C + c];
      float acc = bc;
      for (int i = 0; i < Wm1; ++i) acc += hist[i] * wc[i];
      acc += xnew * wc[Wm1];
      for (int i = 0; i < Wm1 - 1; ++i) hist[i] = hist[i + 1];
      hist[Wm1 - 1] = xnew;
      if (activation) acc = acc / (1.0f + __expf(-acc));
      out[t * C + c] = acc;
    }
    if (st) for (int i = 0; i < Wm1; ++i) st[i] = hist[i];  // persist trailing window
  }
}

// ---------------------------------------------------------------------------------------------
// RMSNormGated: y = (x / rms(x)) * weight * silu(z), normalized over the last dim D (norm-before-gate
// = the Qwen3.5 GDN convention). x,z,out: [M, D]; weight: [D]. grid=(M), block=min(D,1024).
__global__ void rmsnorm_gated_kernel(
    const float* __restrict__ x, const float* __restrict__ z, const float* __restrict__ weight,
    float* __restrict__ out, int M, int D, float eps) {
  const int m = blockIdx.x;
  extern __shared__ float red[];
  float partial = 0.0f;
  for (int j = threadIdx.x; j < D; j += blockDim.x) { float xv = x[m * D + j]; partial += xv * xv; }
  red[threadIdx.x] = partial;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = rsqrtf(red[0] / D + eps);
  for (int j = threadIdx.x; j < D; j += blockDim.x) {
    const float zv = z[m * D + j];
    const float gate = zv / (1.0f + __expf(-zv));  // SiLU(z)
    out[m * D + j] = x[m * D + j] * inv * weight[j] * gate;
  }
}

// ============================== launchers (called from bindings.cpp) ==========================
static inline float* FP(const at::Tensor& t) { return t.data_ptr<float>(); }
static inline const float* CFP(const at::Tensor& t) { return t.data_ptr<float>(); }

void launch_gdn_decode(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                       const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
                       const at::Tensor& dt_bias, at::Tensor& ssm_state,
                       const at::Tensor& state_indices, at::Tensor& out,
                       double scale, int64_t use_l2norm) {
  const int B = q.size(0), H = q.size(1), K = q.size(2);
  const int HV = v.size(1), V = v.size(2);
  dim3 grid(B * HV), block(V);
  size_t shmem = (2 * K + V) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(gdn_decode_kernel, grid, block, shmem, stream,
      CFP(q), CFP(k), CFP(v), CFP(a), CFP(b), CFP(A_log), CFP(dt_bias),
      FP(ssm_state), state_indices.data_ptr<long>(), FP(out),
      B, H, HV, K, V, (float)scale, (int)use_l2norm);
}

void launch_gdn_prefill(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                        const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
                        const at::Tensor& dt_bias, const at::Tensor& cu_seqlens,
                        const at::Tensor& state_indices, const at::Tensor& has_initial_state,
                        at::Tensor& ssm_state, at::Tensor& out, double scale, int64_t use_l2norm) {
  const int H = q.size(1), K = q.size(2);
  const int HV = v.size(1), V = v.size(2);
  const int N = state_indices.size(0);
  dim3 grid(N * HV), block(V);
  size_t shmem = (2 * K + V) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(gdn_prefill_kernel, grid, block, shmem, stream,
      CFP(q), CFP(k), CFP(v), CFP(a), CFP(b), CFP(A_log), CFP(dt_bias),
      cu_seqlens.data_ptr<int>(), state_indices.data_ptr<long>(),
      has_initial_state.data_ptr<unsigned char>(), FP(ssm_state), FP(out),
      N, H, HV, K, V, (float)scale, (int)use_l2norm);
}

void launch_gdn_prefill_chunked(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
                                const at::Tensor& dt_bias, const at::Tensor& cu_seqlens,
                                const at::Tensor& state_indices, const at::Tensor& has_initial_state,
                                at::Tensor& ssm_state, at::Tensor& out, double scale,
                                int64_t use_l2norm) {
  const int H = q.size(1), K = q.size(2);
  const int HV = v.size(1), V = v.size(2);
  const int N = state_indices.size(0);
  dim3 grid(N * HV), block(V);
  auto stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(gdn_prefill_chunked_kernel, grid, block, 0, stream,  // static shared mem
      CFP(q), CFP(k), CFP(v), CFP(a), CFP(b), CFP(A_log), CFP(dt_bias),
      cu_seqlens.data_ptr<int>(), state_indices.data_ptr<long>(),
      has_initial_state.data_ptr<unsigned char>(), FP(ssm_state), FP(out),
      N, H, HV, K, V, (float)scale, (int)use_l2norm);
}

// WMMA chunked prefill — fp32 boundary like the rest of gdn_hip (op.py casts q/k/v/a/b/state to
// float), so the templated kernel is instantiated <float,float>; it converts fp32->fp16 internally
// for the matmuls. 256 threads (8 wave32 warps), static shared mem (~62 KB). Requires K==V==128.
void launch_gdn_prefill_wmma(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                             const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
                             const at::Tensor& dt_bias, const at::Tensor& cu_seqlens,
                             const at::Tensor& state_indices, const at::Tensor& has_initial_state,
                             at::Tensor& ssm_state, at::Tensor& out, double scale,
                             int64_t use_l2norm) {
  const int H = q.size(1), K = q.size(2);
  const int HV = v.size(1), V = v.size(2);
  const int N = state_indices.size(0);
  dim3 grid(N * HV), block(GDN_WTHREADS);  // 256 threads = 8 wave32 warps
  auto stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL((gdn_prefill_wmma_kernel<float, float>), grid, block, 0, stream,
      CFP(q), CFP(k), CFP(v), CFP(a), CFP(b), CFP(A_log), CFP(dt_bias),
      cu_seqlens.data_ptr<int>(), state_indices.data_ptr<long>(),
      has_initial_state.data_ptr<unsigned char>(), FP(ssm_state), FP(out),
      N, H, HV, K, V, (float)scale, (int)use_l2norm);
}

void launch_causal_conv1d_update(const at::Tensor& x, const at::Tensor& weight,
                                 const c10::optional<at::Tensor>& bias, at::Tensor& conv_state,
                                 const at::Tensor& state_indices, at::Tensor& out,
                                 int64_t activation) {
  const int B = x.size(0), C = x.size(1), W = weight.size(1);
  dim3 grid(B), block(C < 1024 ? C : 1024);
  auto stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(causal_conv1d_update_kernel, grid, block, 0, stream,
      CFP(x), CFP(weight), bias.has_value() ? CFP(*bias) : nullptr,
      FP(conv_state), state_indices.data_ptr<long>(), FP(out), B, C, W, (int)activation);
}

void launch_causal_conv1d_fwd(const at::Tensor& x, const at::Tensor& weight,
                              const c10::optional<at::Tensor>& bias, const at::Tensor& cu_seqlens,
                              const at::Tensor& state_indices, const at::Tensor& has_initial_state,
                              at::Tensor& conv_state, at::Tensor& out, int64_t activation) {
  const int C = x.size(1), W = weight.size(1);
  const int N = state_indices.size(0);
  const int CT = C < 256 ? C : 256;
  dim3 grid(N, (C + CT - 1) / CT), block(CT);
  auto stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(causal_conv1d_fwd_kernel, grid, block, 0, stream,
      CFP(x), CFP(weight), bias.has_value() ? CFP(*bias) : nullptr,
      cu_seqlens.data_ptr<int>(), state_indices.data_ptr<long>(),
      has_initial_state.data_ptr<unsigned char>(), FP(conv_state), FP(out), N, C, W, (int)activation);
}

void launch_rmsnorm_gated(const at::Tensor& x, const at::Tensor& z, const at::Tensor& weight,
                          at::Tensor& out, double eps) {
  const int M = x.size(0), D = x.size(1);
  dim3 grid(M), block(D < 1024 ? D : 1024);
  size_t shmem = block.x * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(rmsnorm_gated_kernel, grid, block, shmem, stream,
      CFP(x), CFP(z), CFP(weight), FP(out), M, D, (float)eps);
}
