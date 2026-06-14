// Fused CCA decode conv + state-update kernel for AMD RDNA4 (gfx1201).
//
// Replaces the eager-Python decode path in CCA.forward_cuda (the cat + two-stage
// causal conv + conv-state roll/scatter), which a rocprof showed running as a
// large share of ~1.4M tiny ATen launches because vllm::cca is a graph break.
// One thread block per decode token; fp32 throughout (the conv state is fp32).
//
// Per decode token s with state slot = slot[s], pad = is_pad[s]:
//   window[c, 0..TP-1] = pad ? 0 : conv_states[slot, c, 0..TP-1]
//   window[c, TP]      = pad ? 0 : qk_new[s, c]
//   mid[c, t]          = b0[c] + sum_{k<K0} window[c, t+k] * w0[c, k]     (t<Lmid, depthwise)
//   out[s, oc]         = b1[oc] + sum_{di<d} sum_{k<K1}                   (grouped, L_out=1)
//                          w1[oc, di, k] * mid[gi*d+di, k]   (gi=oc/d)
//   conv_states[slot, c, i] = pad ? 0 : window[c, i+1]   (i<TP; roll left, new token at tail)
//
// Shapes: C=in_out_ch=g*d, TP=total_padding, Lmid=TP-K0+2, weights fp32.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void cca_conv_state_decode_kernel(
    const float* __restrict__ qk_new,       // [S, C]
    float* __restrict__ conv_states,        // [num_blocks, C, TP]
    const long* __restrict__ slot,          // [S]
    const unsigned char* __restrict__ is_pad, // [S]
    const float* __restrict__ w0,           // [C, K0]
    const float* __restrict__ b0,           // [C]
    const float* __restrict__ w1,           // [C, d, K1]
    const float* __restrict__ b1,           // [C]
    float* __restrict__ qk_out,             // [S, C]
    int C, int TP, int K0, int K1, int d, int Lmid,
    long s0, long s1, long s2) {            // conv_states strides (elements)

  const int s = blockIdx.x;
  const long slot_s = slot[s];
  const bool pad = is_pad[s] != 0;
  const int win_cols = TP + 1;

  extern __shared__ float smem[];
  float* win = smem;                 // [C * (TP+1)]
  float* mid = smem + C * win_cols;  // [C * Lmid]

  const long state_base = slot_s * s0;

  // 1. Build the window (cached state || new token), pad-masked.
  for (int idx = threadIdx.x; idx < C * win_cols; idx += blockDim.x) {
    const int c = idx / win_cols;
    const int col = idx % win_cols;
    float v;
    if (pad) {
      v = 0.0f;
    } else if (col < TP) {
      v = conv_states[state_base + (long)c * s1 + (long)col * s2];
    } else {
      v = qk_new[(long)s * C + c];
    }
    win[idx] = v;
  }
  __syncthreads();

  // 2. Depthwise conv over the sequence window.
  for (int idx = threadIdx.x; idx < C * Lmid; idx += blockDim.x) {
    const int c = idx / Lmid;
    const int t = idx % Lmid;
    float acc = b0[c];
    const float* wc = w0 + (long)c * K0;
    const float* wn = win + (long)c * win_cols + t;
    for (int k = 0; k < K0; ++k) acc += wn[k] * wc[k];
    mid[idx] = acc;
  }
  __syncthreads();

  // 3. Grouped conv (output channels), L_out == 1.
  for (int oc = threadIdx.x; oc < C; oc += blockDim.x) {
    const int gi = oc / d;
    float acc = b1[oc];
    const float* w1c = w1 + (long)oc * d * K1;   // [d, K1]
    const float* midg = mid + (long)gi * d * Lmid;  // group base in mid
    for (int di = 0; di < d; ++di) {
      const float* mrow = midg + (long)di * Lmid;  // mid[gi*d+di, :]
      const float* wrow = w1c + (long)di * K1;
      for (int k = 0; k < K1; ++k) acc += wrow[k] * mrow[k];
    }
    qk_out[(long)s * C + oc] = acc;
  }

  // 4. Roll the conv-state window left and write the new token at the tail.
  for (int idx = threadIdx.x; idx < C * TP; idx += blockDim.x) {
    const int c = idx / TP;
    const int i = idx % TP;
    conv_states[state_base + (long)c * s1 + (long)i * s2] =
        pad ? 0.0f : win[(long)c * win_cols + (i + 1)];
  }
}

// Full CCA decode fusion: conv + grouped-means + per-head RMS-norm + qk write,
// plus the conv-state roll. One block per (head, token). Replaces the bulk of
// the eager CCA decode pointwise ops (the 1.4M-launch problem). The four Linear
// GEMMs (linear_q/k, val_proj1/2) and the prev_hs gather/scatter stay separate.
//
// Channel/head layout of qk_new [S, C=latent_q+latent_k]:
//   q-heads 0..num_q-1 at [h*d : h*d+d]; k-heads at [latent_q + ki*d : ...].
//   conv groups == heads, so head h owns w0/w1 rows [h*d : h*d+d].
// Per head h, token t (block), thread j = output channel in [0, d):
//   conv_out = two-stage causal conv over this head's window  (== cca_kernel)
//   q-head:  val = conv_out + 0.5*query_pre[h,j] + 0.5*key_base[gi,j]   (gi=h/gqa)
//   k-head:  val = conv_out + 0.5*mean_gg(query_pre[ki*gqa+gg, j]) + 0.5*key_base[ki,j]
//   val = val * rsqrt(sum_j val^2 + eps) * sqrt_d   (per-head RMS-norm)
//   k-head additionally *= temp_eff[ki]
//   qk_out[t, out_col(h)+j] = val ;  conv_states[slot,c,i] roll-updated
__global__ void cca_decode_qk_kernel(
    const float* __restrict__ qk_new,        // [S, C]
    float* __restrict__ conv_states,         // [NB, C, TP]
    const long* __restrict__ slot,           // [S]
    const unsigned char* __restrict__ is_pad,  // [S]
    const float* __restrict__ w0,            // [C, K0]
    const float* __restrict__ b0,            // [C]
    const float* __restrict__ w1,            // [H, d_in, d_out, K1] (transposed)
    const float* __restrict__ b1,            // [C]
    const float* __restrict__ temp_eff,      // [num_k]
    float* __restrict__ qk_out,              // [S, C]
    int C, int TP, int K0, int K1, int d, int Lmid,
    int num_q, int gqa, int latent_q, float sqrt_d, float eps,
    long s0, long s1, long s2) {

  const int h = blockIdx.x;        // head
  const int t = blockIdx.y;        // token
  const int j = threadIdx.x;       // output channel in head, 0..d-1 (blockDim==d)
  const bool is_q = h < num_q;
  const int head_base = h * d;
  const long slot_t = slot[t];
  const bool pad = is_pad[t] != 0;
  const int win_cols = TP + 1;

  extern __shared__ float sm[];
  float* win = sm;                  // [d * (TP+1)]
  float* mid = win + d * win_cols;  // [d * Lmid]
  float* red = mid + d * Lmid;      // [d]

  // 1+2. Each thread owns channel c = head_base + j (blockDim == d): load its
  //      window into smem and run the depthwise conv. No cross-thread reads,
  //      so no sync here — only before the grouped conv (which reads every
  //      channel's mid).
  const int c = head_base + j;
  for (int col = 0; col < win_cols; ++col) {
    float v;
    if (pad) v = 0.0f;
    else if (col < TP) v = conv_states[slot_t * s0 + (long)c * s1 + (long)col * s2];
    else v = qk_new[(long)t * C + c];
    win[j * win_cols + col] = v;
  }
  // depthwise conv: causal weights cached in registers, sliding window
  // unrolled (K0==2 fast path).
  const float bc = b0[c];
  if (K0 == 2) {
    const float w0a = w0[(long)c * 2 + 0];
    const float w0b = w0[(long)c * 2 + 1];
#pragma unroll
    for (int tm = 0; tm < Lmid; ++tm)
      mid[j * Lmid + tm] = bc + win[j * win_cols + tm] * w0a +
                           win[j * win_cols + tm + 1] * w0b;
  } else {
    const float* wc = w0 + (long)c * K0;
    for (int tm = 0; tm < Lmid; ++tm) {
      float acc = bc;
      for (int k = 0; k < K0; ++k) acc += win[j * win_cols + tm + k] * wc[k];
      mid[j * Lmid + tm] = acc;
    }
  }
  __syncthreads();

  // 3. grouped conv (this head): thread j -> output channel c.
  //    w1 is pre-transposed by the caller to [H, d_in, d_out, K1] (the original
  //    [C=H*d_out, d_in, K1] layout strides w1 by d*K1 floats ACROSS threads in
  //    a head, so every di-iteration touches 128 distinct cache lines — fully
  //    uncoalesced and measured ~9x slower). With the transpose, thread j reads
  //    w1t[h, di, j, k] @ ((h*d+di)*d + j)*K1 + k, i.e. consecutive threads hit
  //    consecutive addresses -> one coalesced float2 transaction per di.
  //    K1==Lmid==2 fast path also vectorizes the two taps as float2.
  float val = b1[c];
  if (K1 == 2 && Lmid == 2) {
    const float2* w1t = reinterpret_cast<const float2*>(w1);
    const float2* m2 = reinterpret_cast<const float2*>(mid);
    const long wbase = (long)h * d * d + j;   // float2 units; + di*d below
    // unroll 16: keep ~16 coalesced float2 loads in flight to hide L2 latency
    // (VGPR 24->64, no spills). Measured 25->8us; 16 beats 4/8/full. Only pays
    // off once the loads are coalesced (above) — uncoalesced loads can't pipeline.
#pragma unroll 16
    for (int di = 0; di < d; ++di) {
      float2 wv = w1t[wbase + (long)di * d];
      float2 mv = m2[di];
      val += wv.x * mv.x + wv.y * mv.y;
    }
  } else {
    for (int di = 0; di < d; ++di) {
      const float* wr = w1 + (((long)h * d + di) * d + j) * K1;
      const float* mr = mid + (long)di * Lmid;
      for (int k = 0; k < K1; ++k) val += wr[k] * mr[k];
    }
  }

  // 4. grouped means
  if (is_q) {
    const int gi = h / gqa;
    val += 0.5f * qk_new[(long)t * C + head_base + j];
    val += 0.5f * qk_new[(long)t * C + latent_q + gi * d + j];
  } else {
    const int ki = h - num_q;
    float m = 0.0f;
    for (int gg = 0; gg < gqa; ++gg)
      m += qk_new[(long)t * C + (ki * gqa + gg) * d + j];
    val += 0.5f * (m / (float)gqa);
    val += 0.5f * qk_new[(long)t * C + latent_q + ki * d + j];
  }

  // 5. per-head RMS-norm: wave-level reduction in registers via __shfl_down
  //    (Wave32), then a single cross-wave combine. A 128-channel head spans 4
  //    wave32 waves and AMD shuffles are intra-wave only, so exactly one LDS
  //    combine is unavoidable — there is no pure-shuffle reduction across 4
  //    waves. This is the kernel's only barrier (down from 9).
  float sq = val * val;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sq += __shfl_down(sq, o, 32);  // sum in lane 0
  const int nwaves = blockDim.x >> 5;
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sq;
  __syncthreads();
  float ssq = 0.0f;
#pragma unroll
  for (int w = 0; w < nwaves; ++w) ssq += red[w];
  val *= rsqrtf(ssq + eps) * sqrt_d;
  if (!is_q) val *= temp_eff[h - num_q];

  // 6. write qk_out (q region [0,latent_q) | k region [latent_q,C))
  const int out_col = is_q ? head_base : (latent_q + (h - num_q) * d);
  qk_out[(long)t * C + out_col + j] = val;

  // 7. roll the conv-state window: thread j writes its own channel's TP taps
  for (int i = 0; i < TP; ++i)
    conv_states[slot_t * s0 + (long)c * s1 + (long)i * s2] =
        pad ? 0.0f : win[j * win_cols + (i + 1)];
}

void launch_cca_decode_qk(
    const at::Tensor& qk_new,
    at::Tensor& conv_states,
    const at::Tensor& slot,
    const at::Tensor& is_pad,
    const at::Tensor& w0,
    const at::Tensor& b0,
    const at::Tensor& w1,
    const at::Tensor& b1,
    const at::Tensor& temp_eff,
    at::Tensor& qk_out,
    int64_t num_q,
    int64_t gqa,
    int64_t latent_q,
    double sqrt_d) {
  const int S = qk_new.size(0);
  const int C = qk_new.size(1);
  const int TP = conv_states.size(2);
  const int K0 = w0.size(1);
  // w1 pre-transposed to [H, d_in, d_out, K1]; d_in == d_out == d.
  const int d = w1.size(2);
  const int K1 = w1.size(3);
  const int Lmid = TP - K0 + 2;
  const int H = num_q + temp_eff.size(0);

  const size_t smem = (size_t)d * (TP + 1 + Lmid + 1) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(
      cca_decode_qk_kernel, dim3(H, S), dim3(d), smem, stream,
      qk_new.data_ptr<float>(), conv_states.data_ptr<float>(),
      slot.data_ptr<long>(),
      reinterpret_cast<const unsigned char*>(is_pad.data_ptr<bool>()),
      w0.data_ptr<float>(), b0.data_ptr<float>(),
      w1.data_ptr<float>(), b1.data_ptr<float>(),
      temp_eff.data_ptr<float>(), qk_out.data_ptr<float>(),
      C, TP, K0, K1, d, Lmid, (int)num_q, (int)gqa, (int)latent_q,
      (float)sqrt_d, 1e-12f,
      conv_states.stride(0), conv_states.stride(1), conv_states.stride(2));
}

void launch_cca_conv_state_decode(
    const at::Tensor& qk_new,
    at::Tensor& conv_states,
    const at::Tensor& slot,
    const at::Tensor& is_pad,
    const at::Tensor& w0,
    const at::Tensor& b0,
    const at::Tensor& w1,
    const at::Tensor& b1,
    at::Tensor& qk_out) {

  const int S = qk_new.size(0);
  const int C = qk_new.size(1);
  const int TP = conv_states.size(2);
  const int K0 = w0.size(1);
  const int d = w1.size(1);
  const int K1 = w1.size(2);
  const int Lmid = TP - K0 + 2;

  const int block = 256;
  const size_t smem = (size_t)C * (TP + 1 + Lmid) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  hipLaunchKernelGGL(
      cca_conv_state_decode_kernel, dim3(S), dim3(block), smem, stream,
      qk_new.data_ptr<float>(), conv_states.data_ptr<float>(),
      slot.data_ptr<long>(),
      reinterpret_cast<const unsigned char*>(is_pad.data_ptr<bool>()),
      w0.data_ptr<float>(), b0.data_ptr<float>(),
      w1.data_ptr<float>(), b1.data_ptr<float>(),
      qk_out.data_ptr<float>(), C, TP, K0, K1, d, Lmid,
      conv_states.stride(0), conv_states.stride(1), conv_states.stride(2));
}

// Full CCA PREFILL fusion: the prefill twin of cca_decode_qk. Replaces the eager
// flat-buffer path (scatter token/state cols -> one batched MIOpen Conv1d ->
// gather outputs -> scatter new state -> eager grouped-means + fp32 RMS-norm),
// ~25-30 ATen ops/layer, with ONE kernel emitting normalized q|k directly.
//
// Variable-length requests are packed token-after-token (qk_new[P, C]); each
// prefill token t belongs to request req_id[t] and sits at local position
// seg_pos[t]=L within it. The two-stage causal conv window for token t is the
// TP+1 columns ending at it: column o holds absolute position (L-TP+o) ->
//   abs_pos >= 0 : a real token,  qk_new[t-TP+o]            (same request, coalesced)
//   abs_pos <  0 : a cached col,  init_states[req_id, L+o]  (the prev chunk's state)
// init_states[R, C, TP] is the caller's pre-gathered, has-initial-state-masked
// copy of the old conv state (zero where no initial state). The kernel READS it
// (never the live conv_states) so there is no read/write hazard, and WRITES the
// new conv state only from each request's LAST token (window[1..TP]) — exactly
// the decode state-roll, gated to seg ends; short segments fall back onto the
// cached columns naturally. Conv + grouped-means + per-head RMS-norm are
// byte-identical to cca_decode_qk (same coalesced-w1 + unroll16 hot loop).
__global__ void cca_prefill_qk_kernel(
    const float* __restrict__ qk_new,        // [P, C]
    float* __restrict__ conv_states,         // [NB, C, TP] (write target by slot)
    const float* __restrict__ init_states,   // [R, C, TP] (masked cached state, read)
    const int* __restrict__ seg_pos,         // [P] local position L within request
    const int* __restrict__ req_id,          // [P] request index (indexes init_states)
    const long* __restrict__ slot,           // [P] conv_states write slot per token
    const unsigned char* __restrict__ is_last, // [P] 1 if token is its request's last
    const float* __restrict__ w0,            // [C, K0]
    const float* __restrict__ b0,            // [C]
    const float* __restrict__ w1,            // [H, d_in, d_out, K1] (transposed)
    const float* __restrict__ b1,            // [C]
    const float* __restrict__ temp_eff,      // [num_k]
    float* __restrict__ qk_out,              // [P, C]
    int C, int TP, int K0, int K1, int d, int Lmid,
    int num_q, int gqa, int latent_q, float sqrt_d, float eps,
    long is0, long is1, long is2,            // init_states strides
    long s0, long s1, long s2) {             // conv_states strides

  const int h = blockIdx.x;        // head
  const int t = blockIdx.y;        // prefill token (global)
  const int j = threadIdx.x;       // output channel in head, 0..d-1 (blockDim==d)
  const bool is_q = h < num_q;
  const int head_base = h * d;
  const int c = head_base + j;
  const int L = seg_pos[t];
  const int r = req_id[t];
  const int win_cols = TP + 1;

  extern __shared__ float sm[];
  float* win = sm;                  // [d * (TP+1)]
  float* mid = win + d * win_cols;  // [d * Lmid]
  float* red = mid + d * Lmid;      // [d]

  // 1. Build this channel's window (cached cols from init_states, token cols from
  //    qk_new). No cross-thread reads -> no sync until the grouped conv.
  const long init_base = (long)r * is0 + (long)c * is1;
  for (int o = 0; o < win_cols; ++o) {
    const int abs_pos = L - TP + o;
    float v;
    if (abs_pos >= 0) v = qk_new[(long)(t - TP + o) * C + c];
    else              v = init_states[init_base + (long)(L + o) * is2];
    win[j * win_cols + o] = v;
  }
  // 2. depthwise conv (K0==2 fast path).
  const float bc = b0[c];
  if (K0 == 2) {
    const float w0a = w0[(long)c * 2 + 0];
    const float w0b = w0[(long)c * 2 + 1];
#pragma unroll
    for (int tm = 0; tm < Lmid; ++tm)
      mid[j * Lmid + tm] = bc + win[j * win_cols + tm] * w0a +
                           win[j * win_cols + tm + 1] * w0b;
  } else {
    const float* wc = w0 + (long)c * K0;
    for (int tm = 0; tm < Lmid; ++tm) {
      float acc = bc;
      for (int k = 0; k < K0; ++k) acc += win[j * win_cols + tm + k] * wc[k];
      mid[j * Lmid + tm] = acc;
    }
  }
  __syncthreads();

  // 3. grouped conv (transposed w1, coalesced float2 + unroll16) — see cca_decode_qk.
  float val = b1[c];
  if (K1 == 2 && Lmid == 2) {
    const float2* w1t = reinterpret_cast<const float2*>(w1);
    const float2* m2 = reinterpret_cast<const float2*>(mid);
    const long wbase = (long)h * d * d + j;
#pragma unroll 16
    for (int di = 0; di < d; ++di) {
      float2 wv = w1t[wbase + (long)di * d];
      float2 mv = m2[di];
      val += wv.x * mv.x + wv.y * mv.y;
    }
  } else {
    for (int di = 0; di < d; ++di) {
      const float* wr = w1 + (((long)h * d + di) * d + j) * K1;
      const float* mr = mid + (long)di * Lmid;
      for (int k = 0; k < K1; ++k) val += wr[k] * mr[k];
    }
  }

  // 4. grouped means (read this token's pre-conv q|k) — identical to decode.
  if (is_q) {
    const int gi = h / gqa;
    val += 0.5f * qk_new[(long)t * C + head_base + j];
    val += 0.5f * qk_new[(long)t * C + latent_q + gi * d + j];
  } else {
    const int ki = h - num_q;
    float m = 0.0f;
    for (int gg = 0; gg < gqa; ++gg)
      m += qk_new[(long)t * C + (ki * gqa + gg) * d + j];
    val += 0.5f * (m / (float)gqa);
    val += 0.5f * qk_new[(long)t * C + latent_q + ki * d + j];
  }

  // 5. per-head RMS-norm: wave32 shuffle reduction + one cross-wave LDS combine.
  float sq = val * val;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sq += __shfl_down(sq, o, 32);
  const int nwaves = blockDim.x >> 5;
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sq;
  __syncthreads();
  float ssq = 0.0f;
#pragma unroll
  for (int w = 0; w < nwaves; ++w) ssq += red[w];
  val *= rsqrtf(ssq + eps) * sqrt_d;
  if (!is_q) val *= temp_eff[h - num_q];

  // 6. write qk_out (q region | k region).
  const int out_col = is_q ? head_base : (latent_q + (h - num_q) * d);
  qk_out[(long)t * C + out_col + j] = val;

  // 7. new conv state: only each request's LAST token writes its slot's TP taps
  //    (window[1..TP] == the segment's last TP columns). Reads init_states above,
  //    writes conv_states here -> no aliasing with other tokens' window reads.
  if (is_last[t]) {
    const long sbase = slot[t] * s0 + (long)c * s1;
    for (int k = 0; k < TP; ++k)
      conv_states[sbase + (long)k * s2] = win[j * win_cols + (k + 1)];
  }
}

void launch_cca_prefill_qk(
    const at::Tensor& qk_new,
    at::Tensor& conv_states,
    const at::Tensor& init_states,
    const at::Tensor& seg_pos,
    const at::Tensor& req_id,
    const at::Tensor& slot,
    const at::Tensor& is_last,
    const at::Tensor& w0,
    const at::Tensor& b0,
    const at::Tensor& w1,
    const at::Tensor& b1,
    const at::Tensor& temp_eff,
    at::Tensor& qk_out,
    int64_t num_q,
    int64_t gqa,
    int64_t latent_q,
    double sqrt_d) {
  const int P = qk_new.size(0);
  const int C = qk_new.size(1);
  const int TP = conv_states.size(2);
  const int K0 = w0.size(1);
  const int d = w1.size(2);   // w1 transposed [H, d_in, d_out, K1]
  const int K1 = w1.size(3);
  const int Lmid = TP - K0 + 2;
  const int H = num_q + temp_eff.size(0);

  const size_t smem = (size_t)d * (TP + 1 + Lmid + 1) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(
      cca_prefill_qk_kernel, dim3(H, P), dim3(d), smem, stream,
      qk_new.data_ptr<float>(), conv_states.data_ptr<float>(),
      init_states.data_ptr<float>(),
      seg_pos.data_ptr<int>(), req_id.data_ptr<int>(), slot.data_ptr<long>(),
      reinterpret_cast<const unsigned char*>(is_last.data_ptr<bool>()),
      w0.data_ptr<float>(), b0.data_ptr<float>(),
      w1.data_ptr<float>(), b1.data_ptr<float>(),
      temp_eff.data_ptr<float>(), qk_out.data_ptr<float>(),
      C, TP, K0, K1, d, Lmid, (int)num_q, (int)gqa, (int)latent_q,
      (float)sqrt_d, 1e-12f,
      init_states.stride(0), init_states.stride(1), init_states.stride(2),
      conv_states.stride(0), conv_states.stride(1), conv_states.stride(2));
}
