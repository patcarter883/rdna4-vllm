// rotation_hip.hip — HIP/RDNA4 (gfx1201) port of ParoQuant's pairwise-rotation kernel
// (upstream: z-lab/paroquant paroquant/kernels/cuda/rotation.{cu,cuh}, © 2025 Haisheng Chen, MIT).
//
// WHAT THIS IS: the inference-time hot path of ParoQuant is a single pre-GEMM activation
// rotation — KROT rounds of disjoint within-group Givens rotations + per-channel scale, applied
// to x before a standard INT4 matmul. This file is a faithful HIP port of that kernel plus a
// SELF-CONTAINED correctness harness (no torch / no vLLM), matching the repo's bench_*.hip style.
//
// CORRECTNESS METHODOLOGY (see advisor note):
//   * Test data is generated in LOGICAL form: per [round][group][pair] a (i_local,j_local,angle),
//     per global channel a scale, per [row][channel] an activation.
//   * A single packer encodes the LAYOUT CONTRACT (theta [krot,h/2], idx_ij [krot,h] read as
//     packed int32, scales [h]) into the flat device buffers the kernel consumes — cross-checked
//     against the upstream vLLM plugin's param shapes.
//   * The CPU reference is computed from the LOGICAL data directly (it never re-derives offsets
//     with the kernel's own index math), so a transposed/off-by-one layout bug shows as gross
//     per-channel error instead of passing self-consistently.
//   * Test A (forward vs CPU ref) validates the layout + math. Test B (forward∘inverse ≈ identity)
//     validates orthogonality. A alone catches layout bugs that B (identity-preserving) cannot.
//
//   build: /opt/rocm/bin/hipcc --offload-arch=gfx1201 -O3 -I. rotation_hip.hip -o /tmp/rot_test
//   run:   under a 1-card lease (CLAUDE.md GPU sharing protocol):
//          scripts/gpu-lease.sh -n 1 -- /tmp/rot_test

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <numeric>

#define HIP_CHECK(cmd) do { hipError_t e = (cmd); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

// ============================ ported kernel (rotation.cuh) ============================
// Faithful to upstream: float specialization (scalar) + packed half/bf16 specialization.
// Only changes vs CUDA: HIP headers, __nv_bfloat16{,2} -> __hip_bfloat16{,2}, and the bf16
// scalar hmul routed through float (the bf16 *packing* intrinsics are kept native).

template <typename scalar_t> struct RotateAccess;

template <> struct RotateAccess<float> {
  template <int CTA_M, int GROUP_SIZE, bool USE_SCALE>
  __device__ static void load_group(float *__restrict__ x_grp, const float *__restrict__ x,
                                    const float *__restrict__ scales, const int s, const int h,
                                    const int j, const int g, const int t) {
    const int base0 = g * GROUP_SIZE + t;
    const int base1 = base0 + GROUP_SIZE / 2;
    float scale0 = USE_SCALE ? scales[base0] : float(1);
    float scale1 = USE_SCALE ? scales[base1] : float(1);
#pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        x_grp[t * CTA_M + i] = x[row * h + base0] * scale0;
        x_grp[(t + GROUP_SIZE / 2) * CTA_M + i] = x[row * h + base1] * scale1;
      }
    }
  }
  template <int KROT, int GROUP_SIZE>
  __device__ static void load_coeffs(float reg_theta[KROT], int reg_idx[KROT],
                                     const int16_t *__restrict__ idx_ij,
                                     const float *__restrict__ theta, const int h, const int g,
                                     const int t) {
#pragma unroll
    for (int r = 0; r < KROT; r++) {
      reg_theta[r] = theta[r * h / 2 + g * GROUP_SIZE / 2 + t];
      reg_idx[r] = *reinterpret_cast<const int *>(idx_ij + r * h + g * GROUP_SIZE + 2 * t);
    }
  }
  template <int CTA_M>
  __device__ static void apply_one(float *__restrict__ x_grp, const int ij, const float theta) {
    int16_t i = ij & 0xFFFF, j = ij >> 16;
    float s_, c_;
    __sincosf(theta, &s_, &c_);
#pragma unroll
    for (int m = 0; m < CTA_M; m++) {
      float xi = x_grp[i * CTA_M + m];
      float xj = x_grp[j * CTA_M + m];
      x_grp[i * CTA_M + m] = xi * c_ + xj * s_;
      x_grp[j * CTA_M + m] = xi * (-s_) + xj * c_;
    }
  }
  template <int CTA_M, int GROUP_SIZE>
  __device__ static void store_group(float *__restrict__ out, const float *__restrict__ x_grp,
                                     const int s, const int h, const int j, const int g,
                                     const int t) {
    const int base0 = g * GROUP_SIZE + t;
    const int base1 = base0 + GROUP_SIZE / 2;
#pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        out[row * h + base0] = x_grp[t * CTA_M + i];
        out[row * h + base1] = x_grp[(t + GROUP_SIZE / 2) * CTA_M + i];
      }
    }
  }
};

template <typename HalfT, typename Half2T, typename Traits> struct RotateAccessHalf {
  template <int CTA_M, int GROUP_SIZE, bool USE_SCALE>
  __device__ static void load_group(HalfT *__restrict__ x_grp, const HalfT *__restrict__ x,
                                    const HalfT *__restrict__ scales, const int s, const int h,
                                    const int j, const int g, const int t) {
    const int offset = GROUP_SIZE * g + 2 * t;
    HalfT scale_i, scale_j;
    if constexpr (USE_SCALE) {
      Half2T scale_pair = *reinterpret_cast<const Half2T *>(scales + offset);
      scale_i = Traits::low(scale_pair);
      scale_j = Traits::high(scale_pair);
    } else {
      scale_i = Traits::from_float(1.0f);
      scale_j = Traits::from_float(1.0f);
    }
#pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        Half2T x2 = *reinterpret_cast<const Half2T *>(x + row * h + offset);
        HalfT lo = Traits::hmul(Traits::low(x2), scale_i);
        HalfT hi = Traits::hmul(Traits::high(x2), scale_j);
        x_grp[(2 * t) * CTA_M + i] = lo;
        x_grp[(2 * t + 1) * CTA_M + i] = hi;
      }
    }
  }
  template <int KROT, int GROUP_SIZE>
  __device__ static void load_coeffs(float reg_theta[KROT], int reg_idx[KROT],
                                     const int16_t *__restrict__ idx_ij,
                                     const HalfT *__restrict__ theta, const int h, const int g,
                                     const int t) {
#pragma unroll
    for (int r = 0; r < KROT; r++) {
      reg_theta[r] = Traits::to_float(theta[r * h / 2 + g * GROUP_SIZE / 2 + t]);
      reg_idx[r] = *reinterpret_cast<const int *>(idx_ij + r * h + g * GROUP_SIZE + 2 * t);
    }
  }
  template <int CTA_M>
  __device__ static void apply_one(HalfT *__restrict__ x_grp, const int ij, const float theta) {
    int16_t i = ij & 0xFFFF, j = ij >> 16;
    float s_, c_;
    __sincosf(theta, &s_, &c_);
#pragma unroll
    for (int m = 0; m < CTA_M / 2; ++m) {
      Half2T *pi2 = reinterpret_cast<Half2T *>(x_grp + i * CTA_M + m * 2);
      Half2T *pj2 = reinterpret_cast<Half2T *>(x_grp + j * CTA_M + m * 2);
      float2 xi = Traits::to_float2(*pi2);
      float2 xj = Traits::to_float2(*pj2);
      float2 yi, yj;
      yi.x = fmaf(c_, xi.x, s_ * xj.x);
      yi.y = fmaf(c_, xi.y, s_ * xj.y);
      yj.x = fmaf(c_, xj.x, -s_ * xi.x);
      yj.y = fmaf(c_, xj.y, -s_ * xi.y);
      *pi2 = Traits::from_floats(yi.x, yi.y);
      *pj2 = Traits::from_floats(yj.x, yj.y);
    }
  }
  template <int CTA_M, int GROUP_SIZE>
  __device__ static void store_group(HalfT *__restrict__ out, const HalfT *__restrict__ x_grp,
                                     const int s, const int h, const int j, const int g,
                                     const int t) {
    const int base = GROUP_SIZE * g + 2 * t;
#pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        Half2T out2;
        out2.x = x_grp[(2 * t) * CTA_M + i];
        out2.y = x_grp[(2 * t + 1) * CTA_M + i];
        *reinterpret_cast<Half2T *>(out + row * h + base) = out2;
      }
    }
  }
};

struct HalfTraits {
  __device__ static float2 to_float2(__half2 v) { return __half22float2(v); }
  __device__ static __half2 from_floats(float a, float b) { return __floats2half2_rn(a, b); }
  __device__ static float to_float(__half v) { return static_cast<float>(v); }
  __device__ static __half from_float(float v) { return __half(v); }
  __device__ static __half low(__half2 v) { return __low2half(v); }
  __device__ static __half high(__half2 v) { return __high2half(v); }
  __device__ static __half hmul(__half a, __half b) { return __hmul(a, b); }
};
template <> struct RotateAccess<__half> : RotateAccessHalf<__half, __half2, HalfTraits> {};

struct BFloat16Traits {
  __device__ static float2 to_float2(__hip_bfloat162 v) { return __bfloat1622float2(v); }
  __device__ static __hip_bfloat162 from_floats(float a, float b) {
    return __float22bfloat162_rn(float2{a, b}); // HIP spelling of CUDA __floats2bfloat162_rn
  }
  __device__ static float to_float(__hip_bfloat16 v) { return __bfloat162float(v); }
  __device__ static __hip_bfloat16 from_float(float v) { return __float2bfloat16(v); }
  __device__ static __hip_bfloat16 low(__hip_bfloat162 v) { return __low2bfloat16(v); }
  __device__ static __hip_bfloat16 high(__hip_bfloat162 v) { return __high2bfloat16(v); }
  // bf16 scalar hmul: the packing intrinsics above are native; the lone scalar multiply (load
  // scale) is routed through float — RDNA4 has no need of a special bf16 scalar-mul intrinsic.
  __device__ static __hip_bfloat16 hmul(__hip_bfloat16 a, __hip_bfloat16 b) {
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
  }
};
template <>
struct RotateAccess<__hip_bfloat16>
    : RotateAccessHalf<__hip_bfloat16, __hip_bfloat162, BFloat16Traits> {};

// ============================ ported kernel (rotation.cu) ============================

template <typename scalar_t, int CTA_M, int GROUP_SIZE, int KROT, bool USE_SCALE>
__global__ void rotate_kernel(const scalar_t *__restrict__ x, scalar_t *__restrict__ out,
                              const int16_t *__restrict__ idx_ij, const scalar_t *__restrict__ theta,
                              const scalar_t *__restrict__ scales, int s, int h) {
  static_assert(CTA_M % 2 == 0, "CTA_M must be even");
  static_assert(GROUP_SIZE % 2 == 0, "GROUP_SIZE must be even");
  __shared__ scalar_t x_grp[CTA_M * GROUP_SIZE];

  int j = blockIdx.x;  // row-block
  int g = blockIdx.y;  // group
  int t = threadIdx.x; // pair within group

  RotateAccess<scalar_t>::template load_group<CTA_M, GROUP_SIZE, USE_SCALE>(x_grp, x, scales, s, h,
                                                                            j, g, t);
  float reg_theta[KROT];
  int reg_idx[KROT];
  RotateAccess<scalar_t>::template load_coeffs<KROT, GROUP_SIZE>(reg_theta, reg_idx, idx_ij, theta,
                                                                 h, g, t);
  __syncthreads();
#pragma unroll
  for (int r = 0; r < KROT; r++) {
    RotateAccess<scalar_t>::template apply_one<CTA_M>(x_grp, reg_idx[r], reg_theta[r]);
    __syncthreads();
  }
  RotateAccess<scalar_t>::template store_group<CTA_M, GROUP_SIZE>(out, x_grp, s, h, j, g, t);
}

// =============================== host harness ===============================

constexpr int CTA_M = 4; // upstream fixes CTA_M=4 in its launcher

// Host dtype helpers: convert float<->device-scalar with the device rounding (host-callable).
template <class T> struct HConv;
template <> struct HConv<float> {
  static float to(float f) { return f; }
  static float from(float v) { return v; }
  static const char *name() { return "float"; }
};
template <> struct HConv<__half> {
  static __half to(float f) { return __float2half(f); }
  static float from(__half v) { return __half2float(v); }
  static const char *name() { return "half"; }
};
template <> struct HConv<__hip_bfloat16> {
  static __hip_bfloat16 to(float f) { return __float2bfloat16(f); }
  static float from(__hip_bfloat16 v) { return __bfloat162float(v); }
  static const char *name() { return "bf16"; }
};

// Logical test data — the SINGLE source of truth the reference reads from.
struct Logical {
  int M, H, GS, KROT;
  std::vector<float> x;                            // [M*H]
  std::vector<float> scale;                        // [H]
  // pairs[r][g] = GS/2 (i_local,j_local); angle[r][g] = GS/2 angles. Logical, not packed.
  std::vector<std::vector<std::vector<std::pair<int,int>>>> pairs;
  std::vector<std::vector<std::vector<float>>> angle;
};

static Logical make_logical(int M, int H, int GS, int KROT, uint32_t seed) {
  Logical L; L.M = M; L.H = H; L.GS = GS; L.KROT = KROT;
  int groups = H / GS, half = GS / 2;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> ux(-2.f, 2.f), us(0.5f, 1.5f), ua(-3.14159f, 3.14159f);
  L.x.resize((size_t)M * H);
  for (auto &v : L.x) v = ux(rng);
  L.scale.resize(H);
  for (auto &v : L.scale) v = us(rng);
  L.pairs.assign(KROT, {});
  L.angle.assign(KROT, {});
  for (int r = 0; r < KROT; r++) {
    L.pairs[r].assign(groups, {});
    L.angle[r].assign(groups, {});
    for (int g = 0; g < groups; g++) {
      // INVARIANT (kernel has no intra-round sync): pairs within a round must be a PERFECT
      // MATCHING of the GS channels. Random permutation -> consecutive pairs guarantees it.
      std::vector<int> perm(GS);
      std::iota(perm.begin(), perm.end(), 0);
      std::shuffle(perm.begin(), perm.end(), rng);
      for (int t = 0; t < half; t++) {
        L.pairs[r][g].push_back({perm[2 * t], perm[2 * t + 1]});
        L.angle[r][g].push_back(ua(rng));
      }
      // assert the matching invariant
      std::vector<char> seen(GS, 0);
      for (auto &p : L.pairs[r][g]) { seen[p.first]++; seen[p.second]++; }
      for (int c = 0; c < GS; c++) if (seen[c] != 1) { fprintf(stderr, "matching invariant broken\n"); exit(1); }
    }
  }
  return L;
}

// THE LAYOUT CONTRACT: pack logical -> flat buffers exactly as the kernel/plugin expect.
//   theta  : [KROT, H/2]    index r*(H/2) + g*(GS/2) + t
//   idx_ij : [KROT, H] int16, read as packed int32 at r*H + g*GS + 2t -> (i_low, j_high)
//   scale  : [H]            global channel c
template <class T>
static void pack(const Logical &L, std::vector<T> &theta_f, std::vector<int16_t> &idx_f,
                 std::vector<T> &scale_f) {
  int groups = L.H / L.GS, half = L.GS / 2;
  theta_f.assign((size_t)L.KROT * (L.H / 2), HConv<T>::to(0.f));
  idx_f.assign((size_t)L.KROT * L.H, 0);
  scale_f.resize(L.H);
  for (int c = 0; c < L.H; c++) scale_f[c] = HConv<T>::to(L.scale[c]);
  for (int r = 0; r < L.KROT; r++)
    for (int g = 0; g < groups; g++)
      for (int t = 0; t < half; t++) {
        auto pr = L.pairs[r][g][t];
        theta_f[(size_t)r * (L.H / 2) + g * half + t] = HConv<T>::to(L.angle[r][g][t]);
        idx_f[(size_t)r * L.H + g * L.GS + 2 * t] = (int16_t)pr.first;
        idx_f[(size_t)r * L.H + g * L.GS + 2 * t + 1] = (int16_t)pr.second;
      }
}

// CPU reference computed FROM LOGICAL DATA (independent of the kernel's offset arithmetic).
// Mirrors GPU numerics: round-to-dtype between rounds and on the scaled load; angles rounded to
// dtype (kernel stores theta in scalar_t). use_scale toggles the per-channel scale at load.
template <class T>
static std::vector<float> cpu_reference(const Logical &L, bool use_scale) {
  int groups = L.H / L.GS, half = L.GS / 2;
  std::vector<float> out((size_t)L.M * L.H, 0.f);
  std::vector<float> v(L.GS);
  for (int m = 0; m < L.M; m++) {
    for (int g = 0; g < groups; g++) {
      for (int c = 0; c < L.GS; c++) {
        int gc = g * L.GS + c;
        float xv = HConv<T>::from(HConv<T>::to(L.x[(size_t)m * L.H + gc])); // buffer is dtype
        if (use_scale) {
          float sv = HConv<T>::from(HConv<T>::to(L.scale[gc]));
          // matches Traits::hmul: half=native dtype mul rounded to dtype; bf16=float mul rounded
          v[c] = HConv<T>::from(HConv<T>::to(xv * sv));
        } else {
          v[c] = xv;
        }
      }
      for (int r = 0; r < L.KROT; r++) {
        for (int t = 0; t < half; t++) {
          int i = L.pairs[r][g][t].first, j = L.pairs[r][g][t].second;
          float th = HConv<T>::from(HConv<T>::to(L.angle[r][g][t])); // theta stored in dtype
          float s_ = sinf(th), c_ = cosf(th);
          float vi = v[i], vj = v[j];
          v[i] = HConv<T>::from(HConv<T>::to(vi * c_ + vj * s_));
          v[j] = HConv<T>::from(HConv<T>::to(-vi * s_ + vj * c_));
        }
      }
      for (int c = 0; c < L.GS; c++) out[(size_t)m * L.H + g * L.GS + c] = v[c];
    }
  }
  return out;
}

// Launch the kernel for a packed input, return output as float[M*H].
template <class T, int GS, int KROT, bool USE_SCALE>
static std::vector<float> run_kernel(int M, int H, const std::vector<T> &x_dev_host,
                                     const std::vector<int16_t> &idx_f,
                                     const std::vector<T> &theta_f, const std::vector<T> &scale_f) {
  size_t nx = (size_t)M * H;
  T *dx, *dout, *dtheta, *dscale = nullptr;
  int16_t *didx;
  HIP_CHECK(hipMalloc(&dx, nx * sizeof(T)));
  HIP_CHECK(hipMalloc(&dout, nx * sizeof(T)));
  HIP_CHECK(hipMalloc(&dtheta, theta_f.size() * sizeof(T)));
  HIP_CHECK(hipMalloc(&didx, idx_f.size() * sizeof(int16_t)));
  HIP_CHECK(hipMemcpy(dx, x_dev_host.data(), nx * sizeof(T), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dtheta, theta_f.data(), theta_f.size() * sizeof(T), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(didx, idx_f.data(), idx_f.size() * sizeof(int16_t), hipMemcpyHostToDevice));
  if (USE_SCALE) {
    HIP_CHECK(hipMalloc(&dscale, scale_f.size() * sizeof(T)));
    HIP_CHECK(hipMemcpy(dscale, scale_f.data(), scale_f.size() * sizeof(T), hipMemcpyHostToDevice));
  }
  dim3 grid((M + CTA_M - 1) / CTA_M, H / GS);
  dim3 block(GS / 2);
  rotate_kernel<T, CTA_M, GS, KROT, USE_SCALE>
      <<<grid, block, 0, 0>>>(dx, dout, didx, dtheta, dscale, M, H);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<T> out_host(nx);
  HIP_CHECK(hipMemcpy(out_host.data(), dout, nx * sizeof(T), hipMemcpyDeviceToHost));
  std::vector<float> out(nx);
  for (size_t i = 0; i < nx; i++) out[i] = HConv<T>::from(out_host[i]);
  HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dout)); HIP_CHECK(hipFree(dtheta)); HIP_CHECK(hipFree(didx));
  if (dscale) HIP_CHECK(hipFree(dscale));
  return out;
}

struct Stat { double max_abs; double max_rel; };
static Stat compare(const std::vector<float> &a, const std::vector<float> &b) {
  double ma = 0, mr = 0;
  for (size_t i = 0; i < a.size(); i++) {
    double d = std::fabs((double)a[i] - (double)b[i]);
    ma = std::max(ma, d);
    double den = std::max(1e-4, std::fabs((double)b[i]));
    mr = std::max(mr, d / den);
  }
  return {ma, mr};
}

static int g_fail = 0;
static void report(const std::string &tag, Stat s, double tol_abs) {
  bool ok = s.max_abs <= tol_abs || s.max_rel <= 0.06;
  printf("  [%s] %-46s max_abs=%.4g max_rel=%.4g (tol_abs=%.3g)\n",
         ok ? "PASS" : "FAIL", tag.c_str(), s.max_abs, s.max_rel, tol_abs);
  if (!ok) g_fail = 1;
}

// Build the "inverse" logical: reverse round order, negate angles (matches torch.flip(dim=0)).
static Logical invert_logical(const Logical &L) {
  Logical I = L;
  std::reverse(I.pairs.begin(), I.pairs.end());
  std::reverse(I.angle.begin(), I.angle.end());
  for (auto &rr : I.angle) for (auto &gg : rr) for (auto &a : gg) a = -a;
  return I;
}

// Test A: forward kernel vs CPU reference (validates LAYOUT + math).
// Test B: forward then inverse ≈ identity (validates orthogonality), no scale.
template <class T, int GS, int KROT>
static void run_case(int M, int H, uint32_t seed, double tol_abs) {
  Logical L = make_logical(M, H, GS, KROT, seed);
  char hdr[160];
  snprintf(hdr, sizeof(hdr), "%s GS=%d KROT=%d M=%d H=%d", HConv<T>::name(), GS, KROT, M, H);
  printf("%s\n", hdr);

  // --- Test A, with scale ---
  std::vector<T> theta_f, scale_f; std::vector<int16_t> idx_f;
  pack<T>(L, theta_f, idx_f, scale_f);
  std::vector<T> x_host((size_t)M * H);
  for (size_t i = 0; i < x_host.size(); i++) x_host[i] = HConv<T>::to(L.x[i]);

  auto k_scaled = run_kernel<T, GS, KROT, true>(M, H, x_host, idx_f, theta_f, scale_f);
  auto ref_scaled = cpu_reference<T>(L, true);
  report("A: forward(scale) vs CPU ref", compare(k_scaled, ref_scaled), tol_abs);

  auto k_noscale = run_kernel<T, GS, KROT, false>(M, H, x_host, idx_f, theta_f, scale_f);
  auto ref_noscale = cpu_reference<T>(L, false);
  report("A: forward(no-scale) vs CPU ref", compare(k_noscale, ref_noscale), tol_abs);

  // --- Test B: forward∘inverse ≈ identity (no scale, orthogonality) ---
  Logical Inv = invert_logical(L);
  std::vector<T> itheta_f, iscale_f; std::vector<int16_t> iidx_f;
  pack<T>(Inv, itheta_f, iidx_f, iscale_f);
  std::vector<T> y_host((size_t)M * H);
  for (size_t i = 0; i < y_host.size(); i++) y_host[i] = HConv<T>::to(k_noscale[i]);
  auto z = run_kernel<T, GS, KROT, false>(M, H, y_host, iidx_f, itheta_f, iscale_f);
  std::vector<float> x_rounded((size_t)M * H);
  for (size_t i = 0; i < x_rounded.size(); i++) x_rounded[i] = HConv<T>::from(x_host[i]);
  report("B: forward∘inverse ≈ identity", compare(z, x_rounded), tol_abs);
}

int main() {
  int dev = 0; hipDeviceProp_t prop;
  HIP_CHECK(hipGetDevice(&dev));
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  printf("device: %s (%s)\n\n", prop.name, prop.gcnArchName);

  // KROT compiled by upstream dispatch: 1 and 8 only. Test exactly those.
  // half / bf16: packed path; float: scalar path. GS 128 and 64. Includes M not divisible by 4.
  run_case<__half, 128, 8>(64, 2048, 1, 6e-2);
  run_case<__half, 128, 1>(64, 2048, 2, 2e-2);
  run_case<__half, 64, 8>(40, 2048, 3, 6e-2);
  run_case<__half, 128, 8>(13, 2048, 4, 6e-2); // M % CTA_M != 0
  run_case<__hip_bfloat16, 128, 8>(64, 2048, 5, 3e-1);
  run_case<__hip_bfloat16, 64, 8>(40, 1024, 6, 3e-1);
  run_case<float, 128, 8>(64, 2048, 7, 2e-3);
  run_case<float, 64, 1>(32, 1024, 8, 1e-3);

  printf("\n%s\n", g_fail ? "*** SOME TESTS FAILED ***" : "ALL TESTS PASSED");
  return g_fail;
}
