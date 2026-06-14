// fp16 WMMA throughput ceiling on gfx1201 (RDNA4), mirror of wmma_peak.hip but
// with float16_t fragments. Tells us the fp16 matrix-core peak that hipBLASLt is
// competing against (vs the 370 TFLOP/s fp8 peak).
#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>

constexpr int NACC = 8;
constexpr int ITERS = 4096;

using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma::float16_t, rocwmma::row_major>;
using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, rocwmma::float16_t, rocwmma::col_major>;
using FragAcc = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

__global__ void peak(float* __restrict__ out) {
    __shared__ rocwmma::float16_t tile[16 * 16];
    if (threadIdx.x < 256) tile[threadIdx.x] = (rocwmma::float16_t)(threadIdx.x & 7);
    __syncthreads();
    FragA a; FragB b;
    rocwmma::load_matrix_sync(a, tile, 16);
    rocwmma::load_matrix_sync(b, tile, 16);
    FragAcc acc[NACC];
    #pragma unroll
    for (int i = 0; i < NACC; ++i) rocwmma::fill_fragment(acc[i], 0.0f);
    for (int t = 0; t < ITERS; ++t) {
        #pragma unroll
        for (int i = 0; i < NACC; ++i) rocwmma::mma_sync(acc[i], a, b, acc[i]);
    }
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < NACC; ++i) s += acc[i].x[0];
    if (s == -1.0f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

int main() {
    const int blocks = 64 * 8, warps_per_block = 8, threads = warps_per_block * 32;
    float* d_out; hipMalloc(&d_out, sizeof(float));
    hipEvent_t s, e; hipEventCreate(&s); hipEventCreate(&e);
    peak<<<blocks, threads>>>(d_out); hipDeviceSynchronize();
    hipEventRecord(s);
    const int launches = 20;
    for (int i = 0; i < launches; ++i) peak<<<blocks, threads>>>(d_out);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms = 0; hipEventElapsedTime(&ms, s, e);
    double mma = (double)blocks * warps_per_block * NACC * ITERS * launches;
    double flops = mma * 2.0 * 16 * 16 * 16;
    printf("fp16 WMMA peak: %.1f TFLOP/s  (%.2f ms)\n", flops / (ms / 1e3) / 1e12, ms);
    return 0;
}
