// Measure raw fp8 WMMA throughput ceiling on gfx1201: each warp hammers NACC
// INDEPENDENT mma_sync (no accumulator dependency chain) so we measure
// throughput, not latency. Tells us the achievable peak the GEMM should target.
//
// Build: hipcc -O3 --offload-arch=gfx1201 -I<rocm>/include wmma_peak.hip -o /tmp/peak
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>

constexpr int NACC = 8;
constexpr int ITERS = 4096;

using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma::float8_t, rocwmma::row_major>;
using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, rocwmma::float8_t, rocwmma::col_major>;
using FragAcc = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

__global__ void peak(float* __restrict__ out) {
    __shared__ unsigned char tile[16 * 16];
    if (threadIdx.x < 256) tile[threadIdx.x] = (unsigned char)(threadIdx.x & 7);
    __syncthreads();

    FragA a;
    FragB b;
    rocwmma::load_matrix_sync(a, reinterpret_cast<const rocwmma::float8_t*>(tile), 16);
    rocwmma::load_matrix_sync(b, reinterpret_cast<const rocwmma::float8_t*>(tile), 16);

    FragAcc acc[NACC];
    #pragma unroll
    for (int i = 0; i < NACC; ++i) rocwmma::fill_fragment(acc[i], 0.0f);

    for (int t = 0; t < ITERS; ++t) {
        #pragma unroll
        for (int i = 0; i < NACC; ++i)
            rocwmma::mma_sync(acc[i], a, b, acc[i]);
    }

    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < NACC; ++i) s += acc[i].x[0];
    if (s == -1.0f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;  // prevent DCE
}

int main() {
    const int blocks = 64 * 8;   // saturate 64 CUs
    const int warps_per_block = 8;
    const int threads = warps_per_block * 32;

    float* d_out;
    hipMalloc(&d_out, sizeof(float));

    hipEvent_t s, e;
    hipEventCreate(&s); hipEventCreate(&e);
    // warmup
    peak<<<blocks, threads>>>(d_out);
    hipDeviceSynchronize();

    hipEventRecord(s);
    const int launches = 20;
    for (int i = 0; i < launches; ++i) peak<<<blocks, threads>>>(d_out);
    hipEventRecord(e);
    hipEventSynchronize(e);
    float ms = 0; hipEventElapsedTime(&ms, s, e);

    double total_warps = (double)blocks * warps_per_block;
    double mma = total_warps * NACC * ITERS * launches;
    double flops = mma * 2.0 * 16 * 16 * 16;
    double tflops = flops / (ms / 1e3) / 1e12;
    printf("fp8 WMMA peak: %.1f TFLOP/s  (%.2f ms for %d launches)\n", tflops, ms, launches);
    return 0;
}
