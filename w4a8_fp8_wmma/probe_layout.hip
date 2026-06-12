// Probe the gfx12 WMMA f32 accumulator fragment layout: for each (lane, element)
// learn the (row, col) it maps to in the 16x16 output tile. Needed to apply
// per-group weight scales in-register (no LDS round-trip) in the optimized kernel.
//
// Build (bare metal): hipcc -O2 --offload-arch=gfx1201 -I<rocm>/include \
//                     probe_layout.hip -o /tmp/probe && /tmp/probe
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>

using AccFrag = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

__global__ void probe(const float* __restrict__ known, float* __restrict__ raw,
                      int* __restrict__ nelem) {
    AccFrag acc;
    rocwmma::load_matrix_sync(acc, known, 16, rocwmma::mem_row_major);
    int lane = threadIdx.x;
    if (lane == 0) *nelem = acc.num_elements;
    for (int e = 0; e < acc.num_elements; ++e)
        raw[lane * acc.num_elements + e] = acc.x[e];
}

int main() {
    float h_known[256];
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c)
            h_known[r * 16 + c] = r * 100.0f + c;  // encode (row,col)

    float *d_known, *d_raw;
    int *d_n;
    hipMalloc(&d_known, 256 * sizeof(float));
    hipMalloc(&d_raw, 32 * 16 * sizeof(float));
    hipMalloc(&d_n, sizeof(int));
    hipMemcpy(d_known, h_known, 256 * sizeof(float), hipMemcpyHostToDevice);

    probe<<<1, 32>>>(d_known, d_raw, d_n);
    hipDeviceSynchronize();

    int nelem = 0;
    hipMemcpy(&nelem, d_n, sizeof(int), hipMemcpyDeviceToHost);
    float h_raw[32 * 16];
    hipMemcpy(h_raw, d_raw, 32 * 16 * sizeof(float), hipMemcpyDeviceToHost);

    printf("num_elements per lane = %d\n", nelem);
    printf("lane: (row,col) for each element e=0..%d\n", nelem - 1);
    for (int lane = 0; lane < 32; ++lane) {
        printf("lane %2d:", lane);
        for (int e = 0; e < nelem; ++e) {
            float v = h_raw[lane * nelem + e];
            int r = (int)(v / 100.0f + 0.5f);
            int c = (int)(v - r * 100.0f + 0.5f);
            printf("  e%d=(%2d,%2d)", e, r, c);
        }
        printf("\n");
    }
    return 0;
}
