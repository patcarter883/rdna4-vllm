// Crack the raw fp8 WMMA operand layout with PLAIN 128-bit K-major loads (the
// proven gfx1151-style path, no rocWMMA). Compute one 16x16x16 tile and compare
// to a CPU reference; try several per-lane offset conventions.
// Accumulator output layout is known: col = lane&15, row = (lane>>4)*8 + e.
//
// Build: hipcc -O2 --offload-arch=gfx1201 -I<rocm>/include probe_tr.hip -o /tmp/ptr
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

using v2i = int __attribute__((ext_vector_type(2)));
using v8f = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ unsigned char e4m3(float v) {
    return (unsigned char)(__builtin_amdgcn_cvt_pk_fp8_f32(v, v, 0, false) & 0xFF);
}

// A row-major A[m][k] (m*16+k), Bt row-major Bt[n][k] (n*16+k) = B^T (K-major both).
__global__ void probe(const float* gA, const float* gBt, float* gC, int mode) {
    __shared__ unsigned char A[256];
    __shared__ unsigned char Bt[256];
    int t = threadIdx.x;
    for (int i = t; i < 256; i += 32) { A[i] = e4m3(gA[i]); Bt[i] = e4m3(gBt[i]); }
    __syncthreads();

    const int lane = t;
    int offA, offB;
    if (mode == 0) {              // row=lane&15, K-half=(lane>>4)*8
        offA = (lane & 15) * 16 + (lane >> 4) * 8;
        offB = (lane & 15) * 16 + (lane >> 4) * 8;
    } else if (mode == 1) {       // flat 8 bytes per lane
        offA = lane * 8; offB = lane * 8;
    } else {                      // row=lane&15 but K-half swapped to high lanes via &16
        offA = (lane & 15) * 16 + ((lane & 16) ? 8 : 0);
        offB = (lane & 15) * 16 + ((lane & 16) ? 8 : 0);
    }

    v2i a = *reinterpret_cast<const v2i*>(&A[offA]);
    v2i b = *reinterpret_cast<const v2i*>(&Bt[offB]);
    v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
    acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, acc);

    const int col = lane & 15;
    #pragma unroll
    for (int e = 0; e < 8; ++e)
        gC[((lane >> 4) * 8 + e) * 16 + col] = acc[e];
}

int main() {
    float hA[256], hBt[256];
    float hC[256], ref[256];
    for (int m = 0; m < 16; ++m) for (int k = 0; k < 16; ++k) hA[m*16+k] = (float)((m + k) % 7);
    for (int n = 0; n < 16; ++n) for (int k = 0; k < 16; ++k) hBt[n*16+k] = (float)((n * 2 + k) % 5);
    for (int m = 0; m < 16; ++m) for (int n = 0; n < 16; ++n) {
        float s = 0; for (int k = 0; k < 16; ++k) s += hA[m*16+k] * hBt[n*16+k];
        ref[m*16+n] = s;
    }
    float *dA, *dBt; float* dC;
    hipMalloc(&dA, 256*sizeof(float)); hipMalloc(&dBt, 256*sizeof(float)); hipMalloc(&dC, 256*sizeof(float));
    hipMemcpy(dA, hA, 256*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dBt, hBt, 256*sizeof(float), hipMemcpyHostToDevice);
    for (int mode = 0; mode < 3; ++mode) {
        probe<<<1, 32>>>(dA, dBt, dC, mode);
        hipDeviceSynchronize();
        hipMemcpy(hC, dC, 256*sizeof(float), hipMemcpyDeviceToHost);
        int bad = 0; for (int i = 0; i < 256; ++i) if (fabsf(hC[i]-ref[i]) > 0.5f) bad++;
        printf("mode %d: %3d/256 mismatch | C[0,0]=%.0f/%.0f  C[1,2]=%.0f/%.0f  C[5,9]=%.0f/%.0f\n",
               mode, bad, hC[0], ref[0], hC[18], ref[18], hC[5*16+9], ref[5*16+9]);
    }
    return 0;
}
