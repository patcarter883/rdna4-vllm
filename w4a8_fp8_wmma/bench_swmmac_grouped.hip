// bench_swmmac_grouped.hip — self-consistency validation for the grouped-MoE SWMMAC sibling
// (moe_gemm_swmmac.h). No pruned MoE checkpoint exists, so correctness is proven by self-
// consistency (the validate_swmmac_real.hip method): synthesise per-expert 2:4 weights, repack to
// (Wc, idx) per RESEARCH_swmmac.md §3, and compare grouped-SWMMAC against a scalar grouped
// reference computed on the SAME 2:4-zeroed weight bytes — both via HW e4m3 + fp32 accumulate, so
// a correct operand/index construction matches near-exactly. A wrong construction is grossly off.
//   build: /opt/rocm/bin/hipcc --offload-arch=gfx1201 -O3 -I. bench_swmmac_grouped.hip -o /tmp/bsg
//   run:   under a 1-card lease (docs/GPU_LEASE.md)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "moe_gemm_swmmac.h"
using namespace w4a8_tile;

#define HC(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);} }while(0)
static unsigned rng=2024u; static unsigned nxt(){ rng=rng*1664525u+1013904223u; return rng; }
static unsigned char rand_e4m3(){ unsigned char v=nxt()&0xFF; if((v&0x7F)==0x7F) v&=0x7E; return v; }  // avoid NaN

// scalar grouped reference: reads the 2:4-ZEROED weight bytes Wz (E,N,K), e4m3->f32, sums over K.
__global__ void ref_grouped(const unsigned char* X, const float* sx, const unsigned char* Wz,
                            const float* sw, const int* sti, const int* eid, const int* ntp,
                            float* Y, int T, int N, int K, int top_k, int block_m, int num_valid) {
    const int m = blockIdx.y; const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || m >= ntp[0]) return;
    const int offs = sti[m]; if (offs >= num_valid) return;
    const int src = offs / top_k; const int e = eid[m / block_m];
    const unsigned char* Wz_en = Wz + ((size_t)e * N + n) * K;
    const unsigned char* Xs = X + (size_t)src * K;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += __builtin_amdgcn_cvt_f32_fp8((int)Xs[k], 0) * __builtin_amdgcn_cvt_f32_fp8((int)Wz_en[k], 0);
    Y[(size_t)m * N + n] = sx[src] * sw[(size_t)e * N + n] * acc;
}

static void run(const char* tag, int N, int K, int top_k, int block_m, int P) {
    const int E = 8, KT = K / 32, Khalf = K / 2;
    const int T = P / top_k > 0 ? P / top_k : 1; const int num_valid = T * top_k;

    // ---- synth activations + scales + routing ----
    std::vector<unsigned char> X((size_t)T * K); for (auto& b : X) b = rand_e4m3();
    // small scales: real W4A8 per-token/per-channel scales keep the result O(1); unscaled here the
    // K-long fp8 sum overflows the fp16 OUTPUT (rel is scale-invariant, so this only avoids inf).
    std::vector<float> sx(T); for (auto& s : sx) s = 1e-3f * (1.0f + (nxt() % 100) * 0.01f);
    std::vector<float> sw((size_t)E * N); for (auto& s : sw) s = 1e-3f * (1.0f + (nxt() % 100) * 0.01f);
    std::vector<int> sti(P), eid(P / block_m); int ntp = P;
    for (int r = 0; r < P; ++r) sti[r] = (r < num_valid) ? r : num_valid + 1;
    for (size_t i = 0; i < eid.size(); ++i) eid[i] = (int)(i % E);

    // ---- synth per-expert 2:4 weights: Wz (zeroed bytes) + repack to Wc + idx ----
    std::vector<unsigned char> Wz((size_t)E * N * K, 0);
    std::vector<unsigned char> Wc((size_t)E * N * Khalf, 0);
    std::vector<int> idx((size_t)E * N * KT, 0);
    for (int e = 0; e < E; ++e) for (int n = 0; n < N; ++n) for (int t = 0; t < KT; ++t) {
        int packed_idx = 0;
        for (int g = 0; g < 8; ++g) {                 // 8 groups of 4 within the K=32 tile
            // pick 2 distinct keep-positions p0<p1 of {0,1,2,3}
            int p0 = nxt() & 3, p1 = nxt() & 3; while (p1 == p0) p1 = nxt() & 3;
            if (p0 > p1) { int s = p0; p0 = p1; p1 = s; }
            unsigned char v0 = rand_e4m3(), v1 = rand_e4m3();
            const int kbase = t * 32 + g * 4;
            Wz[((size_t)e * N + n) * K + kbase + p0] = v0;     // kept (others stay 0 = zeroed)
            Wz[((size_t)e * N + n) * K + kbase + p1] = v1;
            const int half = g / 4, jloc = g % 4;
            unsigned char* dst = &Wc[((size_t)e * N + n) * Khalf + t * 16 + half * 8 + jloc * 2];
            dst[0] = v0; dst[1] = v1;                          // ascending-position [v@p0, v@p1]
            packed_idx |= ((p1 << 2) | p0) << (half * 16 + jloc * 4);   // nibble=(p1<<2)|p0
        }
        idx[((size_t)e * N + n) * KT + t] = packed_idx;
    }

    // ---- device ----
    unsigned char *dX, *dWz, *dWc; float *dsx, *dsw, *dYr, *dYs; int *dsti, *deid, *dntp, *didx;
    HC(hipMalloc(&dX, X.size()));   HC(hipMemcpy(dX, X.data(), X.size(), hipMemcpyHostToDevice));
    HC(hipMalloc(&dWz, Wz.size())); HC(hipMemcpy(dWz, Wz.data(), Wz.size(), hipMemcpyHostToDevice));
    HC(hipMalloc(&dWc, Wc.size())); HC(hipMemcpy(dWc, Wc.data(), Wc.size(), hipMemcpyHostToDevice));
    HC(hipMalloc(&didx, idx.size()*4)); HC(hipMemcpy(didx, idx.data(), idx.size()*4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dsx, T*4));        HC(hipMemcpy(dsx, sx.data(), T*4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dsw, sw.size()*4));HC(hipMemcpy(dsw, sw.data(), sw.size()*4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dsti, P*4));       HC(hipMemcpy(dsti, sti.data(), P*4, hipMemcpyHostToDevice));
    HC(hipMalloc(&deid, eid.size()*4)); HC(hipMemcpy(deid, eid.data(), eid.size()*4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dntp, 4));         HC(hipMemcpy(dntp, &ntp, 4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dYr, (size_t)P*N*4)); HC(hipMalloc(&dYs, (size_t)P*N*4));

    // reference
    ref_grouped<<<dim3((N+127)/128, P), 128>>>(dX, dsx, dWz, dsw, dsti, deid, dntp, dYr, T, N, K, top_k, block_m, num_valid);
    // SWMMAC: output to fp16 (non-scatter), then read back as float for compare
    __half* dYs16; HC(hipMalloc(&dYs16, (size_t)P*N*2));
    const int NWARP = block_m / 16;
    dim3 grid(N/16, P/block_m); dim3 block(NWARP*32);
    #define LAUNCH(NW) moe_gemm_swmmac_kernel<false, NW><<<grid, block>>>( \
        dX, dsx, dWc, didx, dsw, dsti, deid, dntp, dYs16, T, N, K, top_k, block_m, num_valid, nullptr, nullptr, 0)
    if (NWARP==1) LAUNCH(1); else if (NWARP==2) LAUNCH(2); else LAUNCH(4);
    #undef LAUNCH
    HC(hipDeviceSynchronize()); HC(hipGetLastError());

    std::vector<float> Yr((size_t)P*N); std::vector<__half> Ys((size_t)P*N);
    HC(hipMemcpy(Yr.data(), dYr, (size_t)P*N*4, hipMemcpyDeviceToHost));
    HC(hipMemcpy(Ys.data(), dYs16, (size_t)P*N*2, hipMemcpyDeviceToHost));
    double max_rel = 0, sum_rel = 0; long cnt = 0, bad = 0;
    for (int m = 0; m < P; ++m) { if (sti[m] >= num_valid) continue;
        for (int n = 0; n < N; ++n) {
            float r = Yr[(size_t)m*N+n], s = __half2float(Ys[(size_t)m*N+n]);
            float rel = fabsf(r - s) / (fabsf(r) + 1e-4f);
            max_rel = fmax(max_rel, rel); sum_rel += rel; ++cnt; if (rel > 0.05f) ++bad; } }
    printf("[%s] N=%d K=%d bm=%d P=%d  mean_rel=%.2e max_rel=%.2e bad(>5%%)=%ld/%ld -> %s\n",
           tag, N, K, block_m, P, sum_rel/cnt, max_rel, bad, cnt, bad==0 ? "PASS" : "FAIL");
    hipFree(dX);hipFree(dWz);hipFree(dWc);hipFree(didx);hipFree(dsx);hipFree(dsw);
    hipFree(dsti);hipFree(deid);hipFree(dntp);hipFree(dYr);hipFree(dYs);hipFree(dYs16);
}

int main() {
    // 35B MoE shapes (fp8 first rung). gemm1 N=1024 K=2048 ; gemm2 N=2048 K=512.
    run("gemm1", 1024, 2048, 8, 64, 512);
    run("gemm2", 2048,  512, 8, 64, 512);
    run("gemm1-bm32", 1024, 2048, 8, 32, 256);
    return 0;
}
