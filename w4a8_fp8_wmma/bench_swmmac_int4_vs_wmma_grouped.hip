// bench_swmmac_int4_vs_wmma_grouped.hip — the SPEEDUP half of the phase-3b validation ceiling
// (TILE_FRAMEWORK_DESIGN §2.5): grouped-SWMMAC-int4 (moe_gemm_swmmac_int4_kernel, ~3 bit/wt) vs the
// grouped WMMA-DENSE *production W4A8 path* (moe_gemm_tiled_kernel<WmmaFp8>, 4 bit/wt) on the real
// 35B MoE shapes. TIMING ONLY — inputs are synthetic (correctness is proven separately by
// bench_swmmac_grouped_int4.hip); both kernels get the same shapes/routing/group-size/zero-points so
// the only difference is the GEMM backend + weight bytes. The non-grouped bench_swmmac_int4.hip
// showed 1.17-1.55x over dense W4A8; this confirms whether it survives the grouped wrapper.
//   build: /opt/rocm/bin/hipcc --offload-arch=gfx1201 -O3 -I. bench_swmmac_int4_vs_wmma_grouped.hip -o /tmp/bsivw
//   run:   under a 1-card lease (CLAUDE.md GPU sharing protocol)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "moe_gemm_tiled.h"          // grouped WMMA-dense (the W4A8 production GEMM)
#include "moe_gemm_swmmac_int4.h"    // grouped SWMMAC-int4 (the phase-3b kernel)
using namespace w4a8_tile;

#define HC(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);} }while(0)
static unsigned rng=1357u; static unsigned nxt(){ rng=rng*1664525u+1013904223u; return rng; }
static unsigned char rand_e4m3(){ unsigned char v=nxt()&0xFF; if((v&0x7F)==0x7F) v&=0x7E; return v; }

template<class F> static double bench(F f, int iters) {
    hipEvent_t evs, eve; hipEventCreate(&evs); hipEventCreate(&eve);
    f(); HC(hipDeviceSynchronize());                     // warmup
    hipEventRecord(evs); for (int i = 0; i < iters; ++i) f(); hipEventRecord(eve);
    HC(hipEventSynchronize(eve)); float ms = 0; hipEventElapsedTime(&ms, evs, eve);
    hipEventDestroy(evs); hipEventDestroy(eve); return ms / iters;
}

static void run(const char* tag, int N, int K, int group_size, int top_k, int block_m, int P, int iters) {
    const int E = 8, KT = K / 32, num_groups = K / group_size, ppr = K / 8;
    const int T = P / top_k > 0 ? P / top_k : 1; const int num_valid = T * top_k;
    constexpr int BN = 64;

    // ---- shared inputs ----
    std::vector<unsigned char> X((size_t)T * K); for (auto& b : X) b = rand_e4m3();
    std::vector<float> as(T); for (auto& s : as) s = 1e-3f;
    std::vector<__half> ws((size_t)E * N * num_groups); for (auto& s : ws) s = __float2half(1e-3f);
    std::vector<int> wz((size_t)E * (N / 8) * num_groups); for (auto& v : wz) v = (int)nxt();   // asym zp (35B)
    std::vector<int> sti(P), eid(P / block_m); int ntp = P;
    for (int r = 0; r < P; ++r) sti[r] = (r < num_valid) ? r : num_valid + 1;
    for (size_t i = 0; i < eid.size(); ++i) eid[i] = (int)(i % E);
    // ---- WMMA-dense weights (K/8 packed int4) ----
    std::vector<int> wp((size_t)E * N * ppr); for (auto& v : wp) v = (int)nxt();
    // ---- SWMMAC-int4 weights (K/4 compressed + idx) ----
    std::vector<unsigned char> Wci4((size_t)E * N * (K / 4)); for (auto& b : Wci4) b = (unsigned char)(nxt() & 0xFF);
    std::vector<int> idx((size_t)E * N * KT); for (auto& v : idx) v = (int)nxt();

    unsigned char *dX, *dWci4; float *das, *doW_s; __half *dws, *doW, *doS; int *dwz, *dwp, *didx, *dsti, *deid, *dntp;
    (void)doW_s;
    HC(hipMalloc(&dX, X.size()));   HC(hipMemcpy(dX, X.data(), X.size(), hipMemcpyHostToDevice));
    HC(hipMalloc(&das, T*4));       HC(hipMemcpy(das, as.data(), T*4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dws, ws.size()*2)); HC(hipMemcpy(dws, ws.data(), ws.size()*2, hipMemcpyHostToDevice));
    HC(hipMalloc(&dwz, wz.size()*4)); HC(hipMemcpy(dwz, wz.data(), wz.size()*4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dwp, wp.size()*4)); HC(hipMemcpy(dwp, wp.data(), wp.size()*4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dWci4, Wci4.size())); HC(hipMemcpy(dWci4, Wci4.data(), Wci4.size(), hipMemcpyHostToDevice));
    HC(hipMalloc(&didx, idx.size()*4)); HC(hipMemcpy(didx, idx.data(), idx.size()*4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dsti, P*4));       HC(hipMemcpy(dsti, sti.data(), P*4, hipMemcpyHostToDevice));
    HC(hipMalloc(&deid, eid.size()*4)); HC(hipMemcpy(deid, eid.data(), eid.size()*4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dntp, 4));         HC(hipMemcpy(dntp, &ntp, 4, hipMemcpyHostToDevice));
    HC(hipMalloc(&doW, (size_t)P*N*2)); HC(hipMalloc(&doS, (size_t)P*N*2));

    // WMMA-dense grouped (BN=64)
    dim3 gridW((N+BN-1)/BN, P/block_m); dim3 blockW((block_m/16)*32);
    size_t smem = (size_t)(block_m+BN)*(group_size+8);
    auto launchW = [&](){ moe_gemm_tiled_kernel<WmmaFp8,false,BN><<<gridW,blockW,smem>>>(
        dX,das,dwp,dws,dwz,dsti,deid,dntp,doW,T,N,K,group_size,top_k,block_m,num_valid,nullptr,nullptr,0); };
    // SWMMAC-int4 grouped (non-tiled: weight re-read per warp from DRAM)
    const int NWARP = block_m/16; dim3 gridS(N/16, P/block_m); dim3 blockS(NWARP*32);
    #define LS(NW) moe_gemm_swmmac_int4_kernel<false,NW><<<gridS,blockS>>>( \
        dX,das,dWci4,didx,dws,dwz,dsti,deid,dntp,doS,T,N,K,group_size,top_k,block_m,num_valid,nullptr,nullptr,0)
    auto launchS = [&](){ if (NWARP==1) LS(1); else if (NWARP==2) LS(2); else LS(4); };
    // SWMMAC-int4 grouped TILED (weight staged once in LDS, reused across the M-tile)
    const size_t smemT = swmmac_int4_tiled_smem_bytes(K);
    #define LT(NW) moe_gemm_swmmac_int4_tiled_kernel<false,NW><<<gridS,blockS,smemT>>>( \
        dX,das,dWci4,didx,dws,dwz,dsti,deid,dntp,doS,T,N,K,group_size,top_k,block_m,num_valid,nullptr,nullptr,0)
    auto launchT = [&](){ if (NWARP==1) LT(1); else if (NWARP==2) LT(2); else LT(4); };

    double tW = bench(launchW, iters), tS = bench(launchS, iters), tT = bench(launchT, iters);
    HC(hipGetLastError());
    printf("[%-12s] N=%-5d K=%-5d g=%d bm=%d P=%-5d | WMMA(4b) %7.4f | SWMMAC(3b) %7.4f (%.2fx) | SWMMAC-TILED %7.4f (%.2fx vs WMMA, %.2fx vs untiled)\n",
           tag, N, K, group_size, block_m, P, tW, tS, tW/tS, tT, tW/tT, tS/tT);
    #undef LS
    #undef LT
    hipFree(dX);hipFree(das);hipFree(dws);hipFree(dwz);hipFree(dwp);hipFree(dWci4);hipFree(didx);
    hipFree(dsti);hipFree(deid);hipFree(dntp);hipFree(doW);hipFree(doS);
}

int main() {
    printf("=== grouped SWMMAC-int4 (~3 bit/wt) vs grouped WMMA-dense W4A8 (4 bit/wt), 35B MoE shapes, g32 ===\n");
    // decode-ish (small P) -> prefill (large P); gemm1 K=2048, gemm2 K=512.
    for (int P : {64, 256, 512, 2048}) {
        run("gemm1", 1024, 2048, 32, 8, 64, P, P <= 512 ? 300 : 100);
        run("gemm2", 2048,  512, 32, 8, 64, P, P <= 512 ? 300 : 100);
    }
    return 0;
}
