// bench_swmmac_grouped_int4.hip — self-consistency validation for the int4-sparse + per-group-scale
// grouped-MoE SWMMAC kernel (moe_gemm_swmmac_int4.h, the production 35B rung / "phase 3b").
//
// No pruned MoE checkpoint exists, so correctness is proven by self-consistency (the
// bench_swmmac_grouped.hip / validate_swmmac_real.hip method) extended for int4 + per-group scale:
//   * synth per-(expert,channel,tile) int4 codes + a per-(channel,tile) ZERO-POINT + per-(channel,
//     tile) SCALE (group_size=32 -> one SWMMAC tile == one scale group, as on the 35B g32);
//   * 2:4-prune along K (keep 2 of every 4), pack the kept codes to (Wci4, idx) per RESEARCH_swmmac
//     §3, and pack the zero-points to wz;
//   * DEVICE-fill the zeroed dense weight Wz = e4m3(code - zp) at kept positions, 0 at dropped,
//     using the SAME int4_signed_to_e4m3 the kernel uses -> the two paths' fp8 weight bytes are
//     bit-identical by construction, so a correct grouped/index/scale wiring matches near-exactly
//     and a wrong one is grossly off.
//   * compare grouped-SWMMAC-int4 against a scalar dense reference (per-group-scaled dot of Wz·X).
//   build: /opt/rocm/bin/hipcc --offload-arch=gfx1201 -O3 -I. bench_swmmac_grouped_int4.hip -o /tmp/bsgi
//   run:   under a 1-card lease (CLAUDE.md GPU sharing protocol)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "moe_gemm_swmmac_int4.h"
using namespace w4a8_tile;

#define HC(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);} }while(0)
static unsigned rng=2024u; static unsigned nxt(){ rng=rng*1664525u+1013904223u; return rng; }
static unsigned char rand_e4m3(){ unsigned char v=nxt()&0xFF; if((v&0x7F)==0x7F) v&=0x7E; return v; }  // avoid NaN
static int rand_nib(){ return (int)(nxt() & 0xF); }   // int4 code 0..15

// Device fill of the zeroed dense weight: Wz[e][n][k] = e4m3(code - zp(e,n,k/group_size)), identical
// to the kernel's int4_signed_to_e4m3. The zero-point is per-GROUP (matching the kernel); dropped
// positions carry code==zp in Wcode -> they fill to e4m3(0)=0.
__global__ void fill_wz(const unsigned char* Wcode, const int* zpf, unsigned char* Wz,
                        int E, int N, int K, int group_size, int num_groups) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)E * N * K) return;
    const int k = (int)(i % K), g = k / group_size;
    const size_t en = i / K;
    Wz[i] = int4_signed_to_e4m3((int)Wcode[i] - zpf[en * num_groups + g]);
}

// Scalar dense reference: per-group-scaled dot of the zeroed dense weight Wz·X (e4m3->f32, fp32
// accum), folding the per-(channel,tile) scale exactly like the kernel, then the per-row act scale.
__global__ void ref_grouped_int4(const unsigned char* X, const float* sx, const unsigned char* Wz,
                                 const __half* ws, const int* sti, const int* eid, const int* ntp,
                                 float* Y, int T, int N, int K, int group_size, int top_k,
                                 int block_m, int num_valid) {
    const int m = blockIdx.y; const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || m >= ntp[0]) return;
    const int offs = sti[m]; if (offs >= num_valid) return;
    const int src = offs / top_k; const int e = eid[m / block_m];
    const int num_groups = K / group_size;
    const unsigned char* Wz_en = Wz + ((size_t)e * N + n) * K;
    const unsigned char* Xs = X + (size_t)src * K;
    const __half* ws_en = ws + ((size_t)e * N + n) * num_groups;
    float total = 0.0f;
    for (int g = 0; g < num_groups; ++g) {
        float acc = 0.0f;
        for (int kk = 0; kk < group_size; ++kk) {
            const int k = g * group_size + kk;
            acc += __builtin_amdgcn_cvt_f32_fp8((int)Xs[k], 0)
                 * __builtin_amdgcn_cvt_f32_fp8((int)Wz_en[k], 0);
        }
        total += __half2float(ws_en[g]) * acc;
    }
    Y[(size_t)m * N + n] = sx[src] * total;
}

static void run(const char* tag, int N, int K, int group_size, int top_k, int block_m, int P) {
    const int E = 8, KT = K / 32, num_groups = K / group_size;
    const int T = P / top_k > 0 ? P / top_k : 1; const int num_valid = T * top_k;

    // ---- activations + per-row act scale + routing ----
    std::vector<unsigned char> X((size_t)T * K); for (auto& b : X) b = rand_e4m3();
    std::vector<float> sx(T); for (auto& s : sx) s = 1e-3f * (1.0f + (nxt() % 100) * 0.01f);
    std::vector<int> sti(P), eid(P / block_m); int ntp = P;
    for (int r = 0; r < P; ++r) sti[r] = (r < num_valid) ? r : num_valid + 1;
    for (size_t i = 0; i < eid.size(); ++i) eid[i] = (int)(i % E);

    // ---- per-expert int4 weights: dense codes Wcode (dropped==zp) + compressed (Wci4, idx) +
    //      packed zero-points wz + per-(channel,tile) scale ws ----
    const int tiles_per_group = group_size / 32;
    std::vector<unsigned char> Wcode((size_t)E * N * K, 0);   // int4 codes 0..15 (dropped = zp)
    std::vector<unsigned char> Wci4((size_t)E * N * (K / 4), 0);
    std::vector<int> idx((size_t)E * N * KT, 0);
    std::vector<int> zpf((size_t)E * N * num_groups, 0);      // per-group plain zp for the device fill
    std::vector<int> wz((size_t)E * (N / 8) * num_groups, 0); // per-group packed zp for the kernel
    std::vector<__half> ws((size_t)E * N * num_groups);
    for (int e = 0; e < E; ++e) for (int n = 0; n < N; ++n) for (int grp = 0; grp < num_groups; ++grp) {
        const int zp = rand_nib();                            // one zero-point per (channel, group)
        zpf[((size_t)e * N + n) * num_groups + grp] = zp;
        wz[((size_t)e * (N / 8) + n / 8) * num_groups + grp] |= (zp & 0xF) << ((n % 8) * 4);
        ws[((size_t)e * N + n) * num_groups + grp] = __float2half(1e-3f * (1.0f + (nxt() % 100) * 0.01f));
        for (int tg = 0; tg < tiles_per_group; ++tg) {        // SWMMAC tiles within this group
            const int t = grp * tiles_per_group + tg;
            // pre-seed this tile's 32 dense codes to zp (so dropped positions dequant to 0)
            for (int kk = 0; kk < 32; ++kk) Wcode[((size_t)e * N + n) * K + t * 32 + kk] = (unsigned char)zp;
            int packed_idx = 0;
            for (int gg = 0; gg < 8; ++gg) {          // 8 groups-of-4 within the K=32 tile
                int p0 = nxt() & 3, p1 = nxt() & 3; while (p1 == p0) p1 = nxt() & 3;
                if (p0 > p1) { int s = p0; p0 = p1; p1 = s; }
                const int c0 = rand_nib(), c1 = rand_nib();         // kept int4 codes
                const int kbase = t * 32 + gg * 4;
                Wcode[((size_t)e * N + n) * K + kbase + p0] = (unsigned char)c0;   // kept (others = zp)
                Wcode[((size_t)e * N + n) * K + kbase + p1] = (unsigned char)c1;
                const int half = gg / 4, jloc = gg % 4;
                // compressed nibbles [code@p0, code@p1] -> 1 byte, ascending-position
                Wci4[((size_t)e * N + n) * (K / 4) + t * 8 + half * 4 + jloc] =
                    (unsigned char)((c0 & 0xF) | ((c1 & 0xF) << 4));
                packed_idx |= ((p1 << 2) | p0) << (half * 16 + jloc * 4);          // idx nibble (p1<<2)|p0
            }
            idx[((size_t)e * N + n) * KT + t] = packed_idx;
        }
    }

    // ---- device ----
    unsigned char *dX, *dWcode, *dWci4, *dWz; float *dsx, *dYr; __half *dws, *dYs16;
    int *dsti, *deid, *dntp, *didx, *dzpf, *dwz;
    HC(hipMalloc(&dX, X.size()));        HC(hipMemcpy(dX, X.data(), X.size(), hipMemcpyHostToDevice));
    HC(hipMalloc(&dWcode, Wcode.size()));HC(hipMemcpy(dWcode, Wcode.data(), Wcode.size(), hipMemcpyHostToDevice));
    HC(hipMalloc(&dWci4, Wci4.size()));  HC(hipMemcpy(dWci4, Wci4.data(), Wci4.size(), hipMemcpyHostToDevice));
    HC(hipMalloc(&dWz, (size_t)E * N * K));
    HC(hipMalloc(&didx, idx.size()*4));  HC(hipMemcpy(didx, idx.data(), idx.size()*4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dzpf, zpf.size()*4));  HC(hipMemcpy(dzpf, zpf.data(), zpf.size()*4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dwz, wz.size()*4));    HC(hipMemcpy(dwz, wz.data(), wz.size()*4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dws, ws.size()*2));    HC(hipMemcpy(dws, ws.data(), ws.size()*2, hipMemcpyHostToDevice));
    HC(hipMalloc(&dsx, T*4));            HC(hipMemcpy(dsx, sx.data(), T*4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dsti, P*4));           HC(hipMemcpy(dsti, sti.data(), P*4, hipMemcpyHostToDevice));
    HC(hipMalloc(&deid, eid.size()*4));  HC(hipMemcpy(deid, eid.data(), eid.size()*4, hipMemcpyHostToDevice));
    HC(hipMalloc(&dntp, 4));             HC(hipMemcpy(dntp, &ntp, 4, hipMemcpyHostToDevice));
    __half *dYs16t; HC(hipMalloc(&dYs16t, (size_t)P*N*2));   // tiled-kernel output
    HC(hipMalloc(&dYr, (size_t)P*N*4));  HC(hipMalloc(&dYs16, (size_t)P*N*2));

    // device-fill the zeroed dense weight (same int4->fp8 conversion as the kernel)
    fill_wz<<<((size_t)E*N*K + 255)/256, 256>>>(dWcode, dzpf, dWz, E, N, K, group_size, num_groups);
    HC(hipDeviceSynchronize()); HC(hipGetLastError());

    // reference (dense, per-group-scaled)
    ref_grouped_int4<<<dim3((N+127)/128, P), 128>>>(dX, dsx, dWz, dws, dsti, deid, dntp, dYr,
                                                    T, N, K, group_size, top_k, block_m, num_valid);
    // SWMMAC-int4 kernel -> fp16 (non-scatter)
    const int NWARP = block_m / 16;
    dim3 grid(N/16, P/block_m); dim3 block(NWARP*32);
    #define LAUNCH(NW) moe_gemm_swmmac_int4_kernel<false, NW><<<grid, block>>>( \
        dX, dsx, dWci4, didx, dws, dwz, dsti, deid, dntp, dYs16, \
        T, N, K, group_size, top_k, block_m, num_valid, nullptr, nullptr, 0)
    if (NWARP==1) LAUNCH(1); else if (NWARP==2) LAUNCH(2); else LAUNCH(4);
    #undef LAUNCH
    HC(hipDeviceSynchronize()); HC(hipGetLastError());
    // TILED SWMMAC-int4 kernel (LDS weight staging) -> fp16. Must match the reference (and thus the
    // non-tiled kernel) bit-exactly: same operands/index/scale/accumulation, only weight read site.
    const size_t smem_t = swmmac_int4_tiled_smem_bytes(K);
    #define LAUNCHT(NW) moe_gemm_swmmac_int4_tiled_kernel<false, NW><<<grid, block, smem_t>>>( \
        dX, dsx, dWci4, didx, dws, dwz, dsti, deid, dntp, dYs16t, \
        T, N, K, group_size, top_k, block_m, num_valid, nullptr, nullptr, 0)
    if (NWARP==1) LAUNCHT(1); else if (NWARP==2) LAUNCHT(2); else LAUNCHT(4);
    #undef LAUNCHT
    HC(hipDeviceSynchronize()); HC(hipGetLastError());

    std::vector<float> Yr((size_t)P*N); std::vector<__half> Ys((size_t)P*N), Yst((size_t)P*N);
    HC(hipMemcpy(Yr.data(), dYr, (size_t)P*N*4, hipMemcpyDeviceToHost));
    HC(hipMemcpy(Ys.data(), dYs16, (size_t)P*N*2, hipMemcpyDeviceToHost));
    HC(hipMemcpy(Yst.data(), dYs16t, (size_t)P*N*2, hipMemcpyDeviceToHost));
    double max_rel = 0, sum_rel = 0; long cnt = 0, bad = 0;
    double max_rel_t = 0; long bad_t = 0, mismatch_st = 0;   // tiled vs ref, and tiled-vs-nontiled
    for (int m = 0; m < P; ++m) { if (sti[m] >= num_valid) continue;
        for (int n = 0; n < N; ++n) {
            float r = Yr[(size_t)m*N+n], s = __half2float(Ys[(size_t)m*N+n]);
            float st = __half2float(Yst[(size_t)m*N+n]);
            float rel = fabsf(r - s) / (fabsf(r) + 1e-4f);
            float rel_t = fabsf(r - st) / (fabsf(r) + 1e-4f);
            max_rel = fmax(max_rel, rel); sum_rel += rel; ++cnt; if (rel > 0.05f) ++bad;
            max_rel_t = fmax(max_rel_t, rel_t); if (rel_t > 0.05f) ++bad_t;
            // bit-exact gate: tiled must equal non-tiled (same fp16 bits) — pure LDS-vs-DRAM read site
            if (__half2float(Ys[(size_t)m*N+n]) != st) ++mismatch_st; } }
    printf("[%s] N=%d K=%d g=%d bm=%d P=%d  mean_rel=%.2e max_rel=%.2e bad(>5%%)=%ld/%ld -> %s\n",
           tag, N, K, group_size, block_m, P, sum_rel/cnt, max_rel, bad, cnt, bad==0 ? "PASS" : "FAIL");
    printf("[%s-TILED] max_rel=%.2e bad(>5%%)=%ld/%ld  vs-nontiled fp16-mismatch=%ld -> %s\n",
           tag, max_rel_t, bad_t, cnt, mismatch_st,
           (bad_t==0 && mismatch_st==0) ? "PASS (bit-exact)" : "FAIL");
    hipFree(dX);hipFree(dWcode);hipFree(dWci4);hipFree(dWz);hipFree(didx);hipFree(dzpf);hipFree(dwz);
    hipFree(dws);hipFree(dsx);hipFree(dsti);hipFree(deid);hipFree(dntp);hipFree(dYr);hipFree(dYs16);
    hipFree(dYs16t);
}

int main() {
    // 35B MoE shapes, g32 (one SWMMAC tile == one scale group). gemm1 N=1024 K=2048 ; gemm2 N=2048 K=512.
    run("gemm1",      1024, 2048, 32, 8, 64, 512);
    run("gemm2",      2048,  512, 32, 8, 64, 512);
    run("gemm1-bm32", 1024, 2048, 32, 8, 32, 256);
    run("gemm1-g64",  1024, 2048, 64, 8, 64, 512);   // group spanning 2 SWMMAC tiles
    return 0;
}
