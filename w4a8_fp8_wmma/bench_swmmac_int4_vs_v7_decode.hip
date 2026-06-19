// bench_swmmac_int4_vs_v7_decode.hip — the HONEST decode-regime A/B that the phase-3b commit
// flagged as missing. The prior vs-WMMA bench (bench_swmmac_int4_vs_wmma_grouped.hip) reported a
// 12-18x SWMMAC "win" at small P, but that number is BASELINE-INFLATED: the WMMA-tiled kernel is
// weight-staging-bound at small M, and production never dispatches it there — it dispatches the
// GEMV (moe_gemv_v7_kernel, run_moe_gemm version==7). So the only meaningful decode comparison is
// SWMMAC-int4 vs v7-GEMV, both in the regime they actually run.
//
// Two things this bench gets right that the vs-WMMA bench did not:
//   1. v7 runs in its PRODUCTION config — the auto-picked nw/bk/cols/MMAX/full-K from
//      run_moe_gemm version==7 (moe_kernel.hip ~L1133-1178), lifted verbatim, NOT a naive launch.
//   2. REALISTIC decode routing — a few real tokens (T) routed top_k-ways and moe_align-padded to
//      block_m, so most padded rows are EMPTY. v7 compacts to the M_real real rows (its whole
//      point); SWMMAC computes all 16 token-rows/warp regardless. Feeding num_valid==P (every row
//      real, as the old bench did) would hide exactly that difference.
// TIMING ONLY (correctness is bench_swmmac_grouped_int4.hip). Each kernel gets its own synthetic
// weights in its native layout: v7 = dense int4 (E,N,K/8); SWMMAC = 2:4-compressed (E,N,K/4)+idx.
//   build: /opt/rocm/bin/hipcc --offload-arch=gfx1201 -O3 -I. bench_swmmac_int4_vs_v7_decode.hip -o /tmp/bsv7
//   run:   under a 1-card lease (CLAUDE.md GPU sharing protocol)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "moe_gemm_swmmac_int4.h"   // grouped SWMMAC-int4 (+ tiled) — the kernels under test
using namespace w4a8_tile;

#define HC(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);} }while(0)
static unsigned rng=99173u; static unsigned nxt(){ rng=rng*1664525u+1013904223u; return rng; }
static unsigned char rand_e4m3(){ unsigned char v=nxt()&0xFF; if((v&0x7F)==0x7F) v&=0x7E; return v; }

// ===========================================================================================
// v7 GEMV kernel — COPIED VERBATIM from moe_kernel.hip (moe_gemv_v7_kernel, ~L865-1042). It is
// self-contained (PACK_FACTOR/v4i_t/v2f_t from tile_config.h via moe_gemm_swmmac_int4.h). Kept
// here (not #included) because moe_kernel.hip pulls in ATen/torch and only builds in-container;
// this header-only bench compiles standalone with hipcc. KEEP IN SYNC if the production kernel
// changes (it is timing-only, so drift only mis-measures, never mis-serves).
// ===========================================================================================
namespace v7ref {
template<bool SCATTER, int NWARPS, int MMAX, int COLS>
__global__ void __launch_bounds__(NWARPS * 32) moe_gemv_v7_kernel(
    const unsigned char* __restrict__ x_fp8,
    const float*         __restrict__ act_scales,
    const int*           __restrict__ w_packed,
    const __half*        __restrict__ w_scales,
    const int*           __restrict__ w_zeros,
    const int*           __restrict__ sorted_token_ids,
    const int*           __restrict__ expert_ids,
    const int*           __restrict__ num_tokens_post_padded,
    __half*              __restrict__ out,
    int T, int N, int K, int group_size, int top_k, int block_m,
    int num_valid_tokens,
    float*               __restrict__ output_scatter,
    const float*         __restrict__ topk_weights,
    int out_top_k, int BK) {

    const int block_idx = blockIdx.y;
    const int row0 = block_idx * block_m;
    const int P_valid = num_tokens_post_padded[0];
    if (row0 >= P_valid) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int col_base = (blockIdx.x * NWARPS + warp_id) * COLS;

    const int e = expert_ids[block_idx];
    const int num_groups = K / group_size;
    const int ppr = K / PACK_FACTOR;
    const int* wq_e = w_packed + (long)e * N * ppr;
    const __half* ws_e = w_scales + (long)e * N * num_groups;
    const int* wz_e = (w_zeros != nullptr)
                      ? w_zeros + (long)e * (N / 8) * num_groups : nullptr;

    __shared__ int s_src[MMAX], s_pad[MMAX], s_offs[MMAX], s_mreal;
    if (tid == 0) {
        int c = 0;
        for (int r = 0; r < block_m; ++r) {
            const int offs = sorted_token_ids[row0 + r];
            if (offs < num_valid_tokens) {
                s_offs[c] = offs;
                s_pad[c] = row0 + r;
                s_src[c] = SCATTER ? (row0 + r) : (offs / top_k);
                ++c;
            }
        }
        s_mreal = c;
    }
    __syncthreads();
    const int M_real = s_mreal;

    extern __shared__ __attribute__((aligned(16))) unsigned char v7_a[];

    float acc[COLS][MMAX];
    #pragma unroll
    for (int c = 0; c < COLS; ++c)
        #pragma unroll
        for (int m = 0; m < MMAX; ++m) acc[c][m] = 0.0f;

    for (int kc = 0; kc < K; kc += BK) {
        const int ck = (K - kc < BK) ? (K - kc) : BK;
        const int tile = M_real * ck;
        for (int i = tid; i < tile; i += NWARPS * 32) {
            const int r = i / ck, kk = i - r * ck;
            v7_a[i] = x_fp8[(long)s_src[r] * K + kc + kk];
        }
        __syncthreads();
        const int ppr_chunk = ck / PACK_FACTOR;
        const int kw = kc / PACK_FACTOR;
        for (int base = lane * 4; base < ppr_chunk; base += 32 * 4) {
            const int k0 = base * PACK_FACTOR;
            const int g = (kc + k0) / group_size;
            v4i_t wv[COLS];
            float wsc_c[COLS];
            int zp_c[COLS];
            #pragma unroll
            for (int c = 0; c < COLS; ++c) {
                const int nc = col_base + c;
                if (nc < N) {
                    wv[c] = *reinterpret_cast<const v4i_t*>(&wq_e[(long)nc * ppr + kw + base]);
                    wsc_c[c] = __half2float(ws_e[(long)nc * num_groups + g]);
                    zp_c[c] = (wz_e != nullptr)
                        ? ((wz_e[(nc / 8) * num_groups + g] >> ((nc % 8) * 4)) & 0xF) : 8;
                } else {
                    wv[c] = v4i_t{0, 0, 0, 0}; wsc_c[c] = 0.0f; zp_c[c] = 8;
                }
            }
            #pragma unroll
            for (int c = 0; c < COLS; ++c) {
                if (col_base + c >= N) continue;
                const v4i_t wvc = wv[c];
                const float wsc = wsc_c[c];
                const int zp = zp_c[c];
                if constexpr (COLS == 1) {
                    float wf[32];
                    #pragma unroll
                    for (int t = 0; t < 32; ++t)
                        wf[t] = (float)(((wvc[t >> 3] >> ((t & 7) * 4)) & 0xF) - zp);
                    for (int r = 0; r < M_real; ++r) {
                        const unsigned int* ar4 =
                            reinterpret_cast<const unsigned int*>(v7_a + r * ck + k0);
                        float p = 0.0f;
                        #pragma unroll
                        for (int q = 0; q < 8; ++q) {
                            const v2f_t lo = __builtin_amdgcn_cvt_pk_f32_fp8(ar4[q], false);
                            const v2f_t hi = __builtin_amdgcn_cvt_pk_f32_fp8(ar4[q], true);
                            const int b = q * 4;
                            p += lo[0] * wf[b] + lo[1] * wf[b + 1]
                               + hi[0] * wf[b + 2] + hi[1] * wf[b + 3];
                        }
                        acc[c][r] += p * wsc;
                    }
                    continue;
                }
                for (int r = 0; r < M_real; ++r) {
                    const unsigned int* ar4 =
                        reinterpret_cast<const unsigned int*>(v7_a + r * ck + k0);
                    float p = 0.0f;
                    #pragma unroll
                    for (int q = 0; q < 8; ++q) {
                        const unsigned int a4 = ar4[q];
                        const int wword = wvc[q >> 1];
                        const int nb = (q & 1) * 4;
                        const v2f_t lo = __builtin_amdgcn_cvt_pk_f32_fp8(a4, false);
                        const v2f_t hi = __builtin_amdgcn_cvt_pk_f32_fp8(a4, true);
                        p += lo[0] * (float)(((wword >> ((nb + 0) * 4)) & 0xF) - zp);
                        p += lo[1] * (float)(((wword >> ((nb + 1) * 4)) & 0xF) - zp);
                        p += hi[0] * (float)(((wword >> ((nb + 2) * 4)) & 0xF) - zp);
                        p += hi[1] * (float)(((wword >> ((nb + 3) * 4)) & 0xF) - zp);
                    }
                    acc[c][r] += p * wsc;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int c = 0; c < COLS; ++c) {
        for (int off = 16; off > 0; off >>= 1)
            for (int r = 0; r < M_real; ++r)
                acc[c][r] += __shfl_xor(acc[c][r], off);
        const int nc = col_base + c;
        if (lane != 0 || nc >= N) continue;
        for (int r = 0; r < M_real; ++r) {
            const float asc = act_scales[s_src[r]];
            if constexpr (SCATTER) {
                const int token = s_offs[r] / out_top_k;
                const float w = topk_weights[s_offs[r]];
                atomicAdd(&output_scatter[(long)token * N + nc], w * acc[c][r] * asc);
            } else {
                out[(long)s_pad[r] * N + nc] = __float2half(acc[c][r] * asc);
            }
        }
    }
}
}  // namespace v7ref

// v7 production launcher — auto nw/bk/cols/MMAX exactly as run_moe_gemm version==7 (non-scatter).
static void launch_v7(const unsigned char* xf, const float* asc, const int* wq, const __half* wsc,
                      const int* wz, const int* sti, const int* eid, const int* ntp, __half* out,
                      int T, int N, int K, int group_size, int top_k, int block_m, int num_valid,
                      int P) {
    int nw = (K <= 1024) ? 8 : 32;
    if (nw != 4 && nw != 8 && nw != 16) nw = 32;
    const int mreal_cap = (block_m < T) ? block_m : T;
    int bk = (49152 / mreal_cap / group_size) * group_size;
    if (bk < group_size) bk = group_size;
    if (bk > K) bk = K;
    int cols = 1;
    if (K <= 1024) cols = (mreal_cap <= 8) ? 4 : (mreal_cap <= 16 ? 2 : 1);
    const int per_block = nw * cols;
    dim3 grid((N + per_block - 1) / per_block, P / block_m);
    const size_t smem7 = (size_t)mreal_cap * bk;
    #define GEMV7(NW, MM, CO) v7ref::moe_gemv_v7_kernel<false, NW, MM, CO> \
        <<<grid, dim3(NW * 32), smem7>>>( \
        xf, asc, wq, wsc, wz, sti, eid, ntp, out, T, N, K, group_size, \
        top_k, block_m, num_valid, nullptr, nullptr, 0, bk)
    #define GEMV7_CO(NW, MM) do { \
        if (cols == 8) GEMV7(NW, MM, 8); \
        else if (cols == 4) GEMV7(NW, MM, 4); \
        else if (cols == 2) GEMV7(NW, MM, 2); \
        else GEMV7(NW, MM, 1); } while (0)
    #define GEMV7_MM(NW) do { \
        if (mreal_cap <= 8) GEMV7_CO(NW, 8); \
        else if (mreal_cap <= 16) GEMV7_CO(NW, 16); \
        else if (mreal_cap <= 32) GEMV7_CO(NW, 32); \
        else GEMV7_CO(NW, 64); } while (0)
    if (nw == 4) GEMV7_MM(4);
    else if (nw == 16) GEMV7_MM(16);
    else if (nw == 32) GEMV7_MM(32);
    else GEMV7_MM(8);
    #undef GEMV7_MM
    #undef GEMV7_CO
    #undef GEMV7
}

template<class F> static double bench(F f, int iters) {
    hipEvent_t evs, eve; hipEventCreate(&evs); hipEventCreate(&eve);
    f(); HC(hipDeviceSynchronize());
    hipEventRecord(evs); for (int i = 0; i < iters; ++i) f(); hipEventRecord(eve);
    HC(hipEventSynchronize(eve)); float ms = 0; hipEventElapsedTime(&ms, evs, eve);
    hipEventDestroy(evs); hipEventDestroy(eve); return ms / iters;
}

// Build a REALISTIC moe_align layout: T tokens, each routed top_k ways (expert (t+k)%E), grouped by
// expert and padded to block_m. Returns sorted_token_ids (offs = t*top_k+k, padding = num_valid),
// expert_ids (one per block), and P (padded total). This is the decode sparsity v7 exploits.
static int build_routing(int T, int top_k, int E, int block_m,
                         std::vector<int>& sti, std::vector<int>& eid) {
    const int num_valid = T * top_k;
    std::vector<std::vector<int>> per_expert(E);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < top_k; ++k)
            per_expert[(t + k) % E].push_back(t * top_k + k);   // offs value
    sti.clear(); eid.clear();
    for (int e = 0; e < E; ++e) {
        auto& v = per_expert[e];
        if (v.empty()) continue;
        const int nblk = (int)((v.size() + block_m - 1) / block_m);
        for (int b = 0; b < nblk; ++b) {
            eid.push_back(e);
            for (int r = 0; r < block_m; ++r) {
                const size_t idx = (size_t)b * block_m + r;
                sti.push_back(idx < v.size() ? v[idx] : num_valid);   // pad = invalid
            }
        }
    }
    return (int)sti.size();   // P
}

static void run(const char* tag, int N, int K, int group_size, int top_k, int block_m, int T, int iters) {
    const int E = 8, KT = K / 32, num_groups = K / group_size, ppr = K / 8;
    std::vector<int> sti, eid;
    const int P = build_routing(T, top_k, E, block_m, sti, eid);
    const int num_valid = T * top_k;
    int ntp = P;

    std::vector<unsigned char> X((size_t)T * K); for (auto& b : X) b = rand_e4m3();
    std::vector<float> as(T); for (auto& s : as) s = 1e-3f;
    std::vector<__half> ws((size_t)E * N * num_groups); for (auto& s : ws) s = __float2half(1e-3f);
    std::vector<int> wz((size_t)E * (N / 8) * num_groups); for (auto& v : wz) v = (int)nxt();
    std::vector<int> wp((size_t)E * N * ppr); for (auto& v : wp) v = (int)nxt();             // v7 dense int4
    std::vector<unsigned char> Wci4((size_t)E * N * (K / 4)); for (auto& b : Wci4) b = (unsigned char)(nxt() & 0xFF);
    std::vector<int> idx((size_t)E * N * KT); for (auto& v : idx) v = (int)nxt();             // SWMMAC

    unsigned char *dX, *dWci4; float *das; __half *dws, *doV, *doS, *doT; int *dwz, *dwp, *didx, *dsti, *deid, *dntp;
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
    HC(hipMalloc(&doV, (size_t)P*N*2)); HC(hipMalloc(&doS, (size_t)P*N*2)); HC(hipMalloc(&doT, (size_t)P*N*2));

    // v7 GEMV (production dispatch) — dense int4 weights
    auto launchV = [&](){ launch_v7(dX, das, dwp, dws, dwz, dsti, deid, dntp, doV,
                                    T, N, K, group_size, top_k, block_m, num_valid, P); };
    // SWMMAC-int4 grouped (non-tiled) — 2:4-compressed int4 weights
    const int NWARP = block_m/16; dim3 gridS(N/16, P/block_m); dim3 blockS(NWARP*32);
    #define LS(NW) moe_gemm_swmmac_int4_kernel<false,NW><<<gridS,blockS>>>( \
        dX,das,dWci4,didx,dws,dwz,dsti,deid,dntp,doS,T,N,K,group_size,top_k,block_m,num_valid,nullptr,nullptr,0)
    auto launchS = [&](){ if (NWARP==1) LS(1); else if (NWARP==2) LS(2); else LS(4); };
    // SWMMAC-int4 TILED (LDS weight staging)
    const size_t smemT = swmmac_int4_tiled_smem_bytes(K);
    #define LT(NW) moe_gemm_swmmac_int4_tiled_kernel<false,NW><<<gridS,blockS,smemT>>>( \
        dX,das,dWci4,didx,dws,dwz,dsti,deid,dntp,doT,T,N,K,group_size,top_k,block_m,num_valid,nullptr,nullptr,0)
    auto launchT = [&](){ if (NWARP==1) LT(1); else if (NWARP==2) LT(2); else LT(4); };

    double tV = bench(launchV, iters), tS = bench(launchS, iters), tT = bench(launchT, iters);
    HC(hipGetLastError());
    printf("[%-8s] N=%-5d K=%-5d g=%d bm=%d T=%-4d P=%-5d | v7-GEMV %7.4f | SWMMAC %7.4f (%.2fx) | SWMMAC-TILED %7.4f (%.2fx)\n",
           tag, N, K, group_size, block_m, T, P, tV, tS, tV/tS, tT, tV/tT);
    #undef LS
    #undef LT
    hipFree(dX);hipFree(das);hipFree(dws);hipFree(dwz);hipFree(dwp);hipFree(dWci4);hipFree(didx);
    hipFree(dsti);hipFree(deid);hipFree(dntp);hipFree(doV);hipFree(doS);hipFree(doT);
}

int main() {
    printf("=== DECODE A/B: SWMMAC-int4 (3b, 2:4) vs v7-GEMV (4b dense), 35B MoE g32, realistic routing ===\n");
    printf("    (>1.0x in the SWMMAC columns means SWMMAC beats v7; v7 is the production decode path)\n");
    // decode batch T = 1..64 real tokens, block_m=16 (the decode moe_align block). gemm1 K=2048, gemm2 K=512.
    for (int T : {1, 4, 8, 16, 32, 64}) {
        run("gemm1", 1024, 2048, 32, 8, 16, T, 500);
        run("gemm2", 2048,  512, 32, 8, 16, T, 500);
    }
    return 0;
}
