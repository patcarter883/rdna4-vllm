// bench_tiled_vs_v5.hip — §2.4 acceptance gate as a standalone HIP microbench (no torch).
// Launches the ORIGINAL grouped-v5 kernel and the policy-templated moe_gemm_tiled_kernel<WmmaFp8>
// on the real 35B MoE shapes, asserts BIT-EXACT output, and times both (perf-neutrality).
// Build:  /opt/rocm/bin/hipcc --offload-arch=gfx1201 -O3 -I. bench_tiled_vs_v5.hip -o /tmp/bench_tiled
// Run under a 1-card lease (see docs/GPU_LEASE.md).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "moe_gemm_tiled.h"   // w4a8_tile::moe_gemm_tiled_kernel + WmmaFp8

// ---- ORIGINAL grouped-v5 kernel, verbatim, in its own namespace (no torch) ----
namespace orig {
constexpr int PACK_FACTOR = 8; constexpr float E4M3_MAX = 448.0f; constexpr int WMMA_DIM = 16;
__device__ __forceinline__ unsigned char f32_to_e4m3(float v) {
    int p = __builtin_amdgcn_cvt_pk_fp8_f32(v, v, 0, false); return (unsigned char)(p & 0xFF); }
__device__ __forceinline__ unsigned char int4_signed_to_e4m3(int s){ return f32_to_e4m3((float)s); }
using v2i_t = int __attribute__((ext_vector_type(2)));
using v8f_t = float __attribute__((ext_vector_type(8)));
constexpr int MV5_MAX_WARPS = 8; constexpr int LDS_PAD = 8;
template<bool SCATTER, int BN>
__global__ void __launch_bounds__(MV5_MAX_WARPS*32) moe_gemm_v5_kernel(
    const unsigned char* __restrict__ x_fp8, const float* __restrict__ act_scales,
    const int* __restrict__ w_packed, const __half* __restrict__ w_scales,
    const int* __restrict__ w_zeros, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ num_tokens_post_padded,
    __half* __restrict__ out, int T, int N, int K, int group_size, int top_k, int block_m,
    int num_valid_tokens, float* __restrict__ output_scatter,
    const float* __restrict__ topk_weights, int out_top_k) {
    constexpr int NFRAG = BN / WMMA_DIM;
    const int block_idx = blockIdx.y; const int row0 = block_idx * block_m;
    if (row0 >= num_tokens_post_padded[0]) return;
    const int block_n = blockIdx.x * BN; const int tid = threadIdx.x;
    const int warp_id = tid >> 5; const int lane = tid & 31; const int lrow = lane & 15;
    const int khalf = (lane >> 4) * 8; const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8; const int n_warps = block_m / WMMA_DIM;
    if (warp_id >= n_warps) return;
    const int e = expert_ids[block_idx]; const int num_groups = K / group_size;
    const int ppr = K / PACK_FACTOR; const int* wq_e = w_packed + (long)e * N * ppr;
    const __half* ws_e = w_scales + (long)e * N * num_groups;
    const int* wz_e = (w_zeros != nullptr) ? w_zeros + (long)e * (N/8) * num_groups : nullptr;
    const int BK = group_size; const int LDSBK = BK + LDS_PAD;
    extern __shared__ unsigned char smem[]; unsigned char* A_tile = smem;
    unsigned char* B_tile = smem + block_m * LDSBK;
    float running[NFRAG][8];
    #pragma unroll
    for (int f=0; f<NFRAG; ++f) for (int ee=0; ee<8; ++ee) running[f][ee]=0.0f;
    const int total_threads = n_warps * 32;
    for (int g=0; g<num_groups; ++g) { const int k0 = g*BK;
        for (int idx=tid*4; idx<block_m*BK; idx+=total_threads*4) {
            const int r=idx/BK, k=idx%BK; unsigned int v=0u;
            const int ot=sorted_token_ids[row0+r];
            if (ot<num_valid_tokens){ const int src=SCATTER?(row0+r):(ot/top_k);
                v=*reinterpret_cast<const unsigned int*>(&x_fp8[(long)src*K+k0+k]); }
            *reinterpret_cast<unsigned int*>(&A_tile[r*LDSBK+k])=v; }
        const int njobs = BN*(BK/PACK_FACTOR);
        for (int j=tid; j<njobs; j+=total_threads) {
            const int n=j/(BK/PACK_FACTOR), k8=j%(BK/PACK_FACTOR); const int an=block_n+n;
            int word=0, zp=8;
            if (an<N){ word=wq_e[an*ppr+(k0/PACK_FACTOR)+k8];
                if (wz_e!=nullptr){ const int pk=wz_e[(an/8)*num_groups+g]; zp=(pk>>((an%8)*4))&0xF; } }
            unsigned char* dst=&B_tile[n*LDSBK+k8*PACK_FACTOR];
            #pragma unroll
            for (int jj=0; jj<PACK_FACTOR; ++jj)
                dst[jj]=(an<N)?int4_signed_to_e4m3(((word>>(jj*4))&0xF)-zp):0; }
        __syncthreads();
        v8f_t acc[NFRAG];
        #pragma unroll
        for (int f=0; f<NFRAG; ++f) acc[f]=v8f_t{0,0,0,0,0,0,0,0};
        const int k_sub=BK/WMMA_DIM;
        const int a_base=(warp_id*WMMA_DIM+lrow)*LDSBK+khalf; const int b_base=lrow*LDSBK+khalf;
        for (int kt=0; kt<k_sub; ++kt){ const int ko=kt*WMMA_DIM;
            v2i_t a_cur=*reinterpret_cast<const v2i_t*>(&A_tile[a_base+ko]);
            #pragma unroll
            for (int f=0; f<NFRAG; ++f){ v2i_t b_cur=*reinterpret_cast<const v2i_t*>(&B_tile[b_base+f*WMMA_DIM*LDSBK+ko]);
                acc[f]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_cur,b_cur,acc[f]); } }
        #pragma unroll
        for (int f=0; f<NFRAG; ++f){ const int abs_n=block_n+f*WMMA_DIM+frag_col;
            const float wsc=(abs_n<N)?__half2float(ws_e[abs_n*num_groups+g]):0.0f;
            #pragma unroll
            for (int ee=0; ee<8; ++ee) running[f][ee]+=acc[f][ee]*wsc; }
        __syncthreads(); }
    #pragma unroll
    for (int ee=0; ee<8; ++ee){ const int r=warp_id*WMMA_DIM+frag_row0+ee;
        if (r>=block_m) continue; const int row_pad=row0+r; const int ot=sorted_token_ids[row_pad];
        if (ot>=num_valid_tokens) continue; const int src=SCATTER?row_pad:(ot/top_k);
        const float asc=act_scales[src];
        #pragma unroll
        for (int f=0; f<NFRAG; ++f){ const int abs_n=block_n+f*WMMA_DIM+frag_col;
            if (abs_n<N) out[(long)row_pad*N+abs_n]=__float2half(running[f][ee]*asc); } }
}
// ---- ORIGINAL grouped-v6 kernel (A-shuffle, B-only LDS, gtile), verbatim ----
template<bool SCATTER, int BN>
__global__ void __launch_bounds__(MV5_MAX_WARPS*32) moe_gemm_v6_kernel(
    const unsigned char* __restrict__ x_fp8, const float* __restrict__ act_scales,
    const int* __restrict__ w_packed, const __half* __restrict__ w_scales,
    const int* __restrict__ w_zeros, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ num_tokens_post_padded,
    __half* __restrict__ out, int T, int N, int K, int group_size, int top_k, int block_m,
    int num_valid_tokens, float* __restrict__ output_scatter,
    const float* __restrict__ topk_weights, int out_top_k, int gtile) {
    constexpr int NFRAG = BN / WMMA_DIM;
    const int block_idx = blockIdx.y; const int row0 = block_idx * block_m;
    if (row0 >= num_tokens_post_padded[0]) return;
    const int block_n = blockIdx.x * BN; const int tid = threadIdx.x;
    const int warp_id = tid >> 5; const int lane = tid & 31; const int lrow = lane & 15;
    const int khalf = (lane >> 4) * 8; const int frag_col = lane & 15;
    const int frag_row0 = (lane >> 4) * 8; const int n_warps = block_m / WMMA_DIM;
    if (warp_id >= n_warps) return;
    const int e = expert_ids[block_idx]; const int num_groups = K / group_size;
    const int ppr = K / PACK_FACTOR; const int* wq_e = w_packed + (long)e * N * ppr;
    const __half* ws_e = w_scales + (long)e * N * num_groups;
    const int* wz_e = (w_zeros != nullptr) ? w_zeros + (long)e * (N/8) * num_groups : nullptr;
    const int BK = group_size; const int LDSBK = BK + LDS_PAD;
    extern __shared__ unsigned char smem[]; unsigned char* B_tile = smem;
    const int a_my_row = row0 + warp_id * WMMA_DIM + (lane >> 1);
    const int a_offs = sorted_token_ids[a_my_row]; const bool a_valid = a_offs < num_valid_tokens;
    const int a_src = SCATTER ? a_my_row : (a_offs / top_k);
    const int a_kh = (lane & 1) * 8; const int a_src_lane = 2 * (lane & 15) + (lane >> 4);
    float running[NFRAG][8];
    #pragma unroll
    for (int f=0; f<NFRAG; ++f) for (int ee=0; ee<8; ++ee) running[f][ee]=0.0f;
    const int total_threads = n_warps * 32; const int k_sub = BK / WMMA_DIM;
    const int b_base = lrow * LDSBK + khalf; const int njobs = BN * (BK / PACK_FACTOR);
    for (int gt=0; gt<num_groups; gt+=gtile) {
        const int gend = (gt+gtile<num_groups)?(gt+gtile):num_groups;
        for (int gi=gt; gi<gend; ++gi) { const int k0=gi*BK;
            unsigned char* Bg = B_tile + (long)(gi-gt)*BN*LDSBK;
            for (int j=tid; j<njobs; j+=total_threads) {
                const int n=j/(BK/PACK_FACTOR), k8=j%(BK/PACK_FACTOR); const int an=block_n+n;
                int word=0, zp=8;
                if (an<N){ word=wq_e[an*ppr+(k0/PACK_FACTOR)+k8];
                    if (wz_e!=nullptr){ const int pk=wz_e[(an/8)*num_groups+gi]; zp=(pk>>((an%8)*4))&0xF; } }
                unsigned char* dst=&Bg[n*LDSBK+k8*PACK_FACTOR];
                #pragma unroll
                for (int jj=0; jj<PACK_FACTOR; ++jj)
                    dst[jj]=(an<N)?int4_signed_to_e4m3(((word>>(jj*4))&0xF)-zp):0; } }
        __syncthreads();
        for (int gi=gt; gi<gend; ++gi) { const int k0=gi*BK;
            unsigned char* Bg = B_tile + (long)(gi-gt)*BN*LDSBK;
            v8f_t acc[NFRAG];
            #pragma unroll
            for (int f=0; f<NFRAG; ++f) acc[f]=v8f_t{0,0,0,0,0,0,0,0};
            for (int kt=0; kt<k_sub; ++kt) { const int gk=k0+kt*WMMA_DIM+a_kh;
                v2i_t av=v2i_t{0,0};
                if (a_valid) av=*reinterpret_cast<const v2i_t*>(&x_fp8[(long)a_src*K+gk]);
                v2i_t a_cur; a_cur[0]=__shfl(av[0],a_src_lane); a_cur[1]=__shfl(av[1],a_src_lane);
                #pragma unroll
                for (int f=0; f<NFRAG; ++f) {
                    v2i_t b_cur=*reinterpret_cast<const v2i_t*>(&Bg[b_base+f*WMMA_DIM*LDSBK+kt*WMMA_DIM]);
                    acc[f]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_cur,b_cur,acc[f]); } }
            #pragma unroll
            for (int f=0; f<NFRAG; ++f){ const int abs_n=block_n+f*WMMA_DIM+frag_col;
                const float wsc=(abs_n<N)?__half2float(ws_e[abs_n*num_groups+gi]):0.0f;
                #pragma unroll
                for (int ee=0; ee<8; ++ee) running[f][ee]+=acc[f][ee]*wsc; } }
        __syncthreads(); }
    #pragma unroll
    for (int ee=0; ee<8; ++ee){ const int r=warp_id*WMMA_DIM+frag_row0+ee;
        if (r>=block_m) continue; const int row_pad=row0+r; const int ot=sorted_token_ids[row_pad];
        if (ot>=num_valid_tokens) continue; const int src=SCATTER?row_pad:(ot/top_k);
        const float asc=act_scales[src];
        #pragma unroll
        for (int f=0; f<NFRAG; ++f){ const int abs_n=block_n+f*WMMA_DIM+frag_col;
            if (abs_n<N) out[(long)row_pad*N+abs_n]=__float2half(running[f][ee]*asc); } }
}
} // namespace orig

#define HC(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);} }while(0)

// deterministic pseudo-random
static unsigned rng=12345u; static unsigned nxt(){ rng=rng*1664525u+1013904223u; return rng; }

template<int BN>
static void run_shape(const char* tag, int N, int K, int group_size, int top_k,
                      int block_m, int P, int iters) {
    const int E = 8;                       // a few experts for the block grid
    const int num_groups = K/group_size;
    const int ppr = K/8;                    // PACK_FACTOR (8 int4 per int32)
    const int T = P/top_k > 0 ? P/top_k : 1;
    const int num_valid = T*top_k;

    // ---- host buffers ----
    std::vector<unsigned char> x_fp8(T*K);
    std::vector<float> act_scales(T);
    std::vector<int> w_packed((long)E*N*ppr);
    std::vector<__half> w_scales((long)E*N*num_groups);
    std::vector<int> sorted_ids(P), expert_ids(P/block_m);
    int ntp = P;
    // random e4m3 bytes, but avoid the two NaN codes (0x7F / 0xFF) so outputs are finite.
    for (auto& b : x_fp8) { unsigned char v = (unsigned char)(nxt() & 0xFF);
        if ((v & 0x7F) == 0x7F) v &= 0x7E; b = v; }
    for (auto& s : act_scales) s = 0.01f + (nxt()%100)*1e-4f;
    for (auto& w : w_packed) w = (int)nxt();
    for (auto& s : w_scales) s = __float2half(0.005f + (nxt()%50)*1e-4f);
    for (int r=0;r<P;++r) sorted_ids[r]= (r<num_valid)? r : num_valid+1;  // all valid here
    for (size_t i=0;i<expert_ids.size();++i) expert_ids[i]= (int)(i % E);

    // ---- device buffers ----
    unsigned char* d_x; float* d_as; int* d_wp; __half* d_ws; int* d_sid; int* d_eid; int* d_ntp;
    __half* d_o5; __half* d_ot;
    HC(hipMalloc(&d_x, x_fp8.size()));         HC(hipMemcpy(d_x,x_fp8.data(),x_fp8.size(),hipMemcpyHostToDevice));
    HC(hipMalloc(&d_as, T*sizeof(float)));     HC(hipMemcpy(d_as,act_scales.data(),T*sizeof(float),hipMemcpyHostToDevice));
    HC(hipMalloc(&d_wp, w_packed.size()*4));   HC(hipMemcpy(d_wp,w_packed.data(),w_packed.size()*4,hipMemcpyHostToDevice));
    HC(hipMalloc(&d_ws, w_scales.size()*2));   HC(hipMemcpy(d_ws,w_scales.data(),w_scales.size()*2,hipMemcpyHostToDevice));
    HC(hipMalloc(&d_sid, P*4));                HC(hipMemcpy(d_sid,sorted_ids.data(),P*4,hipMemcpyHostToDevice));
    HC(hipMalloc(&d_eid, expert_ids.size()*4));HC(hipMemcpy(d_eid,expert_ids.data(),expert_ids.size()*4,hipMemcpyHostToDevice));
    HC(hipMalloc(&d_ntp, 4));                  HC(hipMemcpy(d_ntp,&ntp,4,hipMemcpyHostToDevice));
    HC(hipMalloc(&d_o5, (long)P*N*2));         HC(hipMalloc(&d_ot, (long)P*N*2));

    dim3 block((block_m/16)*32);
    dim3 grid((N+BN-1)/BN, P/block_m);
    size_t smem = (size_t)(block_m+BN)*(group_size+8);

    auto launch5 = [&](){ orig::moe_gemm_v5_kernel<false,BN><<<grid,block,smem>>>(
        d_x,d_as,d_wp,d_ws,nullptr,d_sid,d_eid,d_ntp,d_o5,T,N,K,group_size,top_k,block_m,num_valid,nullptr,nullptr,0); };
    auto launchT = [&](){ w4a8_tile::moe_gemm_tiled_kernel<w4a8_tile::WmmaFp8,false,BN><<<grid,block,smem>>>(
        d_x,d_as,d_wp,d_ws,nullptr,d_sid,d_eid,d_ntp,d_ot,T,N,K,group_size,top_k,block_m,num_valid,nullptr,nullptr,0); };

    launch5(); launchT(); HC(hipDeviceSynchronize()); HC(hipGetLastError());

    // ---- bit-exact compare ----
    std::vector<__half> o5((long)P*N), ot((long)P*N);
    HC(hipMemcpy(o5.data(),d_o5,(long)P*N*2,hipMemcpyDeviceToHost));
    HC(hipMemcpy(ot.data(),d_ot,(long)P*N*2,hipMemcpyDeviceToHost));
    // BIT-exact compare on the raw 16-bit patterns (NaN-safe: identical bits == equal).
    long mism=0; float maxabs=0;
    for (long i=0;i<(long)P*N;++i){
        unsigned short ba=*reinterpret_cast<unsigned short*>(&o5[i]);
        unsigned short bb=*reinterpret_cast<unsigned short*>(&ot[i]);
        if (ba!=bb){ ++mism; maxabs=fmaxf(maxabs, fabsf(__half2float(o5[i])-__half2float(ot[i]))); } }

    // ---- timing ----
    hipEvent_t e0,e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    for(int w=0;w<5;++w){launch5();launchT();} HC(hipDeviceSynchronize());
    HC(hipEventRecord(e0)); for(int i=0;i<iters;++i) launch5(); HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
    float t5; HC(hipEventElapsedTime(&t5,e0,e1));
    HC(hipEventRecord(e0)); for(int i=0;i<iters;++i) launchT(); HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
    float tt; HC(hipEventElapsedTime(&tt,e0,e1));

    printf("[%s BN=%d v5] N=%d K=%d g=%d P=%d  mismatch=%ld maxabs=%.3e  v5=%.4f ms  tiled=%.4f ms  ratio=%.4f\n",
           tag, BN, N, K, group_size, P, mism, maxabs, t5/iters, tt/iters, (tt/t5));

    // ---- v6 A/B (A-shuffle + gtile=4), same buffers ----
    int gtile = 4; { const int ng = K/group_size; if (gtile > ng) gtile = ng; }
    size_t smem6 = (size_t)BN*(group_size+8)*gtile;
    auto launch6 = [&](){ orig::moe_gemm_v6_kernel<false,BN><<<grid,block,smem6>>>(
        d_x,d_as,d_wp,d_ws,nullptr,d_sid,d_eid,d_ntp,d_o5,T,N,K,group_size,top_k,block_m,num_valid,nullptr,nullptr,0,gtile); };
    auto launchT6 = [&](){ w4a8_tile::moe_gemm_tiled_v6_kernel<w4a8_tile::WmmaFp8,false,BN><<<grid,block,smem6>>>(
        d_x,d_as,d_wp,d_ws,nullptr,d_sid,d_eid,d_ntp,d_ot,T,N,K,group_size,top_k,block_m,num_valid,nullptr,nullptr,0,gtile); };
    launch6(); launchT6(); HC(hipDeviceSynchronize()); HC(hipGetLastError());
    HC(hipMemcpy(o5.data(),d_o5,(long)P*N*2,hipMemcpyDeviceToHost));
    HC(hipMemcpy(ot.data(),d_ot,(long)P*N*2,hipMemcpyDeviceToHost));
    long mism6=0; float maxabs6=0;
    for (long i=0;i<(long)P*N;++i){ unsigned short ba=*reinterpret_cast<unsigned short*>(&o5[i]),
        bb=*reinterpret_cast<unsigned short*>(&ot[i]);
        if (ba!=bb){ ++mism6; maxabs6=fmaxf(maxabs6, fabsf(__half2float(o5[i])-__half2float(ot[i]))); } }
    for(int w=0;w<5;++w){launch6();launchT6();} HC(hipDeviceSynchronize());
    HC(hipEventRecord(e0)); for(int i=0;i<iters;++i) launch6(); HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
    float t6; HC(hipEventElapsedTime(&t6,e0,e1));
    HC(hipEventRecord(e0)); for(int i=0;i<iters;++i) launchT6(); HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
    float tt6; HC(hipEventElapsedTime(&tt6,e0,e1));
    printf("[%s BN=%d v6] N=%d K=%d g=%d P=%d  mismatch=%ld maxabs=%.3e  v6=%.4f ms  tiled=%.4f ms  ratio=%.4f\n",
           tag, BN, N, K, group_size, P, mism6, maxabs6, t6/iters, tt6/iters, (tt6/t6));

    hipFree(d_x);hipFree(d_as);hipFree(d_wp);hipFree(d_ws);hipFree(d_sid);hipFree(d_eid);hipFree(d_ntp);hipFree(d_o5);hipFree(d_ot);
}

int main(){
    const int iters=200;
    // 35B MoE: gemm1 N=1024,K=2048 ; gemm2 N=2048,K=512 ; g=32, block_m=64, top_k=8.
    printf("=== prefill-ish P=512 ===\n");
    run_shape<64 >("gemm1", 1024, 2048, 32, 8, 64, 512, iters);
    run_shape<128>("gemm1", 1024, 2048, 32, 8, 64, 512, iters);
    run_shape<64 >("gemm2", 2048,  512, 32, 8, 64, 512, iters);
    run_shape<128>("gemm2", 2048,  512, 32, 8, 64, 512, iters);
    printf("=== larger P=2048 ===\n");
    run_shape<64 >("gemm1", 1024, 2048, 32, 8, 64, 2048, iters);
    run_shape<128>("gemm2", 2048,  512, 32, 8, 64, 2048, iters);
    return 0;
}
