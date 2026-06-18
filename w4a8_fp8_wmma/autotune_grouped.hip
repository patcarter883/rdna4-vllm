// autotune_grouped.hip — TileConfig autotune harness for the grouped-MoE WMMA path.
// Sweeps {version 5/6} x {BN 64,128} x {gtile (v6)} per shape on real data, times each, and emits
// a winner-cache JSON keyed by (N,K,group) — the unified mechanism that folds the scattered
// VLLM_W4A8_MOE_{BN,GTILE,...} env knobs + crossover_cache.json into one TileConfig sweep.
// Build:  /opt/rocm/bin/hipcc --offload-arch=gfx1201 -O3 -I. autotune_grouped.hip -o /tmp/autotune
// Run under a 1-card lease; redirect stdout JSON lines to tile_crossover_cache.jsonl.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "moe_gemm_tiled.h"
using namespace w4a8_tile;

#define HC(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);} }while(0)
static unsigned rng=999u; static unsigned nxt(){ rng=rng*1664525u+1013904223u; return rng; }

struct Cfg { int version; int BN; int gtile; };  // a point in the TileConfig sweep space

// time one config (ms/iter); returns -1 if the launch is invalid for the shape.
static float time_cfg(const Cfg& c, unsigned char* d_x, float* d_as, int* d_wp, __half* d_ws,
                      int* d_sid, int* d_eid, int* d_ntp, __half* d_o,
                      int T,int N,int K,int group_size,int top_k,int block_m,int P,int num_valid,int iters){
    dim3 block((block_m/16)*32);
    dim3 grid((N+c.BN-1)/c.BN, P/block_m);
    int gt = c.gtile; { const int ng=K/group_size; if(gt>ng) gt=ng; if(gt<1) gt=1; }
    auto launch = [&](){
        if (c.version==5){ size_t sm=(size_t)(block_m+c.BN)*(group_size+8);
            if(c.BN==128) moe_gemm_tiled_kernel<WmmaFp8,false,128><<<grid,block,sm>>>(d_x,d_as,d_wp,d_ws,nullptr,d_sid,d_eid,d_ntp,d_o,T,N,K,group_size,top_k,block_m,num_valid,nullptr,nullptr,0);
            else          moe_gemm_tiled_kernel<WmmaFp8,false,64 ><<<grid,block,sm>>>(d_x,d_as,d_wp,d_ws,nullptr,d_sid,d_eid,d_ntp,d_o,T,N,K,group_size,top_k,block_m,num_valid,nullptr,nullptr,0);
        } else { size_t sm=(size_t)c.BN*(group_size+8)*gt;
            if(c.BN==128) moe_gemm_tiled_v6_kernel<WmmaFp8,false,128><<<grid,block,sm>>>(d_x,d_as,d_wp,d_ws,nullptr,d_sid,d_eid,d_ntp,d_o,T,N,K,group_size,top_k,block_m,num_valid,nullptr,nullptr,0,gt);
            else          moe_gemm_tiled_v6_kernel<WmmaFp8,false,64 ><<<grid,block,sm>>>(d_x,d_as,d_wp,d_ws,nullptr,d_sid,d_eid,d_ntp,d_o,T,N,K,group_size,top_k,block_m,num_valid,nullptr,nullptr,0,gt);
        }
    };
    launch(); if (hipDeviceSynchronize()!=hipSuccess || hipGetLastError()!=hipSuccess) return -1.0f;
    for(int w=0;w<5;++w) launch(); HC(hipDeviceSynchronize());
    hipEvent_t e0,e1; HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    HC(hipEventRecord(e0)); for(int i=0;i<iters;++i) launch(); HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
    float t; HC(hipEventElapsedTime(&t,e0,e1));
    hipEventDestroy(e0); hipEventDestroy(e1);
    return t/iters;
}

static void autotune_shape(const char* tag,int N,int K,int group_size,int top_k,int block_m,int P,int iters){
    const int E=8; const int num_groups=K/group_size; const int ppr=K/8;
    const int T=P/top_k>0?P/top_k:1; const int num_valid=T*top_k;
    std::vector<unsigned char> x(T*K); std::vector<float> as(T);
    std::vector<int> wp((long)E*N*ppr); std::vector<__half> ws((long)E*N*num_groups);
    std::vector<int> sid(P), eid(P/block_m); int ntp=P;
    for(auto&b:x){ unsigned char v=nxt()&0xFF; if((v&0x7F)==0x7F) v&=0x7E; b=v; }
    for(auto&s:as) s=0.01f+(nxt()%100)*1e-4f;
    for(auto&w:wp) w=(int)nxt();
    for(auto&s:ws) s=__float2half(0.005f+(nxt()%50)*1e-4f);
    for(int r=0;r<P;++r) sid[r]=(r<num_valid)?r:num_valid+1;
    for(size_t i=0;i<eid.size();++i) eid[i]=(int)(i%E);
    unsigned char* d_x; float* d_as; int* d_wp; __half* d_ws; int* d_sid; int* d_eid; int* d_ntp; __half* d_o;
    HC(hipMalloc(&d_x,x.size()));      HC(hipMemcpy(d_x,x.data(),x.size(),hipMemcpyHostToDevice));
    HC(hipMalloc(&d_as,T*4));          HC(hipMemcpy(d_as,as.data(),T*4,hipMemcpyHostToDevice));
    HC(hipMalloc(&d_wp,wp.size()*4));  HC(hipMemcpy(d_wp,wp.data(),wp.size()*4,hipMemcpyHostToDevice));
    HC(hipMalloc(&d_ws,ws.size()*2));  HC(hipMemcpy(d_ws,ws.data(),ws.size()*2,hipMemcpyHostToDevice));
    HC(hipMalloc(&d_sid,P*4));         HC(hipMemcpy(d_sid,sid.data(),P*4,hipMemcpyHostToDevice));
    HC(hipMalloc(&d_eid,eid.size()*4));HC(hipMemcpy(d_eid,eid.data(),eid.size()*4,hipMemcpyHostToDevice));
    HC(hipMalloc(&d_ntp,4));           HC(hipMemcpy(d_ntp,&ntp,4,hipMemcpyHostToDevice));
    HC(hipMalloc(&d_o,(long)P*N*2));

    Cfg space[] = { {5,64,0},{5,128,0}, {6,64,1},{6,64,2},{6,64,4},{6,64,8},
                    {6,128,1},{6,128,2},{6,128,4},{6,128,8} };
    Cfg best{-1,-1,-1}; float best_ms=1e30f;
    for (const Cfg& c : space){
        float ms = time_cfg(c,d_x,d_as,d_wp,d_ws,d_sid,d_eid,d_ntp,d_o,T,N,K,group_size,top_k,block_m,P,num_valid,iters);
        if (ms>0 && ms<best_ms){ best_ms=ms; best=c; }
    }
    // JSON winner-cache line (keyed by shape).
    printf("{\"tag\":\"%s\",\"N\":%d,\"K\":%d,\"group\":%d,\"block_m\":%d,\"P\":%d,"
           "\"best\":{\"version\":%d,\"BN\":%d,\"gtile\":%d},\"ms\":%.5f}\n",
           tag,N,K,group_size,block_m,P,best.version,best.BN,best.gtile,best_ms);
    hipFree(d_x);hipFree(d_as);hipFree(d_wp);hipFree(d_ws);hipFree(d_sid);hipFree(d_eid);hipFree(d_ntp);hipFree(d_o);
}

int main(){
    const int it=200;
    // 35B MoE shapes across the decode->prefill P range.
    int Ps[] = {64, 256, 512, 2048};
    for (int P : Ps){
        autotune_shape("gemm1", 1024, 2048, 32, 8, 64, P, it);  // gate|up
        autotune_shape("gemm2", 2048,  512, 32, 8, 64, P, it);  // down
    }
    return 0;
}
