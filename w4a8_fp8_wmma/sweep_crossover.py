import time, torch
import w4a8_fp8_wmma
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import triton_w4a16_gemm
dev='cuda'; G=128
def pack(w):
    N,K=w.shape; o=torch.zeros(N,K//8,dtype=torch.int32,device=dev)
    for j in range(8): o|=(w[:,j::8]&0xF)<<(j*4)
    return o
def bench(fn,it=30,wu=8):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/it
empty=torch.empty(0,dtype=torch.int32,device=dev)
print(f"{'N':>6} {'K':>6} | crossover M and ratio (ours/triton) per M")
for (K,N) in [(4096,2048),(4096,4096),(4096,11008),(11008,4096),(14336,4096)]:
    w4=torch.randint(0,16,(N,K),dtype=torch.int8,device=dev); owp=pack(w4)
    wt=w4.t().contiguous(); tbq=torch.zeros(K,N//8,dtype=torch.int32,device=dev)
    for j in range(8): tbq|=(wt[:,j::8]&0xF)<<(j*4)
    sc=torch.rand(N,K//G,dtype=torch.float16,device=dev)*0.02+0.001; tsc=sc.t().contiguous()
    row=[]; cross=None
    for M in [512,1024,1536,2048,2560,3072,4096]:
        x=torch.randn(M,K,dtype=torch.float16,device=dev)
        to=bench(lambda:torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm(x,owp,sc,empty,5))
        tt=bench(lambda:triton_w4a16_gemm(a=x,b_q=tbq,scales=tsc,qzeros=None,group_size=G,zp_bias=8))
        r=tt/to  # time ratio = ours_tflops/triton_tflops
        row.append(f"M{M}:{r:.2f}")
        if cross is None and r>=1.0: cross=M
    print(f"{N:>6} {K:>6} | cross~{cross} | "+" ".join(row))
