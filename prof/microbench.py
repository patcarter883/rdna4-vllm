"""M=1 decode micro-bench: stock Triton vs our gfx1201-Triton vs our v10/v5 HIP op,
on Qwen2.5-Coder-7B-AWQ dense shapes. Times the KERNEL only (no serving wrapper),
to see which path the dense decode should use and whether our kernels are slow at M=1.
"""
import torch, time
import w4a8_fp8_wmma  # loads the op
from w4a8_fp8_wmma.triton_w4a16_gfx1201 import triton_w4a16_gemm_gfx1201 as tri_gfx
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import triton_w4a16_gemm as tri_stock

dev = torch.device("cuda")

def pack_nk8(w, N, K):   # (N,K)->(N,K//8) our op layout
    o = torch.zeros(N, K//8, dtype=torch.int32, device=dev)
    for j in range(8): o |= (w[:, j::8] & 0xF) << (j*4)
    return o
def pack_kn8(w, N, K):   # (N,K)->(K,N//8) triton layout
    wt = w.t().contiguous()
    o = torch.zeros(K, N//8, dtype=torch.int32, device=dev)
    for j in range(8): o |= (wt[:, j::8] & 0xF) << (j*4)
    return o

def timed(fn, it=50):
    for _ in range(8): fn()
    torch.cuda.synchronize(); s=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-s)/it*1e6  # us

SHAPES = [(4608,3584,128),(3584,3584,128),(37888,3584,128),(3584,18944,128)]
for M in (1, 8):
    print(f"\n==== M={M} (decode) ====")
    print(f"{'N,K,g':>16} | {'stock-Tri':>10} | {'gfx1201-Tri':>11} | {'v10':>8} | {'v5':>8}  (us, lower=better)")
    for N,K,g in SHAPES:
        w=torch.randint(0,16,(N,K),dtype=torch.int8,device=dev)
        x=torch.randn(M,K,dtype=torch.float16,device=dev)
        wnk=pack_nk8(w,N,K); wkn=pack_kn8(w,N,K)
        s_nk=torch.randn(N,K//g,dtype=torch.float16,device=dev).abs()*0.02+0.001
        s_kn=s_nk.t().contiguous()
        zp=torch.empty(0,dtype=torch.int32,device=dev)
        ts  = timed(lambda: tri_stock(a=x,b_q=wkn,scales=s_kn,qzeros=None,group_size=g,zp_bias=8))
        tg  = timed(lambda: tri_gfx(x,wkn,s_kn,None,g,8))
        t10 = timed(lambda: torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm(x,wnk,s_nk,zp,10))
        t5  = timed(lambda: torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm(x,wnk,s_nk,zp,5))
        print(f"{f'{N},{K},{g}':>16} | {ts:10.1f} | {tg:11.1f} | {t10:8.1f} | {t5:8.1f}")
print("\nstock-Tri = vLLM Triton W4A16 (the served stock kernel); gfx1201-Tri = our tuned Triton")
PY = None
