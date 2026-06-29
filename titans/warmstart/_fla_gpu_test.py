import torch, time
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
import fla
from fla.utils import device as fla_device  # fla's resolved device
print("fla version", getattr(fla,"__version__","?"), "fla device ->", fla_device)
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
B,T,Hv,Dk,Dv = 1, 512, 32, 128, 128   # Qwen3.5-4B linear_attn: 32 v-heads, head 128
dev='cuda'
q=torch.randn(B,T,Hv,Dk,device=dev,dtype=torch.bfloat16)
k=torch.randn(B,T,Hv,Dk,device=dev,dtype=torch.bfloat16)
v=torch.randn(B,T,Hv,Dv,device=dev,dtype=torch.bfloat16)
g=(-torch.rand(B,T,Hv,device=dev)).float()       # log-decay, negative
beta=torch.rand(B,T,Hv,device=dev,dtype=torch.bfloat16)
o,state=chunk_gated_delta_rule(q,k,v,g=g,beta=beta,use_qk_l2norm_in_kernel=True,output_final_state=True)
torch.cuda.synchronize()
print("RAN: out", tuple(o.shape), o.device, o.dtype, "| finite", bool(torch.isfinite(o).all()))
t0=time.time()
for _ in range(30):
    o,_=chunk_gated_delta_rule(q,k,v,g=g,beta=beta,use_qk_l2norm_in_kernel=True,output_final_state=True)
torch.cuda.synchronize()
print(f"30-iter chunk_gated_delta_rule: {(time.time()-t0)*1000/30:.2f} ms/call (T={T})")
