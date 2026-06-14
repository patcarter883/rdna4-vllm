import math, torch, cca_op
torch.manual_seed(0)
C,d,K0,K1,TP=1280,128,2,2,2
num_q,num_k=8,2; gqa=num_q//num_k; latent_q=num_q*d; S=16; NB=2048
sqrt_d=math.sqrt(d); dev="cuda"
qk_new=torch.randn(S,C,device=dev); conv_states=torch.randn(NB,C,TP,device=dev)
w0=torch.randn(C,K0,device=dev); b0=torch.randn(C,device=dev)
w1=torch.randn(C,d,K1,device=dev)*0.05; b1=torch.randn(C,device=dev)
temp_eff=torch.rand(num_k,device=dev)+0.5
H=C//d; w1=w1.view(H,d,d,K1).permute(0,2,1,3).contiguous()
slot=torch.randint(1,NB,(S,),device=dev,dtype=torch.long)
is_pad=torch.zeros(S,device=dev,dtype=torch.bool)
def once(): return torch.ops.zaya_cca.cca_decode_qk(qk_new,conv_states,slot,is_pad,w0,b0,w1,b1,temp_eff,num_q,gqa,latent_q,sqrt_d)
for _ in range(30): once()
torch.cuda.synchronize()
for _ in range(200): once()
torch.cuda.synchronize()
