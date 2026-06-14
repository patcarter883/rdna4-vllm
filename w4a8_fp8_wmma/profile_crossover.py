"""AOT crossover profiler -> crossover_cache.json (consumed by vllm_adapter._crossover_for).

For each (N,K,group) it measures v10 (W4A8 fp8-WMMA) vs the SERVED mid-M fallback
(gfx1201-tuned Triton W4A16) and records the LOWEST M whose entire >=M suffix beats
Triton within a 2% noise margin. v10's mid-M crossover is non-monotonic and
shape-dependent, so a single threshold must be the start of the contiguous winning
*suffix* -- guaranteeing the dispatch never regresses below Triton once it engages v10.
Unknown shapes stay null -> Triton.

IMPORTANT: timing uses HIP graph replay (ms_graph), NOT eager Python loops.
Under production (cudagraphs on) stock Triton benefits from graph capture -- its
per-kernel launch overhead is amortised once, whereas our W4A8 kernel is already
GPU-compute-bound per call and sees ~no benefit. Eager timings flatter us and
produce crossovers that are too low (we win shapes in eager that lose under graphs).
ms_graph exposes the production regime for both sides so the cache is correct.

Run: HIP_VISIBLE_DEVICES=0 python profile_crossover.py   (writes crossover_cache.json)
"""
import torch, time, os, json, sys
torch.ops.load_library(os.path.join(os.path.dirname(__file__),
    "w4a8_fp8_wmma/_C.cpython-312-x86_64-linux-gnu.so"))
op = torch.ops.w4a8_fp8_wmma
sys.path.insert(0, os.path.dirname(__file__))
from triton_w4a16_ref import triton_w4a16_gemm as tri_stock
dev = 'cuda'
EPS = 1.02           # within 2% of stock counts as "not worse" (parity)
MGRID = [48, 64, 96, 128, 160, 192, 224, 256]
# (N=out, K=in) for the common dense int4 served models. Extend freely.
SHAPES = [
    (4096, 4096), (11008, 4096), (4096, 11008),       # Llama-7B
    (4096, 14336), (14336, 4096), (6144, 4096), (28672, 4096), (4096, 4096),  # Llama3-8B MLP/attn
    (5120, 5120), (13824, 5120), (5120, 13824),       # 13B
    (8192, 8192),
]
def pack_NK8(W):
    N,K=W.shape; b=torch.zeros(N,K//8,dtype=torch.int32,device=W.device)
    for j in range(8): b|=((W[:,j::8].int())&0xF)<<(j*4)
    return b
def pkt(W):
    N,K=W.shape; b=torch.zeros(K,N//8,dtype=torch.int32,device=W.device)
    for j in range(8): b|=((W[j::8,:].t().int())&0xF)<<(j*4)
    return b
def ms_graph(fn, it=200):
    # Eager warmup: ensures Triton JIT compiles and both kernels are warm in L2.
    for _ in range(20): fn()
    torch.cuda.synchronize()
    # Capture a HIP replay graph -- same API as CUDA graphs on PyTorch/ROCm.
    # The captured graph holds tensor addresses fixed; fn() must not allocate
    # inside the capture window (all tensors are pre-allocated in crossover()).
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    # Time pure replays: this is what production cudagraph mode looks like.
    t = time.perf_counter()
    for _ in range(it): g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it * 1e3
def crossover(N,K,G):
    W=torch.randint(0,16,(N,K),dtype=torch.int32,device=dev); nkg=K//G
    sc=(torch.rand(N,nkg,device=dev)*0.02+0.002).half(); tsc=sc.t().contiguous()
    wq=pack_NK8(W); bq=pkt(W); ze=torch.empty(0,dtype=torch.int32,device=dev)
    ratios={}
    for M in MGRID:
        x=(torch.randn(M,K,device=dev)*0.4).half()
        # Graph-captured timing: both kernels timed under graph replay so the
        # ratio reflects production (not eager) performance. Stock Triton's
        # graph speedup is exposed here; our kernel is graph-invariant.
        t_ours = ms_graph(lambda: op.mmq_fp8_gemm(x,wq,sc,ze,10))
        t_tri  = ms_graph(lambda: tri_stock(x,bq,tsc,None,G,8))
        ratios[M] = t_ours / t_tri
    # lowest M whose entire suffix (all M'>=M in grid) is within EPS
    best=None
    for i,M in enumerate(MGRID):
        if all(ratios[m]<=EPS for m in MGRID[i:]): best=M; break
    return best, ratios
table={}
seen=set()
for (N,K) in SHAPES:
  if N%16 or K%16: continue
  for G in (128,32):
    if K%G: continue
    key=f"{N},{K},{G}"
    if key in seen: continue
    seen.add(key)
    if G in (32,128):  # v10_ok
        co,ratios=crossover(N,K,G)
        table[key]=co
        rs=" ".join(f"M{m}:{ratios[m]:.2f}" for m in MGRID)
        print(f"{key:20s} crossover={co}  | {rs}")
# write next to vllm_adapter.py (the package dir), where _load_crossover_table looks
_pkg = os.path.join(os.path.dirname(__file__), "w4a8_fp8_wmma")
_out = os.path.join(_pkg if os.path.isdir(_pkg) else os.path.dirname(__file__),
                    "crossover_cache.json")
with open(_out, "w") as f:
    json.dump(table,f,indent=1,sort_keys=True)
print("-> wrote", _out)
print(f"\nwrote crossover_cache.json: {sum(1 for v in table.values() if v)}/{len(table)} shapes engage v10 below M=256")
