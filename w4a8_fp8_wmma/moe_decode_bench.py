"""MoE grouped-GEMM decode microbench (op-level, no vllm) on gfx1201 host.

Measures gemm1 (the w13 projection) for the grouped kernels across decode token
counts T and block_m, at the Qwen3.6 expert shape, plus the achieved
weight-read bandwidth vs the card's ~640 GB/s peak — to see how much headroom a
decode-specialized (GEMV-style) kernel has over the WMMA tile kernels (v5/v6),
which pad M up to block_m and waste rows at small per-expert load.

  HIP_VISIBLE_DEVICES=0 python moe_decode_bench.py
  VERS=5,6,7 BLOCK_MS=16,32,64 python moe_decode_bench.py
"""
import os, time, torch
import w4a8_fp8_wmma  # noqa: F401

mmq = torch.ops.w4a8_fp8_wmma.mmq_fp8_moe_gemm
PEAK_GBs = float(os.environ.get("PEAK_GBS", "640"))  # RX 9070 XT GDDR6


def pack_uint4(w):  # (E,N,K) int -> (E,N,K//8) int32
    E, N, K = w.shape
    w = w.to(torch.int32)
    p = torch.zeros((E, N, K // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        p |= (w[:, :, i::8] & 0xF) << (i * 4)
    return p


def moe_align(topk_ids, block_m, E):
    T, top_k = topk_ids.shape
    num_valid = T * top_k
    flat = topk_ids.reshape(-1)
    sorted_ids, expert_ids = [], []
    n_active = 0
    for e in range(E):
        slots = torch.nonzero(flat == e, as_tuple=False).flatten().tolist()
        n = len(slots)
        if n == 0:
            continue
        n_active += 1
        npad = ((n + block_m - 1) // block_m) * block_m
        for i in range(npad):
            sorted_ids.append(slots[i] if i < n else num_valid)
        expert_ids.extend([e] * (npad // block_m))
    dev = topk_ids.device
    sti = torch.tensor(sorted_ids, dtype=torch.int32, device=dev)
    eids = torch.tensor(expert_ids, dtype=torch.int32, device=dev)
    ntp = torch.tensor([sti.numel()], dtype=torch.int32, device=dev)
    return sti, eids, ntp, n_active


def bench(fn, it=50, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it


def main():
    dev = "cuda"
    torch.manual_seed(0)
    E = int(os.environ.get("BENCH_E", "128"))
    hidden = int(os.environ.get("BENCH_H", "2304"))
    inter = int(os.environ.get("BENCH_I", "896"))
    g = int(os.environ.get("BENCH_G", "32"))
    top_k = int(os.environ.get("BENCH_TK", "8"))
    N13, K13 = 2 * inter, hidden
    vers = [int(v) for v in os.environ.get("VERS", "5,6").split(",")]
    block_ms = [int(b) for b in os.environ.get("BLOCK_MS", "16,32,64").split(",")]
    Ts = [int(t) for t in os.environ.get("TS", "1,2,4,8,16,32,64").split(",")]

    w13 = torch.randint(0, 16, (E, N13, K13), dtype=torch.int8, device=dev)
    w13p = pack_uint4(w13)
    s13 = torch.rand(E, N13, K13 // g, device=dev, dtype=torch.float16) * 0.02 + 0.005
    print(f"E={E} hidden={hidden} inter={inter} g={g} top_k={top_k} "
          f"N13={N13} K13={K13} peak={PEAK_GBs}GB/s dev={torch.cuda.get_device_name(0)}")
    wbytes_per_expert = N13 * K13 * 0.5  # int4
    for ver in vers:
        for bm in block_ms:
            print(f"\n--- gemm1  ver={ver}  block_m={bm} ---")
            print(f"{'T':>4} {'P':>6} {'nexp':>5} {'us':>9} {'GB/s':>7} {'%peak':>6}")
            for T in Ts:
                x = torch.randn(T, hidden, dtype=torch.float16, device=dev) * 0.5
                tids = torch.stack(
                    [torch.randperm(E, device=dev)[:top_k] for _ in range(T)]).to(torch.int32)
                sti, eids, ntp, nact = moe_align(tids, bm, E)
                P = sti.numel()
                # declaration order: (x, w_packed, scales, w_zeros, sorted_token_ids,
                #                     expert_ids, num_tokens_post_padded, top_k, block_m, version)
                try:
                    fn = lambda: mmq(x, w13p, s13, None, sti, eids, ntp, top_k, bm, ver)
                    fn()
                except Exception as e:
                    print(f"{T:>4} {P:>6} {nact:>5}  FAILED {type(e).__name__}: {e}")
                    break
                us = bench(fn) * 1e6
                gbs = nact * wbytes_per_expert / (us * 1e-6) / 1e9
                print(f"{T:>4} {P:>6} {nact:>5} {us:>9.1f} {gbs:>7.0f} {100*gbs/PEAK_GBs:>5.0f}%")


if __name__ == "__main__":
    main()
