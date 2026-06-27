"""GPU validation for the TiDAR serving path (run under a 1-card lease).

Two independent checks, both weight-independent:

A) attn_hip kernel correctness in bf16-ULP units.  The parity gate's flat max|Δ|<=5e-3 trips on
   peaked causal rows where |out| reaches ~2-4 and a single bf16 ULP is ~1.5e-2 — not a kernel bug.
   We report max|Δ| (vs the bf16-rounded fp32 reference) in UNITS OF bf16 ULP at the output
   magnitude. A correct bf16 kernel is ~1-2 ULP; a real layout/mask bug is hundreds. (cosine vs
   fp32 — the engine's standing oracle — is printed alongside; a bug tanks it well below 0.999.)

B) The TiDAR additive-bias mask drives a real bf16 scaled-dot-product attention ON DEVICE identically
   (to bf16 ULP) to a boolean-masked reference — the on-GPU correctness gate for the structured-mask
   path that both attn_hip (smem_S bias) and triton_attn (Route B) will share.

Part D (Route B) loads BOTH the stock kernel (image's /app/vllm/...) and the patched overlay
(zaya/tidar/triton_overlay/triton_unified_attention.py) directly by file path, so the overlay is
NOT mounted over site-packages -- the stock file must stay pristine for the tank-check that the
unpatched kernel FAILS the bidirectional case.

Run:
  scripts/gpu-lease.sh -n 1 -- bash -c 'docker run --rm --device /dev/kfd --device /dev/dri \
    --group-add video --security-opt seccomp=unconfined --security-opt label=disable \
    --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
    -v /home/pat/code/vllm-gfx1201-tidar-serve/zaya/tidar:/tidar \
    -v /home/pat/code/vllm-gfx1201-attn-hip/attn_hip:/attn_hip \
    -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
    --entrypoint bash vllm22-w4a8:combined -lc \
    "source /app/.venv/bin/activate && PYTHONPATH=/attn_hip/..:/tidar \
     python /tidar/gpu_validate.py"'
"""
from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/tidar")
from tidar_mask import (  # noqa: E402
    MaskDescriptor,
    additive_bias,
    build_allow_matrix,
    build_square_allow_matrix,
    square_additive_bias,
)

DEV = "cuda"
torch.manual_seed(0)


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """ULP (spacing) of bf16 at each |x|: 2^(exp-7), 7 mantissa bits. Floor at the smallest normal."""
    ax = x.abs().clamp(min=1e-30)
    exp = torch.floor(torch.log2(ax))
    return torch.pow(2.0, exp - 7.0)


# ---------------------------------------------------------------------------------------------
# A) attn_hip kernel correctness, reported in bf16-ULP units
# ---------------------------------------------------------------------------------------------
def ref_attention(q, k, v, scale, causal, sliding_window):
    S, Hq, D = q.shape
    Hk = k.shape[1]
    rep = Hq // Hk
    qf = q.float().permute(1, 0, 2)
    kf = k.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    vf = v.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    attn = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    i = torch.arange(S, device=q.device)
    if causal:
        mask = i[:, None] < i[None, :]
        if sliding_window > 0:
            mask = mask | ((i[:, None] - i[None, :]) >= sliding_window)
        attn = attn.masked_fill(mask[None], float("-inf"))
    out = torch.matmul(F.softmax(attn, dim=-1), vf)
    return out.permute(1, 0, 2).contiguous()


def check_kernel(name, S, Hq, Hk, D, causal=1, sw=0) -> bool:
    import attn_hip  # noqa: F401  (registers torch.ops.attn_hip)

    scale = D ** -0.5
    q = torch.randn(S, Hq, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(S, Hk, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(S, Hk, D, device=DEV, dtype=torch.bfloat16)
    got = torch.ops.attn_hip.flash_prefill(q, k, v, scale, causal, sw).float()
    ref = ref_attention(q, k, v, scale, causal, sw)
    ref_b = ref.bfloat16().float()
    cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    # bf16-ULP-correct bound: atol covers near-zero outputs (where abs error is tiny but a pure
    # ULP-ratio explodes); rtol ~= 2 bf16 ULP covers the peaked causal rows where |out|~2-4 and a
    # single ULP is ~1.5e-2. A real layout/mask bug fails cosine AND blows past this.
    # worst error among meaningful (|out|>0.1) elements, in bf16 ULP at that element. A correct
    # fp32-internal/bf16-output flash kernel lands a few ULP off an fp32 SDPA reference on peaked
    # rows (different accumulation order); ULP_MAX=6 is the honest "correct bf16 kernel" bar. A real
    # layout/mask bug tanks cosine AND blows worst-ULP to hundreds.
    big = ref_b.abs() > 0.1
    worst_ulp = ((got - ref_b).abs() / bf16_ulp(ref_b))[big].max().item() if big.any() else 0.0
    ok = (cos >= 0.9995) and (worst_ulp <= 6.0)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} cos={cos:.6f}  worst={worst_ulp:.2f} bf16-ULP")
    return ok


# ---------------------------------------------------------------------------------------------
# B) TiDAR structured-mask bias drives bf16 attention == boolean-masked reference, on device
# ---------------------------------------------------------------------------------------------
def sdpa_with_bias(q, k, v, scale, bias):
    scores = (q.float() @ k.float().t()) * scale + bias  # bias has -inf in denied cells
    return torch.softmax(scores, dim=-1) @ v.float()


def check_mask(name, block_len, prefix_len, D=128) -> bool:
    d = MaskDescriptor(prefix_len=prefix_len, block_len=block_len)
    ql, kl = d.q_len, d.kv_len
    q = torch.randn(ql, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(kl, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(kl, D, device=DEV, dtype=torch.bfloat16)
    scale = D ** -0.5

    bias = additive_bias(d, dtype=torch.float32, device=DEV)
    got = sdpa_with_bias(q, k, v, scale, bias)

    allow = build_allow_matrix(d, device=DEV)
    ref_scores = (q.float() @ k.float().t()) * scale
    ref = torch.softmax(ref_scores.masked_fill(~allow, float("-inf")), dim=-1) @ v.float()

    # Both are fp32 paths fed identical inputs -> must match to ~fp32 eps.
    dmax = (got - ref).abs().max().item()
    ok = dmax <= 1e-4
    print(f"  [{'PASS' if ok else 'FAIL'}] mask {name:20s} q_len={ql:4d} kv={kl:4d}  "
          f"max|Δ|={dmax:.2e}  rows-masked-ok={bool(allow.any(1).all())}")
    return ok


# ---------------------------------------------------------------------------------------------
# C) The REAL attn_hip kernel, driven with a square TiDAR mask_bias, == boolean-masked reference
# ---------------------------------------------------------------------------------------------
def check_kernel_tidar(name, block_len, prefix_len, Hq=8, Hk=2, D=128) -> bool:
    import attn_hip  # noqa: F401

    d = MaskDescriptor(prefix_len=prefix_len, block_len=block_len)
    L = d.prefix_len + d.q_len
    scale = D ** -0.5
    q = torch.randn(L, Hq, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(L, Hk, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(L, Hk, D, device=DEV, dtype=torch.bfloat16)

    sq_bias = square_additive_bias(d, dtype=torch.float32, device=DEV)  # [L, L]
    got = torch.ops.attn_hip.flash_prefill(q, k, v, scale, 0, 0, sq_bias).float()  # causal=0

    # fp32 reference: GQA-expanded SDPA with the same square boolean mask.
    allow = build_square_allow_matrix(d, device=DEV)
    rep = Hq // Hk
    qf = q.float().permute(1, 0, 2)
    kf = k.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    vf = v.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    sc = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    sc = sc.masked_fill(~allow[None], float("-inf"))
    ref = torch.matmul(F.softmax(sc, dim=-1), vf).permute(1, 0, 2).contiguous()

    # Compare ONLY the new-region rows (prefix rows are computed-and-ignored). bf16 kernel -> ULP bound.
    new = slice(d.prefix_len, L)
    g, r = got[new], ref[new]
    rb = r.bfloat16().float()
    cos = F.cosine_similarity(g.flatten(), r.flatten(), dim=0).item()
    big = rb.abs() > 0.1
    worst_ulp = ((g - rb).abs() / bf16_ulp(rb))[big].max().item() if big.any() else 0.0
    ok = (cos >= 0.999) and (worst_ulp <= 6.0)  # same honest bf16 bar as Part A
    print(f"  [{'PASS' if ok else 'FAIL'}] kernel+mask {name:16s} L={L:4d}  "
          f"cos={cos:.6f}  worst={worst_ulp:.2f} bf16-ULP")
    return ok


# ---------------------------------------------------------------------------------------------
# D) triton_attn (Route B) unified_attention with the TiDAR qq_bias gate == boolean-masked SDPA
#
# The kernel applies the causal seq_mask, THEN adds qq_bias. Stock: -inf + finite = -inf, so a
# query can never attend a query-key to its RIGHT -> the replica block's bidirectional siblings are
# unreachable. The overlay (triton_overlay/triton_unified_attention.py) OR's the causal seq_mask
# back to True for keys in the query-query region, so qq_bias alone defines allow/deny there while
# PREFIX keys stay causal. This part loads BOTH the stock kernel (read-only mirror) and the patched
# overlay, drives each with the SAME paged single-sequence KV cache + qq_bias slice, and compares to
# a boolean-masked fp32 SDPA reference. PASS(patched) + FAIL(stock) on the bidirectional case proves
# the gate is what fixes it. Two regressions (qq_bias=None stock-causal; a strictly-causal qq_bias)
# assert the gate is a no-op outside the bidirectional region.
# ---------------------------------------------------------------------------------------------
import importlib.util  # noqa: E402

# Where the two kernel sources live inside the container (mounted by the run command).
STOCK_UA = "/app/vllm/vllm/v1/attention/ops/triton_unified_attention.py"  # read-only stock
PATCHED_UA = "/tidar/triton_overlay/triton_unified_attention.py"          # the gate overlay


def _load_unified_attention(path: str):
    """Import a specific triton_unified_attention.py file as a throwaway module and return its
    unified_attention fn. The module imports its helpers from the installed vllm package (unchanged),
    so only the kernel/launcher body differs between stock and patched."""
    spec = importlib.util.spec_from_file_location(f"_ua_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.unified_attention


def _kernel_has_gate(path: str) -> bool:
    with open(path) as f:
        return "is_qq" in f.read()


def _run_unified_paged(unified_attention, q_new, k_all, v_all, scale, qq_bias):
    """Run unified_attention for ONE sequence whose KV = [prefix | new-token region].

    q_new: [q_len, Hq, D]   -- queries = the new-token region only.
    k_all/v_all: [seq_len, Hk, D] -- prefix keys + new-token keys, seq_len = prefix_len + q_len.
    qq_bias: [q_len, q_len] 0/-inf over the query-query (new-token) key columns, or None.
    Returns out [q_len, Hq, D] (fp32).
    """
    seq_len, Hk, D = k_all.shape
    q_len, Hq, _ = q_new.shape
    block_size = 16
    num_blocks = (seq_len + block_size - 1) // block_size
    # Paged KV cache: [num_blocks, block_size, Hk, D]; one sequence occupying blocks 0..num_blocks-1.
    key_cache = torch.zeros(num_blocks, block_size, Hk, D, device=DEV, dtype=q_new.dtype)
    value_cache = torch.zeros(num_blocks, block_size, Hk, D, device=DEV, dtype=q_new.dtype)
    for t in range(seq_len):
        key_cache[t // block_size, t % block_size] = k_all[t]
        value_cache[t // block_size, t % block_size] = v_all[t]
    block_table = torch.arange(num_blocks, device=DEV, dtype=torch.int32).view(1, num_blocks)
    cu_seqlens_q = torch.tensor([0, q_len], device=DEV, dtype=torch.int32)
    seqused_k = torch.tensor([seq_len], device=DEV, dtype=torch.int32)
    out = torch.empty(q_len, Hq, D, device=DEV, dtype=q_new.dtype)
    unified_attention(
        q=q_new,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=q_len,
        seqused_k=seqused_k,
        max_seqlen_k=seq_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        qq_bias=qq_bias,
    )
    return out.float()


def _ref_sdpa_masked(q_new, k_all, v_all, scale, allow):
    """fp32 GQA-expanded SDPA over the [q_len, kv_len] boolean allow matrix. Returns [q_len, Hq, D]."""
    q_len, Hq, D = q_new.shape
    Hk = k_all.shape[1]
    rep = Hq // Hk
    qf = q_new.float().permute(1, 0, 2)                                  # [Hq, q_len, D]
    kf = k_all.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)    # [Hq, kv_len, D]
    vf = v_all.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    sc = torch.matmul(qf, kf.transpose(-1, -2)) * scale                  # [Hq, q_len, kv_len]
    sc = sc.masked_fill(~allow[None], float("-inf"))
    return torch.matmul(torch.softmax(sc, dim=-1), vf).permute(1, 0, 2).contiguous()


def _ulp_metric(got, ref_b):
    """(cos, worst-ULP, n>6ULP) over |ref|>0.1 elements. bf16 reference, same metric as Part A/C.

    A correct bf16 flash kernel lands a handful of ULP off an fp32 SDPA reference on the most-peaked
    rows (different accumulation order) -- Part A already shows up to 5 ULP on ragged/SWA. A real
    mask bug tanks cosine AND blows worst-ULP to hundreds-thousands across MANY elements (see the
    stock tank-check: cos 0.75-0.99, worst 32-5253 ULP). So we gate on cosine + worst-ULP + the
    COUNT of >6-ULP outliers: a few tail-rounding outliers are fine, a mask error is everywhere."""
    cos = F.cosine_similarity(got.flatten(), ref_b.flatten(), dim=0).item()
    big = ref_b.abs() > 0.1
    if not big.any():
        return cos, 0.0, 0
    ulp = ((got - ref_b).abs() / bf16_ulp(ref_b))[big]
    return cos, ulp.max().item(), int((ulp > 6.0).sum().item())


def check_unified_tidar(name, block_len, prefix_len, Hq=2, Hk=2, D=128):
    """Patched overlay == boolean-masked SDPA on a TiDAR-masked sequence; stock kernel must FAIL it."""
    d = MaskDescriptor(prefix_len=prefix_len, block_len=block_len)
    ql, kl = d.q_len, d.kv_len
    scale = D ** -0.5
    q = torch.randn(ql, Hq, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(kl, Hk, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(kl, Hk, D, device=DEV, dtype=torch.bfloat16)

    # qq_bias = the query-query (new-token) columns of the TiDAR additive bias: [q_len, q_len].
    full_bias = additive_bias(d, dtype=torch.float32, device=DEV)        # [q_len, kv_len]
    qq_bias = full_bias[:, prefix_len:kl].contiguous()                   # [q_len, q_len]

    allow = build_allow_matrix(d, device=DEV)                           # [q_len, kv_len] ground truth
    ref = _ref_sdpa_masked(q, k, v, scale, allow)
    ref_b = ref.bfloat16().float()

    ok_all = True
    for tag, ua_path in (("patched", PATCHED_UA), ("stock", STOCK_UA)):
        ua = _load_unified_attention(ua_path)
        got = _run_unified_paged(ua, q, k, v, scale, qq_bias)
        cos, worst, n_out = _ulp_metric(got, ref_b)
        # Correct bf16 kernel: high cosine + only a handful of tail-rounding outliers >6 ULP.
        # A mask bug fails cosine and produces outliers EVERYWHERE (n_out in the hundreds).
        passed = (cos >= 0.9995) and (n_out <= 2)
        if tag == "patched":
            ok = passed
            tstr = "PASS" if ok else "FAIL"
        else:
            # tank-check: stock must NOT reproduce the bidirectional region.
            bidir_exists = block_len > 1  # a 1x1 replica has no right-siblings to expose the bug
            ok = (not passed) if bidir_exists else passed
            tstr = "tank-OK(fails)" if (bidir_exists and not passed) else (
                "tank-OK(deg)" if (not bidir_exists and passed) else "tank-BAD(passes!)")
        ok_all &= ok
        print(f"  [{('PASS' if ok else 'FAIL')}] unified {tag:7s} {name:14s} "
              f"q={ql:4d} kv={kl:4d}  cos={cos:.6f} worst={worst:.2f}ULP n>6={n_out}  ({tstr})")
    return ok_all


def check_unified_regression_causal(name, q_len=24, prefix_len=16, Hq=2, Hk=2, D=128):
    """Regression: with qq_bias=None the patched gate is dead-code-eliminated (USE_QQ_BIAS False), so
    the patched and stock kernels must be BYTE-IDENTICAL on a plain causal sequence."""
    scale = D ** -0.5
    kl = prefix_len + q_len
    q = torch.randn(q_len, Hq, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(kl, Hk, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(kl, Hk, D, device=DEV, dtype=torch.bfloat16)
    got_p = _run_unified_paged(_load_unified_attention(PATCHED_UA), q, k, v, scale, None)
    got_s = _run_unified_paged(_load_unified_attention(STOCK_UA), q, k, v, scale, None)
    dmax = (got_p - got_s).abs().max().item()
    ok = dmax == 0.0
    print(f"  [{'PASS' if ok else 'FAIL'}] regr qq_bias=None  {name:14s} byte-identical "
          f"patched==stock  max|Δ|={dmax:.2e}")
    return ok


def check_unified_regression_tree_causal(name, q_len=24, prefix_len=16, Hq=2, Hk=2, D=128):
    """Regression: a strictly-causal qq_bias (tree-attn-style) must give the SAME result patched vs
    stock -- OR-ing the causal mask back on then adding an already-causal bias is a no-op there."""
    scale = D ** -0.5
    kl = prefix_len + q_len
    q = torch.randn(q_len, Hq, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(kl, Hk, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(kl, Hk, D, device=DEV, dtype=torch.bfloat16)
    # Lower-triangular 0 / upper -inf over the [q_len, q_len] query-query block: strictly causal.
    idx = torch.arange(q_len, device=DEV)
    causal_qq = torch.where(idx[:, None] >= idx[None, :], 0.0, float("-inf")).to(torch.float32)
    got_p = _run_unified_paged(_load_unified_attention(PATCHED_UA), q, k, v, scale, causal_qq)
    got_s = _run_unified_paged(_load_unified_attention(STOCK_UA), q, k, v, scale, causal_qq)
    dmax = (got_p - got_s).abs().max().item()
    ok = dmax == 0.0
    print(f"  [{'PASS' if ok else 'FAIL'}] regr causal-qq    {name:14s} no-op gate "
          f"patched==stock  max|Δ|={dmax:.2e}")
    return ok


def main():
    print("=== A) attn_hip flash_prefill correctness in bf16-ULP units ===")
    okA = True
    okA &= check_kernel("causal D128 S64  Hq16/Hk2", 64, 16, 2, 128)
    okA &= check_kernel("causal D128 S128 Hq16/Hk2", 128, 16, 2, 128)
    okA &= check_kernel("causal D128 S100 ragged   ", 100, 16, 2, 128)
    okA &= check_kernel("noncausal D128 S96 Hq8/Hk8", 96, 8, 8, 128, causal=0)
    okA &= check_kernel("SWA=64 D128 S160 Hq16/Hk2 ", 160, 16, 2, 128, sw=64)
    okA &= check_kernel("causal D64  S128 Hq8/Hk1  ", 128, 8, 1, 64)
    print("  ->", "KERNEL CORRECT (<=~2 ULP)" if okA else "KERNEL SUSPECT")

    print("\n=== B) TiDAR structured-mask bias == boolean-masked reference (bf16 inputs, fp32 SDPA) ===")
    okB = True
    for B in (4, 8, 16):
        for P in (0, 64, 512):
            okB &= check_mask(f"B={B} P={P}", B, P)
    print("  ->", "MASK BIAS CORRECT" if okB else "MASK BIAS WRONG")

    print("\n=== C) REAL attn_hip kernel + square TiDAR mask_bias == boolean-masked SDPA reference ===")
    okC = True
    for B in (4, 8):
        for P in (0, 64, 200):
            okC &= check_kernel_tidar(f"B={B} P={P}", B, P)
    print("  ->", "KERNEL+MASK CORRECT" if okC else "KERNEL+MASK WRONG")

    print("\n=== D) triton_attn unified_attention qq_bias gate == boolean-masked SDPA (Route B) ===")
    print(f"  (patched={PATCHED_UA} has gate: {_kernel_has_gate(PATCHED_UA)}; "
          f"stock={STOCK_UA} has gate: {_kernel_has_gate(STOCK_UA)})")
    okD = True
    for B in (4, 8, 16):
        for P in (0, 64, 512):
            okD &= check_unified_tidar(f"B={B} P={P}", B, P)
    print("  -- regressions (gate is a no-op outside the bidirectional region) --")
    okD &= check_unified_regression_causal("q24 P16")
    okD &= check_unified_regression_tree_causal("q24 P16")
    print("  ->", "ROUTE-B GATE CORRECT" if okD else "ROUTE-B GATE WRONG")

    print("\nRESULT:", "ALL PASS" if (okA and okB and okC and okD) else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
