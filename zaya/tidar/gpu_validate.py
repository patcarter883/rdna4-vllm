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
from tidar_attn_metadata import (  # noqa: E402
    build_tidar_mask_meta,
    clear_active_tidar_mask,
    get_active_tidar_mask,
    set_active_tidar_mask,
    update_active_tidar_mask_,
    wrap_unified_attention,
)
from tidar_proposer import TidarProposer  # noqa: E402

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


# ---------------------------------------------------------------------------------------------
# E) WIRED PATH (step 3): the metadata builder + active-mask carrier + backend hook drive the real
# kernel. Part D proved the kernel honors an explicitly-passed qq_bias; Part E proves the serving
# plumbing actually delivers that qq_bias to it. We build the mask via build_tidar_mask_meta (the
# serving builder, NOT a hand-rolled slice), install it on the carrier, and call the HOOK-WRAPPED
# unified_attention WITHOUT an explicit qq_bias — exactly what the standard self.attn backend does
# (it never passes qq_bias). The wrapper must inject the active mask's qq_bias so the result both
# (a) matches boolean-masked SDPA and (b) is byte-identical to Part D's explicit-qq_bias path. With
# the carrier cleared, the wrapper must be a no-op (byte-identical to stock) — a plain decode step.
# ---------------------------------------------------------------------------------------------
def check_wired_tidar(name, block_len, prefix_len, Hq=2, Hk=2, D=128):
    scale = D ** -0.5
    meta = build_tidar_mask_meta(prefix_len, block_len, device=DEV, dtype=torch.float32)
    d = meta.descriptor
    ql, kl = d.q_len, d.kv_len
    q = torch.randn(ql, Hq, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(kl, Hk, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(kl, Hk, D, device=DEV, dtype=torch.bfloat16)

    patched_ua = _load_unified_attention(PATCHED_UA)
    wrapped_ua = wrap_unified_attention(patched_ua)

    ref = _ref_sdpa_masked(q, k, v, scale, build_allow_matrix(d, device=DEV))
    ref_b = ref.bfloat16().float()

    # (1) Wired: carrier-injected qq_bias, hook-wrapped UA, NO explicit qq_bias passed.
    set_active_tidar_mask(meta)
    try:
        got_wired = _run_unified_paged(wrapped_ua, q, k, v, scale, None)
    finally:
        clear_active_tidar_mask()
    cos, worst, n_out = _ulp_metric(got_wired, ref_b)
    ok_sdpa = (cos >= 0.9995) and (n_out <= 2)

    # (2) Byte-identical to the explicit-qq_bias path (Part D): the wrap only changed where qq_bias
    # came from, nothing about the math.
    got_explicit = _run_unified_paged(patched_ua, q, k, v, scale, meta.qq_bias)
    dmax = (got_wired - got_explicit).abs().max().item()
    ok_ident = dmax == 0.0

    ok = ok_sdpa and ok_ident
    print(f"  [{'PASS' if ok else 'FAIL'}] wired  {name:14s} q={ql:4d} kv={kl:4d}  "
          f"cos={cos:.6f} worst={worst:.2f}ULP n>6={n_out}  ==explicit(max|Δ|={dmax:.2e})")
    return ok


def check_wired_regression_nomask(name, q_len=24, prefix_len=16, Hq=2, Hk=2, D=128):
    """Carrier cleared ⇒ the hook-wrapped UA must be byte-identical to stock (a plain decode step)."""
    scale = D ** -0.5
    kl = prefix_len + q_len
    q = torch.randn(q_len, Hq, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(kl, Hk, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(kl, Hk, D, device=DEV, dtype=torch.bfloat16)
    wrapped_ua = wrap_unified_attention(_load_unified_attention(STOCK_UA))
    clear_active_tidar_mask()
    got_w = _run_unified_paged(wrapped_ua, q, k, v, scale, None)
    got_s = _run_unified_paged(_load_unified_attention(STOCK_UA), q, k, v, scale, None)
    dmax = (got_w - got_s).abs().max().item()
    ok = dmax == 0.0
    print(f"  [{'PASS' if ok else 'FAIL'}] wired regr no-mask {name:10s} byte-identical "
          f"wrapped==stock  max|Δ|={dmax:.2e}")
    return ok


# ---------------------------------------------------------------------------------------------
# F) RUNNER-PATH β=1 DECODE LOOP (proposer/runner integration — this item's headline gate).
#
# Parts D/E proved one forward through the hooked kernel honors the carrier mask. Part F proves the
# WIRED LOOP is lossless: a full β=1 TiDAR decode driven through TidarProposer (hook installed once;
# carrier set BEFORE / cleared AFTER each block forward via proposer.run_block) commits the SAME
# token stream as plain AR-greedy — token-for-token. This is the GATE-B losslessness property, now
# through the live carrier/hook + the real RDNA4 triton_attn kernel (not the standalone CPU loop).
#
# Model: a small deterministic attention LM (fixed random q/k/v/head projections over an embedding).
# Every attention call — the block forward AND the AR verify/greedy forwards — runs through the
# PATCHED unified_attention via _run_unified_paged, so the kernel under test is exactly the one
# ZayaAttention.forward -> self.attn -> triton_attn dispatches to on the serve. The block forward's
# mask is delivered ONLY by the proposer carrier (no explicit qq_bias), exercising the full step-3
# bridge; the AR forwards use no carrier (plain causal), exercising the null path each step.
#
# WHY a controlled model and not the converted checkpoint here: driving the REAL ZAYA checkpoint
# through this same triton_attn carrier on the live vLLM runner needs (a) TP=2 / a >16 GB fit (this
# item is capped at -n 1) and (b) position_ids for the fused [S|R_0..R_{B-1}] replica block — design
# §7.6, the one un-pinned off-by-one. The standalone β=1 coherence gate (coherence_gate.py, real
# checkpoint, CPU) already pins losslessness on the real weights by re-deriving R_0 from a fresh
# causal forward each step (no replica position_ids). Part F is the missing complement: it pins that
# the *wiring* (proposer carrier + hooked real kernel) is itself lossless, holding the weights fixed
# and controlled. Real-checkpoint-through-the-runner is §7.6-gated and reported as the blocker.
# ---------------------------------------------------------------------------------------------
class _AttnLM:
    """Deterministic attention next-token LM with paged-kernel attention. Single KV head, Hq query
    heads; q/k/v/head are fixed random projections. logits[t] = head(attn_out[t]). The model itself
    is irrelevant to the gate's truth (β=1 verifies against its OWN AR argmax) — it only has to be a
    consistent function whose attention goes through the kernel under test."""

    def __init__(self, vocab=48, D=128, Hq=2, Hk=2, seed=0):
        g = torch.Generator(device=DEV).manual_seed(seed)
        r = lambda *s: torch.randn(*s, generator=g, device=DEV, dtype=torch.float32) * 0.25
        self.V, self.D, self.Hq, self.Hk = vocab, D, Hq, Hk
        self.emb = r(vocab, D)
        self.Wq, self.Wk, self.Wv = r(D, Hq * D), r(D, Hk * D), r(D, Hk * D)
        self.head = r(D, vocab)
        self.scale = D ** -0.5

    def _qkv(self, toks):
        h = self.emb[torch.as_tensor(toks, device=DEV, dtype=torch.long)]   # [T, D]
        T = h.shape[0]
        q = (h @ self.Wq).view(T, self.Hq, self.D).bfloat16()
        k = (h @ self.Wk).view(T, self.Hk, self.D).bfloat16()
        v = (h @ self.Wv).view(T, self.Hk, self.D).bfloat16()
        return q, k, v

    def _logits_from_attn(self, out):                                       # out [T, Hq, D] fp32
        return out.mean(dim=1) @ self.head                                  # [T, V]

    def ar_logits(self, toks):
        """Plain causal forward through the hooked kernel (NO carrier). Returns [T, V]."""
        q, k, v = self._qkv(toks)
        clear_active_tidar_mask()
        out = _run_unified_paged(self._ua, q, k, v, self.scale, None)
        return self._logits_from_attn(out)

    def greedy(self, prompt, n_new):
        toks = list(prompt)
        for _ in range(n_new):
            toks.append(int(self.ar_logits(toks)[-1].argmax()))
        return toks

    def block_drafts(self, committed, B, mask_id):
        """Diffusion draft from a [committed | S=mask*B | R...] block forward whose TiDAR mask is
        delivered SOLELY by the proposer carrier. Read the S rows' argmax = the B drafts."""
        prefix_len = len(committed)
        prop = TidarProposer(block_len=B)
        d = MaskDescriptor(prefix_len=prefix_len, block_len=B)
        ql = d.q_len
        # Sequence = committed (prefix) + the q_len-token query block (S then replicas), all = mask_id
        # in the new region (values are irrelevant to the gate; the mask + projections define it).
        seq = list(committed) + [mask_id] * ql
        q, k, v = self._qkv(seq)
        q_new = q[prefix_len:]                                              # [ql, Hq, D] — queries
        with prop.run_block(prefix_len, device=DEV, dtype=torch.float32):   # carrier set, cleared after
            out = _run_unified_paged(self._ua, q_new, k, v, self.scale, None)
        s_logits = self._logits_from_attn(out)[:B]                         # S block rows
        return s_logits.argmax(dim=-1).tolist()


def check_runner_loop(name, B, n_new=10, prompt=(5, 9, 2, 7), seed=0, mask_id=0):
    """β=1 TiDAR decode through the proposer/hooked kernel == AR-greedy, token-for-token."""
    m = _AttnLM(seed=seed)
    m._ua = _load_unified_attention(PATCHED_UA)
    prop = TidarProposer(block_len=B)

    ar = m.greedy(list(prompt), n_new)[len(prompt):]

    committed = list(prompt)
    target = len(prompt) + n_new
    drafts = m.block_drafts(committed, B, mask_id)
    accepts = []
    while len(committed) < target:
        L = len(committed)
        # AR verify rows L-1..L+B-1 over [committed | drafts] (B+1 rows incl. the bonus row).
        full = m.ar_logits(committed + drafts)
        p_ar = full[L - 1: L + B].float()
        k, bonus = prop.verify_commit(drafts, p_ar, beta=1.0)
        committed = committed + drafts[:k] + [bonus]
        accepts.append(k)
        if len(committed) >= target:
            break
        drafts = m.block_drafts(committed, B, mask_id)
    td = committed[len(prompt):target]

    # carrier must be clean after the loop (every run_block cleared it)
    from tidar_attn_metadata import get_active_tidar_mask
    carrier_clean = get_active_tidar_mask() is None
    ok = (ar == td) and carrier_clean
    avg = sum(accepts) / len(accepts) if accepts else 0.0
    print(f"  [{'PASS' if ok else 'FAIL'}] runner-loop {name:10s} B={B} "
          f"{'IDENTICAL' if ar == td else 'DIVERGED'} carrier-clean={carrier_clean} "
          f"steps={len(accepts)} avg-accept={avg:.2f}/{B}")
    if ar != td:
        print(f"        AR   : {ar}\n        TiDAR: {td}")
    return ok


# ---------------------------------------------------------------------------------------------
# G) §31g FULL-CUDAGRAPH-CAPTURE of the carrier + hooked-kernel block forward (step 6 / §5).
#
# Parts D/E/F proved the EAGER carrier+hook path is lossless. Part G adds the one piece the eager
# path skips and that capture REQUIRES: the in-place, static-ADDRESS active-mask carrier
# (`update_active_tidar_mask_`) plus a HIP-graph capture of the hooked block forward at a fixed
# capture size, then bit-equality eager-vs-replay. This is the weight-independent §31g gate at -n 1.
#
# It mirrors Part F's surface exactly — `_AttnLM` projections, `_run_unified_paged` paged-KV layout,
# the PATCHED unified_attention via `_load_unified_attention` (the kernel ZayaAttention.forward ->
# self.attn -> triton_attn dispatches to) — so the kernel under capture is the real serving kernel.
# The block forward's qq_bias is delivered SOLELY through `update_active_tidar_mask_(...)` into a
# carrier buffer allocated ONCE for a fixed (block_len, prefix_len) and copied-in each step — NOT the
# fresh-alloc `build_tidar_mask_meta` path the eager Part F/E use.
#
# Two sub-gates:
#   G1 (outside any graph): after an in-place carrier copy-in, the hooked unified_attention output ==
#      the fresh-alloc `build_tidar_mask_meta` build path (max|Δ|=0), AND the carrier buffer's
#      id()/data_ptr() are UNCHANGED across two updates. This exercises the §5 "persistent
#      static-address buffer" contract the design names but never exercised.
#   G2 (capture==replay): capture the carrier+hooked block forward under torch.cuda.graph at a FIXED
#      q_len = block_len*(1+block_len); for a NEW step, copy new q/k/v into the static input buffers +
#      update_active_tidar_mask_ the carrier (fixed address), g.replay(), read the static out buffer.
#      GATE: replayed out == the same EAGER hooked call, bit-equal (max|Δ|=0), for ≥2 distinct
#      mask/input fills. This is the §5 "k-variability does not break capture" property: shape is
#      fixed at block_len*(1+block_len) regardless of accept-length k; only post-forward index math
#      (eviction/selection) varies, and that is not in the captured region.
#
# MEASUREMENT DISCIPLINE (memory [[profiler-bypasses-cudagraph-replay]]): we do NOT use
# torch.profiler launch-count to "prove" capture — the profiler bypasses graph replay. The honest
# weight-independent capture signal at -n 1 is eager==replay BIT-EQUALITY (a replayed graph that did
# not honor the in-place carrier swap, or that re-allocated, would diverge). The live-runner
# FULL_DECODE_ONLY dispatch probe + real throughput (zaya/dflash/{dispatch_probe.sh,
# analyze_launch_count.py}) are §7.6 / TP=2-gated and are the explicit follow-on, NOT done here.
# ---------------------------------------------------------------------------------------------
def _make_carrier_meta(block_len, prefix_len, *, device, dtype=torch.float32):
    """Build a TiDAR mask meta (fresh alloc) for (block_len, prefix_len) — the per-step source the
    in-place carrier copies FROM. Identical layout to build_tidar_mask_meta (the eager path)."""
    return build_tidar_mask_meta(prefix_len, block_len, device=device, dtype=dtype)


def _static_paged_kv(seq_len, Hk, D, dtype, block_size=16):
    """Allocate the static paged KV cache + block_table for a single seq of length seq_len (fixed
    address; we copy new k/v in each step). Mirrors _run_unified_paged's layout exactly."""
    num_blocks = (seq_len + block_size - 1) // block_size
    key_cache = torch.zeros(num_blocks, block_size, Hk, D, device=DEV, dtype=dtype)
    value_cache = torch.zeros(num_blocks, block_size, Hk, D, device=DEV, dtype=dtype)
    block_table = torch.arange(num_blocks, device=DEV, dtype=torch.int32).view(1, num_blocks)
    return key_cache, value_cache, block_table, block_size, num_blocks


def _fill_paged_kv_(key_cache, value_cache, block_size, k_all, v_all):
    """Copy k_all/v_all [seq_len, Hk, D] into the static paged cache IN PLACE (no realloc)."""
    seq_len = k_all.shape[0]
    for t in range(seq_len):
        key_cache[t // block_size, t % block_size].copy_(k_all[t])
        value_cache[t // block_size, t % block_size].copy_(v_all[t])


def check_capture_carrier_static(name, block_len, prefix_len, Hq=2, Hk=2, D=128):
    """G1: in-place static-address carrier correctness OUTSIDE any graph.

    After update_active_tidar_mask_ copies a new step's qq_bias INTO the active carrier buffer, the
    hooked unified_attention output must equal the fresh-alloc build_tidar_mask_meta path (max|Δ|=0),
    and the carrier buffer's Python id() + data_ptr() must be UNCHANGED across two in-place updates.
    """
    scale = D ** -0.5
    d = MaskDescriptor(prefix_len=prefix_len, block_len=block_len)
    ql, kl = d.q_len, d.kv_len
    q = torch.randn(ql, Hq, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(kl, Hk, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(kl, Hk, D, device=DEV, dtype=torch.bfloat16)

    patched_ua = _load_unified_attention(PATCHED_UA)
    wrapped_ua = wrap_unified_attention(patched_ua)

    # Reference: the fresh-alloc serving builder + the SAME hook (Part E's path), no in-place carrier.
    fresh = build_tidar_mask_meta(prefix_len, block_len, device=DEV, dtype=torch.float32)
    set_active_tidar_mask(fresh)
    try:
        ref = _run_unified_paged(wrapped_ua, q, k, v, scale, None)
    finally:
        clear_active_tidar_mask()

    # Carrier path: allocate ONCE (an all-zero-shaped meta), then copy-in via update_active_tidar_mask_.
    carrier = build_tidar_mask_meta(prefix_len, block_len, device=DEV, dtype=torch.float32)
    carrier.qq_bias.zero_()  # scribble so the copy-in is doing real work, not a no-op
    set_active_tidar_mask(carrier)
    try:
        held = get_active_tidar_mask()
        id0, ptr0 = id(held.qq_bias), held.qq_bias.data_ptr()
        # First in-place update with the real step's values.
        update_active_tidar_mask_(_make_carrier_meta(block_len, prefix_len, device=DEV))
        b1 = get_active_tidar_mask().qq_bias
        id1, ptr1 = id(b1), b1.data_ptr()
        got1 = _run_unified_paged(wrapped_ua, q, k, v, scale, None)
        # Second in-place update (identical values) — address must STILL be unchanged.
        update_active_tidar_mask_(_make_carrier_meta(block_len, prefix_len, device=DEV))
        b2 = get_active_tidar_mask().qq_bias
        id2, ptr2 = id(b2), b2.data_ptr()
        got2 = _run_unified_paged(wrapped_ua, q, k, v, scale, None)
    finally:
        clear_active_tidar_mask()

    addr_stable = (ptr0 == ptr1 == ptr2) and (id0 == id1 == id2)
    d1 = (got1 - ref).abs().max().item()
    d2 = (got2 - ref).abs().max().item()
    ok = addr_stable and d1 == 0.0 and d2 == 0.0
    print(f"  [{'PASS' if ok else 'FAIL'}] carrier-static {name:12s} q={ql:4d} kv={kl:4d}  "
          f"addr-stable={addr_stable} max|Δ|vs-fresh={max(d1, d2):.2e}")
    return ok


def check_capture_replay(name, block_len, prefix_len, Hq=2, Hk=2, D=128):
    """G2: capture the carrier+hooked block forward under a HIP graph at a FIXED capture size, then
    replay with new inputs/mask copied into the static buffers and assert replay == eager bit-equal.

    The captured callable is the HOOKED unified_attention reading the static-address carrier + static
    paged-KV + static q/out buffers. For each fill: copy new q/k/v into the static buffers,
    update_active_tidar_mask_ the carrier (fixed address), g.replay(), compare static out to a fresh
    EAGER hooked call on the same values.
    """
    scale = D ** -0.5
    d = MaskDescriptor(prefix_len=prefix_len, block_len=block_len)
    ql, kl = d.q_len, d.kv_len  # ql = block_len*(1+block_len) — the FIXED capture size

    patched_ua = _load_unified_attention(PATCHED_UA)
    wrapped_ua = wrap_unified_attention(patched_ua)

    # Static input/output buffers (fixed address — the graph closes over these).
    q_static = torch.zeros(ql, Hq, D, device=DEV, dtype=torch.bfloat16)
    key_cache, value_cache, block_table, block_size, _ = _static_paged_kv(kl, Hk, D, torch.bfloat16)
    cu_seqlens_q = torch.tensor([0, ql], device=DEV, dtype=torch.int32)
    seqused_k = torch.tensor([kl], device=DEV, dtype=torch.int32)
    out_static = torch.empty(ql, Hq, D, device=DEV, dtype=torch.bfloat16)

    # Carrier allocated ONCE at the fixed (block_len, prefix_len); copied-in each step.
    carrier = build_tidar_mask_meta(prefix_len, block_len, device=DEV, dtype=torch.float32)
    set_active_tidar_mask(carrier)
    carrier_ptr0 = carrier.qq_bias.data_ptr()

    def _forward():
        # Reads the static-address carrier (hook injects carrier.qq_bias) + static q/KV; writes out_static.
        wrapped_ua(
            q=q_static, k=key_cache, v=value_cache, out=out_static,
            cu_seqlens_q=cu_seqlens_q, max_seqlen_q=ql, seqused_k=seqused_k, max_seqlen_k=kl,
            softmax_scale=scale, causal=True, window_size=(-1, -1), block_table=block_table,
            softcap=0.0, q_descale=None, k_descale=None, v_descale=None, qq_bias=None,
        )

    try:
        # Warmup on a side stream (required before capture so lazy inits / autotune don't get captured).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _forward()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _forward()

        ok_all = True
        worst = 0.0
        for fi in range(2):  # >=2 distinct mask/input fills
            qf = torch.randn(ql, Hq, D, device=DEV, dtype=torch.bfloat16, generator=None)
            kf = torch.randn(kl, Hk, D, device=DEV, dtype=torch.bfloat16)
            vf = torch.randn(kl, Hk, D, device=DEV, dtype=torch.bfloat16)
            # Copy new inputs into the static buffers + update the carrier IN PLACE (fixed address).
            q_static.copy_(qf)
            _fill_paged_kv_(key_cache, value_cache, block_size, kf, vf)
            update_active_tidar_mask_(_make_carrier_meta(block_len, prefix_len, device=DEV))
            assert carrier.qq_bias.data_ptr() == carrier_ptr0, "carrier address moved across replay!"
            g.replay()
            torch.cuda.synchronize()
            replayed = out_static.float().clone()
            # Eager reference on the SAME values through the same hooked kernel (fresh-alloc internals).
            set_active_tidar_mask(carrier)  # carrier already holds this step's qq_bias
            eager = _run_unified_paged(wrapped_ua, qf, kf, vf, scale, None)
            dmax = (replayed - eager).abs().max().item()
            worst = max(worst, dmax)
            ok_all &= (dmax == 0.0)
    finally:
        clear_active_tidar_mask()

    print(f"  [{'PASS' if ok_all else 'FAIL'}] capture==replay {name:10s} q={ql:4d} kv={kl:4d}  "
          f"fixed-cap={ql} fills=2  max|Δ|eager-vs-replay={worst:.2e}")
    return ok_all


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

    print("\n=== E) WIRED path: builder + carrier + backend hook deliver qq_bias to the kernel ===")
    okE = True
    for B in (4, 8, 16):
        for P in (0, 64, 512):
            okE &= check_wired_tidar(f"B={B} P={P}", B, P)
    print("  -- regression (carrier cleared ⇒ hook is a no-op) --")
    okE &= check_wired_regression_nomask("q24 P16")
    print("  ->", "WIRED PATH CORRECT" if okE else "WIRED PATH WRONG")

    print("\n=== F) RUNNER-PATH β=1 decode loop (proposer carrier + hooked kernel) == AR-greedy ===")
    okF = True
    for B in (1, 4, 8):
        for seed in (0, 1):
            okF &= check_runner_loop(f"seed{seed}", B, n_new=10, seed=seed)
    print("  -- null-safety (carrier off / cleared ⇒ a plain decode step is byte-identical to stock) --")
    okF &= check_wired_regression_nomask("runner-null")
    print("  ->", "RUNNER LOOP LOSSLESS" if okF else "RUNNER LOOP DIVERGED")

    print("\n=== G) §31g FULL-cudagraph-capture of the carrier + hooked block forward ===")
    okG = True
    print("  -- G1: in-place static-address carrier == fresh-alloc build (max|Δ|=0), address stable --")
    for B in (4, 8):
        for P in (0, 64, 512):
            okG &= check_capture_carrier_static(f"B={B} P={P}", B, P)
    print("  -- G2: capture==replay bit-equality at fixed q_len=block_len*(1+block_len) --")
    for B in (4, 8, 16):  # 16 included if it fits (q_len=272)
        for P in (0, 64):
            okG &= check_capture_replay(f"B={B} P={P}", B, P)
    print("  ->", "CUDAGRAPH CAPTURE LOSSLESS" if okG else "CUDAGRAPH CAPTURE DIVERGED")

    print("\nRESULT:", "ALL PASS"
          if (okA and okB and okC and okD and okE and okF and okG) else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
