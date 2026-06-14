"""vLLM MPLinearKernel adapter for the W4A8-FP8 WMMA HIP op (gfx1201).

Routes 4-bit (compressed-tensors / GPTQ-style uint4b8, optionally AWQ uint4 with
zero points) linear layers through torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm, which
expands int4 weights to fp8 e4m3 in-register and runs RDNA4's FP8 WMMA units.

Tensor contract (compressed_tensors_wNa16-style, verified against the gfx1151 MMQ
reference):
  w_q   weight_packed   [N, K//8]        int32  (8 uint4b8 per int32, low nibble first)
  w_s   weight_scale    [N, K//group]    fp16/bf16
  w_zp  zero_points      packed [N//8, K//group] int32, or None for symmetric
  g_idx                  not supported (no activation reordering)

register() in register.py inserts this at the front of _POSSIBLE_KERNELS[ROCM].
"""
import json
import os

import torch

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

_NEVER = 1 << 30  # sentinel crossover meaning "always use Triton"
_CROSSOVER_TABLE: dict | None = None


def _force_mode() -> str:
    """Override the per-shape tuning gate (the AOT crossover cache).

    VLLM_ROCM_W4A8_FORCE:
      'auto' (default) — consult the crossover cache (never regress vs stock)
      'on'  / '1'      — ALWAYS run our W4A8 kernel (ignore the cache; for
                         measurement, or when you know it wins). Implies no
                         Triton fallback copy is needed.
      'off' / '0'      — ALWAYS use the stock Triton path (never run ours).
    Applies symmetrically to the dense (here) and MoE (moe_experts) gates."""
    v = os.environ.get("VLLM_ROCM_W4A8_FORCE", "auto").strip().lower()
    if v in ("on", "1", "true"):
        return "on"
    if v in ("off", "0", "false"):
        return "off"
    return "auto"


def _v6_band() -> tuple[int, int] | None:
    """Optional mid-M band routed to kernel v6 (b128 double-K loads).

    v6 issues TWO back-to-back WMMAs per b128 (16-fp8) LDS read, halving the LDS
    load-instruction count vs v5's b64-per-subtile. It loads BIT-IDENTICAL
    operands to v5 (a permuted LDS layout, see the v6 kernel header) so it is
    bit-exact vs v5 — only the instruction mix changes. It is the ROADMAP Task-4
    lever for the mid-M band (~512-2048) where the served path currently trails
    Triton (DIARY Act VI/VIII, 0.66-0.9x). v6 reuses v5's V2_BM=256/V2_BN=64 tile
    and needs group_size % 32 == 0 (k_sub even for the dual-subtile pairing).

    OFF by default: returns None unless BOTH band bounds are set, so the served
    dispatch is byte-identical to before until v6 is benchmarked on the target GPU
    (the pathway never regresses unmeasured). To engage, after measuring v6's
    crossover (bench_v6.py / bench_dense_vs_triton.py):
      VLLM_ROCM_W4A8_V6_MIN_M=<lo> VLLM_ROCM_W4A8_V6_MAX_M=<hi>
    routes M in [lo, hi] (with gs%32==0) to v6 instead of the v10/v5 it would
    otherwise pick. The band is closed [lo, hi]; set hi huge for an open suffix."""
    lo = os.environ.get("VLLM_ROCM_W4A8_V6_MIN_M")
    hi = os.environ.get("VLLM_ROCM_W4A8_V6_MAX_M")
    if lo is None or hi is None:
        return None
    try:
        lo_i, hi_i = int(lo), int(hi)
    except ValueError:
        return None
    if hi_i < lo_i:
        return None
    return (lo_i, hi_i)


def _layout_mode() -> str:
    """Weight-layout mode — controls whether the Triton-fallback weight copy is built.

    The Triton fallback (both stock `triton_w4a16_gemm` and our gfx1201-tuned
    `triton_w4a16_gemm_gfx1201`) reads weights in (K, N//8) packing — a *second*
    full copy alongside our native (N, K//8). The two Triton paths share the SAME
    copy (identical b_q/scales/qzeros contract), so it is one copy, not two: it
    EXACTLY doubles dense weight VRAM (same int32 byte count as native). On 16GB
    cards a ~10GB-weights model (e.g. Qwen3.6-27B at TP=2) OOMs at load building it,
    and that VRAM is otherwise KV cache -> concurrency/throughput.

    Modes (VLLM_ROCM_W4A8_LAYOUT, default 'single'):
      'single'  — single-layout (NEW DEFAULT): build NO second copy. Frees ~50% of
                  dense weight VRAM for KV cache. All M route through our v11/v10/v5
                  (apply_weights makes the ladder gap-free). Trades the dense M=16-64
                  small-band, where the gfx1201-tuned Triton wins ~1.2-1.4x at g=128
                  (and is parity at g<=64), for the VRAM — i.e. parity-or-slightly-
                  below stock ONLY in that narrow band, >= stock everywhere else.
      'tuned'   — build the (K, N//8) copy and route small-M (M<=32 & g>64) to the
                  gfx1201-tuned Triton, mid-M to stock Triton. The strictly->=-stock
                  pathway in EVERY regime (PIECE2_V*); costs 2x dense weight VRAM.
      'full'    — alias of 'tuned' (kept for clarity / future stock-only variants).

    Reversible: set VLLM_ROCM_W4A8_LAYOUT=tuned to restore the strictly->=-stock
    small-M behavior at the 2x-VRAM cost. The legacy VLLM_ROCM_W4A8_NO_TRITON_
    FALLBACK is still honored (1 -> 'single', 0 -> 'tuned') so existing deploys
    that pinned the old default behavior keep it."""
    legacy = os.environ.get("VLLM_ROCM_W4A8_NO_TRITON_FALLBACK")
    if legacy is not None:
        return "single" if legacy == "1" else "tuned"
    v = os.environ.get("VLLM_ROCM_W4A8_LAYOUT", "single").strip().lower()
    if v in ("tuned", "full"):
        return "tuned"
    return "single"


def _no_triton_fallback() -> bool:
    """True when NO Triton-fallback weight copy should be built (single-layout).

    The (K, N//8) copy is what the stock + gfx1201-tuned Triton fallbacks read; in
    single-layout mode it is skipped and ALL M route through v11/v10/v5 (the
    apply_weights ladder is gap-free). See _layout_mode for the VRAM tradeoff."""
    return _layout_mode() == "single"


def _autotune_enabled() -> bool:
    """On a crossover-cache MISS, run a quick A/B microbench at model load to
    learn this exact (N,K,group)'s crossover, then persist it (subsequent loads
    are O(1) lookups). VLLM_ROCM_W4A8_AUTOTUNE: 'on' (default) / 'off'. Off ->
    unknown shapes stay _NEVER (always stock Triton), the prior behaviour."""
    v = os.environ.get("VLLM_ROCM_W4A8_AUTOTUNE", "on").strip().lower()
    return v not in ("0", "off", "false", "no")


# Winning-suffix M grid for the load-time autotune A/B (matches profile_crossover.py
# so the cached value the autotuner writes is the same shape the AOT tuner would).
_AUTOTUNE_MGRID = (48, 64, 96, 128, 160, 192, 224, 256)
_AUTOTUNE_EPS = 1.02  # within 2% of stock counts as "not worse" (parity)


def _winning_suffix_crossover(ratios: dict, mgrid=_AUTOTUNE_MGRID,
                              eps: float = _AUTOTUNE_EPS):
    """Pure selection (no GPU): given ratios[M] = our_time / triton_time over the
    sweep grid, return the LOWEST M whose ENTIRE >=M suffix is within `eps`
    (ours never worse than stock by more than the noise margin), else None.

    v10's mid-M crossover is non-monotonic + shape-dependent (it can win at M=48,
    dip at M=96, win again at M=192), so a single threshold must be the start of
    the contiguous winning *suffix* -- the dispatch then never regresses below
    stock once it engages. Mirrors profile_crossover.crossover()'s suffix rule."""
    for i, m in enumerate(mgrid):
        if all(ratios.get(mm) is not None and ratios[mm] <= eps
               for mm in mgrid[i:]):
            return m
    return None


def _persist_crossover(N: int, K: int, group: int, value) -> None:
    """Merge {f"{N},{K},{group}": value} into the crossover cache JSON and update
    the in-process table. value is the crossover M (int) or None (= never). Best-
    effort: a write failure must not break load (we still use the value in-mem)."""
    key = f"{N},{K},{group}"
    table = _load_crossover_table()
    table[key] = value
    path = os.environ.get(
        "VLLM_ROCM_W4A8_FP8_WMMA_CACHE",
        os.path.join(os.path.dirname(__file__), "crossover_cache.json"))
    try:
        with open(path, "w") as f:
            json.dump(table, f, indent=1, sort_keys=True)
    except OSError:
        pass


def _autotune_crossover(layer: torch.nn.Module, w_q_name: str,
                        N: int, K: int, group: int):
    """Load-time A/B microbench (GPU): time our v10 fp8-WMMA vs the served stock
    Triton W4A16 across _AUTOTUNE_MGRID for THIS exact (N,K,group), pick the
    winning-suffix crossover, persist it, and return it. ROBUSTNESS: ANY failure
    returns None (-> caller falls back to _NEVER = stock) so the served pathway
    never regresses. Reuses the weights already on `layer` (no extra VRAM).

    Mirrors profile_crossover.py's measurement (v10 vs stock Triton, 2% suffix
    rule) but at load time for the actual served shape, instead of an AOT sweep
    over a hand-listed shape table. v10 needs group in {32,128}; for any other
    group it returns None (the caller leaves _NEVER, as before)."""
    import time

    if group not in (32, 128):
        return None  # v10 (the benched kernel) is compiled only for these groups

    import w4a8_fp8_wmma  # noqa: F401  (ensure the op is loaded)
    from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
        triton_w4a16_gemm,
    )

    w_q = getattr(layer, w_q_name).data                # (N, K//8)
    w_s = layer._w4a8_fp8_w_s                           # (N, K//group) fp16
    tri_bq = getattr(layer, "_w4a8_tri_bq", None)      # (K, N//8)
    tri_s = getattr(layer, "_w4a8_tri_s", None)        # (K//group, N)
    if tri_bq is None or tri_s is None:
        return None  # no Triton-layout weights to A/B against (single-layout mode)
    tri_zp = getattr(layer, "_w4a8_tri_zp", None)
    dev = w_q.device
    empty = torch.empty(0, dtype=torch.int32, device=dev)

    def _ms(fn, it=50):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(it):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / it

    ratios = {}
    for m in _AUTOTUNE_MGRID:
        x = (torch.randn(m, K, device=dev) * 0.4).to(torch.float16)
        t_o = _ms(lambda: torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm(x, w_q, w_s, empty, 10))
        t_t = _ms(lambda: triton_w4a16_gemm(x, tri_bq, tri_s, tri_zp, group, 8))
        ratios[m] = t_o / t_t if t_t > 0 else None
    co = _winning_suffix_crossover(ratios)
    _persist_crossover(N, K, group, co)
    return co


def _load_crossover_table() -> dict:
    """Load the AOT crossover cache (O(1) per-shape Triton<->FP8 thresholds).
    Path: $VLLM_ROCM_W4A8_FP8_WMMA_CACHE or crossover_cache.json next to this
    file. Keys are "N,K,group" -> crossover M (or null = never)."""
    global _CROSSOVER_TABLE
    if _CROSSOVER_TABLE is not None:
        return _CROSSOVER_TABLE
    path = os.environ.get(
        "VLLM_ROCM_W4A8_FP8_WMMA_CACHE",
        os.path.join(os.path.dirname(__file__), "crossover_cache.json"))
    table = {}
    try:
        with open(path) as f:
            table = json.load(f)
    except (OSError, ValueError):
        table = {}
    _CROSSOVER_TABLE = table
    return table


def _crossover_for(N: int, K: int, group: int) -> int:
    """O(1) crossover lookup. Env override forces a fixed M for all shapes.
    Unknown shapes -> _NEVER (always Triton), so the pathway stays >= Triton."""
    env = os.environ.get("VLLM_ROCM_W4A8_FP8_WMMA_MIN_M")
    if env:
        return int(env)
    v = _load_crossover_table().get(f"{N},{K},{group}")
    if v is None:
        return _NEVER
    return int(v)

SUPPORTED_QUANT_TYPES = [
    scalar_types.uint4b8,  # symmetric, implicit zero point = 8
    scalar_types.uint4,    # asymmetric, explicit per-group zero points (AWQ)
]


def _on_gfx12x() -> bool:
    try:
        from vllm.platforms.rocm import on_gfx12x
        return on_gfx12x()
    except Exception:
        return False


class RocmW4A8Fp8WmmaLinearKernel(MPLinearKernel):
    """FP8 WMMA (16x16x16) kernel for 4-bit weights on RDNA4 / gfx1201."""

    SUPPORTED_QUANT_TYPES = SUPPORTED_QUANT_TYPES

    @classmethod
    def get_min_capability(cls) -> int:
        return 0  # gated by on_gfx12x() in can_implement instead

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        ok, reason = cls._can_implement_inner(c)
        import logging
        logging.getLogger(__name__).warning(
            "[w4a8_fp8_wmma] can_implement -> %s (%s) | wt=%s act=%s g=%s zp=%s "
            "gidx=%s part=%s",
            ok, reason, c.weight_type, c.act_type, c.group_size,
            c.zero_points, c.has_g_idx, c.partition_weight_shape,
        )
        return ok, reason

    @classmethod
    def _can_implement_inner(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "ROCm only"
        if not _on_gfx12x():
            return False, "requires gfx12x (RDNA4)"
        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return False, f"unsupported weight_type {c.weight_type}"
        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, f"act_type must be fp16/bf16, got {c.act_type}"
        if c.has_g_idx:
            return False, "act reordering (g_idx) not supported"
        gs = c.group_size
        K = c.partition_weight_shape[0]
        if gs == -1:
            return False, "per-channel (group_size=-1) not supported"
        if gs % 16 != 0 or gs > 128:
            return False, f"group_size must be multiple of 16 and <=128, got {gs}"
        if K % gs != 0:
            return False, f"K={K} not divisible by group_size={gs}"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Normalize weights to the op's layout: w_q (N, K//8) int32, scales
        (N, K//group) fp16.

        Two source layouts are handled:
          - compressed-tensors: already (N, K//8) / (N, K//group).
          - AutoGPTQ: qweight (K//8, N) packed along K, scale (K//group, N).
            Repacked here (transpose + re-pack) so the op sees its native layout.
        """
        c = self.config
        K, N = c.partition_weight_shape  # (in, out)
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)
        wq = w_q.data
        ws = w_s.data

        if tuple(wq.shape) == (K // 8, N):
            # ---- AutoGPTQ layout -> our (N, K//8) ----
            dev = wq.device
            shifts = (torch.arange(8, device=dev, dtype=torch.int32) * 4).view(1, 8, 1)
            # (K//8, N) -> (K//8, 8, N) -> (K, N): nibble j of group is k = k8*8+j
            unpacked = ((wq.unsqueeze(1) >> shifts) & 0xF).reshape(K, N)
            w_kn = unpacked.t().contiguous()  # (N, K)
            repacked = torch.zeros((N, K // 8), dtype=torch.int32, device=dev)
            for j in range(8):
                repacked |= (w_kn[:, j::8] & 0xF) << (j * 4)
            self._transform_param(layer, self.w_q_name, lambda _p: repacked)
            ws = ws.t().contiguous()  # (K//group, N) -> (N, K//group)
        elif tuple(wq.shape) != (N, K // 8):
            raise RuntimeError(
                f"unexpected w_q shape {tuple(wq.shape)}; expected (N,K//8)="
                f"{(N, K // 8)} or GPTQ (K//8,N)={(K // 8, N)}")
        elif not wq.is_contiguous():
            self._transform_param(layer, self.w_q_name, lambda p: p.contiguous())

        w_s_fp16 = (
            ws.to(torch.float16).contiguous()
            if ws.dtype != torch.float16 else ws.contiguous()
        )
        layer._w4a8_fp8_w_s = w_s_fp16
        if w_zp is not None and not w_zp.is_contiguous():
            self._transform_param(layer, self.w_zp_name, lambda p: p.contiguous())

        # ---- Triton W4A16 fallback weights (small-M band where Triton wins).
        # Both the stock triton_w4a16_gemm AND our gfx1201-tuned variant read b_q
        # [K, N//8], scales [K//group, N] — ONE shared copy, built from our (N, K//8)
        # / (N, K//group) layout. SKIPPED by default (single-layout mode): this is a
        # SECOND full weight copy (Triton packing) that EXACTLY doubles dense weight
        # VRAM (same int32 byte count as native) and OOMs ~10GB-weight models on 16GB
        # cards — VRAM that is otherwise KV cache. The default frees it; set
        # VLLM_ROCM_W4A8_LAYOUT=tuned to build it and keep the strictly->=-stock
        # small-M path (2x VRAM). See _layout_mode for the full tradeoff.
        if _no_triton_fallback() or _force_mode() == "on":
            # single-layout, or FORCE=on (always runs our kernel) -> the Triton copy
            # is dead weight. apply_weights routes ALL M through v11/v10/v5 (gap-free).
            layer._w4a8_tri_bq = None
            layer._w4a8_tri_s = None
            layer._w4a8_tri_zp = None
        else:
            wq_now = getattr(layer, self.w_q_name).data  # (N, K//8)
            dev = wq_now.device
            # Unpack/repack in N-row blocks so the transient is O(BLK*K), not the
            # full O(N*K) int32 that previously spiked peak VRAM at load.
            N8 = N // 8
            tri_bq = torch.zeros((K, N8), dtype=torch.int32, device=dev)
            shifts = (torch.arange(8, device=dev, dtype=torch.int32) * 4).view(1, 8, 1)
            BLK = 1024 - (1024 % 8)  # multiple of 8 (packing granularity along N)
            for nb in range(0, N, BLK):
                ne = min(nb + BLK, N)
                blk = wq_now[nb:ne]                                   # (b, K//8)
                # (b, K//8) -> (b, K//8, 8) -> (b, K): nibble j of int32 is k=k8*8+j
                up = ((blk.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF).reshape(ne - nb, K)
                w_kn = up.t().contiguous()                            # (K, b)
                for j in range(8):
                    tri_bq[:, nb // 8:ne // 8] |= (w_kn[:, j::8] & 0xF) << (j * 4)
            layer._w4a8_tri_bq = tri_bq.contiguous()           # (K, N//8)
            layer._w4a8_tri_s = w_s_fp16.t().contiguous()      # (K//group, N)

            # AWQ asymmetric zeros for the Triton fallback (decode path). Our op's
            # zero layout is (N//8, K//group) N-packed, standard nibble order;
            # triton_w4a16_gemm wants (K//group, N//8) with the same N-packing and
            # nibble order, i.e. a plain transpose. Symmetric (uint4b8) keeps None
            # so Triton uses the implicit zp_bias=8. Without this, decode (small M)
            # silently dropped the per-group zero points -> wrong AWQ outputs.
            if c.zero_points and w_zp is not None:
                zp_now = getattr(layer, self.w_zp_name).data  # (N//8, K//group)
                layer._w4a8_tri_zp = zp_now.t().contiguous()  # (K//group, N//8)
            else:
                layer._w4a8_tri_zp = None

        # Per-layer calibration: the ours-vs-Triton crossover depends on (N,K),
        # not just M (larger N lowers it, larger K raises it; some shapes never
        # cross). Measure THIS layer's crossover once so apply_weights engages the
        # FP8 kernel only where it is actually faster -> pathway is always >=
        # Triton. Cached per (N, K, group) since shapes repeat across layers.
        gs = c.group_size if c.group_size != -1 else K
        # O(1) lookup of the precomputed crossover (AOT Profile & Cache). On a
        # cache HIT this is benchmark-free. On a MISS (a shape that wasn't AOT-
        # tuned) the kernel would otherwise be DEAD WEIGHT (always-Triton); so if
        # VLLM_ROCM_W4A8_AUTOTUNE is on (default) we run a quick load-time A/B
        # microbench for THIS exact (N,K,group), persist the winning-suffix
        # crossover to the cache (subsequent loads are O(1)), and use it. ROBUST:
        # any autotune failure -> _NEVER (always Triton), so the served pathway is
        # always >= Triton, tuned or not. Env override / FORCE bypass the cache.
        layer._w4a8_min_m = _crossover_for(N, K, gs)
        miss = (
            layer._w4a8_min_m == _NEVER
            and not os.environ.get("VLLM_ROCM_W4A8_FP8_WMMA_MIN_M")
            and _force_mode() == "auto"
            and _autotune_enabled()
            and f"{N},{K},{gs}" not in _load_crossover_table()
        )
        if miss:
            try:
                co = _autotune_crossover(layer, self.w_q_name, N, K, gs)
                layer._w4a8_min_m = _NEVER if co is None else int(co)
            except Exception:  # pragma: no cover - defensive; never regress
                _persist_crossover(N, K, gs, None)  # cache the miss -> O(1) next
                layer._w4a8_min_m = _NEVER

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        N = c.partition_weight_shape[1]
        K = c.partition_weight_shape[0]
        out_shape = x.shape[:-1] + (N,)

        x_2d = x.reshape(-1, x.shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        M = x_2d.size(0)
        orig_dtype = x_2d.dtype

        gs = c.group_size if c.group_size != -1 else K
        # Kernel selection by M (all bit-exact; each wins its regime):
        #   decode (M<=DECODE_MAX): v11 streaming GEMV   -> beats Triton ~1.4-1.9x
        #   prefill (M>=prefill_min): v10 A-shuffle WMMA -> beats fp16/Triton
        #   mid-M band (opt-in): v6 b128 double-K WMMA   -> ROADMAP Task 4 lever
        #   middle / unsupported shapes: Triton W4A16 fallback (safe).
        # NB seam (actquant-fusion task): the op stages activations to fp8 ONCE in
        # the host launcher (compute_act_fp8_and_scales_kernel) and ALL versions
        # (v6 included) consume the same x_fp8/act_scales -- this adapter only picks
        # the weight-side kernel + load width, never the activation staging.
        # v10 needs group_size in {32,128} (compile-time); v11 needs K%1024==0,
        # group_size%32==0, and M*K<=65536 (activations fit LDS).
        v10_ok = gs in (32, 128)
        # v11 is now K-TILED (kernel stages only (M,BK) in LDS) -> no M*K cap; correct
        # for any K%1024==0 at M<=16 (validated bit-exact vs v5, incl. K=17408 decode).
        # BUT v11 is a GEMV: GPU microbench (N=5120,K=17408) shows it only WINS at M=1
        # (162us vs triton 283 / v10 440); it scales ~110+50*M us and loses to Triton
        # by M>=4. So gate v11 to tiny M; mid-M decode stays on Triton (fallback) and
        # large M on v10. (default 2; was wrongly 16. Real crossover is shape-dependent.)
        v11_ok = (K % 1024 == 0) and (gs % 32 == 0) and (M <= 16)
        decode_max = int(os.environ.get("VLLM_ROCM_W4A8_DECODE_MAX_M", "2"))
        # v10 reliably beats fp16/Triton from ~M=256 even untuned; engage there
        # (or earlier if the AOT crossover cache proved a lower threshold).
        v10_min = int(os.environ.get("VLLM_ROCM_W4A8_V10_MIN_M", "256"))
        cached = getattr(layer, "_w4a8_min_m", _NEVER)
        prefill_min = min(cached, v10_min) if v10_ok else cached

        use_v11 = v11_ok and M <= decode_max
        use_v10 = (not use_v11) and v10_ok and M >= prefill_min
        use_v5 = (not use_v11) and (not use_v10) and M >= cached  # gs not 32/128

        # Single-layout (default): no (K,N//8) fallback weights exist, so the Triton
        # else-branch below would deref None -> ALWAYS route through our kernel here,
        # gap-free (v10 covers any M for gs in {32,128}; else v5), overriding even a
        # FORCE=off (which has no stock-Triton weights to fall back to in this mode).
        have_fallback = getattr(layer, "_w4a8_tri_bq", None) is not None
        # Tuning-gate override (VLLM_ROCM_W4A8_FORCE):
        #   off -> always stock Triton (needs the fallback weights, which were built
        #          since FORCE!=on AND mode='tuned' at load); on / single-layout ->
        #          always our kernel, gap-free.
        force = _force_mode()
        if force == "off" and have_fallback:
            use_v11 = use_v10 = use_v5 = False
        elif (force == "on" or not have_fallback) and not (use_v11 or use_v10 or use_v5):
            if v10_ok:
                use_v10 = True
            else:
                use_v5 = True

        # v6 mid-M band (b128 double-K). OFF unless VLLM_ROCM_W4A8_V6_{MIN,MAX}_M
        # are BOTH set (see _v6_band). When engaged for M in [lo,hi] with gs%32==0,
        # route to v6 IN PLACE OF the kernel the ladder already chose -- v6 is
        # bit-exact vs v5/v10 (identical operands; only LDS load width differs), so
        # this never changes numerics. It only re-targets a WMMA M-range that was
        # already going to our kernel (so force=='off' still stays on Triton, and an
        # M that would fall back to Triton is NOT yanked onto v6 -- the band tunes
        # ours-vs-ours, not ours-vs-Triton, which the crossover cache already gates).
        use_v6 = False
        v6_band = _v6_band()
        if (v6_band is not None and (gs % 32 == 0)
                and (use_v10 or use_v5)
                and v6_band[0] <= M <= v6_band[1]):
            use_v6 = True
            use_v10 = use_v5 = False

        if use_v11 or use_v10 or use_v6 or use_v5:
            w_q, _w_s_native, w_zp, _ = self._get_weight_params(layer)
            w_s = layer._w4a8_fp8_w_s
            x16 = x_2d if x_2d.dtype == torch.float16 else x_2d.to(torch.float16)
            if c.zero_points and w_zp is not None:
                zp_in = w_zp
            else:
                zp_in = torch.empty(0, dtype=torch.int32, device=x.device)
            ver = 11 if use_v11 else (10 if use_v10 else (6 if use_v6 else 5))
            out = torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm(x16, w_q, w_s, zp_in, ver)
            if orig_dtype != torch.float16:
                out = out.to(orig_dtype)
        else:
            # Triton W4A16 fallback (mid-M, or shapes v10/v11 can't take). At small M
            # the stock vLLM Triton runs a gfx1151-tuned config (BLOCK_K clamped to 64)
            # that is suboptimal on gfx1201 (64 CU) -- exactly the dense M=16-32 band
            # where our fp8-WMMA HIP kernels are at the raw WMMA/HBM limit (~20 variants
            # confirm). For M<=32 we route to a gfx1201-tuned Triton (BLOCK_K = full
            # group) that is 1.2-1.4x faster than the stock config at g=128 and >= it
            # elsewhere, so the served pathway EXCEEDS stock in that last regime too.
            # Above M=32 the stock config is already best, so use it unchanged.
            tri_zp = getattr(layer, "_w4a8_tri_zp", None)
            zp_bias = c.weight_type.bias if c.weight_type.has_bias() else 0
            # gfx1201-tuned Triton ONLY where it strictly beats stock: small M (<=32)
            # AND group_size > 64 (the stock clamps BLOCK_K to 64, so for g=128 it runs
            # the half-group tile -> 1.2-1.4x slower than BLOCK_K=full-group here). For
            # g<=64 the tuned config IS the stock config, so use stock directly (no
            # separate path). M>32: stock's tile is already best. Net: fallback >= stock.
            if M <= 32 and gs > 64:
                from w4a8_fp8_wmma.triton_w4a16_gfx1201 import (
                    triton_w4a16_gemm_gfx1201 as _tri,
                )
            else:
                from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (  # noqa: E501
                    triton_w4a16_gemm as _tri,
                )
            out = _tri(x_2d, layer._w4a8_tri_bq, layer._w4a8_tri_s, tri_zp, gs, zp_bias)

        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        if bias is not None:
            out = out + bias
        return out.reshape(out_shape)
