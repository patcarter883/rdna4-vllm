# SPDX-License-Identifier: Apache-2.0
"""Generic parallel-compile for Triton's @triton.autotune (accelerates the GDN/SSD
cold-compile and any other triton.autotune'd kernel).

Triton's Autotuner.run benchmarks configs SERIALLY: `{cfg: _bench(cfg) for cfg in
pruned}`, and each _bench's first kernel call triggers a Triton->LLVM->ISA compile
(~seconds CPU). On a cold cache (every dev container / kernel rebuild) the GDN SSD
kernels alone are ~58 configs (chunk_scan 23, chunk_state 20, bmm 9, state_passing 6)
compiled one-at-a-time => the dominant chunk of the ~15-30 min FLA-GDN cold boot.

This monkeypatch wraps Autotuner.run so that, on a cache MISS, it first PARALLEL-warms
every pruned config via JITFunction.warmup (compile-only: dtype-mocked args, no launch,
no tensor mutation -> safe), populating the shared Triton cache. The original run then
benchmarks serially and CACHE-HITS every compile — so timing stays serial+accurate
(parallelizing the *timing* would corrupt the autotune pick via GPU contention) while
the expensive compiles fan across threads (LLVM releases the GIL). Thread pool, not
processes: shares the CUDA context (no per-worker HIP init) and needs no pickling of
kernels/args.

Correctness: warmup only compiles (it never runs the kernel or touches data), and the
final config selection is still done by the unchanged serial benchmark. Best-effort:
any warmup failure is swallowed (that config just compiles serially in _bench as today).
Gate: VLLM_TRITON_PARALLEL_COMPILE=1 (default on here). Workers:
VLLM_TRITON_PARALLEL_COMPILE_WORKERS (default 8).
"""
import os


def install() -> None:
    if os.environ.get("VLLM_TRITON_PARALLEL_COMPILE", "1") != "1":
        return
    import triton.runtime.autotuner as AT
    if getattr(AT.Autotuner, "_parallel_compile_patched", False):
        return

    import concurrent.futures as cf

    from triton.runtime.jit import JITFunction

    _orig_run = AT.Autotuner.run

    def _jit(fn):
        seen = 0
        while fn is not None and not isinstance(fn, JITFunction) and seen < 8:
            fn = getattr(fn, "fn", None)
            seen += 1
        return fn if isinstance(fn, JITFunction) else None

    def run(self, *args, **kwargs):
        try:
            if len(self.configs) > 1:
                nargs = dict(zip(self.arg_names, args))
                _a = {k: v for k, v in {**nargs, **kwargs}.items()
                      if k in self.arg_names}
                key = [_a[k] for k in self.keys if k in _a]
                for _, v in _a.items():
                    if hasattr(v, "dtype"):
                        key.append(str(v.dtype))
                key = tuple(key)
                if key not in self.cache:           # cache MISS -> first autotune
                    pruned = self.prune_configs(kwargs)
                    jf = _jit(self.fn)
                    if jf is not None and len(pruned) > 1:
                        nw = min(int(os.environ.get(
                            "VLLM_TRITON_PARALLEL_COMPILE_WORKERS", "8")), len(pruned))

                        def _warm(cfg):
                            try:  # compile-only; never executes the kernel
                                jf.warmup(*args, grid=(1, ),
                                          **{**kwargs, **cfg.all_kwargs()})
                            except Exception:
                                pass  # falls back to serial compile in _bench

                        with cf.ThreadPoolExecutor(max_workers=nw) as ex:
                            list(ex.map(_warm, pruned))
        except Exception:
            pass  # never break autotuning; serial path is unchanged below
        return _orig_run(self, *args, **kwargs)

    AT.Autotuner.run = run
    AT.Autotuner._parallel_compile_patched = True
