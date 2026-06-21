"""Register the W4A8-FP8 WMMA kernel into vLLM's linear-kernel dispatcher.

Call register() once, before any quantized model is loaded. It prepends our
kernel to _POSSIBLE_KERNELS[ROCM] so 4-bit layers on gfx1201 route through the
FP8 WMMA path (when can_implement() accepts them; otherwise vLLM falls through to
the next kernel, e.g. TritonW4A16).

Usage:
    import w4a8_fp8_wmma            # loads the HIP op (torch.ops.w4a8_fp8_wmma)
    from w4a8_fp8_wmma.register import register
    register()
    # ... then construct the vLLM engine / load the model
"""
import os

_REGISTERED = False

# De-numbered env vars whose old numeric names are REMOVED. A stale numeric
# override must fail loudly, never silently mis-dispatch / fall back to stock.
_REMOVED_W4A8_ENV = {
    "VLLM_ROCM_W4A8_FP8_WMMA_MOE_VERSION":
        "use VLLM_ROCM_W4A8_FP8_WMMA_MOE_KERNEL=wmma|scalar|gemv "
        "(A-residence, formerly v5-vs-v6, is now VLLM_W4A8_MOE_A_IN_LDS)",
    "VLLM_ROCM_W4A8_V10_MIN_M":
        "use VLLM_ROCM_W4A8_PREFILL_MIN_M (same int threshold, renamed)",
}


def _check_removed_w4a8_env() -> None:
    """Hard-break for de-numbered env vars, at the one LOUD chokepoint. vLLM's
    load_general_plugins() calls register() UNWRAPPED (only the plugin *import* is
    try/excepted), so a raise here crashes boot — whereas a raise inside the
    register_moe* hooks is swallowed by register()'s own try/except below (logged,
    then MoE silently falls to stock). Covers BOTH the MoE (MOE_VERSION) and dense
    (V10_MIN_M) renames so neither stale override can silently change behavior."""
    bad = [f"{k} is removed — {why}"
           for k, why in _REMOVED_W4A8_ENV.items() if k in os.environ]
    if bad:
        raise RuntimeError("[w4a8_fp8_wmma] removed env var(s) set: " + "; ".join(bad))


def register(verbose: bool = True) -> bool:
    """Prepend RocmW4A8Fp8WmmaLinearKernel to _POSSIBLE_KERNELS[ROCM].

    Returns True if registered (or already registered), False if disabled or the
    platform isn't ROCm/gfx12x. Honors env flag VLLM_ROCM_USE_W4A8_FP8_WMMA
    (default "1"); set to "0" to disable.
    """
    global _REGISTERED
    if _REGISTERED:
        return True
    if os.environ.get("VLLM_ROCM_USE_W4A8_FP8_WMMA", "1") != "1":
        if verbose:
            print("[w4a8_fp8_wmma] disabled via VLLM_ROCM_USE_W4A8_FP8_WMMA=0")
        return False

    # Loud hard-break BEFORE any try/except below (register() is called unwrapped
    # by vLLM's plugin loader, so this raise crashes boot as intended).
    _check_removed_w4a8_env()

    # Ensure the op is loaded.
    import w4a8_fp8_wmma  # noqa: F401

    from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
    from vllm.platforms import PlatformEnum

    from w4a8_fp8_wmma.vllm_adapter import RocmW4A8Fp8WmmaLinearKernel

    lst = _POSSIBLE_KERNELS.setdefault(PlatformEnum.ROCM, [])
    if RocmW4A8Fp8WmmaLinearKernel not in lst:
        lst.insert(0, RocmW4A8Fp8WmmaLinearKernel)
    _REGISTERED = True
    if verbose:
        print("[w4a8_fp8_wmma] registered at _POSSIBLE_KERNELS[ROCM][0]:",
              [k.__name__ for k in lst])

    # mxfp4 (OCP E2M1) DENSE hook — registers our kernel into vLLM's SEPARATE mxfp4 linear-kernel
    # registry (_POSSIBLE_MXFP4_KERNELS, distinct from _POSSIBLE_KERNELS above). vLLM ships no ROCm
    # entry there, so CompressedTensorsW4A4Mxfp4 otherwise crashes at construct on gfx1201; this lets
    # the unmodified scheme route E2M1 dense linears through the W4A8 op. Best-effort.
    try:
        from vllm.model_executor.kernels.linear import register_linear_kernel
        from vllm.platforms import PlatformEnum as _PE
        from w4a8_fp8_wmma.mxfp4_linear import RocmW4A8MxFp4LinearKernel
        from vllm.model_executor.kernels.linear import _POSSIBLE_MXFP4_KERNELS
        if RocmW4A8MxFp4LinearKernel not in _POSSIBLE_MXFP4_KERNELS.get(_PE.ROCM, []):
            register_linear_kernel(RocmW4A8MxFp4LinearKernel, _PE.ROCM, kernel_type="mxfp4")
        if verbose:
            print("[w4a8_fp8_wmma] mxfp4 dense kernel registered at "
                  "_POSSIBLE_MXFP4_KERNELS[ROCM]:",
                  [k.__name__ for k in _POSSIBLE_MXFP4_KERNELS.get(_PE.ROCM, [])])
    except Exception as e:  # pragma: no cover - defensive
        if verbose:
            print(f"[w4a8_fp8_wmma] mxfp4 dense hook not installed: {e}")

    # MoE (fused experts) hook — patches the WNA16 MoE oracle so AWQ-4bit expert
    # layers on gfx12x route to our grouped FP8-WMMA kernel. Best-effort: failures
    # here must not break the (already-registered) dense path.
    try:
        from w4a8_fp8_wmma.moe_experts import register_moe
        register_moe(verbose=verbose)
    except Exception as e:  # pragma: no cover - defensive
        if verbose:
            print(f"[w4a8_fp8_wmma] MoE hook not installed: {e}")

    # Compressed-tensors MoE (the real target models, e.g. Qwen3.6-35B-A3B):
    # patches CompressedTensorsWNA16MoEMethod. Separate try/except so a failure
    # here can't break the AWQ or dense paths.
    try:
        from w4a8_fp8_wmma.moe_experts import register_moe_ct
        register_moe_ct(verbose=verbose)
    except Exception as e:  # pragma: no cover - defensive
        if verbose:
            print(f"[w4a8_fp8_wmma] CT MoE hook not installed: {e}")

    # MoeWNA16 MoE (the AWQ/GPTQ WNA16 fallback path; what Qwen3.6-35B-A3B takes
    # on gfx1201). Separate try/except.
    try:
        from w4a8_fp8_wmma.moe_experts import register_moe_wna16
        register_moe_wna16(verbose=verbose)
    except Exception as e:  # pragma: no cover - defensive
        if verbose:
            print(f"[w4a8_fp8_wmma] WNA16 MoE hook not installed: {e}")

    # GPTQ MoE (auto_gptq). AutoGPTQMoEMethod uses the same wna16 oracle as AWQ but
    # in its own namespace, so register_moe (awq_marlin-only) left GPTQ on Marlin.
    # This closes that gap for symmetric GPTQ-4bit MoE. Separate try/except.
    try:
        from w4a8_fp8_wmma.moe_experts import register_moe_gptq
        register_moe_gptq(verbose=verbose)
    except Exception as e:  # pragma: no cover - defensive
        if verbose:
            print(f"[w4a8_fp8_wmma] GPTQ MoE hook not installed: {e}")

    # mxfp4 (OCP E2M1) MoE — gpt-oss experts -> grouped FP8-WMMA. Separate try/except so a failure
    # here can't break the AWQ/GPTQ/CT/dense paths. (Weight decode GPU-validated; e2e serve TBD.)
    try:
        from w4a8_fp8_wmma.moe_experts import register_moe_mxfp4
        register_moe_mxfp4(verbose=verbose)
    except Exception as e:  # pragma: no cover - defensive
        if verbose:
            print(f"[w4a8_fp8_wmma] mxfp4 MoE hook not installed: {e}")

    # --- cold-boot accelerators (parallel kernel compilation) ----------------
    # These only fire on a COLD Triton cache (dev containers, kernel/.so rebuilds);
    # warm production boots skip autotune entirely, so they are no-ops there. Both
    # are best-effort + fall back to the unchanged serial path on any failure, and
    # never change the tuning RESULT (they only parallelize the compile step).
    #
    # (1) Generic Triton @triton.autotune parallel-compile — DEFAULT ON. Thread
    #     pool (shares the CUDA context, no spawn/HIP-init cost). Accelerates the
    #     GDN/SSD ssd_* autotune (~58 configs = the bulk of the 15-30min FLA-GDN
    #     cold boot) and any other triton.autotune kernel. Validated 3.15x,
    #     best-config unchanged. Disable: VLLM_TRITON_PARALLEL_COMPILE=0.
    try:
        from w4a8_fp8_wmma.triton_autotune_parallel import install as _tap_install
        _tap_install()
    except Exception as e:  # pragma: no cover - defensive
        if verbose:
            print(f"[w4a8_fp8_wmma] triton parallel-compile not installed: {e}")

    # (2) vLLM attention startup autotuner parallel pre-warm — DEFAULT ON
    #     (VLLM_ATTN_AUTOTUNE_PARALLEL=0 to disable). Process pool (spawns short-lived
    #     GPU workers during startup); ~2.77x on the ~42s cold attention autotune.
    #     On a warm cache the workers cache-hit, adding only small spawn overhead.
    #     Covers vLLM's CUSTOM attention autotuner, which (1) can't touch (it's not a
    #     triton.autotune kernel).
    try:
        from w4a8_fp8_wmma.attn_autotune_parallel import install as _aap_install
        _aap_install()
    except Exception as e:  # pragma: no cover - defensive
        if verbose:
            print(f"[w4a8_fp8_wmma] attn parallel autotune not installed: {e}")

    return True
