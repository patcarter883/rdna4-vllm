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

    return True
