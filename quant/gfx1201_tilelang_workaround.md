# gfx1201 tilelang abort — RESOLVED by the server sessions (root cause below)

> **UPDATE (2026-06-12): real root cause + proper fix found by the
> vllm-rocm714-gfx1250 sessions — it is NOT a gfx1201 hardware gap.**
>
> It's an **apache-tvm-ffi ABI/version skew**: the base image ships
> `apache-tvm-ffi 0.1.12`, but `tilelang 0.1.10` bundles its own TVM
> (`libtvm_compiler.so`) built against `apache-tvm-ffi 0.1.10`. At static-init
> both register the `__ffi_repr__` TypeAttr for type index 130 →
> double-registration → `terminate called after throwing 'tvm::ffi::Error'`,
> which unwinds through pydantic_core's Rust frame during config construction
> and surfaces as the `Rust cannot catch foreign exceptions` abort (RC=134).
> pip's constraint was satisfied (`apache-tvm-ffi>=0.1.10,~=0.1.0`), so this
> was an ABI mismatch, not a version-range violation — and it would bite the
> image on any GPU.
>
> **Proper fix (now baked into `Dockerfile.rocm.gfx1201-inject`):**
> `pip install apache-tvm-ffi==0.1.10` (a pip step after the vLLM wheel
> install). After that, `import vllm` + `import tilelang` coexist cleanly.
>
> Consequence for ZAYA: once `vllm-gfx1201:latest` is rebuilt with that pin,
> the `has_tilelang()→False` overlay below is **no longer needed** — drop it.
> It remains only as an interim for images built before the pin. The other
> ZAYA fixes (RoutedExperts isinstance, expert loader, router bf16) are
> unaffected and still required.
>
> Note: my original attribution below ("TVM native lib aborts on gfx1201") was
> wrong about the *cause*; the symptom chain it documents is still accurate.

---

# gfx1201 tilelang abort — interim workaround (superseded; see update above)

## Symptom

Serving a **quantized** model (compressed-tensors, AWQ, GPTQ — anything with a
`quantization_config`) on `vllm-gfx1201:latest` aborts during startup with:

```
fatal runtime error: Rust cannot catch foreign exceptions, aborting
!!!!!!! Segfault encountered !!!!!!!
```

bf16/unquantized models are unaffected.

## Root cause

`ModelConfig._verify_quantization` → `get_quantization_config` eagerly does
`from vllm.models.deepseek_v4 import DeepseekV4FP8Config`, which transitively
imports `vllm/model_executor/layers/mhc.py` → `has_tilelang()` →
`_has_module("tilelang")` → **imports `tilelang`, whose TVM native lib aborts
on gfx1201** (a C++/HIP foreign exception through a Rust frame — uncatchable).

The import happens for *every* quantized model because it is on the
`get_quantization_config` path, not behind any arch/quant gate. This is why
the other sessions run quantized models (`rdna4-w4a8-awq`) on a **different**
image (`kyuz0/vllm-therock-gfx1201:latest`), not `vllm-gfx1201:latest`.

Confirmed via `PYTHONFAULTHANDLER=1`; the abort is in
`tilelang/3rdparty/tvm/python/tvm/libinfo.py:load_lib_ctypes`.

## Workaround (used by the ZAYA FP8 overlay)

Make `has_tilelang()` return `False` without importing tilelang. ZAYA does not
use tilelang kernels, so this is safe for ZAYA serving:

```bash
# extract, patch, and bind-mount over the wheel file:
cid=$(docker create vllm-gfx1201:latest)
docker cp "$cid:/opt/python/lib/python3.12/site-packages/vllm/utils/import_utils.py" import_utils.py
docker rm "$cid"
# edit has_tilelang() body to `return False`
# then add to `docker run`:  -v $PWD/import_utils.py:/opt/python/lib/python3.12/site-packages/vllm/utils/import_utils.py:ro
```

## Proper fix (for the vllm-rocm714-gfx1250 sessions)

Either (a) make `_has_module("tilelang")` resilient — guard the import so a
native-load abort can't kill the process (hard, it's an abort not an
exception), or (b) make the `tilelang` build importable on gfx1201, or
(c) gate the `deepseek_v4`/`mhc` import in `get_quantization_config` behind an
arch check so non-deepseek quantized models don't pull in tilelang. (c) is the
least invasive and unblocks all quantized serving on `vllm-gfx1201:latest`.
