"""Idempotently patch vLLM's TritonAttentionImpl.forward to route pure-decode bf16 attention
through the native attn_decode HIP kernel when VLLM_ATTN_DECODE_HIP=1.

Insertion is anchored on a stable line (not line numbers) so it survives minor file drift; re-running
is a no-op. Run inside the image build:  python apply_vllm_patch.py
"""
import os
import sys

import vllm
V = os.path.dirname(vllm.__file__)
TARGET = os.path.join(V, "v1", "attention", "backends", "triton_attn.py")

ANCHOR = "        block_table = attn_metadata.block_table\n"
MARKER = "# --- attn_decode native HIP decode path"

BLOCK = '''
        # --- attn_decode native HIP decode path (opt-in: VLLM_ATTN_DECODE_HIP=1) ----------
        # Routes a pure-decode bf16 batch through torch.ops.attn_decode.flash_decode_paged (one
        # AOT HIP kernel, no Triton). Lazy one-time init; any unsupported case bails to Triton.
        import os as _os_ad
        global _ATTN_DECODE_ROUTE
        try:
            _ATTN_DECODE_ROUTE
        except NameError:
            _ATTN_DECODE_ROUTE = None
            if _os_ad.environ.get("VLLM_ATTN_DECODE_HIP", "0") == "1":
                try:
                    from vllm.model_executor.layers.attn_decode import op as _ad_op       # decode .so
                    from vllm.model_executor.layers.attn_prefill_paged import op as _ap_op  # prefill .so
                    from vllm.model_executor.layers.attn_decode.vllm_route import maybe_hip_attention
                    _ATTN_DECODE_ROUTE = maybe_hip_attention
                    print("[attn_hip] native HIP attention ENABLED (bf16 decode + paged prefill).", flush=True)
                except Exception as _e_ad:  # build/load failure -> stay on Triton
                    _ATTN_DECODE_ROUTE = None
                    print("[attn_hip] VLLM_ATTN_DECODE_HIP=1 but load failed:", _e_ad, flush=True)
        if _ATTN_DECODE_ROUTE is not None and _ATTN_DECODE_ROUTE(
                self, layer, query, kv_cache, attn_metadata, output, output_scale):
            return output
        # ----------------------------------------------------------------------------------
'''


def main() -> int:
    src = open(TARGET).read()
    if MARKER in src:
        print(f"[apply_vllm_patch] already patched: {TARGET}")
        return 0
    if ANCHOR not in src:
        print(f"[apply_vllm_patch] ANCHOR not found in {TARGET} — vLLM forward changed; aborting.",
              file=sys.stderr)
        return 1
    # insert AFTER the first anchor occurrence (inside forward, before the tuning-params block)
    src = src.replace(ANCHOR, ANCHOR + BLOCK, 1)
    open(TARGET, "w").write(src)
    print(f"[apply_vllm_patch] patched {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
