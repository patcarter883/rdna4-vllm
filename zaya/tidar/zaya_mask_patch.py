#!/usr/bin/env python3
"""Inject a TiDAR structured 4D attention mask into ZAYA's CCA attention.

ZAYA (Zyphra's transformers fork, `modeling_zaya`) does NOT accept a custom 4D
attention mask through the public `attention_mask` argument — that argument is
reserved for the 2D CCA-conv padding mask and the forward raises if it is not 2D.
Instead ZAYA builds its causal mask *internally* via `create_causal_mask(...)` and
passes it to each attention module as the dict entry `mask_mapping["causal"]`,
which `eager_attention_forward` adds straight onto the attention weights.

So we inject the TiDAR mask by monkeypatching `create_causal_mask` to return our
precomputed additive bias. The TiDAR mask must REPLACE (not AND with) the causal
mask: TiDAR's intra-block bidirectional attention is not a subgraph of causal.

All 80 ZAYA1 layers are `"hybrid"` (full causal); `sliding_window` is None, so no
`"hybrid_sliding"` layers exist and patching `create_causal_mask` alone suffices
(we patch the sliding variant identically for safety). Same mechanism for the mask
self-test (`check_zaya_mask.py`) and the trainer, so train ↔ check ↔ serve agree.

Usage:
    import zaya_mask_patch as zmp
    zmp.install()                      # once, after importing transformers
    zmp.set_bias(bias_4d)              # [B|1, 1, L, L] additive (0 keep / min block)
    out = model(input_ids=ids, attention_mask=None, position_ids=pos, use_cache=False)
"""
import torch

_BIAS = None


def install():
    """Patch ZAYA's mask factory so attention's causal mask == the bias set below."""
    import transformers.models.zaya.modeling_zaya as mz

    def _tidar_causal_mask(**kwargs):
        # Single-device (CPU / one GPU): _BIAS is already on the right device → return as-is (the
        # coherence-gate path, unchanged). Multi-GPU device_map (the throughput run): each layer
        # calls this on ITS device, so the bias must follow — infer the layer device from the first
        # tensor kwarg (input_embeds / cache_position / position_ids) and move there. The [1,1,L,L]
        # move is cheap vs the layer's 8B-weight forward and is a no-op when devices already match.
        if _BIAS is None:
            return None
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                return _BIAS if _BIAS.device == v.device else _BIAS.to(v.device)
        return _BIAS

    mz.create_causal_mask = _tidar_causal_mask
    mz.create_sliding_window_causal_mask = _tidar_causal_mask


def set_bias(bias_4d):
    """Set the additive 4D bias used by the next forward(s). None -> default causal."""
    global _BIAS
    _BIAS = bias_4d


def bias_from_tidar(attn_bias_2d, dtype, device):
    """[2S,2S] {0,-inf} TiDAR bias -> [1,1,2S,2S] additive in `dtype` on `device`."""
    neg = torch.finfo(dtype).min
    m = torch.where(attn_bias_2d < 0,
                    torch.tensor(neg, dtype=dtype),
                    torch.tensor(0.0, dtype=dtype))
    return m.view(1, 1, *attn_bias_2d.shape).to(device)
