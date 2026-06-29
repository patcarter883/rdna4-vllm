"""Shared RDNA4 Titans helpers: model build (with the gfx1201-safe flags) + the
overlapping-parameter fixups. Used by phase0_smoke (validated) and train_enwik8."""

from __future__ import annotations

import torch
import torch.nn as nn


# The MAC constructor kwargs we persist to config.json so a serving loader can rebuild
# the exact architecture. Memory kwargs are nested under "neural_memory_kwargs".
def build_mac_model(cfg: dict):
    from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

    mem = cfg["neural_memory_kwargs"]
    neural_memory_model = MemoryMLP(dim=mem["dim_head"], depth=cfg["neural_memory_depth"])

    model = MemoryAsContextTransformer(
        num_tokens=cfg["num_tokens"],
        dim=cfg["dim"],
        depth=cfg["depth"],
        segment_len=cfg["segment_len"],
        num_persist_mem_tokens=cfg["num_persist_mem_tokens"],
        num_longterm_mem_tokens=cfg["num_longterm_mem_tokens"],
        neural_memory_layers=tuple(cfg["neural_memory_layers"]),
        neural_memory_segment_len=cfg["neural_memory_segment_len"],
        neural_memory_batch_size=cfg["neural_memory_batch_size"],
        neural_mem_weight_residual=cfg["neural_mem_weight_residual"],
        neural_memory_qkv_receives_diff_views=cfg["neural_memory_qkv_receives_diff_views"],
        use_flex_attn=False,            # RDNA4: no FlexAttention
        sliding_window_attn=cfg["sliding_window_attn"],
        neural_memory_model=neural_memory_model,
        neural_memory_kwargs=dict(
            dim_head=mem["dim_head"],
            heads=mem["heads"],
            attn_pool_chunks=mem["attn_pool_chunks"],
            qk_rmsnorm=mem["qk_rmsnorm"],
            momentum=mem["momentum"],
            momentum_order=mem["momentum_order"],
            default_step_transform_max_lr=mem["default_step_transform_max_lr"],
            use_accelerated_scan=False,  # RDNA4: pure-torch AssocScan
            per_parameter_lr_modulation=mem["per_parameter_lr_modulation"],
            per_head_learned_parameters=mem["per_head_learned_parameters"],
        ),
    )
    return model


def materialize_overlapping_params(model) -> int:
    """The per-head learned memory init is an einops.repeat (stride-0 broadcast) aliasing
    the memory submodule's weights — a single Parameter whose storage overlaps itself,
    which in-place optimizers reject (bites on CPU; .to('cuda') happens to materialize it).
    Replace each with an independent contiguous clone. Returns count replaced."""
    n = 0
    for module in model.modules():
        plist = getattr(module, "memory_model_parameters", None)
        if not isinstance(plist, nn.ParameterList):
            continue
        for i in range(len(plist)):
            p = plist[i]
            if not p.is_contiguous() or torch._debug_has_internal_overlap(p) != 0:
                plist[i] = nn.Parameter(p.detach().clone().contiguous(),
                                        requires_grad=p.requires_grad)
                n += 1
    return n


def dedup_params(model) -> list:
    """One Parameter per underlying storage (drops dead memory_model submodule aliases)."""
    seen, out = set(), []
    for p in model.parameters():
        key = p.data_ptr()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def default_config(num_tokens=256, dim=512, depth=8, seq_len=512) -> dict:
    """Default ~30M MAC config for the enwik8 tiny convergence run."""
    return dict(
        arch="titans-mac",
        num_tokens=num_tokens,
        dim=dim,
        depth=depth,
        seq_len=seq_len,
        segment_len=32,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=4,
        neural_memory_layers=[2, 4, 6],
        neural_memory_depth=2,
        neural_memory_segment_len=4,
        neural_memory_batch_size=128,
        neural_mem_weight_residual=True,
        neural_memory_qkv_receives_diff_views=True,
        sliding_window_attn=False,
        neural_memory_kwargs=dict(
            dim_head=64,
            heads=4,
            attn_pool_chunks=True,
            qk_rmsnorm=True,
            momentum=True,
            momentum_order=1,
            default_step_transform_max_lr=1e-1,
            per_parameter_lr_modulation=True,
            per_head_learned_parameters=True,
        ),
    )
