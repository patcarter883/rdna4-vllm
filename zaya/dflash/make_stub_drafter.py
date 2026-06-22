#!/usr/bin/env python3
"""Generate a random-weight DFlash drafter sized for ZAYA1-8B (M1 plumbing stub).

This is NOT a trained drafter — it exists only to validate the ZAYA DFlash
*plumbing* end-to-end (target aux-hidden-state extraction -> DFlashProposer ->
context-KV precompute -> bidirectional mask-block verify). Acceptance will be
~random. The real CCA-aware drafter is M2; a trained softmax fallback is M3+.

Format mirrors the known-good z-lab/Qwen3.5-4B-DFlash speculator
(architectures=["DFlashDraftModel"] -> qwen3_dflash.DFlashQwen3ForCausalLM):
  - model_type "qwen3"; tie_word_embeddings=true so the drafter BORROWS ZAYA's
    embed_tokens + lm_head (has_own_embed_tokens/lm_head default False) -> the
    checkpoint omits both, and the drafter hidden_size MUST equal ZAYA's (2048)
    so the borrowed [vocab, 2048] tensors fit.
  - target_hidden_size defaults to hidden_size (2048 == ZAYA hidden).
  - dflash_config.target_layer_ids: the runner taps ZAYA layers (id+1); fc input
    = target_hidden * len(target_layer_ids).

Run on CPU (no GPU/lease needed):
  python zaya/dflash/make_stub_drafter.py --out /home/pat/code/_models/ZAYA1-8B-DFlash-stub
"""
import argparse
import json
import os

import torch
from safetensors.torch import save_file

# ZAYA1-8B target geometry (zaya/overlay/.../configs/zaya.py + model).
ZAYA_HIDDEN = 2048
ZAYA_VOCAB = 262272
ZAYA_HEAD_DIM = 128
ZAYA_NUM_LAYERS = 80
ZAYA_MAX_POS = 131072

# Drafter geometry (stub: 1 layer, small MLP).
DRAFT_LAYERS = 1
DRAFT_HEADS = ZAYA_HIDDEN // ZAYA_HEAD_DIM  # 16
DRAFT_KV_HEADS = 2
DRAFT_INTERMEDIATE = 4096
# Three taps spread across ZAYA's 80 layers. The runner converts each to id+1,
# so these map to ZAYA residual-stream depths {2, 40, 77}.
TARGET_LAYER_IDS = [1, 39, 76]
MASK_TOKEN_ID = 0  # any valid id < vocab; the mask block just needs an embedding


def build_config() -> dict:
    return {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "head_dim": ZAYA_HEAD_DIM,
        "hidden_act": "silu",
        "hidden_size": ZAYA_HIDDEN,
        "intermediate_size": DRAFT_INTERMEDIATE,
        "num_hidden_layers": DRAFT_LAYERS,
        "num_attention_heads": DRAFT_HEADS,
        "num_key_value_heads": DRAFT_KV_HEADS,
        "layer_types": ["full_attention"] * DRAFT_LAYERS,
        "max_position_embeddings": ZAYA_MAX_POS,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
        "tie_word_embeddings": True,
        "vocab_size": ZAYA_VOCAB,
        "num_target_layers": ZAYA_NUM_LAYERS,
        "dflash_config": {
            "mask_token_id": MASK_TOKEN_ID,
            "target_layer_ids": TARGET_LAYER_IDS,
        },
        "dtype": "bfloat16",
        "_note": "RANDOM-WEIGHT M1 PLUMBING STUB — not a trained drafter.",
    }


def build_state_dict() -> dict:
    h = ZAYA_HIDDEN
    hd = ZAYA_HEAD_DIM
    q = DRAFT_HEADS * hd          # 2048
    kv = DRAFT_KV_HEADS * hd      # 256
    inter = DRAFT_INTERMEDIATE
    n_aux = len(TARGET_LAYER_IDS)
    g = torch.Generator().manual_seed(0)

    def randn(*shape, std=0.02):
        return (torch.randn(*shape, generator=g) * std).to(torch.bfloat16)

    def ones(*shape):
        return torch.ones(*shape, dtype=torch.bfloat16)

    sd: dict[str, torch.Tensor] = {}
    # Aux-state combiner + final norms (no embed_tokens / lm_head: tied/borrowed).
    sd["fc.weight"] = randn(h, h * n_aux)
    sd["hidden_norm.weight"] = ones(h)
    sd["norm.weight"] = ones(h)
    for i in range(DRAFT_LAYERS):
        p = f"layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = ones(h)
        sd[f"{p}.post_attention_layernorm.weight"] = ones(h)
        sd[f"{p}.self_attn.q_proj.weight"] = randn(q, h)
        sd[f"{p}.self_attn.k_proj.weight"] = randn(kv, h)
        sd[f"{p}.self_attn.v_proj.weight"] = randn(kv, h)
        sd[f"{p}.self_attn.o_proj.weight"] = randn(h, q)
        sd[f"{p}.self_attn.q_norm.weight"] = ones(hd)
        sd[f"{p}.self_attn.k_norm.weight"] = ones(hd)
        sd[f"{p}.mlp.gate_proj.weight"] = randn(inter, h)
        sd[f"{p}.mlp.up_proj.weight"] = randn(inter, h)
        sd[f"{p}.mlp.down_proj.weight"] = randn(h, inter)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="/home/pat/code/_models/ZAYA1-8B-DFlash-stub",
        help="output drafter directory (real disk, never tmpfs)",
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = build_config()
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    sd = build_state_dict()
    save_file(sd, os.path.join(args.out, "model.safetensors"))

    n_params = sum(t.numel() for t in sd.values())
    print(f"wrote {args.out}")
    print(f"  config: {len(cfg)} keys, dflash_config.target_layer_ids="
          f"{cfg['dflash_config']['target_layer_ids']}")
    print(f"  weights: {len(sd)} tensors, {n_params/1e6:.1f}M params")
    print("  (embed_tokens + lm_head intentionally omitted — borrowed from ZAYA)")


if __name__ == "__main__":
    main()
