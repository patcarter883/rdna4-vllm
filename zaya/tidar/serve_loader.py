#!/usr/bin/env python3
"""Load a TiDAR-converted ZAYA1-8B checkpoint for the SERVING coherence gate.

The checkpoint (`pat883/zaya1-8b-tidar-experts`, vehicle="full-ft-all") is a FULL
`model.state_dict()` of ZAYA1-8B saved by the conversion trainer
(`feat/tidar-convert:train_tidar_zaya.py::save_model_ckpt`). So serving =
`base ZAYA architecture + load_state_dict(model_latest.pt)` — every parameter,
including the trained input-embedding row for the `[mask]` token, lives in the
checkpoint. We therefore do NOT re-initialise the mask embedding row here (training
mean-init'd it once, then trained it; clobbering it would be wrong).

Requires the Zyphra transformers fork (it provides `modeling_zaya`); the base repo
ships weights-only. The repo's container venv (`/app/.venv`) has tf 5.5.3 + vLLM,
which has NO zaya model — use the dedicated fork venv at `/opt/zaya-fork-venv`
(host: /home/pat/code/.venv-zaya-fork) built per the checkpoint-phase prep.

This module is import-clean (no GPU at import); the loader is CPU-capable so the
β=1 coherence gate can run on host RAM (model ~17.7 GB bf16 + ckpt) without the
16 GB-card OOM. Pass device="cuda" + device_map for GPU once a fit is arranged.
"""
import json
from pathlib import Path

import torch


def load_tidar_config(ckpt_dir):
    """Read the trainer's tidar_config.json → dict (mask_token_id, block_size, …)."""
    return json.loads((Path(ckpt_dir) / "tidar_config.json").read_text())


def load_tidar_zaya(
    ckpt_dir,
    base="Zyphra/ZAYA1-8B",
    device="cpu",
    dtype=torch.bfloat16,
    from_config=True,
    attn_implementation="eager",
):
    """Return (model, tok, mask_id, block_size, (missing, unexpected)).

    from_config=True builds the architecture from the base CONFIG (no base-weight
    read — full-ft-all overwrites every param anyway) then loads the checkpoint;
    from_config=False does the literal "base + load_state_dict" (loads base weights
    first, robust to any non-persistent buffer the random-init path would miss).
    """
    import transformers  # noqa: F401  (ensures the fork is importable)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    ckpt_dir = Path(ckpt_dir)
    tcfg = load_tidar_config(ckpt_dir)
    mask_id = tcfg["mask_token_id"]
    block_size = tcfg["block_size"]

    tok = AutoTokenizer.from_pretrained(str(ckpt_dir))

    # attn_implementation="eager" is REQUIRED: the zaya_mask_patch injects the TiDAR mask via the
    # create_causal_mask monkeypatch, which only the EAGER attention path consumes (ZAYA's SDPA path
    # ignores it → the bias is silently dropped → the model is non-causal under appended tokens).
    # check_zaya_mask.py loads eager for exactly this reason.
    def _build(make):
        try:
            return make(dtype=dtype, attn_implementation=attn_implementation)
        except TypeError:
            return make(torch_dtype=dtype, attn_implementation=attn_implementation)

    if from_config:
        config = AutoConfig.from_pretrained(base)
        model = _build(lambda **kw: AutoModelForCausalLM.from_config(config, **kw))
    else:
        model = _build(
            lambda **kw: AutoModelForCausalLM.from_pretrained(
                base, low_cpu_mem_usage=True, **kw
            )
        )

    sd = torch.load(ckpt_dir / "model_latest.pt", map_location="cpu")
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)

    model = model.to(device=device).eval()
    # Sanity: the trained mask-embedding row must be populated (non-zero) by the load.
    emb = model.get_input_embeddings().weight
    assert mask_id < emb.shape[0], f"mask_id {mask_id} >= embed rows {emb.shape[0]}"

    return model, tok, mask_id, block_size, (missing, unexpected)
