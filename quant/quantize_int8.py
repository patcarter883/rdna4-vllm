"""Direct INT8 (W8A8) quantizer for ZAYA1-8B safetensors.

ZAYA1 has no HuggingFace ``transformers`` implementation (the repo ships only
config + safetensors; vLLM provides the modeling code), so the standard
``llm-compressor`` flow — which does ``AutoModelForCausalLM.from_pretrained`` and
calibrates on GPU — cannot load it. This script instead quantizes the weight
tensors *directly* in the safetensors files, producing a ``compressed-tensors``
checkpoint vLLM loads natively. No model instantiation, no calibration, no GPU,
so it does not contend with the running inference server.

Scheme (matches vLLM's ``CompressedTensorsW8A8Int8`` / ``...Int8MoEMethod``):
- **weights**: per-output-channel symmetric INT8, static (a ``weight_scale``
  per output row stored alongside each quantized weight).
- **activations**: per-token symmetric INT8, quantized *dynamically* at runtime
  (nothing stored). On gfx1100 both the Linear and MoE INT8 paths run through
  Triton kernels (``triton_scaled_mm`` / ``TritonExperts``), which use native
  RDNA3 WMMA int8 — a memory *and* compute win.

v1 quantizes only the **MoE experts** (``zaya_block.experts.local_experts.*.
linear_fc{1,2}``) — the dominant share of the 8B weights and the freed VRAM that
lets ``--max-num-seqs`` rise from 6 toward 16. Attention/CCA projections, the
MoE router, norms, and the (tied) embedding are left in bf16: the router and the
CCA state path are accuracy-sensitive, and the embedding is not a decode GEMM.

Validation (load + accuracy) requires a GPU window — see the plan's
"needs-window" steps.

Usage:
    .venv-quant/bin/python -m quant.quantize_int8 \\
        --src ~/.cache/huggingface/hub/models--Zyphra--ZAYA1-8B/snapshots/<hash> \\
        --dst ~/models/ZAYA1-8B-int8
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Expert FFN weights: the only tensors quantized in v1.
QUANT_RE = re.compile(
    r"\.zaya_block\.experts\.local_experts\.\d+\.linear_fc[12]\.weight$"
)

# Files copied verbatim (tokenizer, templates, etc.). config.json is rewritten.
COPY_FILES = [
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
]

# Everything not quantized in v1. Matched by vLLM against runtime module names;
# targeting "Linear" would otherwise also catch these.
IGNORE = [
    "lm_head",
    "re:.*embed_tokens.*",
    "re:.*self_attn.*",  # CCA input projections + conv + o_proj (bf16)
    "re:.*router.*",  # MoE router (down_proj / router_mlp), accuracy-sensitive
]


def quant_config() -> dict:
    """compressed-tensors W8A8 dynamic-token INT8 config block."""
    return {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "quantization_status": "compressed",
        "ignore": IGNORE,
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "channel",
                    "dynamic": False,
                    "group_size": None,
                    "observer": "minmax",
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                    "group_size": None,
                    "observer": None,
                },
                "output_activations": None,
            }
        },
    }


def per_channel_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel (dim 0) INT8 quantization of a (out, in)."""
    wf = w.to(torch.float32)
    absmax = wf.abs().amax(dim=1, keepdim=True)  # (out, 1)
    scale = (absmax / 127.0).clamp(min=1e-8)
    q = torch.round(wf / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def shard_files(src: Path) -> list[Path]:
    files = sorted(src.glob("model-*-of-*.safetensors"))
    if not files:
        single = src / "model.safetensors"
        if single.exists():
            return [single]
        raise FileNotFoundError(f"no safetensors found in {src}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="ZAYA1-8B snapshot dir")
    ap.add_argument("--dst", required=True, help="output dir for the INT8 model")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    weight_map: dict[str, str] = {}
    total_size = 0
    n_quant = 0
    expert_before = 0  # bf16 bytes of expert weights
    expert_after = 0  # int8 + scale bytes of expert weights

    files = shard_files(src)
    for path in files:
        out_name = path.name
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if QUANT_RE.search(key):
                    expert_before += t.numel() * t.element_size()
                    q, scale = per_channel_int8(t)
                    out_tensors[key] = q
                    out_tensors[key[: -len(".weight")] + ".weight_scale"] = scale
                    n_quant += 1
                    expert_after += q.numel() * q.element_size()
                    expert_after += scale.numel() * scale.element_size()
                else:
                    out_tensors[key] = t
        for name, tensor in out_tensors.items():
            weight_map[name] = out_name
            total_size += tensor.numel() * tensor.element_size()
        save_file(out_tensors, str(dst / out_name), metadata={"format": "pt"})
        print(f"  wrote {out_name}  ({len(out_tensors)} tensors)")

    # index.json
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    # config.json with quantization_config
    config = json.loads((src / "config.json").read_text())
    config["quantization_config"] = quant_config()
    (dst / "config.json").write_text(json.dumps(config, indent=2))

    for name in COPY_FILES:
        srcf = src / name
        if srcf.exists():
            shutil.copy2(srcf, dst / name)

    print("\n" + "=" * 60)
    print(f"quantized expert tensors : {n_quant}")
    print(
        f"expert weights  bf16 -> int8 : "
        f"{expert_before / 1e9:.2f} GB -> {expert_after / 1e9:.2f} GB "
        f"(saved {(expert_before - expert_after) / 1e9:.2f} GB)"
    )
    print(f"total checkpoint size    : {total_size / 1e9:.2f} GB")
    print(f"output                   : {dst}")
    print("=" * 60)


if __name__ == "__main__":
    main()
