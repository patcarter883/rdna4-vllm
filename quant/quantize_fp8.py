"""Direct FP8 (W8A8, e4m3) quantizer for ZAYA1-8B safetensors.

The RDNA4 (gfx1201) counterpart of ``quantize_int8.py`` — same direct
safetensors rewrite (ZAYA1 has no ``transformers`` modeling code, so the
calibrate-on-GPU ``llm-compressor`` flow can't load it; see that script's
docstring), but emitting OCP **float8_e4m3fn** expert weights. No model
instantiation, no calibration, no GPU.

Scheme (matches vLLM's ``CompressedTensorsW8A8Fp8`` /
``CompressedTensorsW8A8Fp8MoEMethod`` per-channel + dynamic-token path):
- **weights**: per-output-channel symmetric FP8 e4m3, static (`weight_scale`
  per output row, float32).
- **activations**: per-token symmetric FP8, quantized dynamically at runtime
  (nothing stored).

gfx1201 reports OCP e4m3 (``RocmPlatform.fp8_dtype() == float8_e4m3fn``; the
fnuz variant is MI300-only), and RDNA4 has native FP8 WMMA — unlike the int8
MoE path on RDNA3/gfx1100, which vLLM's Triton kernels miscompute (see
README). Whether the fp8 fused-MoE path is numerically sound on gfx1201 is
exactly what the deploy runbook validates.

Like v1 of the int8 script, only the **MoE experts**
(``zaya_block.experts.local_experts.*.linear_fc{1,2}``) are quantized — ~95%
of the 8B weights. Attention/CCA projections, the MoE router, norms, and the
tied embedding stay bf16.

Usage:
    .venv-quant/bin/python -m quant.quantize_fp8 \\
        --src ~/.cache/huggingface/hub/models--Zyphra--ZAYA1-8B/snapshots/<hash> \\
        --dst ~/models/ZAYA1-8B-fp8
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0

# Expert FFN weights: the only tensors quantized.
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

# Everything not quantized. Matched by vLLM against runtime module names;
# targeting "Linear" would otherwise also catch these.
IGNORE = [
    "lm_head",
    "re:.*embed_tokens.*",
    "re:.*self_attn.*",  # CCA input projections + conv + o_proj (bf16)
    "re:.*router.*",  # MoE router (down_proj / router_mlp), accuracy-sensitive
]


def quant_config() -> dict:
    """compressed-tensors W8A8 dynamic-token FP8 config block."""
    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "quantization_status": "compressed",
        "ignore": IGNORE,
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "channel",
                    "dynamic": False,
                    "group_size": None,
                    "observer": "minmax",
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
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


def per_channel_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel (dim 0) FP8-e4m3 quantization of (out, in)."""
    wf = w.to(torch.float32)
    absmax = wf.abs().amax(dim=1, keepdim=True)  # (out, 1)
    scale = (absmax / FP8_MAX).clamp(min=1e-12)
    q = (wf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
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
    ap.add_argument("--dst", required=True, help="output dir for the FP8 model")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    weight_map: dict[str, str] = {}
    total_size = 0
    n_quant = 0
    expert_before = 0  # bf16 bytes of expert weights
    expert_after = 0  # fp8 + scale bytes of expert weights
    max_err = 0.0  # worst relative dequant error across expert tensors

    files = shard_files(src)
    for path in files:
        out_name = path.name
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if QUANT_RE.search(key):
                    expert_before += t.numel() * t.element_size()
                    q, scale = per_channel_fp8(t)
                    out_tensors[key] = q
                    out_tensors[key[: -len(".weight")] + ".weight_scale"] = scale
                    n_quant += 1
                    expert_after += q.numel() * q.element_size()
                    expert_after += scale.numel() * scale.element_size()
                    deq = q.to(torch.float32) * scale
                    ref = t.to(torch.float32)
                    err = (deq - ref).norm() / ref.norm().clamp(min=1e-12)
                    max_err = max(max_err, err.item())
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
        f"expert weights  bf16 -> fp8 : "
        f"{expert_before / 1e9:.2f} GB -> {expert_after / 1e9:.2f} GB "
        f"(saved {(expert_before - expert_after) / 1e9:.2f} GB)"
    )
    print(f"worst per-tensor rel. err: {max_err:.4%}")
    print(f"total checkpoint size    : {total_size / 1e9:.2f} GB")
    print(f"output                   : {dst}")
    print("=" * 60)


if __name__ == "__main__":
    main()
