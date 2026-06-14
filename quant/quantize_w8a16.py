"""Weight-only INT8 (W8A16) quantizer for ZAYA1-8B MoE experts.

The W8A8 attempt produced garbage: the weights dequant at ~0.9% error (fine),
so the damage came from the *dynamic per-token activation* int8. This recipe
keeps int8 weights but leaves activations in bf16 (weight-only), which still
frees ~8GB VRAM and cuts the decode weight-bandwidth bottleneck, without the
activation-quant loss.

vLLM serves weight-only int8 MoE through the compressed-tensors **wNa16** path,
which on ROCm uses the Triton ``CompressedTensorsWNA16MoEMethod`` (confirmed:
``get_moe_method`` routes ``is_rocm()`` to the non-Marlin kernel). That kernel
requires the GPTQ-style ``pack-quantized`` layout:
- **group** quantization (group_size along the input dim), symmetric int8,
- weights transposed to (in, out) and bit-packed 4×int8 -> int32 along the in
  dim (``compressed_tensors.pack_to_int32``),
- a per-(group,out) ``weight_scale``.

ZAYA stores the gate+up projection *merged* as ``linear_fc1`` (dense (2I, H));
we emit the merged packed/scale tensors and let the (patched) ZAYA loader split
gate/up along the out dim. ``linear_fc2`` (down, (H, I)) is single (w2).
``weight_shape``/``g_idx`` are Marlin-only and intentionally omitted.

Usage:
    .venv-quant/bin/python -m quant.quantize_w8a16 \\
        --src <ZAYA1-8B snapshot> --dst ~/models/ZAYA1-8B-w8a16 --group-size 128
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from safetensors import safe_open
from safetensors.torch import save_file

QUANT_RE = re.compile(
    r"\.zaya_block\.experts\.local_experts\.\d+\.linear_fc[12]\.weight$"
)

COPY_FILES = [
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
]

IGNORE = [
    "lm_head",
    "re:.*embed_tokens.*",
    "re:.*self_attn.*",
    "re:.*router.*",
]


def quant_config(group_size: int) -> dict:
    """compressed-tensors wNa16 (weight-only int8, group) config."""
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "ignore": IGNORE,
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": group_size,
                    "dynamic": False,
                    "observer": "minmax",
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
    }


def group_quantize(w: torch.Tensor, group_size: int):
    """Symmetric per-(out-channel, in-group) int8 quantization.

    *w* is dense (out, in). Returns (q int8 (out, in), scale (out, in//gs)).
    """
    out, inp = w.shape
    assert inp % group_size == 0, f"in dim {inp} not divisible by group {group_size}"
    wf = w.to(torch.float32).reshape(out, inp // group_size, group_size)
    absmax = wf.abs().amax(dim=-1, keepdim=True)  # (out, groups, 1)
    scale = (absmax / 127.0).clamp(min=1e-8)
    q = torch.round(wf / scale).clamp(-127, 127).to(torch.int8)
    q = q.reshape(out, inp)
    scale = scale.squeeze(-1)  # (out, groups)
    return q, scale


def pack_expert(w: torch.Tensor, group_size: int):
    """Dense (out, in) -> wNa16 checkpoint layout (out-major, int32-packed).

    Returns (weight_packed int32 (out, in//4), weight_scale bf16 (out, groups)).
    Stored out-major (untransposed): vLLM's FusedMoE weight_loader transposes
    is_transposed params on load, and gate/up split along dim 0 (the out dim).
    """
    q, scale = group_quantize(w, group_size)  # q (out,in), scale (out, groups)
    packed = pack_to_int32(q, num_bits=8, packed_dim=1).contiguous()  # (out, in//4)
    scale = scale.contiguous().to(torch.bfloat16)  # (out, groups)
    return packed, scale


def shard_files(src: Path) -> list[Path]:
    files = sorted(src.glob("model-*-of-*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors found in {src}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    gs = args.group_size

    weight_map: dict[str, str] = {}
    total_size = 0
    n_quant = 0
    before = after = 0

    for path in shard_files(src):
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if QUANT_RE.search(key):
                    before += t.numel() * t.element_size()
                    packed, scale = pack_expert(t, gs)
                    base = key[: -len(".weight")]
                    out_tensors[base + ".weight_packed"] = packed
                    out_tensors[base + ".weight_scale"] = scale
                    n_quant += 1
                    after += packed.numel() * 4 + scale.numel() * 2
                else:
                    out_tensors[key] = t
        for name, tensor in out_tensors.items():
            weight_map[name] = path.name
            total_size += tensor.numel() * tensor.element_size()
        save_file(out_tensors, str(dst / path.name), metadata={"format": "pt"})
        print(f"  wrote {path.name} ({len(out_tensors)} tensors)")

    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map})
    )
    config = json.loads((src / "config.json").read_text())
    config["quantization_config"] = quant_config(gs)
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    for name in COPY_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)

    print("\n" + "=" * 60)
    print(f"quantized expert tensors : {n_quant} (group_size={gs})")
    print(
        f"expert weights bf16 -> int8(packed) : "
        f"{before / 1e9:.2f} GB -> {after / 1e9:.2f} GB"
    )
    print(f"total checkpoint size : {total_size / 1e9:.2f} GB")
    print(f"output : {dst}")
    print("=" * 60)


if __name__ == "__main__":
    main()
