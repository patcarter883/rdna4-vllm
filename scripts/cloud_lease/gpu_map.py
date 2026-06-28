"""GPU shorthand -> per-provider provisioning ids.

⚠ Plan / gpuType ids DRIFT and MUST be verified live before the first real run:
    cloud-lease --provider vultr  --list-gpus      (GET /v2/plans)
    cloud-lease --provider runpod --list-gpus      (GET /v1/gpu-types ... ish)
The maps below are best-effort as of 2026-06; an unmapped shorthand fails loudly with a hint to
run --list-gpus rather than guessing.

RunPod identifies GPUs by display-name-like ids ("NVIDIA A100 80GB PCIe"), which are stable-ish.
Vultr identifies by plan id (e.g. "vcg-a100-...") which is NOT guessable — leave UNVERIFIED entries
as None so provisioning refuses until you fill them from --list-gpus.
"""

# Vultr cloud-GPU plan ids — VERIFY via --list-gpus, then fill in. None => refuse to provision.
# Confirmed live 2026-06-28 via --list-gpus.
# NOTE: Vultr fractional cloud-GPU is A16 / A40 / L40S only; the biggest SINGLE card is 48 GB.
# There is NO 80 GB single card here (A100/H100 are bare-metal, a separate endpoint) → for an
# 80 GB full fine-tune use RunPod (a100-80). Region availability varies (see comments).
_VULTR = {
    "a16-2": "vcg-a16-2c-8g-2vram",          # 2 GB slice, $0.059/hr — cheap throwaway/smoke (many regions)
    "a16-16": "vcg-a16-6c-64g-16vram",       # full A16, 16 GB
    "a40-48": "vcg-a40-24c-120g-48vram",     # full A40, 48 GB, $1.712/hr — ⚠ no region avail as of 2026-06-28
    "l40s-48": "vcg-l40s-16c-180g-48vram",   # full L40S, 48 GB, $1.671/hr — atl only (use --region atl)
    "a100-80": None,                          # not on Vultr vcg (bare-metal only) → use RunPod for 80 GB
}

# RunPod gpuTypeId values (display-name form). Confirm against --list-gpus.
_RUNPOD = {
    "a100-80": "NVIDIA A100 80GB PCIe",
    "a6000-48": "NVIDIA RTX A6000",
    "a40-48": "NVIDIA A40",
    "l40s-48": "NVIDIA L40S",
    "h100-80": "NVIDIA H100 80GB HBM3",
    "rtx4090-24": "NVIDIA GeForce RTX 4090",
}


def _resolve(table, gpu, provider):
    if gpu not in table:
        raise SystemExit(
            f"cloud-lease: unknown GPU shorthand '{gpu}' for {provider}. "
            f"Known: {', '.join(sorted(table))}. Run `--list-gpus` to see live ids.")
    val = table[gpu]
    if val is None:
        raise SystemExit(
            f"cloud-lease: GPU '{gpu}' is not yet mapped for {provider} "
            f"(plan id unverified). Run `--provider {provider} --list-gpus`, then set it "
            f"in scripts/cloud_lease/gpu_map.py.")
    return val


def vultr_plan(gpu):
    return _resolve(_VULTR, gpu, "vultr")


def runpod_gpu(gpu):
    return _resolve(_RUNPOD, gpu, "runpod")
