#!/usr/bin/env bash
# cloud-lease — cloud sibling of gpu-lease. Provisions ONE cloud GPU, syncs code/data up, runs a
# command over SSH, periodically syncs checkpoints down, and GUARANTEES teardown of the instance on
# exit / Ctrl-C / crash (a leaked instance bills money — this is the whole point). See
# docs/cloud-lease/design.md.
#
# This is a thin shim; the engine is Python (scripts/cloud_lease/), stdlib-only.
#
# USAGE
#   cloud-lease.sh --provider {vultr|runpod|vastai} --gpu <shorthand> [opts] -- CMD...
#
# EXAMPLES
#   # Planned full fine-tune on Vultr A100-80, mirror checkpoints locally every 5 min:
#   scripts/cloud-lease.sh --provider vultr --gpu a100-80 --name tidar-ft \
#     --checkpoint-dir ./ckpt/tidar --remote-out /workspace/repo/out \
#     -- python -m zaya.tidar.train_tidar_zaya --out out --ckpt-every 50
#
#   # Ad-hoc LoRA burst on a RunPod community (spot) A6000:
#   scripts/cloud-lease.sh --provider runpod --gpu a6000-48 --spot --name lora \
#     --checkpoint-dir ./ckpt/lora --remote-out /workspace/repo/out \
#     -- python -m zaya.tidar.train_tidar_zaya --out out
#
#   # Discover live plan/gpu ids for a provider (to pick --gpu / fill gpu_map):
#   scripts/cloud-lease.sh list-gpus --provider vultr
#
# Keys: env ($VULTR_API_KEY / $RUNPOD_API_KEY) or ~/.config/cloud-lease/<provider>.env (chmod 600).
# NEVER pass a key on the command line. Spot tiers REQUIRE the trainer's checkpoint/resume.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
# `list-gpus`/`status` may be passed through directly; otherwise default to the `lease` subcommand.
case "${1:-}" in
  lease|status|list-gpus|-h|--help) exec python3 -m cloud_lease.cli "$@" ;;
  *)                                 exec python3 -m cloud_lease.cli lease "$@" ;;
esac
