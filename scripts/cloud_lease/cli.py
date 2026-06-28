"""cloud-lease CLI. Subcommands:
    lease    provision a GPU, run a command, sync checkpoints, guaranteed teardown
    status   reconcile records vs live instances; --reap leaks
    list-gpus  print live provider plan/gpu ids (to fill gpu_map / pick --gpu)

Driven by the bash shims scripts/cloud-lease.sh (lease) and scripts/cloud-status.sh (status).
"""
import argparse
import os
import sys

from . import config, status
from .runpod import RunPodBackend
from .vastai import VastBackend
from .vultr import VultrBackend

BACKENDS = {"vultr": VultrBackend, "runpod": RunPodBackend, "vastai": VastBackend}
PROVIDERS = list(BACKENDS)


def make_backend(provider):
    required = provider != "vastai"
    return BACKENDS[provider](config.api_key(provider, required=required))


def _cmd_lease(a):
    if not a.command:
        sys.exit("cloud-lease: no command given (use '-- CMD…')")
    from .lease import Lease
    return Lease(make_backend(a.provider), a).run()


def _cmd_status(a):
    return status.run(make_backend, PROVIDERS, reap=a.reap)


def _cmd_list_gpus(a):
    from . import http
    if a.provider == "vultr":
        st, body = http.request("GET", "https://api.vultr.com/v2/plans?type=all&per_page=500",
                                 config.api_key("vultr"))
        for p in body.get("plans", []):
            if "gpu" in str(p.get("type", "")).lower() or p.get("gpu_vram_gb"):
                print(f"{p.get('id'):28} {p.get('gpu_type', '')} "
                      f"vram={p.get('gpu_vram_gb', '?')}GB ${p.get('monthly_cost', '?')}/mo")
    elif a.provider == "runpod":
        st, body = http.request("GET", "https://rest.runpod.io/v1/gpu-types",
                                 config.api_key("runpod"))
        items = body if isinstance(body, list) else body.get("gpuTypes", body)
        for g in (items or []):
            print(f"{g.get('id', g)}")
    else:
        sys.exit("cloud-lease: --list-gpus supports vultr/runpod")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="cloud-lease")
    sub = p.add_subparsers(dest="cmd", required=True)

    lease = sub.add_parser("lease", help="provision, run, sync, guaranteed teardown")
    lease.add_argument("--provider", required=True, choices=PROVIDERS)
    lease.add_argument("--gpu", required=True, help="shorthand, e.g. a100-80 (see --list-gpus)")
    lease.add_argument("--region", default=None)
    lease.add_argument("--spot", action="store_true", help="interruptible/community (needs resume)")
    lease.add_argument("--name", default="run", help="label; gets a cloud-lease- prefix")
    lease.add_argument("--workdir", default=os.getcwd(), help="local dir synced up (default: cwd)")
    lease.add_argument("--remote-workdir", default="/workspace/repo")
    lease.add_argument("--dataset", default=None,
                       help="local path (rsync) OR hf://repo / s3:// / r2:// / https:// (pulled on box)")
    lease.add_argument("--remote-dataset", default="/workspace/data")
    lease.add_argument("--checkpoint-dir", default=None,
                       help="local dir mirrored to/from the remote --remote-out (resume + safety)")
    lease.add_argument("--remote-out", default=None,
                       help="dir on the instance the trainer writes checkpoints to")
    lease.add_argument("--sync-every", dest="sync_every", type=int, default=300,
                       help="seconds between checkpoint sync-downs (default 300)")
    lease.add_argument("--provision-timeout", type=int, default=900)
    lease.add_argument("--setup", default=None, help="one-shot shell run on the box before CMD")
    lease.add_argument("--setup-timeout", dest="setup_timeout", type=int, default=1800,
                       help="seconds before a hung --setup is killed -> teardown (a wedged GPU pod "
                            "can hang torch's HIP init forever on a no-timeout ssh; default 1800)")
    lease.add_argument("--env", action="append", default=[], dest="env",
                       help="NAME or NAME=VALUE exported on the box before setup+CMD (repeatable). "
                            "A bare NAME pulls VALUE from OUR environment, so a secret (e.g. "
                            "HF_TOKEN) stays off the command line; values are written to a mode-600 "
                            "remote file via stdin and sourced before each remote step.")
    lease.add_argument("--restart-on-crash", dest="restart_on_crash", type=int, default=0,
                       help="relaunch CMD up to N times on a non-zero exit (the trainer resumes from "
                            "its box-local checkpoint). For fast runs this is cheaper than syncing.")
    lease.add_argument("--exclude", action="append", default=[".git", "__pycache__", "*.pyc"],
                       help="rsync excludes for the working-dir upload (repeatable)")
    lease.add_argument("command", nargs=argparse.REMAINDER,
                       help="-- CMD… run on the instance in --remote-workdir")
    lease.set_defaults(func=_cmd_lease)

    st = sub.add_parser("status", help="reconcile records vs live instances")
    st.add_argument("--reap", action="store_true", help="destroy STRAY/ORPHAN instances")
    st.set_defaults(func=_cmd_status)

    lg = sub.add_parser("list-gpus", help="print live provider plan/gpu ids")
    lg.add_argument("--provider", required=True, choices=PROVIDERS)
    lg.set_defaults(func=_cmd_list_gpus)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    # argparse REMAINDER keeps a leading '--'; drop it.
    if getattr(a, "command", None) and a.command and a.command[0] == "--":
        a.command = a.command[1:]
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
