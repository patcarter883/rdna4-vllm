"""Lease state records.

There is NO kernel-held lock in the cloud — the provider API is the source of truth. These
records are the LOCAL half used by cloud-status to reconcile: which live instances do we have a
launcher for, which are orphaned (launcher dead), which records are stale. See design §7.

Written atomically (tmp + os.replace) so a crash mid-write can't corrupt a record.
"""
import json
import os
import time

from . import config


def _path(provider, instance_id):
    return config.STATE_DIR / f"{provider}-{instance_id}.json"


def write(provider, instance_id, **fields):
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "provider": provider,
        "instance_id": str(instance_id),
        "pid": os.getpid(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        **fields,
    }
    p = _path(provider, instance_id)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=2))
    os.replace(tmp, p)
    return rec


def remove(provider, instance_id):
    try:
        _path(provider, instance_id).unlink()
    except FileNotFoundError:
        pass


def all_records():
    if not config.STATE_DIR.exists():
        return []
    out = []
    for f in config.STATE_DIR.glob("*.json"):
        try:
            out.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    return out


def pid_alive(pid):
    """True if `pid` is a live process. kill(pid, 0): no error => alive (ours),
    PermissionError => alive (not ours), ProcessLookupError => dead.

    Guard non-positive pids FIRST: a corrupt/missing-pid record yields pid<=0, and
    os.kill(0/-1, 0) targets the whole process group (always 'succeeds') — which would
    misclassify a real orphan as alive and skip reaping it. Treat those as dead."""
    try:
        pid = int(pid)
    except (ValueError, TypeError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
