"""cloud_lease — cloud sibling of gpu-lease.

A small, dependency-free (stdlib-only) engine that provisions a single cloud GPU instance,
syncs code/data up, runs a command over SSH, periodically syncs checkpoints back down, and
GUARANTEES teardown of the instance on exit / Ctrl-C / crash. See docs/cloud-lease/design.md.

Front-ends: scripts/cloud-lease.sh (lease) and scripts/cloud-status.sh (reconcile/reap).
"""

__version__ = "0.1.0"
