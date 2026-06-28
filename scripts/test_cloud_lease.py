"""Offline test of the cloud-lease TEARDOWN GUARANTEE — the one safety property that matters
(a leaked instance bills money). No cloud, no network: a FakeBackend records destroy() calls and
SSH is stubbed to no-ops. Run: PYTHONPATH=scripts python3 scripts/test_cloud_lease.py
"""
import os
import sys
import tempfile
import types

# Isolate config (managed SSH key, state dir) into a temp dir before importing the package.
_TMP = tempfile.mkdtemp(prefix="cloud-lease-test-")
os.environ["CLOUD_LEASE_HOME"] = _TMP

import cloud_lease.ssh as ssh          # noqa: E402
from cloud_lease.backend import Backend, Instance  # noqa: E402
from cloud_lease.lease import Lease     # noqa: E402

# --- stub SSH so nothing touches the network --------------------------------------------------
ssh.wait_ssh = lambda inst, timeout: None
ssh.run = lambda inst, cmd, check=True, quiet=False: types.SimpleNamespace(returncode=0)
ssh.rsync_up = lambda inst, local, remote, excludes=(): None
ssh.rsync_down = lambda inst, remote, local, check=True: None


class FakeBackend(Backend):
    name = "fake"

    def __init__(self, run_behavior="ok"):
        super().__init__(token="x")
        self.destroyed = []
        self.provisioned = None
        self.label = None
        self._behavior = run_behavior

    def provision(self, gpu, region, spot, label, sshkey_pub):
        self.provisioned = "inst-123"
        self.label = label
        return "inst-123"

    def wait_ready(self, instance_id, timeout):
        return Instance(id=instance_id, ssh_host="10.0.0.1", ssh_port=22, ssh_user="root")

    def destroy(self, instance_id):
        self.destroyed.append(instance_id)

    def list_live(self):
        return []


def _args(command):
    return types.SimpleNamespace(
        gpu="a100-80", region=None, spot=False, name="test",
        workdir=_TMP, remote_workdir="/workspace/repo",
        dataset=None, remote_dataset="/workspace/data",
        checkpoint_dir=None, remote_out=None, sync_every=300,
        provision_timeout=10, setup=None, exclude=[], command=command)


def _run(behavior, command, stream_impl):
    be = FakeBackend(behavior)
    ssh.run_stream = stream_impl
    rc = Lease(be, _args(command)).run()
    return be, rc


results = []


def check(name, cond):
    results.append((name, cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


print("teardown guarantee:")

# 1. normal exit (rc 0) -> destroyed exactly once
be, rc = _run("ok", ["true"], lambda i, c: types.SimpleNamespace(returncode=0))
check("normal exit destroys once", be.destroyed == ["inst-123"])
check("normal exit returns command rc", rc == 0)

# 2. command exits non-zero -> still destroyed, rc propagated
be, rc = _run("fail", ["false"], lambda i, c: types.SimpleNamespace(returncode=7))
check("non-zero exit still destroys", be.destroyed == ["inst-123"])
check("non-zero exit propagates rc", rc == 7)

# 3. exception mid-run -> destroyed via finally, THEN the error propagates (correct: teardown is
#    guaranteed, but an unexpected error should still surface with its traceback).
def _boom(i, c):
    raise RuntimeError("kernel panic")
be = FakeBackend()
ssh.run_stream = _boom
propagated = False
try:
    Lease(be, _args(["x"])).run()
except RuntimeError:
    propagated = True
check("exception mid-run still destroys", be.destroyed == ["inst-123"])
check("exception propagates after teardown", propagated)

# 4. signal/Ctrl-C (modelled as KeyboardInterrupt from the run) -> destroyed, rc 130
def _sig(i, c):
    raise KeyboardInterrupt
be, rc = _run("ok", ["x"], _sig)
check("signal interrupt still destroys", be.destroyed == ["inst-123"])
check("signal interrupt returns 130", rc == 130)

# 5. label carries the cloud-lease- prefix (so cloud-status can see/reap only ours)
be, _ = _run("ok", ["true"], lambda i, c: types.SimpleNamespace(returncode=0))
check("instance label is cloud-lease- prefixed", be.label == "cloud-lease-test")

# 6. idempotent teardown — a double-fire destroys only once
be = FakeBackend()
lease = Lease(be, _args(["true"]))
lease.instance_id = "inst-123"
lease.instance = Instance(id="inst-123", ssh_host="10.0.0.1")
lease._teardown()
lease._teardown()
check("teardown is idempotent (destroy once)", be.destroyed == ["inst-123"])

ok = all(c for _, c in results)
print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'} ({sum(c for _, c in results)}/{len(results)})")
sys.exit(0 if ok else 1)
