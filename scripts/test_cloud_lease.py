"""Offline test of the cloud-lease TEARDOWN GUARANTEE — the one safety property that matters
(a leaked instance bills money). No cloud, no network: a FakeBackend records destroy() calls and
SSH is stubbed to no-ops. Run: PYTHONPATH=scripts python3 scripts/test_cloud_lease.py
"""
import os
import sys
import tempfile
import time
import types

# Isolate config (managed SSH key, state dir) into a temp dir before importing the package.
_TMP = tempfile.mkdtemp(prefix="cloud-lease-test-")
os.environ["CLOUD_LEASE_HOME"] = _TMP

import cloud_lease.ssh as ssh          # noqa: E402
from cloud_lease.backend import Backend, Instance  # noqa: E402
from cloud_lease.lease import Lease     # noqa: E402

time.sleep = lambda *a, **k: None       # don't actually sleep through destroy-retry backoff

# --- stub SSH so nothing touches the network (signatures match the real ones) -----------------
ssh.wait_ssh = lambda inst, timeout: None
ssh.run = lambda inst, cmd, check=True, quiet=False, timeout=None: types.SimpleNamespace(returncode=0)
ssh.rsync_up = lambda inst, local, remote, excludes=(), delete=True: None
ssh.rsync_down = lambda inst, remote, local, check=True: None
ssh.write_file = lambda inst, path, content, mode=None, timeout=30: None
# The run path is now run_detached (replaced run_stream); delegate to the per-test run_stream stub so
# the existing stubs/cases still drive it, returning the command's int rc.
ssh.run_detached = lambda inst, cmd, workdir, **k: ssh.run_stream(inst, cmd).returncode


class FakeBackend(Backend):
    name = "fake"

    def __init__(self, destroy_fail=0, provision_raises=False):
        super().__init__(token="x")
        self.destroyed = []
        self.label = None
        self._destroy_fail = destroy_fail
        self._provision_raises = provision_raises
        self._live = []          # simulates provider-side billing instances, keyed by label

    def provision(self, gpu, region, spot, label, sshkey_pub):
        self.label = label
        self._live.append({"id": "inst-123", "label": label})   # provider created (billing) it
        if self._provision_raises:
            raise RuntimeError("connection dropped after the instance was created")
        return "inst-123"

    def wait_ready(self, instance_id, timeout):
        return Instance(id=instance_id, ssh_host="10.0.0.1", ssh_port=22, ssh_user="root")

    def destroy(self, instance_id):
        if self._destroy_fail > 0:
            self._destroy_fail -= 1
            raise RuntimeError("transient destroy failure")
        self.destroyed.append(instance_id)
        self._live = [x for x in self._live if x["id"] != instance_id]

    def list_live(self):
        return list(self._live)


def _args(command):
    return types.SimpleNamespace(
        gpu="a100-80", region=None, spot=False, name="test",
        workdir=_TMP, remote_workdir="/workspace/repo",
        dataset=None, remote_dataset="/workspace/data",
        checkpoint_dir=None, remote_out=None, sync_every=300,
        provision_timeout=10, setup=None, setup_timeout=1800, exclude=[], env=[],
        restart_on_crash=0, command=command)


def _run(command, stream_impl, **be_kw):
    be = FakeBackend(**be_kw)
    ssh.run_stream = stream_impl
    rc = Lease(be, _args(command)).run()
    return be, rc


results = []


def check(name, cond):
    results.append((name, cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


print("teardown guarantee:")

# 1. normal exit (rc 0) -> destroyed exactly once
be, rc = _run(["true"], lambda i, c: types.SimpleNamespace(returncode=0))
check("normal exit destroys once", be.destroyed == ["inst-123"])
check("normal exit returns command rc", rc == 0)

# 2. command exits non-zero -> still destroyed, rc propagated
be, rc = _run(["false"], lambda i, c: types.SimpleNamespace(returncode=7))
check("non-zero exit still destroys", be.destroyed == ["inst-123"])
check("non-zero exit propagates rc", rc == 7)

# 3. exception mid-run -> destroyed via finally, THEN the error propagates.
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
be, rc = _run(["x"], _sig)
check("signal interrupt still destroys", be.destroyed == ["inst-123"])
check("signal interrupt returns 130", rc == 130)

# 5. label carries the cloud-lease- prefix + a unique suffix (cloud-status reaps only ours)
be, _ = _run(["true"], lambda i, c: types.SimpleNamespace(returncode=0))
check("instance label is cloud-lease- prefixed", be.label.startswith("cloud-lease-test"))

# 6. idempotent teardown — a double-fire destroys only once
be = FakeBackend()
lease = Lease(be, _args(["true"]))
lease.instance_id = "inst-123"
lease.instance = Instance(id="inst-123", ssh_host="10.0.0.1")
lease._teardown()
lease._teardown()
check("teardown is idempotent (destroy once)", be.destroyed == ["inst-123"])

# 7. (#4) destroy retries transient failures instead of giving up after one
be, rc = _run(["true"], lambda i, c: types.SimpleNamespace(returncode=0), destroy_fail=2)
check("destroy retried through transient failures", be.destroyed == ["inst-123"])

# 8. (#2) signal/crash mid-provision (id never captured) -> orphan swept by label and destroyed
be = FakeBackend(provision_raises=True)
ssh.run_stream = lambda i, c: types.SimpleNamespace(returncode=0)
try:
    Lease(be, _args(["x"])).run()
except RuntimeError:
    pass
check("mid-provision orphan swept by label", be.destroyed == ["inst-123"])

# 9. (#5) remote command args with spaces are shell-quoted, not re-split
captured = {}
def _capture(i, command):
    captured["cmd"] = command
    return types.SimpleNamespace(returncode=0)
_run(["python", "train.py", "--note", "hello world"], _capture)
check("command args are shell-quoted", "'hello world'" in captured.get("cmd", ""))

# 10. (--restart-on-crash) a non-zero exit relaunches; the retry succeeds -> final rc 0, run twice
calls = {"n": 0}
def _flaky(i, c):
    calls["n"] += 1
    return types.SimpleNamespace(returncode=0 if calls["n"] > 1 else 7)
be = FakeBackend()
ssh.run_stream = _flaky
args = _args(["true"]); args.restart_on_crash = 2
rc = Lease(be, args).run()
check("restart-on-crash relaunches then succeeds (rc 0)", rc == 0)
check("restart-on-crash ran the command twice", calls["n"] == 2)
check("restart-on-crash still destroys", be.destroyed == ["inst-123"])

# 11. (--env) bare NAME pulls from our env, written to a mode-600 remote file, sourced before CMD
os.environ["CLOUD_LEASE_TEST_SECRET"] = "s3 cr3t!"   # spaces/metachars -> must be shell-quoted
written = {}
ssh.write_file = lambda inst, path, content, mode=None, timeout=30: written.update(
    path=path, content=content, mode=mode)
runcmd = {}
ssh.run_detached = lambda inst, cmd, workdir, **k: (runcmd.update(cmd=cmd) or types.SimpleNamespace(returncode=0)).returncode
be = FakeBackend()
ssh.run_stream = lambda i, c: types.SimpleNamespace(returncode=0)
args = _args(["true"]); args.env = ["CLOUD_LEASE_TEST_SECRET", "MISSING_VAR_XYZ"]
Lease(be, args).run()
check("env: secret written to remote file (shell-quoted)",
      "export CLOUD_LEASE_TEST_SECRET='s3 cr3t!'" in written.get("content", ""))
check("env: remote file is mode 600", written.get("mode") == "600")
check("env: command sources the env file", runcmd.get("cmd", "").startswith("source "))
check("env: missing local var is skipped (not written)", "MISSING_VAR_XYZ" not in written.get("content", ""))
# restore the delegating stub for any later reuse
ssh.run_detached = lambda inst, cmd, workdir, **k: ssh.run_stream(inst, cmd).returncode

ok = all(c for _, c in results)
print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'} ({sum(c for _, c in results)}/{len(results)})")
sys.exit(0 if ok else 1)
