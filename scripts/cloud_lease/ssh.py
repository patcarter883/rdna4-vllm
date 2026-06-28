"""SSH / rsync helpers — provider-independent. Backends only do provision/wait/destroy;
all data movement and command execution goes through the same SSH path here, keyed off the
Instance's ssh coordinates."""
import shlex
import subprocess
import sys
import time

from . import config

_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", f"UserKnownHostsFile={config.KNOWN_HOSTS}",
    "-o", "ServerAliveInterval=30",
    "-o", "ConnectTimeout=10",
    "-i", str(config.SSH_KEY),
]


def _target(inst):
    return f"{inst.ssh_user}@{inst.ssh_host}"


def _ssh_e(inst):
    # The -e string rsync hands to ssh. Paths here have no spaces (under ~/.config).
    return "ssh -p %d %s" % (inst.ssh_port, " ".join(_OPTS))


def run(inst, command, check=True, quiet=False):
    """Run a one-shot command on the instance (captured/quiet)."""
    cmd = ["ssh", "-p", str(inst.ssh_port), *_OPTS, _target(inst), f"bash -lc {shlex.quote(command)}"]
    kw = {"check": check}
    if quiet:
        kw["stdout"] = subprocess.DEVNULL
        kw["stderr"] = subprocess.DEVNULL
    return subprocess.run(cmd, **kw)


def run_stream(inst, command):
    """Run the (long-lived) training command, streaming its stdout/stderr to our terminal."""
    cmd = ["ssh", "-tt", "-p", str(inst.ssh_port), *_OPTS, _target(inst),
           f"bash -lc {shlex.quote(command)}"]
    return subprocess.run(cmd)


def rsync_up(inst, local, remote, excludes=()):
    ex = []
    for x in excludes:
        ex += ["--exclude", x]
    cmd = ["rsync", "-az", "--delete", *ex, "-e", _ssh_e(inst),
           local.rstrip("/") + "/", f"{_target(inst)}:{remote}"]
    return subprocess.run(cmd, check=True)


def rsync_down(inst, remote, local, check=True):
    cmd = ["rsync", "-az", "-e", _ssh_e(inst),
           f"{_target(inst)}:{remote.rstrip('/')}/", local.rstrip("/") + "/"]
    return subprocess.run(cmd, check=check)


def wait_ssh(inst, timeout):
    """Poll until the instance answers SSH (the provider can report 'running' before sshd is up)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = run(inst, "true", check=False, quiet=True)
        if r.returncode == 0:
            return
        time.sleep(5)
    raise TimeoutError(f"cloud-lease: SSH to {_target(inst)}:{inst.ssh_port} not ready in {timeout}s")
