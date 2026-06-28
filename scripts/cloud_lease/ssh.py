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


def run(inst, command, check=True, quiet=False, timeout=None, capture=False):
    """Run a one-shot command on the instance. `timeout` (seconds) bounds the whole command — without
    it a wedged remote (held apt lock, dead mirror) can hang the lease indefinitely on a billing box.
    `capture=True` returns stdout/stderr as BYTES on the CompletedProcess."""
    cmd = ["ssh", "-p", str(inst.ssh_port), *_OPTS, _target(inst), f"bash -lc {shlex.quote(command)}"]
    kw = {"check": check, "timeout": timeout}
    if capture:
        kw["stdout"] = subprocess.PIPE
        kw["stderr"] = subprocess.PIPE
    elif quiet:
        kw["stdout"] = subprocess.DEVNULL
        kw["stderr"] = subprocess.DEVNULL
    return subprocess.run(cmd, **kw)


def write_file(inst, remote_path, content, mode=None, timeout=30):
    """Write `content` to remote_path by piping it over STDIN (so the data never appears in the
    remote process argv / `ps` listing — important for secrets like an HF token). Optional chmod."""
    quoted = shlex.quote(remote_path)
    sh = f"umask 077; cat > {quoted}" + (f" && chmod {mode} {quoted}" if mode else "")
    cmd = ["ssh", "-p", str(inst.ssh_port), *_OPTS, _target(inst), f"bash -lc {shlex.quote(sh)}"]
    subprocess.run(cmd, input=content.encode(), check=True, timeout=timeout,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_stream(inst, command):
    """Run the (long-lived) training command, streaming its stdout/stderr to our terminal."""
    cmd = ["ssh", "-tt", "-p", str(inst.ssh_port), *_OPTS, _target(inst),
           f"bash -lc {shlex.quote(command)}"]
    return subprocess.run(cmd)


_MARK = "<<<CLEASE_LOG>>>"


def run_detached(inst, command, workdir, poll=10, max_unreachable=900):
    """Run the long-lived training command DETACHED on the box (survives our SSH dropping), then
    reconnect-poll: stream new log output, RETRY transient SSH failures, and return the command's
    real exit code once it finishes. A network blip no longer kills the run (the detached process
    keeps going on the box); only a prolonged outage (> max_unreachable s of failed polls) gives up
    — the trainer's own checkpoint/resume covers that rare case. Replaces the single fragile stream."""
    rlog, rrc, rpid = f"{workdir}/_run.log", f"{workdir}/_run.rc", f"{workdir}/_run.pid"
    inner = f"cd {shlex.quote(workdir)} && ( {command} ); echo $? > {shlex.quote(rrc)}"
    wrapped = f"echo $$ > {shlex.quote(rpid)}; {inner}"
    launch = (f"rm -f {shlex.quote(rlog)} {shlex.quote(rrc)} {shlex.quote(rpid)}; "
              f"setsid bash -lc {shlex.quote(wrapped)} > {shlex.quote(rlog)} 2>&1 < /dev/null & "
              f"for i in $(seq 1 10); do [ -s {shlex.quote(rpid)} ] && break; sleep 0.2; done; "
              f"cat {shlex.quote(rpid)}")
    pid = run(inst, launch, capture=True, timeout=60).stdout.decode(errors="replace").strip()
    print(f"cloud-lease: launched detached (pid {pid}); reconnect-polling every {poll}s",
          file=sys.stderr, flush=True)
    offset, last_ok = 0, time.time()
    probe = (f"if kill -0 {pid} 2>/dev/null; then echo ALIVE; else echo DEAD; fi; "
             f"cat {shlex.quote(rrc)} 2>/dev/null; printf '{_MARK}'; "  # printf: no trailing newline
             f"tail -c +{{off}} {shlex.quote(rlog)} 2>/dev/null")
    while True:
        try:
            out = run(inst, probe.format(off=offset + 1), capture=True, timeout=30).stdout
            last_ok = time.time()
        except (subprocess.SubprocessError, OSError):
            if time.time() - last_ok > max_unreachable:
                print(f"cloud-lease: box unreachable {int(time.time()-last_ok)}s — giving up "
                      f"(resume later via the same launcher)", file=sys.stderr, flush=True)
                return 124
            time.sleep(poll)
            continue
        head, _, delta = out.partition(_MARK.encode())
        if delta:
            sys.stdout.buffer.write(delta)
            sys.stdout.flush()
            offset += len(delta)
        lines = head.decode(errors="replace").split("\n")
        alive = bool(lines) and lines[0].strip() == "ALIVE"
        if not alive:
            rc = lines[1].strip() if len(lines) > 1 and lines[1].strip() else None
            if rc is None:                       # rc file can lag a beat behind the process dying
                time.sleep(2)
                try:
                    rc = run(inst, f"cat {shlex.quote(rrc)} 2>/dev/null",
                             capture=True, timeout=20).stdout.decode(errors="replace").strip()
                except (subprocess.SubprocessError, OSError):
                    rc = ""
            try:
                return int(rc)
            except (TypeError, ValueError):
                return 0
        time.sleep(poll)


def rsync_up(inst, local, remote, excludes=(), delete=True):
    """Push local -> remote. `delete=True` mirrors (removes remote files absent locally) — correct
    for the code/dataset push to a fresh dir, but DANGEROUS for a checkpoint restore into a
    possibly pre-seeded remote dir, so callers restoring pass delete=False."""
    ex = []
    for x in excludes:
        ex += ["--exclude", x]
    cmd = ["rsync", "-az", *(["--delete"] if delete else []), *ex, "-e", _ssh_e(inst),
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
