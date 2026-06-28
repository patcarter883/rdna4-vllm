"""Lease orchestration — the cloud analogue of gpu-lease's flock.

Lifecycle (design §5): provision -> wait_ready -> sync_up (+restore checkpoint) -> run
-> periodic checkpoint sync_down -> GUARANTEED destroy on any exit path.

The teardown guarantee lives here, in Python try/finally + SIGINT/SIGTERM handlers — NOT in a
bash trap around the process. Every normal/error/signal exit runs _teardown(); only SIGKILL can
skip it, which is exactly what cloud-status --reap exists to catch.
"""
import os
import shlex
import signal
import subprocess
import sys
import threading
import time

from . import config, ssh, state

# Signals we route into teardown, and block during teardown so they can't interrupt destroy().
_CATCH = [s for s in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None),
                      getattr(signal, "SIGHUP", None)) if s is not None]


def _log(msg):
    print(f"cloud-lease: {msg}", file=sys.stderr, flush=True)


def _nonempty_dir(path):
    return os.path.isdir(path) and any(os.scandir(path))


class Lease:
    def __init__(self, backend, args):
        self.backend = backend
        self.a = args
        self.instance = None
        self.instance_id = None
        self.label = None
        self._stop = threading.Event()
        self._tore_down = False

    # --- teardown (idempotent) -----------------------------------------------------------------
    def _teardown(self):
        if self._tore_down:
            return
        self._tore_down = True
        # Block the catchable signals so a second Ctrl-C / SIGTERM / SIGHUP can't unwind out of
        # destroy() mid-call (the one action billing safety depends on). Best-effort on non-POSIX.
        try:
            signal.pthread_sigmask(signal.SIG_BLOCK, _CATCH)
        except (AttributeError, ValueError, OSError):
            pass
        a = self.a
        if self.instance_id is not None:
            if self.instance and a.checkpoint_dir and a.remote_out:
                _log("final checkpoint sync-down (best-effort)…")
                try:
                    ssh.rsync_down(self.instance, a.remote_out, a.checkpoint_dir, check=False)
                except Exception as e:  # noqa: BLE001 — teardown must not raise past destroy
                    _log(f"WARN final sync failed: {e}")
            _log(f"destroying instance {self.instance_id}…")
            self._destroy_with_retry(self.instance_id)
        else:
            # We have NO id — but a signal/crash mid-provision may have created a billing instance
            # before provision() returned. Find it by our unique label and destroy it.
            self._sweep_orphans_by_label()

    def _destroy_with_retry(self, iid, attempts=4):
        """Destroy is idempotent and billing-critical, so retry transient failures (a network
        blip now surfaces as HttpError via request_ok rather than being silently swallowed)."""
        for n in range(1, attempts + 1):
            try:
                self.backend.destroy(iid)
                state.remove(self.backend.name, iid)
                _log(f"instance {iid} destroyed; lease released.")
                return True
            except Exception as e:  # noqa: BLE001
                _log(f"destroy attempt {n}/{attempts} failed: {e}")
                if n < attempts:
                    time.sleep(min(2 ** n, 15))
        _log(f"WARN destroy FAILED after {attempts} attempts -> instance {iid} may still be "
             f"BILLING. Run: cloud-status --reap")
        return False

    def _sweep_orphans_by_label(self):
        if not self.label:
            return
        try:
            live = self.backend.list_live()
        except Exception as e:  # noqa: BLE001
            _log(f"WARN orphan sweep could not list instances: {e} — run cloud-status --reap")
            return
        for inst in live:
            if inst.get("label") == self.label:
                _log(f"orphan sweep: leaked instance {inst.get('id')} ({self.label}) — destroying")
                self._destroy_with_retry(inst.get("id"))

    # --- periodic checkpoint pull --------------------------------------------------------------
    def _sync_loop(self):
        a = self.a
        while not self._stop.wait(a.sync_every):
            try:
                ssh.rsync_down(self.instance, a.remote_out, a.checkpoint_dir, check=False)
                _log("checkpoint synced down")
            except Exception as e:  # noqa: BLE001
                _log(f"periodic sync failed (will retry): {e}")

    # --- main ----------------------------------------------------------------------------------
    def run(self):
        a = self.a

        def _on_signal(signum, _frame):
            _log(f"signal {signum} received — tearing down")
            raise KeyboardInterrupt
        for _sig in _CATCH:
            signal.signal(_sig, _on_signal)

        pub = config.ensure_ssh_key()
        try:
            # Unique per-process label: carries LABEL_PREFIX (so cloud-status only ever
            # sees/reaps ours) AND a pid suffix so the teardown orphan-sweep can pinpoint the
            # instance THIS process created even if we never captured its id.
            self.label = f"{config.LABEL_PREFIX}{a.name}-{os.getpid()}"
            label = self.label
            _log(f"provisioning {a.gpu} on {self.backend.name}"
                 f"{' (spot)' if a.spot else ''} as '{label}'…")
            self.instance_id = self.backend.provision(a.gpu, a.region, a.spot, label, pub)
            state.write(self.backend.name, self.instance_id, gpu=a.gpu,
                        region=a.region or "", label=label, spot=a.spot,
                        cmd=shlex.join(a.command))
            _log(f"instance {self.instance_id} created; waiting for it to come up…")
            self.instance = self.backend.wait_ready(self.instance_id, a.provision_timeout)
            ssh.wait_ssh(self.instance, a.provision_timeout)
            _log(f"SSH ready: {self.instance.ssh_user}@{self.instance.ssh_host}:{self.instance.ssh_port}")

            self._ensure_remote_rsync()

            _log(f"syncing working dir -> {a.remote_workdir}")
            ssh.run(self.instance, f"mkdir -p {shlex.quote(a.remote_workdir)}", quiet=True)
            ssh.rsync_up(self.instance, a.workdir, a.remote_workdir, excludes=a.exclude)

            if a.dataset:
                self._stage_dataset()

            if a.checkpoint_dir and a.remote_out:
                ssh.run(self.instance, f"mkdir -p {shlex.quote(a.remote_out)}", quiet=True)
                if _nonempty_dir(a.checkpoint_dir):
                    _log("restoring checkpoint up to instance (resume)…")
                    # delete=False: never mirror-delete files already in remote_out (the image
                    # may pre-seed it) — only add our checkpoint.
                    ssh.rsync_up(self.instance, a.checkpoint_dir, a.remote_out, delete=False)
                os.makedirs(a.checkpoint_dir, exist_ok=True)
                threading.Thread(target=self._sync_loop, daemon=True).start()

            env_prefix = self._write_remote_env()

            if a.setup:
                _log("running setup…")
                # Bound setup so a wedged box (e.g. torch's HIP init hanging in kfd_create_process on a
                # dud GPU pod) FAILS -> teardown, instead of billing forever on a no-timeout ssh.run.
                # Generous headroom for the real work (torch upgrade + a multi-GB model download).
                ssh.run(self.instance, f"cd {shlex.quote(a.remote_workdir)} && {env_prefix}{a.setup}",
                        timeout=a.setup_timeout)

            # shlex.join quotes each token, so args with spaces/metachars survive the remote
            # bash -lc re-parse instead of being re-split or interpreted.
            remote_cmd = shlex.join(a.command)
            _log(f"launching: {remote_cmd}")
            # run DETACHED + reconnect-poll so a transient SSH/network drop doesn't kill the run
            # (the old run_stream returned 255 and tore down on any blip). On a non-zero exit,
            # optionally RELAUNCH (--restart-on-crash): the trainer resumes from its box-local
            # checkpoint, so a crash costs a few steps, not the whole run nor a 17.7GB sync.
            attempt = 0
            while True:
                rc = ssh.run_detached(self.instance, f"{env_prefix}{remote_cmd}", a.remote_workdir)
                _log(f"command exited rc={rc}")
                if rc == 0 or attempt >= a.restart_on_crash:
                    break
                attempt += 1
                _log(f"non-zero exit — relaunching (attempt {attempt}/{a.restart_on_crash}); "
                     "trainer resumes from its box-local checkpoint")
            self._stop.set()
            return rc
        except KeyboardInterrupt:
            self._stop.set()
            return 130
        finally:
            self._stop.set()
            self._teardown()

    def _write_remote_env(self):
        """Resolve --env specs and write them to a mode-600 file on the box, returning a shell
        prefix that sources it. A bare NAME pulls VALUE from OUR environment, so a secret (e.g.
        HF_TOKEN, for the trainer's push_to_hub) never touches the cloud-lease command line nor the
        box's process list — it's piped over stdin into the file. Returns "" when no --env given."""
        a = self.a
        pairs, missing = [], []
        for spec in a.env:
            if "=" in spec:
                k, v = spec.split("=", 1)
            else:
                k, v = spec, os.environ.get(spec)
                if v is None:
                    missing.append(spec)
                    continue
            pairs.append((k, v))
        if missing:
            _log(f"--env not set locally, skipping: {', '.join(missing)}")
        if not pairs:
            return ""
        body = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in pairs)
        remote = f"{a.remote_workdir}/.cloud_lease_env"
        ssh.write_file(self.instance, remote, body, mode="600")
        _log(f"wrote {len(pairs)} env var(s) to the box: {', '.join(k for k, _ in pairs)}")
        return f"source {shlex.quote(remote)}; "

    # --- bootstrap -----------------------------------------------------------------------------
    def _ensure_remote_rsync(self):
        """rsync is a hard dependency of the sync workflow but many ML images (e.g.
        runpod/pytorch) don't ship it. Install it idempotently before the first sync."""
        check = ("command -v rsync >/dev/null 2>&1 && exit 0; "
                 "(apt-get update -qq && apt-get install -y -qq rsync) >/dev/null 2>&1; "
                 "(command -v yum >/dev/null 2>&1 && yum -y -q install rsync) >/dev/null 2>&1; "
                 "command -v rsync >/dev/null 2>&1")
        # Bound it: a held dpkg lock / dead mirror must not hang the lease on a billing box.
        try:
            r = ssh.run(self.instance, check, check=False, quiet=True, timeout=300)
        except subprocess.TimeoutExpired:
            raise RuntimeError("rsync bootstrap timed out (apt/yum hung) — use an image "
                               "that ships rsync, or pre-install it via --setup")
        if r.returncode != 0:
            raise RuntimeError("rsync unavailable on the instance and could not be installed "
                               "(tried apt-get/yum) — use an image that ships rsync")
        _log("rsync present on instance")

    # --- dataset staging (design §13) ----------------------------------------------------------
    def _stage_dataset(self):
        a = self.a
        d = a.dataset
        if d.startswith(("hf://", "s3://", "r2://", "http://", "https://")):
            # Option C/D: pull on the instance (datacenter bandwidth), no home upload.
            ssh.run(self.instance, f"mkdir -p {shlex.quote(a.remote_dataset)}", quiet=True)
            if d.startswith("hf://"):
                repo = d[len("hf://"):]
                _log(f"pulling dataset from HF on the instance: {repo}")
                ssh.run(self.instance,
                        f"pip install -q huggingface_hub && "
                        f"huggingface-cli download --repo-type dataset {shlex.quote(repo)} "
                        f"--local-dir {shlex.quote(a.remote_dataset)}")
            else:
                # s3://, r2://, http(s):// — needs rclone/curl configured via --setup.
                _log(f"fetching dataset on the instance: {d} (expects rclone/curl from --setup)")
                ssh.run(self.instance,
                        f"rclone copy {shlex.quote(d)} {shlex.quote(a.remote_dataset)} "
                        f"|| curl -fsSL {shlex.quote(d)} -o {shlex.quote(a.remote_dataset)}/data")
        else:
            # Option A (v1 default): rsync the local path up.
            _log(f"syncing local dataset -> {a.remote_dataset}")
            ssh.run(self.instance, f"mkdir -p {shlex.quote(a.remote_dataset)}", quiet=True)
            ssh.rsync_up(self.instance, d, a.remote_dataset)
