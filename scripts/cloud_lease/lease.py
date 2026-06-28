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
import sys
import threading

from . import config, ssh, state


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
        self._stop = threading.Event()
        self._tore_down = False

    # --- teardown (idempotent) -----------------------------------------------------------------
    def _teardown(self):
        if self._tore_down or self.instance_id is None:
            return
        self._tore_down = True
        iid = self.instance_id
        a = self.a
        if self.instance and a.checkpoint_dir and a.remote_out:
            _log("final checkpoint sync-down (best-effort)…")
            try:
                ssh.rsync_down(self.instance, a.remote_out, a.checkpoint_dir, check=False)
            except Exception as e:  # noqa: BLE001 — teardown must not raise past destroy
                _log(f"WARN final sync failed: {e}")
        _log(f"destroying instance {iid}…")
        try:
            self.backend.destroy(iid)
            state.remove(self.backend.name, iid)
            _log(f"instance {iid} destroyed; lease released.")
        except Exception as e:  # noqa: BLE001
            _log(f"WARN destroy FAILED: {e}")
            _log(f"  -> the instance may still be BILLING. Run: cloud-status --reap")

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
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)

        pub = config.ensure_ssh_key()
        try:
            label = config.LABEL_PREFIX + a.name   # all our instances carry the prefix so
                                                   # cloud-status only ever sees/reaps ours
            _log(f"provisioning {a.gpu} on {self.backend.name}"
                 f"{' (spot)' if a.spot else ''} as '{label}'…")
            self.instance_id = self.backend.provision(a.gpu, a.region, a.spot, label, pub)
            state.write(self.backend.name, self.instance_id, gpu=a.gpu,
                        region=a.region or "", label=label, spot=a.spot,
                        cmd=" ".join(a.command))
            _log(f"instance {self.instance_id} created; waiting for it to come up…")
            self.instance = self.backend.wait_ready(self.instance_id, a.provision_timeout)
            ssh.wait_ssh(self.instance, a.provision_timeout)
            _log(f"SSH ready: {self.instance.ssh_user}@{self.instance.ssh_host}:{self.instance.ssh_port}")

            _log(f"syncing working dir -> {a.remote_workdir}")
            ssh.run(self.instance, f"mkdir -p {shlex.quote(a.remote_workdir)}", quiet=True)
            ssh.rsync_up(self.instance, a.workdir, a.remote_workdir, excludes=a.exclude)

            if a.dataset:
                self._stage_dataset()

            if a.checkpoint_dir and a.remote_out:
                ssh.run(self.instance, f"mkdir -p {shlex.quote(a.remote_out)}", quiet=True)
                if _nonempty_dir(a.checkpoint_dir):
                    _log("restoring checkpoint up to instance (resume)…")
                    ssh.rsync_up(self.instance, a.checkpoint_dir, a.remote_out)
                os.makedirs(a.checkpoint_dir, exist_ok=True)
                threading.Thread(target=self._sync_loop, daemon=True).start()

            if a.setup:
                _log("running setup…")
                ssh.run(self.instance, f"cd {shlex.quote(a.remote_workdir)} && {a.setup}")

            remote_cmd = " ".join(a.command)
            _log(f"launching: {remote_cmd}")
            rc = ssh.run_stream(
                self.instance,
                f"cd {shlex.quote(a.remote_workdir)} && {remote_cmd}").returncode
            _log(f"command exited rc={rc}")
            self._stop.set()
            return rc
        except KeyboardInterrupt:
            self._stop.set()
            return 130
        finally:
            self._stop.set()
            self._teardown()

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
