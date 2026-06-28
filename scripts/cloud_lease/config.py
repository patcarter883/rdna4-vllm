"""Config: key/SSH-key handling and well-known paths.

Keys are read ONLY from env vars or ~/.config/cloud-lease/<provider>.env (chmod 600) — never
from the command line (process-list leak) and never from the repo. See design §8.
"""
import os
import pathlib
import subprocess
import sys

CONFIG_DIR = pathlib.Path(os.environ.get("CLOUD_LEASE_HOME", os.path.expanduser("~/.config/cloud-lease")))
STATE_DIR = CONFIG_DIR / "state"
KNOWN_HOSTS = CONFIG_DIR / "known_hosts"
SSH_KEY = CONFIG_DIR / "id_cloud_lease"          # private key (chmod 600)
SSH_PUB = CONFIG_DIR / "id_cloud_lease.pub"

# Every instance cloud-lease creates is labelled with this prefix. cloud-status only ever
# looks at instances whose label starts with it, so a user's UNRELATED nodes on the same
# account (e.g. a production video-streaming headend) are invisible to status/reap.
LABEL_PREFIX = "cloud-lease-"

_KEY_ENV = {"vultr": "VULTR_API_KEY", "runpod": "RUNPOD_API_KEY", "vastai": "VAST_API_KEY"}


def _load_env_file(p: pathlib.Path) -> dict:
    env = {}
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def api_key(provider: str, required: bool = True):
    var = _KEY_ENV[provider]
    if os.environ.get(var):
        return os.environ[var]
    val = _load_env_file(CONFIG_DIR / f"{provider}.env").get(var)
    if val:
        return val
    if required:
        sys.exit(f"cloud-lease: no API key for '{provider}': set ${var} "
                 f"or put it in {CONFIG_DIR}/{provider}.env (chmod 600)")
    return None


def ensure_ssh_key() -> str:
    """Generate the cloud-lease-managed keypair on first use; return the public key text."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not SSH_KEY.exists():
        print(f"cloud-lease: generating managed SSH key at {SSH_KEY}", file=sys.stderr)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-q",
                        "-C", "cloud-lease", "-f", str(SSH_KEY)], check=True)
        SSH_KEY.chmod(0o600)
    return SSH_PUB.read_text().strip()
