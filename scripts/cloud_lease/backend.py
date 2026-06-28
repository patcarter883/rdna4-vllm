"""Backend interface — the five verbs every provider implements (design §3).

Data movement (sync_up / run / sync_down) is NOT here: it is provider-independent and lives in
ssh.py, driven by the Instance coords a backend returns from wait_ready(). A backend is purely
the provider-specific control plane: create / poll / destroy / list.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Instance:
    id: str
    ssh_host: str
    ssh_port: int = 22
    ssh_user: str = "root"
    raw: dict = field(default_factory=dict)


class Backend(ABC):
    name = "base"

    def __init__(self, token):
        self.token = token

    @abstractmethod
    def provision(self, gpu, region, spot, label, sshkey_pub) -> str:
        """Create the instance; return its provider id. `label` is already cloud-lease-prefixed."""

    @abstractmethod
    def wait_ready(self, instance_id, timeout) -> Instance:
        """Poll the provider until the instance is running with an SSH endpoint; return Instance."""

    @abstractmethod
    def destroy(self, instance_id) -> None:
        """Terminate the instance. MUST be idempotent (the teardown trap may double-fire)."""

    @abstractmethod
    def list_live(self) -> list:
        """Return [{id,label,gpu,ip,status}, ...] for cloud-lease-owned instances ONLY
        (filtered by the LABEL_PREFIX), so unrelated account instances are never touched."""
