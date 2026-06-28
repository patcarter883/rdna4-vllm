"""Vultr backend — direct REST (api.vultr.com/v2). The 'standing / planned-run' provider.

⚠ Some constants need a one-time live confirmation (first smoke test validates the wire format):
  - os_id for Ubuntu 24.04 LTS x64 (GET /v2/os) — set below, verify.
  - GPU plan ids live in gpu_map._VULTR (currently None => provisioning refuses until filled).
Vultr's H100 is contract/prepaid-priced, not freely on-demand — use Vultr for A100-class on-demand.
"""
import time

from . import http
from .backend import Backend, Instance
from .config import LABEL_PREFIX
from .gpu_map import vultr_plan

API = "https://api.vultr.com/v2"
_UBUNTU_2404 = 2284  # os_id — VERIFY via GET /v2/os


class VultrBackend(Backend):
    name = "vultr"

    def _sshkey_id(self, pub):
        st, body = http.request("GET", f"{API}/ssh-keys?per_page=500", self.token)
        for k in body.get("ssh_keys", []):
            if k.get("ssh_key", "").strip() == pub.strip():
                return k["id"]
        st, body = http.request("POST", f"{API}/ssh-keys", self.token,
                                 {"name": "cloud-lease", "ssh_key": pub})
        if st >= 300:
            raise RuntimeError(f"vultr ssh-key create failed [{st}]: {body}")
        return body["ssh_key"]["id"]

    def provision(self, gpu, region, spot, label, sshkey_pub):
        plan = vultr_plan(gpu)            # raises with guidance if unmapped
        region = region or "ewr"
        key = self._sshkey_id(sshkey_pub)
        body = {
            "region": region, "plan": plan, "os_id": _UBUNTU_2404,
            "label": label, "hostname": label,
            "sshkey_id": [key], "backups": "disabled",
        }
        st, resp = http.request("POST", f"{API}/instances", self.token, body)
        if st >= 300:
            raise RuntimeError(f"vultr provision failed [{st}]: {resp}")
        return resp["instance"]["id"]

    def wait_ready(self, instance_id, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            st, resp = http.request("GET", f"{API}/instances/{instance_id}", self.token)
            inst = resp.get("instance", {})
            ip = inst.get("main_ip", "")
            if (inst.get("status") == "active" and inst.get("power_status") == "running"
                    and ip and ip != "0.0.0.0"):
                return Instance(id=instance_id, ssh_host=ip, ssh_port=22,
                                ssh_user="root", raw=inst)
            time.sleep(5)
        raise TimeoutError(f"vultr instance {instance_id} not ready in {timeout}s")

    def destroy(self, instance_id):
        st, _ = http.request("DELETE", f"{API}/instances/{instance_id}", self.token)
        if st not in (204, 404):
            raise RuntimeError(f"vultr destroy {instance_id} returned [{st}]")

    def list_live(self):
        st, resp = http.request("GET", f"{API}/instances?per_page=500", self.token)
        out = []
        for i in resp.get("instances", []):
            if not str(i.get("label", "")).startswith(LABEL_PREFIX):
                continue  # never surface unrelated account instances
            out.append({"id": i["id"], "label": i.get("label", ""),
                        "gpu": i.get("plan", ""), "ip": i.get("main_ip", ""),
                        "status": i.get("status", "")})
        return out
