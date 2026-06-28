"""Vultr backend — direct REST (api.vultr.com/v2). The 'standing / planned-run' provider.

⚠ Some constants need a one-time live confirmation (first smoke test validates the wire format):
  - os_id for Ubuntu 24.04 LTS x64 (GET /v2/os) — set below, verify.
  - GPU plan ids live in gpu_map._VULTR (currently None => provisioning refuses until filled).
Vultr's H100 is contract/prepaid-priced, not freely on-demand — use Vultr for A100-class on-demand.
"""
import time

from . import http
from .backend import Backend, Instance
from .gpu_map import vultr_plan

API = "https://api.vultr.com/v2"
_UBUNTU_2404 = 2284  # os_id — VERIFY via GET /v2/os


class VultrBackend(Backend):
    name = "vultr"

    def _sshkey_id(self, pub):
        st, body = http.request_ok("GET", f"{API}/ssh-keys?per_page=500", self.token)
        for k in body.get("ssh_keys", []):
            if k.get("ssh_key", "").strip() == pub.strip():
                return k["id"]
        st, body = http.request_ok("POST", f"{API}/ssh-keys", self.token,
                                   {"name": "cloud-lease", "ssh_key": pub})
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
        st, resp = http.request_ok("POST", f"{API}/instances", self.token, body)
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
        # request_ok raises on status 0 (connection failure) too, so a transient blip surfaces
        # as a retryable error to the caller's destroy-retry loop instead of being swallowed.
        http.request_ok("DELETE", f"{API}/instances/{instance_id}", self.token, ok=(204, 404))

    def list_live(self):
        st, resp = http.request("GET", f"{API}/instances?per_page=500", self.token)
        out = []
        for i in resp.get("instances", []):
            if not self._owned(i.get("label")):
                continue  # never surface unrelated account instances
            out.append({"id": i["id"], "label": i.get("label", ""),
                        "gpu": i.get("plan", ""), "ip": i.get("main_ip", ""),
                        "status": i.get("status", "")})
        return out
