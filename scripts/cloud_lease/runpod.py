"""RunPod backend — direct REST (rest.runpod.io/v1). The 'ad-hoc burst' provider.

--spot maps to community-cloud + interruptible. Pubkey is injected via the PUBLIC_KEY env the
runpod/pytorch images honour, and SSH is reached on the pod's mapped 22/tcp.

⚠ RunPod's runtime port schema has varied across API versions; the port-extraction below is
defensive but the first smoke test (provision -> nvidia-smi -> destroy) is what confirms it.
"""
import os
import time

from . import http
from .backend import Backend, Instance
from .gpu_map import runpod_gpu

API = "https://rest.runpod.io/v1"
# Default is the CUDA pytorch image; override via CLOUD_LEASE_RUNPOD_IMAGE for a ROCm box (MI300X)
# — use a runpod/pytorch ROCm tag so the PUBLIC_KEY->sshd start script still gives cloud-lease SSH.
_IMAGE = os.environ.get("CLOUD_LEASE_RUNPOD_IMAGE") or \
    "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
# Terminal states a pod can land in instead of coming up (e.g. unplaceable / preempted spot pod).
# wait_ready must bail on these immediately rather than spin to provision_timeout while billing.
_TERMINAL = {"EXITED", "TERMINATED", "FAILED", "DEAD"}


class RunPodBackend(Backend):
    name = "runpod"

    def provision(self, gpu, region, spot, label, sshkey_pub):
        gpu_type = runpod_gpu(gpu)
        body = {
            "name": label,
            "imageName": _IMAGE,
            "gpuTypeIds": [gpu_type],
            "gpuCount": 1,
            "cloudType": "COMMUNITY" if spot else "SECURE",
            "interruptible": bool(spot),
            "containerDiskInGb": 120,   # room for the 17.7GB HF model + a 17.7GB model-only ckpt (+tmp)
            "volumeInGb": 0,
            "ports": ["22/tcp"],
            "env": {"PUBLIC_KEY": sshkey_pub},
        }
        st, resp = http.request_ok("POST", f"{API}/pods", self.token, body)
        return resp["id"]

    @staticmethod
    def _ssh_endpoint(pod):
        # Confirmed live 2026-06-28: the SSH endpoint is top-level publicIp + portMappings
        # ({"22": <external>}); `runtime` is empty. desiredStatus flips to RUNNING ~25s BEFORE
        # these populate, so readiness must gate on the endpoint, not just status.
        ip = pod.get("publicIp") or ""
        mappings = pod.get("portMappings") or {}
        port = mappings.get("22") or mappings.get(22)
        if ip and port:
            return ip, int(port)
        return None, None

    def wait_ready(self, instance_id, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            st, pod = http.request("GET", f"{API}/pods/{instance_id}", self.token)
            status = str(pod.get("desiredStatus") or pod.get("status") or "").upper()
            if status in _TERMINAL:
                raise RuntimeError(f"runpod pod {instance_id} entered terminal state "
                                   f"'{status}' before becoming reachable (preempted/unplaceable?)")
            if status == "RUNNING":
                host, port = self._ssh_endpoint(pod)
                if host:
                    return Instance(id=instance_id, ssh_host=host, ssh_port=port,
                                    ssh_user="root", raw=pod)
            time.sleep(5)
        raise TimeoutError(f"runpod pod {instance_id} not ready (or no SSH port) in {timeout}s")

    def destroy(self, instance_id):
        http.request_ok("DELETE", f"{API}/pods/{instance_id}", self.token, ok=(200, 204, 404))

    def list_live(self):
        st, resp = http.request("GET", f"{API}/pods", self.token)
        pods = resp if isinstance(resp, list) else resp.get("pods", [])
        out = []
        for p in pods:
            if not self._owned(p.get("name")):
                continue
            host, _ = self._ssh_endpoint(p)
            out.append({"id": p.get("id"), "label": p.get("name", ""),
                        "gpu": (p.get("machine") or {}).get("gpuTypeId", ""),
                        "ip": host or "", "status": p.get("desiredStatus", "")})
        return out
