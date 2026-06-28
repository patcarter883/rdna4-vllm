"""RunPod backend — direct REST (rest.runpod.io/v1). The 'ad-hoc burst' provider.

--spot maps to community-cloud + interruptible. Pubkey is injected via the PUBLIC_KEY env the
runpod/pytorch images honour, and SSH is reached on the pod's mapped 22/tcp.

⚠ RunPod's runtime port schema has varied across API versions; the port-extraction below is
defensive but the first smoke test (provision -> nvidia-smi -> destroy) is what confirms it.
"""
import time

from . import http
from .backend import Backend, Instance
from .config import LABEL_PREFIX
from .gpu_map import runpod_gpu

API = "https://rest.runpod.io/v1"
_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


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
            "containerDiskInGb": 60,
            "volumeInGb": 0,
            "ports": ["22/tcp"],
            "env": {"PUBLIC_KEY": sshkey_pub},
        }
        st, resp = http.request("POST", f"{API}/pods", self.token, body)
        if st >= 300:
            raise RuntimeError(f"runpod provision failed [{st}]: {resp}")
        return resp["id"]

    @staticmethod
    def _ssh_endpoint(pod):
        ports = (pod.get("runtime") or {}).get("ports") or pod.get("portMappings") or []
        for p in ports:
            internal = str(p.get("privatePort") or p.get("internalPort") or "")
            if internal == "22":
                host = p.get("ip") or p.get("publicIp") or p.get("host")
                port = p.get("publicPort") or p.get("externalPort") or p.get("port")
                if host and port:
                    return host, int(port)
        return None, None

    def wait_ready(self, instance_id, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            st, pod = http.request("GET", f"{API}/pods/{instance_id}", self.token)
            if str(pod.get("desiredStatus") or pod.get("status")).upper() == "RUNNING":
                host, port = self._ssh_endpoint(pod)
                if host:
                    return Instance(id=instance_id, ssh_host=host, ssh_port=port,
                                    ssh_user="root", raw=pod)
            time.sleep(5)
        raise TimeoutError(f"runpod pod {instance_id} not ready (or no SSH port) in {timeout}s")

    def destroy(self, instance_id):
        st, _ = http.request("DELETE", f"{API}/pods/{instance_id}", self.token)
        if st not in (200, 204, 404):
            raise RuntimeError(f"runpod destroy {instance_id} returned [{st}]")

    def list_live(self):
        st, resp = http.request("GET", f"{API}/pods", self.token)
        pods = resp if isinstance(resp, list) else resp.get("pods", [])
        out = []
        for p in pods:
            if not str(p.get("name", "")).startswith(LABEL_PREFIX):
                continue
            host, _ = self._ssh_endpoint(p)
            out.append({"id": p.get("id"), "label": p.get("name", ""),
                        "gpu": (p.get("machine") or {}).get("gpuTypeId", ""),
                        "ip": host or "", "status": p.get("desiredStatus", "")})
        return out
