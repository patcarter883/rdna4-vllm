"""Vast.ai backend — STUB behind the common interface (design §4/§11 step 6).

Wire this in when a cost-floor LoRA run justifies it. Vast is a marketplace (search asks, bid,
create from an offer), so provision() will: search offers for the GPU, create an instance from the
cheapest viable ask, then poll. list_live() must filter by the cloud-lease label, same as the
others, so nothing unrelated is ever reaped.
"""
from .backend import Backend

_MSG = ("vast.ai backend is a stub. It is intentionally not implemented for v1 "
        "(design §11 step 6). Use --provider vultr or --provider runpod.")


class VastBackend(Backend):
    name = "vastai"

    def provision(self, gpu, region, spot, label, sshkey_pub):
        raise NotImplementedError(_MSG)

    def wait_ready(self, instance_id, timeout):
        raise NotImplementedError(_MSG)

    def destroy(self, instance_id):
        raise NotImplementedError(_MSG)

    def list_live(self):
        return []  # nothing to reconcile while unimplemented
