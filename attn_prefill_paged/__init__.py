"""Native paged/chunked-prefill flash-attention HIP op for gfx1201."""
from .op import flash_prefill_paged

__all__ = ["flash_prefill_paged"]
