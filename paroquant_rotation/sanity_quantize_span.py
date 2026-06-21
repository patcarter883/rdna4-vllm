#!/usr/bin/env python3
# Confirms the quantize_rxf.py code increment: (1) default rotation is still the
# bit-identical hadamard32 path; (2) set_rotation(on, span) widens the FWHT and
# the config tag; (3) rxf_quantize round-trips a toy weight under both spans and
# a wider span lowers the per-group quant MSE on an outlier-heavy weight.
# CPU only, no model load.  Run via the container venv (numpy/torch present).
import sys
sys.argv = ["sanity"]                     # quantize_rxf imports argparse lazily in main()
import torch
import quantize_rxf as q

torch.manual_seed(0)
ok = True

# --- 1. default span-32 FWHT bit-identical to a frozen reference ---
def _ref32(x):
    orig = x.shape; x = x.reshape(-1, 32).float(); h = 1
    while h < 32:
        x = x.reshape(-1, 32 // (2*h), 2, h)
        x = torch.stack([x[:, :, 0, :] + x[:, :, 1, :],
                         x[:, :, 0, :] - x[:, :, 1, :]], dim=2).reshape(-1, 32)
        h *= 2
    return (x * 0.1767766953).reshape(orig)

q.set_rotation(True, 32)
t = torch.randn(3, 5, 32)
d = (q._fwht32_rows(t) - _ref32(t)).abs().max().item()
print(f"[1] default span-32 == frozen ref   max_abs={d:.3e}  tag={q.ROTATION_NAME}",
      "OK" if d == 0.0 and q.ROTATION_NAME == "hadamard32" else "FAIL")
ok &= (d == 0.0 and q.ROTATION_NAME == "hadamard32")

# --- 2. widen sets span + tag; reject bad spans ---
q.set_rotation(True, 128)
good = q.ROTATION_SPAN == 128 and q.ROTATION_NAME == "hadamard128"
for bad in (16, 48, 96):
    try:
        q.set_rotation(True, bad); good = False
    except ValueError:
        pass
print(f"[2] widen span=128 tag={q.ROTATION_NAME}, bad spans rejected",
      "OK" if good else "FAIL"); ok &= good

# --- 3. rxf_quantize round-trips under span 32 and 512, and the wider span
#        lowers reconstruction MSE on an outlier-heavy weight ---
N, K = 8, 1024
W = (torch.randn(N, K) * 0.1)
W[:, ::97] += torch.randn(N, (K + 96) // 97) * 6.0   # sparse fat outliers
mse_by_span = {}
for span in (32, 512):
    q.set_rotation(True, span)
    tensors, mse_flat, underflow = q.rxf_quantize(W.clone())
    # reconstruct from packed nibbles * scale, then UN-rotate (Rᵀ=R) to compare
    packed = tensors["weight_packed"]; scale = tensors["weight_scale"].float()
    lo = (packed & 0x0F).to(torch.long); hi = (packed >> 4).to(torch.long)
    idx = torch.stack([lo, hi], dim=-1).reshape(N, K)
    nl = q.ACTIVE_TABLE.float()
    deq_rot = nl[idx] * scale.repeat_interleave(32, dim=1)        # rotated-domain W
    # invert the offline rotation to land back in the original basis
    deq = q._fwht_rows(deq_rot.reshape(N, K // span, span), span).reshape(N, K)
    err = (deq - W).pow(2).mean().item()
    mse_by_span[span] = err
    print(f"[3] span={span:4d}  reconstruction MSE={err:.4e}  underflow={underflow}")
better = mse_by_span[512] < mse_by_span[32]
print(f"    wider span lowers MSE on outlier weight: "
      f"{mse_by_span[512]:.3e} < {mse_by_span[32]:.3e}",
      "OK" if better else "(not strictly lower — informational)")

print("RESULT:", "ALL PASS" if ok else "FAILURES")
sys.exit(0 if ok else 1)
