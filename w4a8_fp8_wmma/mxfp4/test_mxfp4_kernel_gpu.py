"""GPU bit-exactness test: the e2m1 kernel path decodes mxfp4 correctly on RDNA4.

Two complementary checks (the host test_mxfp4_decode.py already proved convert.py + the LUT are
bit-exact to vLLM; this proves the HIP kernels honor weight_is_e2m1):

  A. e2m1-vs-int4 EQUIVALENCE (rigorous, all M, no external oracle). For E2M1 codes whose value is
     an integer v in {0,±1,±2,±3,±4,±6}, the symmetric int4 path with code v+8 (zp=8 -> v) produces
     the *same* fp8 weight byte. So the e2m1 kernel and the (already-validated) int4 kernel must
     give BIT-IDENTICAL output on such weights -- both using the kernel's own fp8 act-quant, which
     removes any torch-fp8-rounding oracle gap. This nails the decode + dispatch wiring exactly.

  B. fp8 ORACLE for the NON-integer values (0.5/1.5), which (A) can't cover. A torch fp32 reference
     mirrors the kernel (per-row reciprocal fp8 act-quant x E2M1 weight x fp16 group scale). Run at
     small M where torch's fp8 cast and the hardware cvt agree exactly; this confirms 0.5/1.5 decode
     through the live kernels.

Run under a single-card GPU lease, building the extension against this worktree:
  scripts/gpu-lease.sh -n 1 -- docker run --rm --device=/dev/kfd --device=/dev/dri \
    --group-add video --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES \
    -v $PWD/w4a8_fp8_wmma:/src:ro --entrypoint bash vllm22-w4a8:combined -lc \
    'cp -r /src /b && cd /b && source /app/.venv/bin/activate && pip install . --no-build-isolation \
     --no-deps -q && mkdir /t && cp -r mxfp4 /t && cd /t && python -m mxfp4.test_mxfp4_kernel_gpu'
"""
import torch

import w4a8_fp8_wmma as w4a8
from mxfp4.convert import (
    FP4_E2M1_LUT, OCP_MX_BLOCK_SIZE, convert_mxfp4_weight, pack_codes_to_int32,
    unpack_e2m1_nibbles,
)

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_ok = True
# E2M1 codes whose codebook value is an integer (excludes ±0.5 = codes 1,9 and ±1.5 = codes 3,11).
INT_CODES = [c for c in range(16) if float(FP4_E2M1_LUT[c]).is_integer()]


def check(name, cond, detail=""):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}{(' -- ' + detail) if detail else ''}")


# ---- A. e2m1 vs int4 bit-exact equivalence (integer-valued codes) ----------------------------
def run_equiv(kernel, M, N, K, group_size=OCP_MX_BLOCK_SIZE, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(0, len(INT_CODES), (N, K), generator=g)
    codes_e2m1 = torch.tensor(INT_CODES, dtype=torch.int64)[idx]            # (N,K) e2m1 codes
    vals = torch.tensor(FP4_E2M1_LUT, dtype=torch.float32)[codes_e2m1]      # integer values
    codes_int4 = (vals + 8).to(torch.int64)                                # symmetric int4 (zp=8)

    w_e2m1 = pack_codes_to_int32(codes_e2m1.to(torch.uint8)).cuda()
    w_int4 = pack_codes_to_int32(codes_int4.to(torch.uint8)).cuda()
    scales = (torch.rand(N, K // group_size, generator=g).to(torch.float16) * 0.5 + 0.25).cuda()
    x = (torch.randn(M, K, generator=g) * 0.5).to(torch.float16).cuda()

    o_e2m1 = w4a8.mmq_fp8_gemm(x, w_e2m1, scales, kernel=kernel, weight_is_e2m1=True)
    o_int4 = w4a8.mmq_fp8_gemm(x, w_int4, scales, kernel=kernel, weight_is_e2m1=False)
    check(f"e2m1==int4  {kernel:<22} M={M:<3} N={N} K={K}", torch.equal(o_e2m1, o_int4),
          f"max|diff|={ (o_e2m1.float()-o_int4.float()).abs().max().item():.3e}")


# ---- B. fp8 oracle for non-integer values (0.5/1.5) at small M --------------------------------
def fp8_dense_ref(x, weight_packed, weight_scale, conv):
    a_scale = (x.float().abs().amax(dim=1) / 448.0).clamp(min=1e-8)
    x_fp8 = (x.float() * (1.0 / a_scale)[:, None]).to(torch.float8_e4m3fn).float()  # match kernel
    codes = unpack_e2m1_nibbles(weight_packed.cpu()).long()
    W = torch.tensor(FP4_E2M1_LUT, dtype=torch.float32)[codes]
    sc = conv["scales"].float().repeat_interleave(OCP_MX_BLOCK_SIZE, dim=1)
    Wd = (W * sc).to(x.device)
    return (x_fp8.to(x.device) @ Wd.t()) * a_scale[:, None].to(x.device)


def run_oracle(kernel, M, N, K, group_size=OCP_MX_BLOCK_SIZE, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, generator=g)  # full codebook
    scale = torch.randint(124, 131, (N, K // group_size), dtype=torch.uint8, generator=g)
    conv = convert_mxfp4_weight(packed, scale)
    x = (torch.randn(M, K, generator=g) * 0.5).to(torch.float16).cuda()
    ref = fp8_dense_ref(x, packed, scale, conv)
    out = w4a8.mmq_fp8_gemm(x, conv["w_packed"].cuda(), conv["scales"].cuda(),
                            kernel=kernel, weight_is_e2m1=True).float()
    rel = ((out - ref).abs() / ref.abs().clamp(min=1.0)).max().item()
    check(f"fp8-oracle  {kernel:<22} M={M:<3} N={N} K={K}", rel < 5e-3, f"max_rel={rel:.2e}")


if __name__ == "__main__":
    print(f"torch {torch.__version__}\n")
    print("A. e2m1 == int4 bit-exact (integer-valued codes, all M)")
    for kern in ("reference_scalar", "prefill_wmma", "prefill_wmma_ashuffle"):
        for M in (4, 16, 64):
            run_equiv(kern, M, 128, 512)
    for M in (1, 2, 4):
        run_equiv("decode_gemv", M, 256, 512)
    print("B. fp8 oracle covers non-integer 0.5/1.5 decode (small M)")
    for kern in ("reference_scalar", "prefill_wmma", "prefill_wmma_ashuffle"):
        run_oracle(kern, 16, 128, 512)
    run_oracle("decode_gemv", 4, 256, 512)
    print("\n" + ("ALL GPU CHECKS PASSED" if _ok else "SOME GPU CHECKS FAILED"))
    raise SystemExit(0 if _ok else 1)
