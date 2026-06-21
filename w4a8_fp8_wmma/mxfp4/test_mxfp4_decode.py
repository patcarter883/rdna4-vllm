"""Host (CPU-only) bit-exactness test for the mxfp4 -> W4A8 decode path.

Validates the central correctness claim WITHOUT any GPU or kernel build:
  1. Every OCP E2M1 codebook value is EXACTLY representable in fp8 e4m3 -> the decode
     e2m1_to_e4m3 (f32->e4m3 of the float LUT) is lossless. This is what makes the existing
     fp8 WMMA core reusable for mxfp4 with only a decode-table swap.
  2. The load-time repack (unpack uint8 nibbles -> (N,K//8) int32) round-trips: re-reading each
     word the way the kernel does ((word>>4j)&0xF) recovers the original E2M1 codes.
  3. The full kernel-decode emulation (code -> e4m3 -> f32 * fp16 group-scale) reproduces vLLM's
     reference dequant (_FP4_E2M1_LUT * 2^(s-127)) BIT-EXACTLY for in-fp16-range scales.
  4. The converter flags E8M0 scale exponents that overflow the kernel's fp16 group-scale store.

Run inside the image venv (CPU only -- NO gpu-lease needed):
  docker run --rm --entrypoint bash vllm22-w4a8:combined -lc \
    'source /app/.venv/bin/activate && cd /work && python -m mxfp4.test_mxfp4_decode' \
    -v /home/pat/code/vllm-gfx1201-mxfp4/w4a8_fp8_wmma:/work
"""
import torch

from mxfp4.convert import (
    FP4_E2M1_LUT, OCP_MX_BLOCK_SIZE, convert_mxfp4_weight, dequant_reference,
    unpack_e2m1_nibbles,
)

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_ok = True


def check(name, cond, detail=""):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}{(' -- ' + detail) if detail else ''}")


def e4m3_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """f32 -> e4m3 -> f32, mirroring the kernel's hardware cvt. CPU cast."""
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def test_lut_exact_in_e4m3():
    print("1. E2M1 codebook is exact in fp8 e4m3 (decode is lossless)")
    lut = torch.tensor(FP4_E2M1_LUT, dtype=torch.float32)
    rt = e4m3_roundtrip(lut)
    check("all 16 E2M1 values round-trip through e4m3 unchanged",
          torch.equal(rt, lut), f"max|err|={float((rt-lut).abs().max()):g}")


def test_vllm_lut_matches():
    print("2. Our LUT matches vLLM's _FP4_E2M1_LUT (if importable)")
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes.\
            compressed_tensors_w4a16_mxfp4 import CompressedTensorsW4A16Mxfp4
        cand = [float(x) for x in CompressedTensorsW4A16Mxfp4._FP4_E2M1_LUT.tolist()]
        check("vLLM _FP4_E2M1_LUT == our LUT",
              len(cand) == len(FP4_E2M1_LUT) and
              all(a == float(b) for a, b in zip(cand, FP4_E2M1_LUT)),
              f"vllm={cand}")
    except Exception as e:  # noqa: BLE001
        print(f"     (vLLM import failed: {type(e).__name__}: {e} -- skipped)")


def test_repack_roundtrip():
    print("3. uint8-nibble unpack -> int32 repack round-trips (kernel read order)")
    torch.manual_seed(0)
    n, k = 4, 64
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    codes0 = unpack_e2m1_nibbles(packed).to(torch.int64)
    scale = torch.randint(120, 135, (n, k // OCP_MX_BLOCK_SIZE), dtype=torch.uint8)
    out = convert_mxfp4_weight(packed, scale)
    w = out["w_packed"].to(torch.int64) & 0xFFFFFFFF      # treat as unsigned bit pattern
    # Re-read exactly as the kernel does: code at K index 8*word+j = (word >> 4j) & 0xF.
    codes1 = torch.empty_like(codes0)
    for word in range(k // 8):
        for j in range(8):
            codes1[:, 8 * word + j] = (w[:, word] >> (4 * j)) & 0xF
    check("recovered codes == original codes", torch.equal(codes0, codes1))


def test_full_dequant_bit_exact():
    print("4. Kernel-decode emulation == vLLM reference dequant (in-range scales)")
    torch.manual_seed(1)
    n, k = 8, 128
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    # E8M0 exponents kept in fp16's exact-power-of-two range [-14, 15] -> bias [113, 142].
    scale = torch.randint(113, 143, (n, k // OCP_MX_BLOCK_SIZE), dtype=torch.uint8)

    ref = dequant_reference(packed, scale)                # _FP4_E2M1_LUT * 2^(s-127), fp32

    out = convert_mxfp4_weight(packed, scale)
    codes = unpack_e2m1_nibbles(packed).to(torch.int64)
    lut = torch.tensor(FP4_E2M1_LUT, dtype=torch.float32)
    w_e4m3 = e4m3_roundtrip(lut[codes])                   # what the kernel decode produces
    sc = out["scales"].to(torch.float32).repeat_interleave(OCP_MX_BLOCK_SIZE, dim=-1)
    emu = w_e4m3 * sc

    check("scale_info.fp16_range_ok for in-range exponents", out["scale_info"]["fp16_range_ok"],
          str(out["scale_info"]))
    check("emulated decode == reference dequant (bit-exact)", torch.equal(emu, ref),
          f"max|err|={float((emu-ref).abs().max()):g}")


def test_scale_overflow_flagged():
    print("5. Converter flags E8M0 exponents that overflow fp16 group-scale store")
    n, k = 2, 64
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    scale = torch.full((n, k // OCP_MX_BLOCK_SIZE), 200, dtype=torch.uint8)  # exp = 73 >> 15
    out = convert_mxfp4_weight(packed, scale)
    check("overflow detected (fp16_range_ok == False)", not out["scale_info"]["fp16_range_ok"],
          str(out["scale_info"]))


if __name__ == "__main__":
    print(f"torch {torch.__version__}\n")
    test_lut_exact_in_e4m3()
    test_vllm_lut_matches()
    test_repack_roundtrip()
    test_full_dequant_bit_exact()
    test_scale_overflow_flagged()
    print("\n" + ("ALL CHECKS PASSED" if _ok else "SOME CHECKS FAILED"))
    raise SystemExit(0 if _ok else 1)
