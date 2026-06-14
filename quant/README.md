# ZAYA1-8B expert quantization

Offline quantization of ZAYA1-8B for faster RSA serving: freed VRAM lets
`--max-num-seqs` rise so a round of N=16 RSA rollouts runs as one wave
(measured with `bench/`).

State of play:

| artifact | scheme | status |
|---|---|---|
| `/home/pat/code/zaya/ZAYA1-8B-INT8/INT8` | **bitsandbytes LLM.int8()** (pre-quantized, SCB scales; attention + experts) | ✅ previously tested working — the known-good INT8 serve candidate |
| `quantize_fp8.py` → `~/models/ZAYA1-8B-fp8` | compressed-tensors **W8A8 FP8 e4m3** (experts only) | staged — needs GPU-window validation on gfx1201 |
| `quantize_int8.py` / `quantize_w8a16.py` | compressed-tensors W8A8/W8A16 int (experts only) | ⛔ archived — Triton quant-MoE miscomputes on the retired gfx1100/RDNA3 card (see bottom); int8 retest on gfx1201 is optional |

The two viable paths trade off differently: **bnb int8** dequantizes to
bf16/fp32 for compute — a pure VRAM win using kernels that are known to work
with this model; **fp8 W8A8** runs the GEMMs in 8-bit — VRAM *and* compute
win, but rides the quantized fused-MoE kernels, which is exactly what broke on
RDNA3 and is unvalidated on RDNA4.

## Target stack: gfx1201 (RDNA4), ROCm 7.14 TheRock

The machine now runs RX 9070 XT + RX 9070 (gfx1201) with the
`vllm-gfx1201:latest` image built by the `vllm-rocm714-gfx1250` sessions; ZAYA
support is overlaid from the `zaya1-therock` branch (see
`../ZAYA1_GFX1201_PORT.md`). Relevant differences from the old RDNA3 setup:

- RDNA4 has native **FP8 WMMA**; gfx1201 uses OCP `float8_e4m3fn`
  (`RocmPlatform.fp8_dtype()`; fnuz is MI300-only).
- The ROCm fork enables **AITER** ops on gfx1201 and is growing RDNA-aware
  quantized-MoE paths (`compressed_tensors_moe_w8a8_fp8.py` per-channel +
  dynamic-token, `rocm_moe.py`, `..._wna16_rdna3.py`, in-progress
  `csrc/quantization/w4a8_fp8_wmma`).
- **The image does not ship `bitsandbytes`** (verified) — serving the bnb
  checkpoint there needs a ROCm bnb build installed in the container first,
  and bnb-on-gfx1201 support is itself unverified.

## Build the FP8 checkpoint (CPU-only, safe anytime)

```bash
# CPU-only venv (kept separate from the rsa client venv):
uv venv --python 3.12 .venv-quant
uv pip install -p .venv-quant/bin/python --torch-backend=cpu torch safetensors numpy

SNAP=$(find ~/.cache/huggingface/hub/models--Zyphra--ZAYA1-8B/snapshots \
        -maxdepth 1 -mindepth 1 -type d | head -1)
.venv-quant/bin/python -m quant.quantize_fp8 --src "$SNAP" --dst ~/models/ZAYA1-8B-fp8
```

Per-output-channel symmetric FP8-e4m3 expert weights + dynamic per-token FP8
activations, emitted as a `compressed-tensors` checkpoint
(`CompressedTensorsW8A8Fp8MoEMethod`, channel/token path). Everything else
stays bf16 (router, attention/CCA projections, norms, tied embedding). The
direct-safetensors approach exists because ZAYA1 ships no `transformers`
modeling code, so `llm-compressor`'s calibrate-on-GPU flow can't load it; the
script needs no GPU and never touches the shared inference container.

## Deploy + validate — needs a coordination window

The GPUs are shared with the other sessions' containers (`rdna4-vllm-therock`,
`rdna4-w4a8-awq`, `qwen36_bench*`); deploy only in an agreed window.

1. **Baseline (bf16).** From the branch root: `docker compose up -d vllm`
   (overlay compose serves bf16 ZAYA on `vllm-gfx1201:latest`), then
   `.venv/bin/python -m bench.run --prometheus http://localhost:9090
   --rsa-n 16 --rsa-k 4 --rsa-t 2 --repeat 3 --out bf16-gfx1201.json`
2. **Coherence gate (cheap, first).**
   `docker compose -f docker-compose.yml -f quant/docker-compose.fp8.yml up -d vllm`,
   wait for `/health`, send one chat completion and *read the output*.
   Garbage = fp8 fused-MoE kernel bug on gfx1201 → stop, record, revert.
   (The RDNA3 canary still applies: fake-quant round-trip was coherent while
   true quantized MoE was garbage, so garbage = kernel bug, not precision.)
3. **Measure FP8.** `bench.run ... --out fp8-gfx1201.json`; compare `latency`,
   `accuracy`, `server.peak_running_seqs` vs bf16. With freed VRAM confirm a
   16-rollout round runs as one wave (`running ≈ 16, waiting ≈ 0`).
4. **Accuracy gate.** AIME-style set (`bench.run --questions aime.jsonl`) on
   both; no regression to adopt.
5. **AITER sweep (optional, same window).** Repeat with
   `VLLM_ROCM_USE_AITER=1` in the compose env — the fork gates AITER on for
   gfx1201 but ZAYA's MoE/CCA is unvalidated under it.
6. **Restore or keep.** Revert to bf16 or hand the winning config to the
   other sessions; confirm `/health` before releasing the window.

### Fallback: the known-good bnb INT8 checkpoint

If FP8 fails the coherence or accuracy gate, the bnb checkpoint is the proven
VRAM-saving serve config:

- Checkpoint: `/home/pat/code/zaya/ZAYA1-8B-INT8/INT8` (LLM.int8(),
  `load_in_8bit`, SCB scales; quantizes attention projections *and* experts).
- Serve with `vllm serve /models/ZAYA1-8B-INT8/INT8 --quantization
  bitsandbytes ...` (same ZAYA flags as the base compose).
- Prereq on the new stack: install a ROCm `bitsandbytes` build into the
  container/image (not shipped in `vllm-gfx1201:latest`) and smoke-test that
  bnb's ROCm backend accepts gfx1201 before burning a window on it.

### If everything 8-bit fails on gfx1201

- Track the other sessions' `w4a8_fp8_wmma` kernel work — a W4A8 ZAYA expert
  checkpoint is the natural follow-on; this script's direct-safetensors
  structure extends to it.
- If per-channel RTN passes coherence but loses accuracy on AIME, a GPTQ pass
  (needs a GPU window + calibration data) is the next lever.

---

## ⛔ Archived: compressed-tensors INT8 on gfx1100/RDNA3 (retired RX 7900)

vLLM's **quantized fused-MoE Triton kernels miscompute on gfx1100/RDNA3** —
they produce garbage. Proven across W8A8 (`TritonExperts`), W8A16
(`TritonWNA16Experts`), and vLLM's own online `--quantization experts_int8`
(which bypasses all our code). The *unquantized* MoE kernel (bf16, and a
fake-quant that rounds experts to int8 then dequantizes to bf16) is coherent,
and the int8 **Linear** kernel (o_proj) is coherent — so int8 *precision* and
the int8 GEMM are fine; only the quantized **MoE** kernels were broken on
RDNA3.

vLLM intentionally gated these off on ROCm at the time (`rocm.py` allow-list
excluded `experts_int8`; `triton_moe.py` gated int8 MoE to `is_cuda()`). Those
gates were **correct** for RDNA3. The three local edits needed to reach the
kernels are archived in `vllm-rocm-quant-patches.diff` (reverted from the
tree — do not re-apply as fixes). Re-enabling INT8 on RDNA3 would require
fixing the Triton quant-MoE kernel upstream.

`quantize_int8.py` (W8A8), `quantize_w8a16.py` (weight-only), and the
`docker-compose.{int8,int8-patch,online-int8,w8a16,fakeq,oproj}.yml` overrides
are kept as the reproducible record of that investigation. Note these
conclusions are **RDNA3-scoped**: the RDNA4 kernel paths are different code,
and the fp8 runbook above is how we find out whether they hold there.
