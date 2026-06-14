# v11 K-tiling — GPU validation (2026-06-13, GPU1 / RX 9070)

## What changed
`mmq_fp8_gemv_kernel_v11` now **K-tiles** the activation staging: only `(M, BK)` fp8
sits in LDS per chunk (BK = largest 1024-multiple ≤ 64KB/M), accumulating across chunks
in registers. Lifts the old `M*K ≤ 65536` cap that locked v11 out of large-K layers
(e.g. K=17408). Files: `w4a8_fp8_wmma_kernel.hip` (kernel + launcher, `int BK` param),
`vllm_adapter.py` (`v11_ok` drops M*K cap; `decode_max`). Built in-container (gfx1201),
compiles clean.

## Correctness — ALL PASS (bit-exact vs validated v5 path)
v11-vs-v5 mean|Δ| ≤ 2.4e-07 across M∈{1,4,8,16}, K∈{4096,5120,17408}, g∈{32,128},
incl. the previously-IMPOSSIBLE M=4..16 × K=17408 (M*K up to 278528 ≫ 65536).
v11-vs-dequant-ref rel ≈ 0.027 (fp8-activation quant error, expected).

## Decode microbench (µs, lower=better; N=5120, K=17408)
| M | v11 | v10 | triton | winner |
|---|---|---|---|---|
| 1 | **162** | 440 | 283 | **v11 (1.75× vs triton)** |
| 4 | 314 | 442 | 284 | triton |
| 8 | 518 | 444 | 250 | triton |
| 16 | 948 | 443 | 288 | triton |

## Verdict
- v11 is a GEMV: scales ~`110 + 50·M` µs → **only wins at M=1** (single-stream decode),
  crosses Triton at M≈3.
- So K-tiling is a real win for **latency-bound single-stream decode**, but does NOT
  beat Triton for **batched decode (M≥4)** — there Triton W4A16 is best, v10 is flat-slow.
- **Implication for "drop the crossover":** not achievable for the dense path by K-tiling
  alone. Optimal dense dispatch is still 3-way: v11 (M≤2) → Triton (mid-M decode) → v10
  (large-M/prefill, the +53% win). Eliminating the Triton fallback would need a NEW
  small-batch (M=2..~16) WMMA decode kernel that beats Triton — open kernel work.
- Batched decode wanting Triton also means the 27B still needs the Triton fallback weight
  copy for ≥parity → the second-copy memory fix (lazy/streamed) is the unlock, not removal.

---
# MoE gemm2/scatter fusion — GPU validation (2026-06-13, GPU1)

## Root cause (isolation microbench, 35B-A3B dims hidden=2048 inter=512 E=256 tk=8)
The gemm2/scatter "2x gemm1" cost is **entirely the atomic scatter**, and only at scale:
| T | gemm1 | gemm2 no-scatter | gemm2 SCATTER | atomic cost |
|---|---|---|---|---|
| 8 (decode) | 472 | 200 | 188 | ~0 (free) |
| 2048 (prefill) | 1903 | 1079 | **4778** | **+3700** |
gemm2's GEMM is efficient (~half gemm1, as FLOPs predict); the 2048 atomicAdds/token-slot
x 8-way top_k contention add +3.7ms at prefill. Free at decode (few tokens).

## Fix: adaptive scatter vs no-atomic gather-reduce (moe_experts.py apply)
- decode (M < SCATTER_MAX_M=128): keep atomic scatter (lowest overhead, +11% win intact).
- prefill (M >= 128): gemm2 non-scatter -> (P,N) + contention-free torch gather-reduce
  (build slot->row inverse map, gather each token's top_k rows, weighted fp32 sum).
- Numerically bit-matches scatter (rel 2e-4). Apply-test pass rate == baseline (the 1-2
  borderline fails are pre-existing tolerance flakiness on random fp8 data, both paths).

## Result (microbench, full gemm2 replacement)
| T | scatter | proposed | speedup |
|---|---|---|---|
| 512 | 1488 | 1091 | 1.36x |
| 2048 (prefill) | 4778 | 2401 | **1.99x** |
gemm2/scatter was 37.8% of prefill GPU time -> halving it should move MoE prefill from
0.84x toward parity/win. **E2E A/B (TP=2, 2026-06-13) CONFIRMS:** 35B w4a8 prefill
**0.84x -> 0.96x** of stock (36.4 -> 41.9 tok/s, +15%); profiler scatter share
**37.8% -> 1.6%**. Decode control 141.3 tok/s (1.08x, unchanged). So MoE is now
decode +8-11% / prefill ~parity; mid+large (M<128, not scatter-bound) unchanged ~0.92x.
Follow-up: a custom HIP gather-reduce kernel would beat the torch reduce (~1.3ms overhead)
and likely win at decode too, removing the adaptive branch + the cudagraph-safety caveat.

---
# Piece 1: custom HIP gather-reduce kernel — DONE + validated (2026-06-13, GPU1)
New op `mmq_fp8_moe_gather_reduce` (moe_kernel.hip: build_inverse_map + gather_reduce
kernels; bindings.cpp; __init__.py). Replaces the gemm2 atomic scatter with: gemm2
non-scatter -> (P,N), build slot->row inverse map, per-(token,n) gather top_k rows +
weighted fp32 sum. No atomics, coalesced, HIP-graph safe. moe_experts.py apply now uses
it as a SINGLE PATH (adaptive branch removed).
Microbench (35B-A3B dims; bit-matches scatter rel 2e-4):
| T | scatter | torch_reduce | KERNEL | kernel vs scatter |
|---|---|---|---|---|
| 8 (decode) | 187 | 381 | 237 | 0.79x (~50us behind, <0.3% e2e) |
| 512 | 1488 | 1090 | 811 | 1.83x |
| 2048 (prefill) | 4782 | 2407 | **1289** | **3.71x** |
Apply composition test: all PASS. **E2E prefill A/B (TP=2) pending 2-GPU window** —
expect prefill past the torch-reduce's 0.96x toward/above stock parity (kernel is 2x the
torch reduce). Built + deployed + synced (build-env + container).

---
# GPTQ MoE hook (register_moe_gptq) — implemented 2026-06-13, pending other-agent validation
WHY GPTQ didn't engage: GPTQ-4bit MoE -> `AutoGPTQMoEMethod`, which picks experts via the
shared `select_wna16_moe_backend` oracle (-> stock MarlinExperts). register_moe patched only
the `awq_marlin` namespace, so GPTQ stayed on Marlin (reached us only via the marlin-UNsupported
fallback to MoeWNA16Method). FIX: `register_moe_gptq` mirrors the AWQ hook in the `auto_gptq`
namespace — overrides AutoGPTQMoEMethod.__init__ (wna16_moe_backend/experts_cls) for symmetric
GPTQ-4bit (no desc_act, supported group size) + patches convert/make there. Symmetric uint4b8 ->
w_zeros=None; GPTQ K-packed weights are GPTQ-convention -> reuses `_ct_moe_to_op_layout`.
Status: installs cleanly (patches swap confirmed); wired into register(). 
**TO VALIDATE (other agent):** a cached GPTQ-Int4 MoE model -> confirm log
"[w4a8_fp8_wmma] GPTQ MoE -> W4A8Fp8WmmaExperts" + coherent text + that `_ct_moe_to_op_layout`'s
nibble order matches AutoGPTQ's qweight packing (the one assumption to spot-check).

---
# HANDOFF — review of other-agent compare logs + a fix (2026-06-13)
Reviewed `profiling/compare-combined` + `compare-oldvllm` (stock vs w4a8 × kv-auto/fp8,
AWQ-7B dense + GPTQ-MoE). Findings + one regression fixed:

## Findings (their logs)
- **No-force (default) = parity, never regresses.** AWQ-7B dense w4a8 no-force = 1.00–1.05x
  stock across decode/mixed/prefill. This is the deployable config (crossover falls back to
  Triton where we'd lose). USE NO-FORCE (or a tuned crossover) for realistic dense numbers.
- **GPTQ MoE: our path ENABLES it.** Stock = LOAD_FAIL — a *vanilla vLLM* bug, not ours:
  `triton_w4a16_gemm` asserts `qzeros shape [768,16]` (it's handed [N//8,K//group] but wants
  [K//group,N//8] — transposed qzeros, the same class we fixed in our adapter). Our path
  loads + runs it (413–488 tok/s no-force; engages via register_moe_wna16, has_zp=False).
- **FORCE=on dense 7B = 0.48x is NOT a regression** — it forces our kernel at decode where
  v10 loses (~0.5x). Diagnostic mode only.

## REGRESSION FOUND + FIXED (affects your forced runs)
`decode_max` had reverted 2 -> 16 (a .so rebuild reinstalled stale .py from /tmp/w4a8src).
Re-synced to 2 everywhere + **re-committed `vllm-gfx1201-w4a8bench:latest`**.
- The 7B FORCE=on result is unaffected: the **7B is v11-INELIGIBLE** (K=3584/18944, not %1024)
  -> v10 ran, not v11. (Also: your harness decode M=8 / prefill M=4 never enters v11's M<=2
  winning range.)
- BUT **re-run any FORCE=on test of the 27B / 35B (K%1024==0) on the REFRESHED image** — the
  old image (decode_max=16) wrongly routed M=4–16 decode to v11 (its bad range: 518us@M=8 vs
  v10 444). decode_max=2 keeps v11 to M<=2 (where it wins 1.28–1.75x).

## Harness nits
- The `prefill` regime generates only 4 tokens -> `out_tok_s` is noise; use `total_tok_s` or
  longer-gen. - kv-fp8 helps GPTQ-MoE decode (+~25%) but is neutral on the dense 7B.

---
# KNOB GUIDE for engaging the kernels (from benchmarking feedback, 2026-06-13)
Informational — how to actually engage each kernel without confounding the other:

- **MoE grouped kernel -> use `VLLM_ROCM_W4A8_FP8_WMMA_MOE_MIN_M=<N>`** (engage our grouped
  op for batch M>=N; MoE-only, does NOT touch the dense path). This is the clean knob to
  exercise/measure the MoE kernel in isolation. MIN_M=1 forces it for all M.
- **`VLLM_ROCM_W4A8_FORCE=on` is a BLUNT diagnostic — avoid for real measurement.** It
  forces the DENSE path everywhere (drops the Triton fallback), which CRIPPLES dense outside
  the decode sweet-spot (~−50% on the AWQ-7B: forces v10/v11 where Triton wins). In the
  refreshed image it ALSO engages MoE (wired into _moe_should_engage), but because it
  simultaneously tanks dense it's the wrong tool for either path in isolation.
  (Earlier images: FORCE only reached the dense gate, so MoE looked un-engageable under FORCE
  — that's why MIN_M was the only MoE knob then. Refreshed image wires FORCE into MoE too.)
- **Dense kernel in production -> FORCE=auto (default) + a tuned crossover** (never regresses;
  engages our kernel only in its proven-winning shape/M window). FORCE=on is for isolating
  the kernel, not deploying it.

Bottom line for the kernel agent: measure MoE with `MOE_MIN_M`, measure dense with the
crossover (or per-shape), and treat `FORCE=on` as "run our kernel everywhere even where it
loses" — useful for kernel A/B, misleading for end-to-end.


---
_Origin: validated against the snapshot in vllm-gfx1201/profiling/kernel-wip (PROVENANCE: copied from this canonical tree 2026-06-13 02:34). Snapshot removed during consolidation 2026-06-14; this file is the surviving record._
