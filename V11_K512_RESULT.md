# v11 K%512 tail — fix down_proj decode falling to v10 WMMA

## Change (3 files)
- `w4a8_fp8_wmma_kernel.hip` (and its hipify twin `_hip.hip`): v11 `TORCH_CHECK`
  relaxed `K % 1024 == 0` → `K % 512 == 0`. The kernel body is UNCHANGED — it already
  clamps the final chunk (`ck=min(BK,K-kc)`) and the inner warp-step tiles by a 32-k
  (4×int32 b128/lane) granularity, so a 512-k tail is consumed exactly by 16 lanes. The
  1024 gate was conservative; 512 keeps the (N,K/8) row int32-offset 16-byte aligned.
- `vllm_adapter.py`: dispatch gate `v11_ok` `K%1024` → `K%512`.

## Why it matters
The 27B (Qwen3.6-27B-AWQ-INT4) `down_proj` has intermediate 17408; TP=2 shards K to
**8704 = 8×1024 + 512**, which failed the old `K%1024` gate, so every decode token ran
down_proj on **v10 (WMMA, pads M=1→16)** instead of the v11 GEMV.

## Validation (gfx1201, rebuilt _C)
- Correctness (`test_v11_k512.py`): ver=11 direct + adapter-dispatch, K∈{512,1536,2560,
  **8704**}, M=1..16, sym + asym (AWQ) — all PASS vs numpy fp8 ref (rel 1e-9..1.8e-3,
  K%1024 regressions still pass). down_proj K=8704 asym = ~1e-6.
- Payoff (`bench_v11_vs_v10_downproj.py`, N=5120/K=8704/gs=32/asym):
  | M | v10 | v11 | speedup | v11 GB/s |
  |---|-----|-----|---------|----------|
  | 1 | 338 µs | 53.7 µs | **6.3×** | 479 (~75% peak) |
  | 2 | 334 µs | 73 µs | 4.6× | 351 |
  | 4 | 336 µs | 126 µs | 2.7× | 204 |
  Matches the live trace (v10 ~360 µs/launch, v11 ~54 µs/launch). v10@M=1 read weights at
  ~76 GB/s; v11 at 479 GB/s.

## Not yet measured
End-to-end serve tok/s A/B. down_proj was a top decode consumer in the trace, and it also
shortens rank0's critical path (so the TP bubble should shrink too) — direction is a real
decode speedup + power cut, but the magnitude needs a full TP=2 serve A/B (warm cache now
covers the 27B). Branch: `feat/v11-k512-tail` (uncommitted).
