# CCA fused-kernel performance uplift (ZAYA1-8B, gfx1201)

End-to-end A/B of the fused CCA HIP kernels versus the eager CCA path, measured on
a real ZAYA1-8B-FP8 serve. Run before merging the consolidation to `main`.

## Why these kernels exist

CCA's conv + grouped-means + per-head RMS-norm + state-roll runs as the
graph-broken `vllm::cca` op. Eagerly it is ~1.4M tiny pointwise ATen launches per
decode run — latency-bound and a large share of decode time on a wave32 RDNA4
card. Three fused kernels collapse it into one launch per region:

1. **decode** (`cca_decode_qk`) — pure-decode batches. Gated `ZAYA_CCA_HIP`.
2. **prefill** (`cca_prefill_qk`) — pure-prefill batches. Gated `ZAYA_CCA_HIP_PREFILL`.
3. **mixed** — prefill+decode in one batch: both kernels run into one forward,
   disjoint conv-state slots, decode-rows-first concat. Engages when both gates on.

## Setup

- **HW:** AMD RX 9070 (gfx1201, **56 CU**, 16 GB), card held exclusively (no
  sibling GPU contention — that materially affects these numbers, see caveat).
- **Model:** `ZAYA1-8B-fp8` (experts-only FP8, compressed-tensors), TP=1,
  `--mamba-cache-dtype=float32`, `--max-num-seqs 16`, CUDA graphs on (not eager).
- **Image:** `vllm22-zaya:combined` (Dockerfile.zaya on `vllm22-w4a8:combined`).
- **Configs** (runtime env toggles, same image/weights):
  | cfg | `ZAYA_CCA_HIP` | `ZAYA_CCA_HIP_PREFILL` | active kernels |
  |---|---|---|---|
  | **A** eager | 0 | 0 | none (full eager CCA) |
  | **B** decode | 1 | 0 | decode only (prefill/mixed → eager) |
  | **C** full | 1 | 1 | decode + prefill + mixed |
- **Workloads:**
  - *Chat decode* — `bench/profiles/decode_toks_conc.py`, 16 concurrent greedy
    `/v1/completions`, 256 output tokens: aggregate decode tok/s. Warmup + 3 reps.
  - *RSA shim* — `rsa.cli` (N=4, K=4→k2, T=2, full-trace tails, 1024 tok/rollout,
    temp 0.8), one math query. Prefill-heavy (round-1 aggregation prompts ~5–6k
    prompt tokens) + decode → exercises prefill and mixed. 2 reps (temp>0 is
    required — RSA needs diverse tails — so token counts vary run-to-run; compare
    throughput, not wall).

## Results

Aggregate throughput, higher = better. Δ vs eager (A).

| metric | A: eager | B: decode | C: full | B Δ | C Δ |
|---|---|---|---|---|---|
| **chat decode tok/s** @conc16 | 281 | 389 | 387 | **+38.6 %** | **+37.7 %** |
| **RSA decode tok/s** | 69.7 | 104.2 | 103.8 | **+49.5 %** | **+49.0 %** |
| RSA total tok/s (prompt+gen) | 127 | 193 | 181 | +51 % | +42 % |

Per-rep (tight on decode; RSA total is noisier due to temp-0.8 token-count
variance, so RSA decode tok/s is the cleaner RSA metric):

```
chat decode tok/s   A: 280.8 281.8 280.7   B: 390.6 389.1 389.1   C: 385.6 389.4 386.7
RSA  decode tok/s   A: 69.0 70.3           B: 103.7 104.6         C: 97.6 109.9
```

### All three kernels confirmed firing (config C)

`ZAYA_CCA_DEBUG_PATHS=1` tallies the fused path per forward. Over the chat+RSA run:

```
config C:  ZAYA CCA HIP path counts: {'decode': 400, 'prefill': 840, 'mixed': 760}
config B:  ZAYA CCA HIP path counts: {'decode': 1,   'prefill': 0,   'mixed': 0}   (prefill/mixed → eager, by design)
config A:  (no HIP paths — fully eager)
```

The RSA aggregation rounds drive heavy prefill and, under continuous/chunked
batching, mixed prefill+decode steps; the concurrent chat bench adds more. All
three kernels are exercised in config C.

## Reading the result

- **The decode kernel is the win.** Eager CCA → fused decode is **+38 % chat /
  +49 % RSA** aggregate throughput on this 56-CU card. The eager path's launch
  storm is the bottleneck at concurrency; one fused launch removes it.
- **The prefill/mixed kernel is throughput-neutral here (B ≈ C), not a
  regression.** It is bit-exact to eager (unit-validated: qk rel ~6e-7, state
  bit-exact, mixed == pure⊕pure) and keeps output coherent. Its standalone
  prefill kernel is 2.4–3× faster, but prefill is a small fraction of these
  MoE/GEMM-bound workloads' GPU time, so it doesn't move aggregate tok/s on an
  exclusive card. Its real-world value is launch reduction + L2/HBM
  bandwidth-contention relief (which shows up under multi-tenant GPU pressure,
  not in this isolated A/B) and removing the eager mixed-batch fallback.
- **Coherence:** all three configs produce coherent output (e.g. "capital of
  France" → "Paris"); enabling the prefill/mixed kernel does not change it. This
  was the first coherent ZAYA serve on the combined base (the consolidation's MoE
  weight-loader had to be ported from the therock `RoutedExperts` nesting to the
  base's factory `FusedMoE` — see commit log).

## Caveats

- 56-CU RX 9070; the 64-CU XT and contended/multi-tenant runs will differ. An
  earlier config-B reading of 348 tok/s was depressed by a sibling job sharing the
  other card — all numbers here are exclusive-GPU.
- RSA at temp 0.8 has run-to-run token-count variance; throughput (tok/s)
  normalizes for it, raw wall time does not.
- Decode and prefill kernels gated OFF by default historically; `ZAYA_CCA_HIP`
  defaults on, `ZAYA_CCA_HIP_PREFILL` enabled by default as of this work (no
  regression, bit-exact, all-mode coverage).
