# Cross-Cutting Improvement Pathways

**Date:** 2026-06-14 · **Status:** living planning doc · **Home:** `main` (project analyses + future planning live here)

A repo-wide audit of *every* work area — W4A8 kernel, ZAYA1 port, quantization/RFP458,
heterogeneous-TP, packaging/CI, and profiling/validation — looking for pathways to
further improvement. The value here is the **cross-cutting** view: themes and
dependencies that span domains, which a per-area review misses. Generated from six
parallel domain audits (read-only) against the repo + `DIARY.md` + the RFP458 plan.

> This is a coordination artifact. The repo + `memory/` + handoff docs are the only
> channel between parallel sessions (DIARY Act IX), so keep it current as items land.

---

## 0. Read this first — the headline e2e numbers are measured on the wrong basis

The project's flagship results — **dense prefill +53%, MoE decode +11%** (DIARY Act VI
correction, `profiling/sweep-2026-06-13/AUDIT.md`, `SWEEP_FINDINGS.md`) — were all
captured with `VLLM_ROCM_W4A8_FORCE=on`. But:

- **Act IX itself calls FORCE "a trap"** that *halved* dense throughput (438→212) by
  ramming the custom kernel onto shapes AUTO correctly routes to Triton.
- The abtest log shows `VLLM_ROCM_W4A8_FORCE` was an **"Unknown vLLM environment
  variable"** on the image that produced those traces.

So the 16-cell table measures **"kernel forced onto every shape,"** not the **served
AUTO pathway**. Every win/loss in the headline table is suspect until re-run in AUTO
mode on the combined image. **Action: one AUTO-mode 16-cell re-run** confirms or
retracts the kernel chapter's conclusions. Treat "dense prefill +53%" as unverified
until then.

(Two correctness gates *are* already cleared and bankable: het-TP greedy-equivalence
passed 2026-06-14 — byte-identical token ids even vs 64:56 — and the dense load-time
OOM is fixed.)

---

## 1. Cross-cutting themes

### Theme 1 — One technical thread unifies half the repo: "dequant register-direct → feed fp8 hardware, never expand to F16"

The same problem, currently solved four times independently:

| Where | State |
|---|---|
| **W4A8 dense** (Act VIII) | **Proven wall.** Stock Triton dequants int4→fp16 *register-direct, fused into `tl.dot`* (~190 GB/s); ours expands int4→fp8 *staged through LDS* (~108 GB/s). v13 (register-direct shuffle), v15 (Marlin repack), v12 (split-K) — **all falsified, all 108 GB/s.** Dense M=16–32 is a compiler-codegen wall, not a missing technique (v16/v17 confirm). |
| **W4A8 MoE** | gemm2 reads w2 at **126 GB/s** vs gemm1's 273 (short-K bursts). *Unfalsified* lever — a repack may help here (bandwidth-bound) where it didn't for dense (latency-bound). |
| **RFP458→FP8** | The whole combining plan. RFP458 has the better *format* but feeds fp16 (`rfp458_kernels.py:327` `.to(compute_type)`); W4A8 has the fp8 *feed* but a weaker format. |
| **ZAYA fp8** | Same silicon, different kernels (compressed-tensors W8A8 e4m3). |

**Convergence target:** one parameterized **"4-bit-pack → e4m3 → fp8-WMMA feed"** path,
format-agnostic (uint4b8/AWQ *or* RFP458 IQ4-NL+block-float), sharing the register-direct
dequant tail, a rank-1 epilogue scale (fp32 accumulator), a per-token-fp8 activation
primitive, and one fidelity harness (Theme 4).

**The binding caveat for all four:** at decode the system is already bandwidth-bound, so
fp8 does **not** cut weight traffic — the win is prefill / latency-hiding / VGPR-LDS
pressure relief. The collaborator's "spend compute to buy accuracy under the bandwidth
roofline" thesis and Act VIII's measured reality are the same coin. A plausible honest
outcome is **"no-go for decode, yes for prefill"** — measure, don't assert.

### Theme 2 — Validated kernels that never actually engage (dispatch-gating)

The MoE chapter's proven v7/v6 wins (M=1–96) **do not run in production**:
`w4a8_fp8_wmma/moe_crossover_cache.json` holds only 2 Mellum2 entries, so the real 35B
TP-sharded shape → unknown → falls back to stock. This *is* the
[[w4a8-moe-kernel-does-not-execute-on-35b]] memory ("our op = 0 calls").

- **Populating that cache for the served shape is effort-S, the highest-leverage,
  lowest-risk action** — it flips the validated MoE win from dormant to live.
- The FORCE trap (§0) is the same disease from the other side.
- **Durable fix: a startup auto-profiler** replacing static JSON — the collaborator's
  attention autotuner (profiles the deployed shape at engine init) is the proven template.
- **Coupling:** het-TP's 64:56 split *changes the MoE cache key* (per-rank `inter` dim),
  so cache-population and het-TP must be validated together.

### Theme 3 — The scarce 2-GPU window is the universal bottleneck; batch it

Everything queues behind the same two cards + warm Triton cache. Proposed single-window
batch (one combined-image 35B TP=2 server, AUTO, cudagraph-on, amortizing the ~20-min
cold compile once):

1. Het-TP COMM A/B (even vs 64:56) **+** cudagraph-mode **+** fidelity-tensor capture — one server.
2. 27B dense + 7B, **AUTO (not FORCE)**, all regimes → resolves §0 *and* the dense A/B in one pass; add the b64 cell to capture/fix the runtime OOM.
3. RFP458 verifier (`quant/rfp458_fp8/run_verify.sh`) — pin one card, seconds.
4. ZAYA coherence gate — one chat, if the ZAYA image is staged.

Honour [[ask-before-gpu-time]] and [[container-testing-protocol]] (mount the per-image
Triton cache). A cheaper *partial* het-TP de-risk needs **no** 2-GPU window:
`VLLM_TP_CU_WEIGHTS="64"` at TP=1 exercises every offset/stamp path against the even baseline.

### Theme 4 — Missing infrastructure: no fidelity harness; contradictory result docs

- **No fp8-dot fidelity harness exists anywhere** (grep-confirmed). Bit-exact tests are
  useless for *lossy* fp8 — the question is "how much accuracy did we trade." RFP458
  Phase-1 needs a Tier-1 MSE-vs-fp16 harness and there is nothing to run it. Build it once,
  generic over reference-vs-candidate paths; it serves RFP458 **and** ZAYA fp8 **and**
  every future quant. **CPU-authorable, no GPU window** — a do-it-now.
- Result numbers are scattered across `results.jsonl`, `DIARY.md`, `AUDIT.md`,
  `SWEEP_FINDINGS.md` and **already contradict each other**. A tiny `analyze.py` that
  regenerates one canonical table from raw jsonl stops the doc/data drift.
- The torch-profiler+kineto method (`profiling/analyze_torch_trace.py`) is reusable across
  kernels — only its ZAYA-centric bucket map needs generalizing.

### Theme 5 — Reproducibility & multi-agent-coordination fragility

- **The floating `tcclaviger/vllm22:dev` base tag is a silent repo-wide risk.** RFP458
  lives in it; het-TP, the `moe_wna16` sed, and the coming RFP458 patch are all
  line-number-sensitive against `/app/vllm`. If the tag moves, every patch bitrots
  simultaneously and the W4A8 ABI can break — nothing detects it. **Pin to a `@sha256:`
  digest** and add a CI digest assertion.
- **CI is broken:** `.github/workflows/build-image.yml` references a non-existent
  `./Dockerfile` and the removed `WITH_W4A8` arg — it cannot succeed today. No non-GPU
  lint/test job exists, though CPU-runnable checks do (`patches/het_tp.py` self-tests,
  numpy conversion tests, `docker compose config -q`).
- **The patch-slot pattern won't scale past ~3 slots** — RFP458 will be the third bespoke
  `RUN` block in `Dockerfile.combined`. Land it, then extract a `patches/manifest` + one
  `apply-patches.sh` loop (folding the inconsistent `moe_wna16` sed in too).
- Since the repo is the only cross-session channel, **stale handoff docs are a correctness
  risk:** `patches/HET_TP_HANDOFF.md` still documents the abandoned `--build-context` flow;
  several het-TP files it cites are uncommitted.

---

## 2. Prioritized action list

Effort S/M/L; "GPU" = needs a (2-)GPU window. Ordered by leverage, non-GPU first.

| # | Action | Theme | Domain | Effort | GPU? | Why it matters |
|---|---|---|---|---|---|---|
| 1 | Build the **fp8-dot fidelity harness** (MSE/cosine/rel-err, reference-vs-candidate) | 4 | quant/profiling | M | no (capture later) | Unblocks RFP458 Phase-1 + ZAYA fp8 + all future quant; nothing exists today |
| 2 | **Pin base image to a `@sha256:` digest** + document bump procedure | 5 | packaging | S | no | Silent repo-wide reproducibility/ABI/bitrot risk; RFP458 lives in that base |
| 3 | **Fix CI** (point at `Dockerfile.combined`, drop `WITH_W4A8`, fix paths) + add a **non-GPU lint/test job** | 5 | packaging | S–M | no | CI cannot run today; regressions land silently until a GPU window is burned |
| 4 | **Canonical results-table generator** from `results.jsonl` | 4 | profiling | S | no | Docs already contradict each other; stops drift |
| 5 | **Upstream the `moe_wna16` tp_size fix** as a vLLM PR | — | distributed | M | no | Genuine arch-agnostic upstream bug; reduces local carry; community value |
| 6 | **Hygiene sweep** (see §3) — delete stale dupes, fix wrong runner, drop tilelang overlay, fix README | 5 | all | S | no | Several are correctness/coordination risks, not just tidiness |
| 7 | **AUTO-mode 16-cell re-run** on the combined image | §0 | profiling | M | **yes** | Confirms/retracts the entire headline e2e table — credibility keystone |
| 8 | **Populate `moe_crossover_cache.json`** for the served 35B shape | 2 | W4A8 MoE | S | **yes** | Flips the validated v7/v6 MoE win from dormant to live |
| 9 | **Het-TP COMM perf A/B** (does 64:56 recover the bubble?) + cudagraph-mode | 2,3 | distributed | M | **yes** | Last gate on het-TP; correctness already passed |
| 10 | **27B dense AUTO A/B** + fix runtime b64 OOM | §0 | W4A8 dense | M | **yes** | The kernel's most favourable workload; "+53% prefill" stands or falls here |
| 11 | **RFP458 verifier baseline** (`run_verify.sh`) | 1 | quant | S | **yes** | Machine-code proof of the fp16 baseline; gates the RFP458→fp8 thesis |
| 12 | **RFP458→FP8 Phase 1** (dense-linear fp8 dot + epilogue scales) | 1 | quant | L | yes + collab | The unified-feed convergence; gated on collaborator reply + the per-N-scale MSE decision |
| 13 | **MoE fused-decode-apply** (gemm1→silu→gemm2→reduce, LDS-resident) | 1 | W4A8 MoE | L | yes | Closes the remaining M≥2 MoE decode gap; reserved worktree exists |
| 14 | **gemm2 short-K repack research** (126→273 GB/s) | 1 | W4A8 MoE | L | yes | Unfalsified (differs from the dense falsification — BW-bound not latency-bound) |
| 15 | **ZAYA coherence gate** + CCA kernel A/B + DP=2/EP profile | — | ZAYA | S–M | **yes** | Converts the whole "architecturally complete, un-bring-up'd" bucket into measurable state |

---

## 3. Quick wins / bugs (hygiene — mostly no GPU)

- **Delete stale duplicate kernel sources** `w4a8_fp8_wmma/*_hip.hip` + `bindings_hip.cpp` — not referenced by `setup.py`, diff from canonical; exactly the stale-build hazard Act IX thought it killed.
- **`moe_wna16.py` qzeros even-split is a het-TP corruption risk** — if het-TP is on and that loader path runs, qweight shards 64:56 while qzeros splits evenly → silent mismatch. The greedy-equivalence run is the detector; prime suspect if 35B MoE tokens diverge. (`HET_TP_PATCH.md` §11 flags this is still needed.)
- **`patches/wait_and_run_het_e2e.sh:30` execs the wrong runner** — the superseded `run_het_e2e.sh` instead of `run_het_e2e_combined.sh`.
- **Drop the `gfx1201_tilelang` overlay** — the `apache-tvm-ffi==0.1.10` ABI pin already fixes the tilelang abort properly; the full-file `has_tilelang()→False` mount is droppable cruft. (Durable alt: arch-gate the `deepseek_v4`/`mhc` import in `get_quantization_config`.)
- **README contradictions** — claims `rocm_attn` is the default (the combined image's headline is `triton_attn`); the `/workspace/test/bench_tp2.py` bench path doesn't resolve in-container.
- **Dead wheel scripts** — `scripts/{build,fetch}-wheels.sh` pinned to the retired TheRock world; README says "There are no wheels to fetch." Move to `scripts/legacy/` or delete.
- **Commit or remove uncommitted handoff files** — `CONTRIBUTING.md` cites `patches/het_e2e_check.py` (untracked); a fresh clone lacks it.
- **Reconcile stale handoff docs** — `HET_TP_HANDOFF.md`/`HET_TP_PATCH.md` reference `--build-context` + `config.py:1283`, both superseded (COPY-from-repo; the split moved to `layer.py` in 0.22.69).

---

## 4. Per-domain top items (detail on request)

**W4A8 kernel** — Proven walls (do not re-attempt): dense M=16–32 (22 variants incl.
v15/v16/v17, a codegen limit); the dense register-direct/Marlin-repack hypothesis is
*falsified*. Live levers: populate the MoE crossover cache (#8); fused MoE decode apply
(#13); gemm2 short-K repack (#14, unfalsified). Open TODO: shared-experts not folded into
the CT MoE GEMM (`moe_experts.py` — unvalidated for shared-expert models). Five reserved
feature worktrees exist (apply-fusion, actquant-fusion, burst-repack, autotune-gate,
single-layout) — queued, not started.

**ZAYA1 port** — Everything is "architecturally complete, never run a forward pass on
gfx1201." Next unchecked box is the cheapest: the **coherence gate** (one chat). Then:
re-profile decode under `ZAYA_CCA_HIP=1` (current profile is stale eager-path), validate
fp8 experts (or fall back to INT8 — needs a ROCm `bitsandbytes` build), commit the DP=2+EP
profile. Hard open problem: **prefix-cache reuse of recurrent conv state** (not
prefix-sliceable; the `all`-mode block-slot layout is the checkpoint substrate). The
`all`-mode bit-lossless spec path boots but outputs garbage — root-caused to a *core-vLLM
runner* change (doesn't thread `num_accepted_tokens` into CCA `build()`). No single ZAYA
ROADMAP — it's distributed across `ZAYA_HANDOFF.md` + `docs/zaya/cca-all-mode-spec-plan.md`
+ `quant/README.md`. **RSA** is a client-side serving capability, orthogonal to kernels —
already works against any vLLM backend.

**Quant / RFP458** — Three independent tracks: W4A8 (int4→fp8-WMMA, serving-time, no
checkpoints), the ZAYA offline quantizers (`quantize_{fp8,int8,w8a16}.py` — ~90% identical
scaffolds, converge them), and RFP458 (collaborator's, in the base image). Pathways:
RFP458→fp8 (#12), the fidelity harness (#1), e4m3-aware both-polarity scale search, a
reusable per-token-fp8 activation primitive (the W8A16 post-mortem proves activation quant
is *the* failure surface, not weights). Retire the `gfx1201_overlay` once the base carries
the tvm-ffi pin.

**Distributed / het-TP** — Landed as a gated-dormant 3-file 0.22.69 patch (linear.py,
parameter.py, fused_moe/layer.py — *not* config.py); CPU self-tests + greedy-equivalence
pass; COMM-perf payoff unmeasured. Pathways: the 2-GPU COMM A/B (#9), upstream the tp_size
fix (#5), the `moe_wna16` qzeros het-offset (§3 corruption risk), the DP=2-vs-TP=2 topology
question (DP avoids the bubble entirely but the 35B can't fit one card — needs a sub-16GB
model). Generalizing the split beyond FFN/MoE has diminishing returns (attention is ~1.6%
and ineligible).

**Packaging / CI** — Stale-snapshot hazard is now well-defended at the build layer
(`.dockerignore`/`.gitignore` block `*.so`; COPY-from-repo + post-install delete). Residual
risks all in §5: broken CI (#3), floating base tag (#2), patch-slot scaling, doc staleness.
No `docs/` dir existed before this file. There is no automated non-GPU test gate.

**Profiling / validation** — Strong reusable assets: the process-per-cell + resume + drain
sweep harness, the warm-at-shape combined-image compare, the per-rank trace bucketer. The
methodology is hard-won and codified (rocprofv3 is blind to Triton; read rank0; eager
ratios hold; real disk not tmpfs). The backlog (§2 items 7–11, 15) all queue behind the
batched window (Theme 3). Infra gaps: the fidelity harness (#1), the canonical results
table (#4), cudagraph-mode profiling, perf-per-watt-at-the-power-wall as a standard metric.

---

## 5. Suggested sequencing

**Now (no GPU):** #1 fidelity harness · #2 pin base digest · #3 fix CI + lint job · #4
results-table generator · #5 upstream tp_size PR · #6 hygiene sweep. These unblock and
de-risk everything downstream and need no card.

**Next GPU window (batched, one server — Theme 3):** #7 AUTO-mode re-run (settles
credibility) + #10 27B dense A/B + #9 het-TP COMM perf + cudagraph-mode + #11 RFP458
baseline + #15 ZAYA coherence.

**Then (substantial):** #8 populate MoE cache → #12 RFP458→fp8 Phase 1 (gated on
collaborator) → #13 MoE fused-decode-apply and #14 gemm2 repack research.

---

*Provenance: synthesized 2026-06-14 from six parallel read-only domain audits
(W4A8 / ZAYA / quant / distributed / packaging / profiling) against the repo, `DIARY.md`,
and `~/.claude/plans/some-discourse-with-my-crispy-lampson.md`. Numbers attributed to the
forced sweep are flagged unverified pending the §0 AUTO-mode re-run.*
