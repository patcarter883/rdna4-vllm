# CAM — Canonical Memory v1 BUILD PLAN (LOCKED 2026-06-29)

> The execution spec for the first *real* canonical memory — the "perfect memory module" that the
> v0/v1 work greenlit. Supersedes the open "pick a v2 dial" decision in `CONTINUANCE.md`: this build
> executes **dial #1 (whitened canonical-Z committee)** + **dial #2 (product-key store)** + a
> **2:4-by-design serve path**, fused into one coherent plan. v0 (`ckpt/cam_v0_L24.pt`) and the
> translator scaffold are reused, not rebuilt.

## 0. Aim — "Titans for everyone"
A single, base-agnostic, high-fidelity, high-capacity associative memory that ANY frozen LLM attaches
to via a tiny translator card. The product is **not** tuned to the models on our dev box and **not**
tuned to one use-case. The end aim is universal: download the memory, fit a translator for your model,
load the head for your purpose.

**Why now / why invest here:** the canonical memory is the one asset every spoke shares. A defect in
it is paid N times and is uncorrectable downstream (no translator recovers fidelity the memory never
encoded). It has the highest leverage in the system, so we perfect it once, properly.

## 1. Architecture (LOCKED)

### 1.1 Canonical-Z hub — d_model = 4096
A base-neutral hub built by a **committee atlas**: sample each committee model's residual geometry at
its proportionally-mapped tap depth, align via **relative representations** (anchor-probe set →
tokenizer- and dimension-agnostic), whiten to isotropy, spherical-code the stored keys.
- **d = 4096** chosen RDNA4-selfishly *and* generally: `4096 = 32×128 = 8×512` → clean WMMA tiling,
  LDS-friendly, **W4A8/SWMMAC-legal (K%512==0, K%32==0)**, and near-identity translators for the
  large 7–8B spoke cluster (d=4096). Universality comes from in-atlas *coverage*, not from inflating
  the hub dim — a 70B (d=8192) spoke's translator bridges 8192→4096 fine as long as 70B-geometry is
  in-atlas. The sparse store keeps the larger hub's per-token serve read cheap.

### 1.2 Committee — 11 members (LOCKED)
The committee is a **spanning atlas of the open-model ecosystem**, architecture-first (attention
mechanism is the highest-leverage geometry axis), skewed **small** (the Titans sweet spot — bolt-on
memory helps low-capacity models most; floor ≈ 1B so the base can integrate the injected signal).
Covariance-balance same-family members so the hub is not Qwen-biased.

| Tier | Model | Mechanism | probe HW |
|------|-------|-----------|----------|
| Small targets (1–8B) | Qwen3.5-4B | GDN hybrid (SSM) | NVIDIA |
| | Gemma-3-4B | soft-cap | NVIDIA |
| | Ministral-8B | GQA | NVIDIA |
| | BitNet b1.58 2B4T | ternary GQA | NVIDIA |
| | LFM2-1.2B | Liquid conv-hybrid | NVIDIA |
| | Laguna-XS (Poolside) | exotic `LagunaForCausalLM` | NVIDIA *(verify load)* |
| | **ZAYA1-8B** | **CCA hybrid** | **AMD — local gfx1201** |
| Mid / MoE | Qwen3.6-35B-A3B | GDN MoE | NVIDIA |
| | DeepSeek-V2-Lite | MLA MoE | NVIDIA |
| | GPT-OSS-20b | GQA + sinks, MoE | NVIDIA |
| Hub ceiling | Llama-3.3-70B | GQA large | NVIDIA (sharded A40) |

Spans **8 attention geometries** (GDN, GQA, GQA+sinks, MLA, soft-cap, conv-hybrid, ternary,
CCA/Laguna-exotic). Notes:
- **Zaya** is in under an explicit owner's exception (not universality-justified). `cca_hip` is
  gfx1201-only and the pure-torch CCA reference is ~1.7M launches/pass (unusable for a real probe), so
  Zaya's probe runs **locally on gfx1201**; probe extraction is per-model decoupled, so its relative-rep
  matrices merge into the cloud-built atlas.
- **Llama-3.3-70B** is the hub-ceiling, a *weak* serving target (large models benefit least from
  bolt-on memory) — kept only so the hub doesn't bottleneck a large spoke. Per the Platonic
  Representation Hypothesis, large-scale geometries converge, so the cheapest mainstream large model is
  the right ceiling; DeepSeek-V3 would fill the large-MLA corner but costs ~20× the probe for marginal
  gain. Architectural diversity is the *small* members' job.

**v2 atlas-refresh candidates (deferred):** pure-Mamba/selective-SSM (Codestral-Mamba / Falcon-Mamba —
niche deployment, only genuinely-new geometry vs the GDN members), and any new committee member; the
atlas is re-whiten-able cheaply.

### 1.3 Memory store + heads
- **Shared product-key (PKM) sparse store** — addressing-sparsity (top-k reads) scales capacity to a
  large store with bounded per-token read cost and low interference. This is the capacity lever; it is
  ORTHOGONAL to the 2:4 weight sparsity below.
- **Multi-head reads over the shared store** — heads specialize by *retrieval mode*: exact-factual (A),
  positional/span (C), recency (B). Heads are cheap (extra query/output projections); trained jointly.
  Specialization without fragmenting the store or the deployment. This is also the right *experiment*:
  it measures whether per-purpose specialists are even needed before we ever split a head out.
- **MAG tap** — zero-init gated cross-attention injecting retrieved memory into the frozen base's
  residual at the tap depth (the validated v0 mechanism), now reading the multi-head PKM store in hub
  space.

### 1.4 2:4-by-design weights (+ dense reference)
The per-token serve weights (the d_hub×d_hub tap projections, the store value matrix) are trained
**2:4-sparse-by-design** (SR-STE mask from init — NOT prune-after-dense, which is the +25% PPL path
that killed SWMMAC for pre-trained models). Served on RDNA4 via the **SWMMAC** kernel.
- **Serve assumption (per owner):** SWMMAC achieves dense-kernel parity, plus the sparsity gain on top
  → a clean serve multiplier (~1.95× fp8 / 1.17–1.55× + ~3 bit/wt int4), including a decode (M=1) win
  via reduced weight bytes. The current untuned 0.91× is treated as known serve-side kernel-tuning
  work, not a research risk.
- **The only residual 2:4 risk is fidelity** (does masked-from-init training hold recall?). The **dense
  reference** arm exists solely as the fidelity control. Cross-vendor bonus: 2:4 also accelerates on
  NVIDIA Sparse Tensor Cores, so the artifact is fast on AMD *and* NVIDIA.
- Port surface (from the SWMMAC audit): the fp8 dense op (`swmmac_op.hip`) is a ~0.5–1 day
  copy-rename-AOT-build into minisgl's vendored-.so set (bit-exact-validated, offline packer exists);
  the int4-sparse op is ~2–4 days (lift unpack + per-group-scale/zp from `moe_gemm_swmmac_int4.h`,
  write the host int4 packer). Tuning to the assumed parity-+-gain is separate serve-side work.

### 1.5 Modular product — two orthogonal axes
```
product = translator(per MODEL)  ×  head(per PURPOSE)
```
The translator handles base-agnosticism ("for everyone"); the head handles purpose. Mix-and-match:
download your model's translator card × the head(s) for your use-case.

## 2. The task (LOCKED) — train the capability, not an application

All three target use-cases are **one core capability** at different configs:

| Use-case | = core configured as |
|----------|----------------------|
| **A — swappable knowledge store** *(1st priority)* | large **static** store, exact factual recall |
| **C — long-context extension** *(2nd)* | **session** store, position-aware span recall |
| **B — personal/episodic** *(3rd)* | core **+ consolidation/forgetting tier** (v2 store-tier dial) |

**Training task = meta-learned associative recall over randomized episodes** (generalizes the v0
DocBuilder, leak-free):
> each episode: write a random (key→value) store → query → recall the values; content generalized far
> beyond the single-token NAME/CARGO toy — variable-length values, NL passages, factual triples, varied
> store sizes and positions; eval on held-out random stores.

Stresses the four axes A/C/B all need: **fidelity** (exact recall), **capacity** (many associations,
low interference), **content generality** (single-token → passage), **position-awareness** (C). The
learned skill transfers to any domain a user later writes in — domain corpora are *deployment*, not
training. A>C>B weighting: emphasize large-store factual fidelity, include long-span/positional, design
the store to *accept* B's consolidation tier (deferred to v2).

**Capacity target (v1): knowledge-store-grade** — ~10k–100k associations per store, up to
passage-length values, position-aware. Product-key-sized to scale to production (1M+) later. This is
the cost-driving knob.

## 3. Build sequence (LOCKED)
```
0. DE-RISK 2:4-by-design vs dense on the EXISTING v0 harness (cheap; the only residual 2:4 risk is
   fidelity, and it is ~independent of the atlas — answer it before any atlas spend).
1–2. Committee probe extraction → canonical-Z atlas (built ONCE; dtype/sparsity-agnostic).
3. Canonical memory training against the hub — TWO arms (dense reference + 2:4-by-design),
   shared PKM store + multi-head reads, on the unified leak-free recall task at knowledge-store-grade.
4. Translator fits per spoke (tiny, ~13M each, parallel across cards).
```
2:4 lives INSIDE step 3 (masked-from-init), NOT as a prune step after the atlas.

## 4. Hardware & budget (LOCKED — live RunPod pricing 2026-06-29; verify at book time, community stock churns)
- **Training:** 4× RTX 3090 **community** ($0.88/hr, 96 GB, FSDP/DP) — same total cost as 1× but ~4×
  faster wall-clock (compute-bound → shards buy time at constant cost). Checkpoint to ride "Low/Medium"
  stock. Cheap single-card fallback: 1× RTX 4090 ($0.34) or 1× RTX 3090 ($0.22, Medium stock).
- **70B-ceiling probe:** 4× A40 secure ($1.76/hr, 192 GB) — cheapest 70B-capable config (beats single
  MI300X $2.19). Cap-≤32B alternative: 1× A100-80 community ($1.19).
- **Other 9 cloud probes:** the cheap NVIDIA box (forward-only, parallel).
- **Zaya probe:** local gfx1201 via the absolute `gpu-lease.sh -n 1` (free; cca_hip is AMD-only).
- **MI300X:** community is a phantom price (commFlag=false); secure $2.19 single-card only — not the
  workhorse. Use only for AMD-stack consistency if ever wanted.
- **Budget:** probe ~$5–15 · atlas ~$2 · training ~$4–20 · translators ~$5 → **~$20–50 total, cap
  ~$100.** Price-no-object ceiling (8× H200 / 4× B200, ~sub-day) is **rejected** — this job is floored
  by critical-batch + data-throughput, not FLOPs, so >~8 fast cards buys nothing here.

## 5. Training stack (LOCKED — speed over vanilla torch, ROCm-aware on the training box)
Job shape: ~95% of FLOPs are in the **frozen base + big-vocab CE**, not the tiny trainable. So:
- **Structural (free):** run below-tap layers in `inference_mode`/`no_grad` (only above-tap layers need
  autograd); **gradient-checkpointing OFF** (not VRAM-bound → recompute is pure waste here).
- **Tier-1:** `torch.compile` (frozen base = static graph; keep the tap injection inside the compiled
  region), **SDPA/flash** attention, **Liger fused linear cross-entropy** (kills the [B,T,V] logit
  materialization — big for 128k/256k vocab), `TORCH_BLAS_PREFER_HIPBLASLT=1` + fused AdamW.
- **Tier-2:** sequence packing (short recall episodes), async data workers / pre-generated corpus
  (a fast cluster starves on DocBuilder otherwise).
- **2:4:** SR-STE mask in plain torch (the sparsity payoff is serve-side SWMMAC, vendor-agnostic at
  train time — no special training kernels needed).
- **Skip:** Unsloth / DeepSpeed-ZeRO / torchao-accelerated-sparse — they optimize a job we're not
  running (base finetuning / large-trainable sharding).
- **Note:** the gfx1201 warm Triton cache does NOT transfer to the training box (NVIDIA / different
  arch) — budget a one-time cold autotune.

## 6. Training-hours estimate (to be replaced by a smoke-run measurement)
Per-step cost ≈ 4B-class frozen base forward + partial backward + Liger-CE, short packed episodes.
Estimate: ~10k–50k meta-training steps (v0 toy used 3k; knowledge-store-grade + multi-head + capacity
stress is harder), ~0.3–0.8 s/step on the 4× 3090 DP cluster → **≈ 2–12 hr wall-clock ≈ $2–12
training compute.** Wide error bars; the per-episode capacity (10k–100k) and steps-to-converge are the
unknowns. **A 1-hr smoke run on the real recipe pins step-time + steps-to-converge and the critical
batch — do this before the full run.** Total project wall-clock to a finished v1 module: ~1 day.

## 7. Deferred to v2 (explicit)
- **B's consolidation/forgetting tier** (multi-tier store dial: episodic + PKM + consolidation).
- **Fully-separate, separately-trained purpose-heads** — only if the unified multi-head measurably
  lags per-purpose specialists.
- **Mamba/pure-SSM committee member** (+ any atlas-refresh members) — cheap re-whiten.
- **SWMMAC kernel tuning** to the assumed dense-parity+gain (serve-side, known levers).
- **Production-scale capacity** (1M+ associations) — a scaling run on the proven core.
- **Energy/abstain calibration**, meta-learned write rule (CAM_DESIGN §6 dials #4/#5).

## 8. Operating rules (unchanged)
Pure HF/torch research in the titans worktree; no minisglang / no `.so` in *training*. Every GPU job via
the absolute `/home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh -n 1` (for the local Zaya probe).
Cloud runs checkpoint + fail-fast. The eventual *serving* primitive (gated tap + SWMMAC) lands in
minisglang later; the training loop does not.
