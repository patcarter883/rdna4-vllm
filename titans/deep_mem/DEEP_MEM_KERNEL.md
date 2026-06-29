# M2.5 — Deep-memory chunked store/retrieve kernel (RDNA4)

Goal: a memory-bounded, fast chunked store+retrieve for the Titans deep-MLP memory, replacing the
lucidrains `torch.func`-grad path that retains the create-graph and OOMs past ~6 segments (M2 finding).
Built the project's way: exact math → graph-free torch reference (the OOM fix + oracle) → Triton hot path,
with parity gates at each stage.

## Design decision: own a clean deep-mem, don't bit-match lucidrains
lucidrains `store_memories` has many options (momentum_order, attn-pool chunks, lookahead values, per-param
lr modulation, weight residual…). We build our OWN minimal deep memory straight from the Titans paper, which
is correct, far simpler, and fully ours to kernelize. Memory = depth-2 `MemoryMLP` per (batch·head):
    pred = gelu(k @ W1) @ W2          W1:[B,d,h]  W2:[B,h,d]

## The math (B = batch*heads, one window of T chunks)
**Surprise (per chunk c, over its tokens):** gradient of weighted MSE wrt (W1,W2), evaluated at the
window-start weights — closed form (DONE, parity-exact vs autograd, `deep_mem_analytic.py`):
    g_pred = (2/D)·lw·(pred − v);  gW2 = zᵀ·g_pred;  gW1 = kᵀ·((g_pred·W2ᵀ)⊙gelu'(a))
**Update recurrence (per chunk, data-dependent θ/η/α from chunk-rep projections):**
    S_c = η_c·S_{c−1} − θ_c·g_c        (momentary surprise + momentum)
    M_c = (1−α_c)·M_{c−1} + S_c        (weight with decay/forgetting)
Two chained gated-linear recurrences over chunks ⇒ associative-scannable ⇒ parallel/Triton-friendly.
**Retrieve:** `out = gelu(q @ W1_c) @ W2_c` with the weights at the relevant chunk boundary.

## Build stages (parity-gated)
1. ✅ **Analytic surprise** — closed-form MLP backward == autograd (1e-16). `deep_mem_analytic.py`.
2. **Store recurrence** — sequential reference + associative-scan implementation, parity-exact
   (proves parallelizability). `store_recurrence.py`.
3. **Graph-free `DeepMemory` module** — analytic surprise + recurrence + retrieve, drop-in replacing
   lucidrains `NeuralMemory` in the M2 adapter. **This fixes the OOM** (no retained graph) → re-run the
   M2 smoke + scale to many segments / long sequences.
4. **Triton kernel** — fuse the per-chunk surprise matmuls + the scan; CPU/torch-parity first, then
   gfx1201. RDNA4 notes: no direct VMEM→LDS (don't port CDNA prefetch); reuse the warm triton cache.
5. Wire into training (M3) at long context; re-confirm M2 loss-drop still holds graph-free.

## Status
- ✅ Stage 1 — analytic surprise == autograd (1e-16). `deep_mem_analytic.py`.
- ✅ Stage 2 — store recurrence: parallel scan == sequential (2e-16). `store_recurrence.py`.
- ✅ Stage 3 — graph-free `DeepMemory` module assembled (`deep_memory.py`): per-token k/v/q + per-token
  loss-weight `lw` + per-chunk gates θ/η/α from chunk-mean reps, analytic surprise at the window-start
  weights, parallel momentum+decay scan, retrieve = gelu(q@W1)@W2. Carries (W1,W2,S1,S2) across segments
  via sequential `forward` calls (scan length = per-segment chunks only → cumprod stays stable).
  Parity-gated on CPU (no GPU):
    - `stage3_parity.py` — `forward` vs an INDEPENDENT autograd+sequential reference: max|dW|≈4e-16; retrieve exact.
    - `stage3_adapter_cpu.py` — adapter deepmem read/ingest cycle over 20 segments: grads reach all 16
      params, state carries, loss 1.01→0.41 (no base model needed).
  Wired into `warmstart/m2_adapter.py` behind `--memory {deepmem,lucidrains}` (default deepmem);
  `--repeat`/`--detach-every` added for segment-scaling. Runner `warmstart/run_m2.sh <cname> [--entry S]`.
  **GPU-validated (2026-06-24, 1 leased card):**
    - Learning gate: deepmem on the frozen 4B, loss **4.20→2.97** over 60 steps, graph-free, **3.94M** params
      (vs lucidrains 17.3M).
    - OOM FIX — isolated memory-backend sweep (`stage3_mem_scaling.py`, no 4B, full cross-segment graph
      retained, real shapes mem_dim=512/h4/chunk16):
        n_seg        4        8       16       32       64
        lucidrains  5.82 GB  10.28 GB  OOM      OOM      OOM
        deepmem     0.61 GB   0.87 GB  1.39 GB  2.43 GB  4.50 GB
      deepmem scales to 64 segments at ~linear, ~10× lower peak; lucidrains' `assoc_scan` create-graph
      OOMs by 16. End-to-end on the real 4B at ~7 segments: **lucidrains OOMs inside `store_memories`/
      `assoc_scan`**, deepmem fits (12.55 GB — dominated by base-activation graph, not the memory).
  **Two distinct OOM axes (settled):** (1) the *memory backend's* retained graph = what stage 3 fixes
  (proven above). (2) *base-activation accumulation* — the smoke loop keeps every segment's frozen-4B
  forward graph until one backward, so it OOMs at ~31 segments regardless of memory backend (the
  deepmem-at-31 OOM was in `lm_head`, not the memory). Axis (2) is an M3 training-loop concern
  (grad-accumulation / per-segment backward / truncated BPTT), orthogonal to the memory kernel.
- Next: Stage 4 — Triton (fuse the per-chunk surprise matmuls + cumprod/cumsum scan); CPU/torch-parity
  first then gfx1201. RDNA4: no VMEM→LDS, reuse the warm triton cache.
</content>
