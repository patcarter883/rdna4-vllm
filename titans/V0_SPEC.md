# CAM V0 — the MAG falsifier (spec)

Status: spec (2026-06-28). Implements the v0 row of CAM_DESIGN.md §6.
One question, cheapest possible answer:

> **Does an additive, zero-initialized Memory-as-Gate (MAG) tap deliver the already-validated
> DeepMemory binding through the FROZEN base, where the boltA Memory-as-Context (input-embeds
> prefix) hit the injection wall (memory ≈ no_memory, 0.000 acc, ~22.6 bits)?**

Pass ⇒ the whole bolt-on thesis is alive; greenlight v1 (translator + 2nd base).
Fail at every injection depth ⇒ a frozen Qwen genuinely cannot be driven by injected memory →
reconsider the "base stays 100% frozen" premise (light-tune, or different base) before more memory work.

This is a TRAINING/research experiment in pure torch + HF transformers (the titans path — NOT
minisglang). No vendored `.so`, no minisglang engine. Runs under the GPU-lease protocol in this
worktree.

---

## 1. What is REUSED unchanged (≈ all of it)
- **`deep_mem/deep_memory.py` `DeepMemory`** — the validated memory core. API: `init_state(B)`,
  `retrieve(q,state)->[B,Lq,mem_dim]`, `forward(x,state)->state`. Untouched.
- **`warmstart/recall_boltA.py` `BoltAdapter`** — the binding path (frozen embed → in_proj → norm →
  DeepMemory → query-conditioned retrieve → attn-pool `readout_q` → K vectors). Its direct
  tied-unembed training (held-out carry 0.86) is **stage 1, already done**. We import it as the
  frozen memory front-end.
- **`recall_deepmem.py`** task: `NAME_CANDIDATES`, `CARGO_CANDIDATES`, `single_token_ids`,
  `DocBuilder`; `builder.build(rng,B,local=False)->ids,ans,apos`, `builder.qa_start/bos/header`.
- **`warmstart/m2_adapter.py` `load_frozen_base()`** — frozen bf16 Qwen3.5-4B, `use_cache=False`,
  grad-checkpointing on. `H=base.config.get_text_config().hidden_size` (2560);
  `n_layers=base.config.get_text_config().num_hidden_layers` (32).
- **`warmstart/run_m2.sh`** runner (docker + entry). New entry = `warmstart/recall_mag.py`.

## 2. The ONE new component — `GatedMemoryTap` (MAG)
A zero-init gated cross-attention tap on one frozen decoder layer (Flamingo/LONGMEM pattern).
This is the *only* trainable module in stage 2.

```
mem bank B_mem = K pooled retrieval vectors (query-conditioned, leak-free) from the FROZEN BoltAdapter
              shape [B, K, mem_dim]   (NOT the tied-unembed out_proj — use the pre-out_proj pooled vectors)

at tap layer L, given residual hidden h_l ∈ [B,T,H]:
   q = Wq · h_l                      # [B,T,H]
   k = Wk · B_mem ;  v = Wv · B_mem  # [B,K,H]   (Wk,Wv map mem_dim→H)
   a = softmax(q kᵀ / √d_head)       # [B,T,K]   cross-attention over the K memory slots
   y = Wo · (a · v)                  # [B,T,H]
   h_l' = h_l + g ⊙ y                # ADDITIVE inject; g zero-init ⇒ exact no-op at step 0
```

- **Gate `g`:** start with a learned **per-channel** vector `g = tanh(γ)`, `γ∈R^H` init **0** ⇒
  `g=0` ⇒ `h_l'=h_l` exactly (base bit-identical at init — the load-bearing stability property,
  CAM_DESIGN §1, ref 2603.16413). Optional upgrade (v0.1): data-dependent `g=σ(Wg·[h_l;y])`.
- **Injection mechanism:** a **forward hook** on `base.model.layers[L]`. HF decoder layers return a
  tuple; the post-hook rewrites `output[0] = h_l'`. The tap stashes `B_mem` as an attribute before
  each `base(...)` call; the hook reads it. (Same module supports a *list* of tap layers for the
  multi-layer escalation — register one hook per L, each with its own gate.)
- **Heads/dim:** reuse a small head count (e.g. 8) over H; `d_head=H/heads`. Keep it tiny — this is
  a delivery valve, not a model.

New file `warmstart/gated_tap.py` (the module) + `warmstart/recall_mag.py` (harness, forked from
recall_boltA.py). `DeepMemory` and `BoltAdapter` are imported, not modified.

## 3. Training protocol (two-stage, gate-only)
- **Stage 1 — binding (DONE):** load a `BoltAdapter` trained by recall_boltA's direct tied-unembed
  loss (carry ≥0.85). Either (a) checkpoint boltA and load it, or (b) re-run boltA's direct-train
  loop inside recall_mag for `--bind-steps` first. Then **freeze the entire BoltAdapter.**
- **Stage 2 — delivery (the experiment):** freeze base + BoltAdapter; train **only**
  `GatedMemoryTap` by **LM loss through the frozen base** on the recall task:
  - per batch: `ids,ans,apos = builder.build(...)`; memory ingests pre-QA context + query-conditioned
    retrieve → `B_mem` (carry=True); set `tap.bank=B_mem`; run base on the leak-free context
    `[header(format only) ; query tokens up to apos]` (exactly boltA's `eval_generative` context);
    CE loss on the answer logit at the final position vs `ans`.
  - AdamW, lr 1e-3, grad-clip 1.0, `--steps 3000`, batch 16 (boltA defaults).
- **Fallback (only if stage-2 gate-only underdelivers):** unfreeze the memory's gates for a light
  joint fine-tune (CAM_DESIGN §0 caveat 2). Keep base frozen.

## 4. Eval — mirror boltA exactly so it's directly comparable
Reuse boltA's `eval_generative` conditions, swapping the prefix-prepend for the MAG tap:
- **local_control** — full doc in-context, tap OFF (ceiling; boltA got acc 0.98).
- **memory** — leak-free context, tap ON, carry=True.
- **no_memory** — leak-free context, tap ON, carry=False (empty/ablated memory state).
Report per-condition **NLL (bits)** and **accuracy**. Also keep the **direct carry** check (tied
unembed on the pooled bank) to confirm the frozen binding still ≈0.9.

**The decisive number:** boltA = `memory ≈ no_memory`. V0 passes if `memory` accuracy ≫ `no_memory`
and approaches the local_control ceiling.

## 5. Decisions & defaults (my recommendations)
- **Tap layer `L`:** default `n_layers//2` (=16). **Sweep `L ∈ {8,16,24}`** in the first run — cheap
  and directly informative (injection depth is the boltA-flagged unknown; "geometric midpoint" is a
  heuristic, not settled — CAM_DESIGN §5.5).
- **Gate granularity:** per-channel `tanh(γ)`, γ init 0. (Scalar = even safer no-op; data-dependent =
  v0.1 upgrade.)
- **K (memory slots):** 16 (boltA default).
- **Single vs multi-layer:** start single. Escalation ladder in §7.
- **Bank source:** pooled **pre-out_proj** retrieval vectors in mem_dim (let `Wk/Wv` learn mem→H).
  This is cleaner than reusing the embedding-space out_proj and gives stage 2 its own valve.

## 6. Instrumentation (catch the known failure modes)
- **Gate value** `mean|g|` per layer over training — must rise from 0; if it stays ~0 →
  cognitive-bypass / gate-collapse (Infini-attention failure mode, CAM_DESIGN §3).
- **Cross-attn entropy** over the K slots — is the tap actually attending to memory vs ignoring it?
- **Per-condition NLL/acc** every eval; **memory−no_memory ΔNLL** as the live signal.
- Short-context guard: confirm tap-ON on a NON-recall sequence doesn't raise base perplexity (the
  "no regression" half of the gate).

## 7. Verdict logic + escalation ladder (forked from boltA's verdict block)
1. `memory_acc > no_memory_acc + 0.15 and memory_acc > 0.5` → **MAG WORKS**: greenlight v1.
2. `memory_acc > no_memory_acc + 0.10 or ΔNLL > 0.5 bits` → **PARTIAL**: real but weak — go
   **multi-layer** (taps at {8,16,24} together), and/or data-dependent gate, and/or unfreeze memory
   gates (§3 fallback).
3. `memory ≈ no_memory` at single layer → **escalate to multi-layer KV inject** (the boltA-recommended
   path) before concluding anything.
4. `memory ≈ no_memory` at ALL depths AND multi-layer → **the frozen-base premise is the wall**:
   reconsider light base-tuning (prefix/LoRA on the base) or a different/more-injectable base. This is
   the genuinely informative negative result and the whole reason v0 is cheap-first.

## 8. Files, run, budget, environment
- New: `warmstart/gated_tap.py` (`GatedMemoryTap`), `warmstart/recall_mag.py` (harness).
- Touch nothing in `deep_mem/`, `recall_boltA.py`, `m2_adapter.py`.
- Run (1 leased card; absolute arbiter path per repo CLAUDE.md):
  ```
  /home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh -n 1 --name titans-magv0 -- \
    titans/warmstart/run_m2.sh titans-magv0 --entry warmstart/recall_mag.py -- \
      --steps 3000 --bind-steps 3000 --tap-layers 8,16,24
  ```
- Budget: ~boltA scale (single card, minutes–low hours). Pure HF/torch; no `.so`, no minisglang.
- Source isolation: edits + run stay in THIS worktree for the whole job (CAM is a research/training
  experiment; only the eventual *serving* tap lands in minisglang later — CAM_DESIGN §4 serving note).

## 9. What each outcome buys
- **Pass:** MAG > MAC confirmed on a frozen base → the architecture pivot is validated end-to-end on
  the same task that produced the boltA wall. v1 (freeze this memory, fit an affine translator to a
  2nd base) becomes the next falsifier.
- **Partial → multi-layer fixes it:** confirms the injection *point* (not the memory) was the limit,
  exactly as boltA hypothesized; informs the serving primitive (how many taps).
- **Fail everywhere:** the most valuable cheap negative — kills "100% frozen base" and redirects to a
  light-tune bolt-on or a more-injectable base, before any kernel/translator/canonical-space spend.
