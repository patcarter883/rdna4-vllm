# Phase 1 — Titans memory ADAPTER in front of a frozen Qwen3.5-4B (bolt-on)

Decided with the user 2026-06-24. Resets the earlier "integrated transplant" drafts: the served
model is a **frozen base + a trainable Titans memory adapter**, NOT a surgically-modified base.

## Decisions locked
- **Form:** **bolt-on memory adapter, base frozen** (not integrated/transplant). The base is never
  modified; we train only the adapter.
- **Injection:** **input-embeds** — the adapter emits K memory vectors prepended to the base's
  `inputs_embeds` (single injection point; the chosen, simplest variant over multi-layer KV).
  Lands directly on our existing `inputs-embeds` serving primitive (minisgl-rdna4 feat/inputs-embeds).
- **Memory core:** **deep-MLP Titans memory** (surprise + momentum + data-dependent decay) — the
  per-sequence MLP weights are test-time STATE; the adapter's projections/gates are the trained params.
  Reopens the Phase-2 deep-mem chunked-store kernel (pure-torch `AssocScan` path first as oracle).
- **Base:** Qwen3.5-4B (frozen). Note: a bolt-on works in front of *a given* frozen base; the adapter
  is keyed to that base's embedding space → retrain the (small) adapter per base, not zero-shot transfer.
- **Eval gate = BOTH:** (a) no general-quality regression (trivially held — base is untouched; the
  adapter must not *hurt* short-context), AND (b) long-context recall lift: frozen base + adapter beats
  frozen base alone beyond the base's effective reach.

## Architecture (the adapter)
Process a long sequence in segments. Per segment t:
1. **Retrieve:** query the current memory state with segment-t (token-embedding-derived) queries →
   retrieved values → `out_proj` into the base's embedding space → **K memory tokens**.
2. **Inject + generate:** frozen base forward on `inputs_embeds = [memory_tokens ; segment_t embeds]`
   → logits. Memory tokens sit as a prefix the base attends over.
3. **Ingest:** the adapter folds segment-t into the memory state via the Titans surprise update
   (deep-MLP, momentum, decay) — for the NEXT segment.

Trainable params (small): `in_proj` (tokens→memory q/k/v), the deep-MLP memory init/structure + gates,
`out_proj` (memory→base embedding space). Base params frozen. **Gradients backprop THROUGH the frozen
base** to the adapter (base gets grad-flow but no optimizer state) — prefix/prompt-tuning style.

Optional warm-start: `in_proj`/gates can init from Qwen3.5-4B's `linear_attn` (`in_proj_qkv`, `A_log`,
`dt_bias`, `in_proj_z`) — same GDN family. The deep-MLP core trains fresh.

## Training
Objective = **LM loss (next-token) over long sequences** — NO teacher needed (the frozen base + the
sequence provide the signal; the adapter learns to inject memory that lowers loss at long range).
Much cheaper than the integrated path: base frozen (no base optimizer state), adapter small.

## Memory budget (frozen 4B on 16 GB)
Base bf16 ≈ 8 GB (forward + grad-flow-through, NO optimizer state). Adapter optimizer tiny. Grad
checkpointing for long-context activations. Deep-memory per-seq MLP state is the new pressure to watch.
Likely fits ONE card; second card for parallel eval / longer context.

## Milestones (M3 = first real training; deep-mem kernel = M2.5)
- **M1 (cheap, 1 card):** *mechanism de-risk* — frozen Qwen3.5-4B + a learnable K-vector prefix as
  `inputs_embeds`. Verify: (i) injection runs (hybrid linear+full layers handle a prefix), (ii)
  **gradients flow to the prefix, base stays frozen** (adapter is trainable through the frozen base),
  (iii) record the frozen-base long-context recall baseline = the bar for gate (b).
- **M2 (cheap):** build the real Titans memory adapter (segmented retrieve/ingest, deep-MLP memory,
  pure-torch) with the M1 injection contract; 100-step LM-loss smoke, loss drops.
- **M2.5:** deep-mem chunked-store/retrieve Triton kernel (CPU/torch-parity first).
- **M3 (explicit green-light):** full adapter training under `gpu-lease`.
- **M4:** both gates vs frozen Qwen3.5-4B → Phase 3 serve (frozen base via existing qwen3_5 path +
  adapter via the inputs-embeds primitive).

## Open design choices (my defaults; flag to change)
- Memory ingests token EMBEDDINGS (model-agnostic-ish) vs base hidden states (richer, more coupled).
  Default: embeddings.
- K (number of injected memory tokens) and segment length — tune in M2.
- Whether to warm-start `in_proj` from Qwen3.5's `linear_attn` (default: yes, it's free).
</content>
