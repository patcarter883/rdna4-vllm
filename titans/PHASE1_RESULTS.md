# Phase 1a — tiny from-scratch convergence run (enwik8): RESULTS

**Status: PASS (2026-06-23) on gfx1201.** Titans MAC trains from scratch and converges on real
data on RDNA4; the checkpoint round-trips and generates.

## Run
- Model: ~31.5M MAC (dim 512, depth 8), neural-memory layers (2,4,6), 2-layer MemoryMLP (dim_head 64).
- Data: byte-level enwik8 (`num_tokens=256`), seq 512, batch 4 × grad-accum 8 (eff 32).
- Budget: 180 min on one card (gpu-lease), AdoptAtan2 lr 2e-4.
- RDNA4 flags: `use_accelerated_scan=False`, `use_flex_attn=False`.

## Convergence (val bits-per-char)
| step | min  | val BPC |
|------|------|---------|
| 0    | 0.1  | 8.50    |
| 200  | 13.8 | 3.70    |
| 600  | 41.9 | 2.44    |
| 1200 | 83.9 | 2.02    |
| 2400 | 169  | **1.83** (best) |

Steady downward trend (mid-run val bounces 2.0–2.45 are normal char-LM small-batch noise).
~14 optimizer steps/min. `model_best.pt` = step 2400.

## Checkpoint + generation (Phase 1a exit check, `generate.py`)
- `config.json` + `state_dict` round-trip is **exact**: `load_state_dict` → missing=0, unexpected=0.
  This de-risks the checkpoint-reconstruction path the serving pipeline (Phases 3/4) needs.
- Generations are locally-fluent English with learned MediaWiki markup (`[[links]]`, `[[Category:…]]`),
  globally incoherent — exactly as expected for a 31M char model at 1.83 BPC.

## Gotchas found & fixed
- **batch-size OOM:** the pure-torch `AssocScan` memory-store path is memory-hungry (batch 8 → 15.3 GB
  on a 16 GB card, first-step OOM). batch 4 fits with headroom; memory ~linear in batch. (Phase 2 kernel
  work should cut this.)
- **`model.sample` arg:** 2nd arg is TOTAL length (prime+new); pass `prime_len + gen_len`.
- **docker stdout buffering:** background-task stdout pipe was block-buffered (0 bytes captured);
  the on-disk `train_log.jsonl` is the reliable monitor. Use `python -u` / `PYTHONUNBUFFERED=1`.

## Artifacts
- `titans/train_enwik8.py`, `titans/generate.py`, `titans/titans_common.py`
- `titans/checkpoints-enwik8/{config.json, model_best.pt, model_last.pt, train_log.jsonl}`

## Next
Phase 3 — serve this checkpoint in minisgl-rdna4. Recommended sub-sequencing: stand up a
**pure-torch** memory forward inside minisgl first (validate the integration + numerics vs this
reference), THEN swap in the Phase-2 kernels. Web-corpus big run (Phase 1b) stays deferred until the
serve pipeline works.
