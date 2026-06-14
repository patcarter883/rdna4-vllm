# Session handoff — 2026-06-14 (RFP458 + cross-cutting audit + no-GPU follow-ups)

Pick-up doc for the next session. Read this, then `docs/IMPROVEMENT_PATHWAYS.md` (the full
plan) and the RFP458 plan at `~/.claude/plans/some-discourse-with-my-crispy-lampson.md`.

## What this session did (in order)
1. **Audited RFP458** — the collaborator's (tcclaviger) 4-bit quant, which lives in the **base
   image** `tcclaviger/vllm22:dev`, not in this repo. Key finding: its runtime kernel dequants
   4-bit → fp16 and runs fp16 `tl.dot` — it **never uses gfx1201 fp8 matrix hardware**. The
   combining plan = keep RFP458's format, route its dequant tail through fp8 WMMA. Full plan +
   line-cited audit in the `~/.claude/plans/...` file. A reply to the collaborator (in Claude's
   voice, attributed) is in §5 of that plan — **not yet sent** (Pat to send / collaborator gated).
2. **Cross-cutting repo audit** (6 parallel agents) → `docs/IMPROVEMENT_PATHWAYS.md` on main.
3. **Did all six no-GPU action items** from that audit (branches below).
4. **Ran the fp8 fidelity simulation** in-container (CPU) — first directional result (below).

## Branch / commit state (ALL off main 233acbb; LOCAL, NOT pushed)
| branch / worktree | commits | what |
|---|---|---|
| `feat/rfp458-fp8` (`…-feat-rfp458-fp8`) | ec18056 | Phase-0 AMDGCN WMMA verifier (`quant/rfp458_fp8/`) — proves fp16-vs-fp8 from machine code. **Not yet run** (needs a GPU window). |
| `chore/audit-followups` (`…-chore-audit-followups`) | 3dac84c, e1fc334, c290b12 | #2 base-image digest pin, #3 CI fix + non-GPU lint job, #4 `scripts/results_table.py`, #5 `patches/upstream/` moe_wna16 PR prep, #6 hygiene (deleted stale `*_hip.hip` dupes, retired wheel scripts, README attn fix). |
| `feat/fidelity-harness` (`…-feat-fidelity-harness`) | 1029892, 3773751 | #1 fp8 fidelity harness (`quant/fidelity/`) + FINDINGS.md from the first run. |

`main` (233acbb) carries `docs/IMPROVEMENT_PATHWAYS.md` + this handoff. main is 2 ahead of origin (not pushed).

## ⚠️ Merge hazard — coordinate before merging `chore/audit-followups`
The main checkout has **concurrent uncommitted edits to `Dockerfile.combined` and
`docker-compose.yml`** (another active session). My `chore/audit-followups` branch **also edits
both** (the digest pin). **Merging will likely conflict** — reconcile with whoever owns those
working-tree changes first. (Also `M CONTRIBUTING.md`, untracked `CLAUDE.md`, `profiling/`, and
the het runner scripts are concurrent — left untouched on purpose.)

## Key findings to carry forward
- **The headline e2e numbers are suspect.** "dense prefill +53% / MoE decode +11%" were measured
  under `VLLM_ROCM_W4A8_FORCE=on` — which Act IX calls a throughput-halving trap and which the
  logs show was an unrecognized env var. They reflect "kernel forced onto every shape," not the
  served AUTO path. **Needs an AUTO-mode 16-cell re-run before anyone claims those wins.** (IMPROVEMENT_PATHWAYS §0.)
- **MoE wins don't engage in production.** `w4a8_fp8_wmma/moe_crossover_cache.json` has only 2
  Mellum2 entries → the real 35B shape falls back to stock. Populating it (GPU-window, effort-S)
  is the highest-leverage MoE action.
- **Fidelity sim (directional, synthetic weights):** per-N ≈ group-tiled (~0.8 dB) → the Phase-1
  "BLOCK_K=64 vs 16" gate is **far less binding than the plan feared**, likely keep BLOCK_K=64;
  and **activation→e4m3 quant dominates the error**, not weight per-group spread → focus accuracy
  work on the activation path. NOT a verdict — synthetic weights don't model mantissa-starved
  channels; no perplexity yet. (`quant/fidelity/FINDINGS.md`.)

## Next steps (prioritized)
**No GPU window needed:**
1. **Add `--weights <safetensors>` to the fidelity harness** → run against REAL RFP458 checkpoint
   weights (healthy + mantissa-starved channels). The single most important upgrade to turn the
   directional fidelity result into a verdict. (Needs an RFP458 checkpoint on the box — ask Pat.)
2. **Send the collaborator reply** (plan §5) — gets the paper/refs, the "458" meaning, and his
   read on an e4m3-aware scale search before Phase-1 kernel work.
3. Merge the three branches once the Dockerfile/compose conflict is reconciled.

**Needs the batched GPU window** (one combined-image 35B TP=2 server — see IMPROVEMENT_PATHWAYS §Theme 3):
4. AUTO-mode 16-cell re-run (settles the credibility issue) + 27B dense A/B.
5. Run `quant/rfp458_fp8/run_verify.sh` (RFP458 fp16 baseline) + populate the MoE crossover cache.
6. Het-TP COMM perf A/B + cudagraph-mode.

**Gated on collaborator + GPU:**
7. RFP458→FP8 kernel Phase 1 (dense-linear fp8 dot) — author as a `patches/rfp458_fp8_vllm.patch`
   (het-TP pattern), NOT in-container, NOT /tmp.

## Conventions reaffirmed this session (see memory)
- Canonical source = this repo; **all work in feature worktrees**; **never /tmp** (volatile).
- Project analyses + future planning live on `main` (hence this doc + IMPROVEMENT_PATHWAYS there).
- RFP458 is the collaborator's base-image code → modify via a **build-gated `patches/` patch**, never edit the container in place.
- Ask before any GPU workload; mount the per-image Triton cache. The repo + `memory/` + handoff docs are the only cross-session channel.
