# Merge plan — landing the in-flight workstreams

Four workstreams (README "In flight", DIARY Acts XIX–XXIII) live on their own
worktrees/branches and have not landed on `main`. The longer they sit, the harder each
rebase onto a moving `main` gets — and PR #4 already carries an inherited commit from
another branch because of exactly this. This file proposes a **landing order** so the
branches converge instead of drifting.

It is a recommendation, not a decree — every land still needs its own hardware validation
(the branch protocol and container-testing protocol in [`CLAUDE.md`](CLAUDE.md) apply).

## Branch state (2026-07)

| Workstream | Branch(es) | Pushed to origin? | PR | Default-on? |
|---|---|---|---|---|
| gdn_hip — native HIP GDN kernels | `feat/gdn-hip` | no (lab worktree) | — | off (`WITH_GDN_HIP` / `VLLM_GDN_HIP=1`) |
| Titans — test-time neural memory | `feat/titans` | **yes** | — | n/a (training workstream) |
| DFlash speculative decoding | `feat/dflash-spec`, `feat/zaya-dflash` | no (lab worktree) | — | off (overlay image) |
| TiDAR — ZAYA1-8B → diffusion | `feat/tidar-convert`, `feat/tidar-serve` | `feat/tidar-serve` **yes** | **#4** | off (new profile) |

> Note: the native HIP serve path (GDN WMMA + attention) already merged to `main`
> (commits `3871e73`, `414d783`, `8cc710a`, `4824764`) and is default-on for `:combined`
> per `VLLM_GDN_HIP=1` / `VLLM_ATTN_DECODE_HIP=1`. `feat/gdn-hip` below is the
> *framework-agnostic `torch.ops`* extension that kills the cold GDN Triton compile — a
> distinct, still-unlanded piece.

## Proposed order

1. **`feat/gdn-hip` first — lowest blast radius, biggest quality-of-life win.**
   It is default-**off** (`WITH_GDN_HIP` build arg + `VLLM_GDN_HIP=1` runtime gate), so
   merging it cannot regress the shipped path, and it removes the 15–30 min cold GDN Triton
   compile cliff that every other workstream (and every cold boot) pays. Numeric parity is
   ≤ ~1e-7 and it serves 4B TP1/TP2 + 35B TP2 coherently. **Blocker to land:** it is not on
   origin yet — push the worktree branch, open a PR, run the CPU lint + a GPU smoke. Open
   perf item (WMMA chunked prefill) can land later; the recurrent path is correct today.

2. **`feat/dflash-spec` (+ `feat/zaya-dflash`) next — it is the base PR #4 is stacked on.**
   PR #4's own description notes it inherits one `feat/zaya-dflash` commit (`627128e`). Landing
   the DFlash bring-up first lets PR #4 retarget onto it and drop that inherited commit, so its
   diff becomes purely the TiDAR-serve work. DFlash is infra-complete (non-causal `TRITON_ATTN`
   patch, boots TP=2, lossless); the open question is acceptance rate by target quant format,
   which is a *tuning* result, not a merge blocker for the (gated, overlay-image) infrastructure.
   **Blocker to land:** push the worktree branches, open a PR.

3. **PR #4 `feat/tidar-serve` — retarget, then land.**
   After step 2 is on `main`, rebase `feat/tidar-serve` onto it (or onto `feat/zaya-dflash`)
   so the inherited commit disappears and the PR shows only the 4 TiDAR-serve commits. It adds a
   new `zaya-tidar` compose profile (off by default), serves the published
   `pat883/zaya1-8b-tidar-experts` checkpoint TP=2 bf16, and the fused-forward speedup is
   de-risked but not yet wired (the `_decode_verify_spec` segmented-conv path is the next phase).
   Land the servable baseline; track the fused-forward speedup as a follow-up.

4. **`feat/titans` — land on its own cadence (independent).**
   It is a *training* workstream (enwik8 from scratch, the GDN-2 arm picked to provisionally
   ship), not part of the serve stack, so it does not block or depend on 1–3. It is already on
   origin. Land when the training-path choice + checkpoint-serving story
   (`minisgl-rdna4`) is settled; keep it off the serve critical path until then.

## Guardrails for each land

- **One concern per PR / worktree** (CLAUDE.md branch protocol) — do not bundle an inherited
  refactor with the feature (this is what PR #4 is untangling).
- **CPU lint gates automatically** now (`.github/workflows/ci.yml`: `docker compose
  config`, patch/AST/shell parse, and the W4A8 numpy conversion tests). A GPU smoke on real
  gfx1201 is still required for anything touching the image or a kernel.
- **Default-off first.** Prefer landing gated/overlay features (`WITH_*` build args, new
  profiles) so `main` and the shipped `:combined` path stay green while the feature bakes.
- **Re-run the ABI import check** after any base bump (CONTRIBUTING "The one hard rule: ABI").
