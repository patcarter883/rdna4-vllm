# Upstream-bound patches

Fixes here are **generic vLLM bugs** (not gfx1201-specific) that we carry locally and
should land upstream so we stop carrying them. Each is a minimal diff + a PR description.

---

## `moe_wna16-tp_size-fallback.patch` — `RoutedExperts` has no `tp_size`

**Bug.** `MoeWNA16Method`'s expert weight loader reads `layer.tp_size` directly when
sharding `w13_qzeros` / `w2_qzeros`. The current MoE container, `RoutedExperts`, does not
expose a `tp_size` attribute — TP now lives on `layer.moe_config.tp_size` (the legacy
`FusedMoE.tp_size` is gone). So loading an AWQ/GPTQ MoE checkpoint that falls back to the
WNA16 path crashes with:

```
AttributeError: 'RoutedExperts' object has no attribute 'tp_size'
```

This is **architecture-agnostic** — it reproduces on any backend that routes an AWQ/GPTQ
MoE through `moe_wna16`; we hit it first on gfx1201 (Qwen3.6-35B-A3B-AWQ) but it is not
RDNA-specific.

**Fix.** Resolve `tp_size` with a fallback to `moe_config.tp_size` once, right after
`tp_rank` is computed, and use that local in the two `qzeros` views. Minimal, behaviour-
preserving on layers that still have `.tp_size`. See the `.patch` file.

**Reproduce.** Serve any AWQ/GPTQ MoE that lands on the WNA16 fallback (e.g.
`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`) with `tensor-parallel-size > 1` on a build where the
MoE container is `RoutedExperts`; model load raises the AttributeError above.

**Test.** Greedy-equivalence before/after the fix: identical output token ids (the change
only fixes an attribute lookup, it does not alter sharding math). A unit test can construct
a `RoutedExperts` layer with `moe_config.tp_size` set and `tp_size` absent and assert the
loader shards `w13_qzeros`/`w2_qzeros` to the right slice.

### How we apply it locally
The combined image applies the *same* fix surgically via `sed` in `Dockerfile.combined`
(step 2), because the base image's copy had diverged with a SiLU-only assertion that a
whole-file patch would clobber. `patches/moe_wna16.py` is the whole-file reference copy.

### Before submitting upstream
This `.patch` was generated against the base image's vendored `moe_wna16.py` (representative,
and it applies cleanly there). **Rebase it onto current `vllm-project/vllm` `main`** before
opening the PR — the surrounding lines may have shifted. Submitting to an external project is
outward-facing: get the go-ahead first, then:

```bash
# in a vllm-project/vllm checkout on a fresh branch
git apply /path/to/moe_wna16-tp_size-fallback.patch   # or re-create the 3-line change by hand
# add a regression test, run the moe_wna16 tests, then open the PR with the description above
```
