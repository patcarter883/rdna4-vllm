# RESEARCH — a weight repack to lengthen the gemm2 coalesced read burst (the MoE-decode 126 GB/s wall)

_Design + honest assessment only. No kernel implementation. CPU-buildable items vs GPU-window
items called out explicitly at the end. References: `PIECE1_V7_GEMV_NOTES.md`,
`PIECE2_SMALLM_DENSE_NOTES.md`, `VALIDATION.md`, `DIARY.md` Act VI & Act VIII, `moe_kernel.hip`
(`moe_gemv_v7_kernel`), `w4a8_fp8_wmma/moe_experts.py` (`_run_grouped_moe`)._

---

## 0. The number we are attacking

From `PIECE1_V7_GEMV_NOTES.md` "Why v7 doesn't beat stock at decode — the gemm2 short-K wall"
(per-step M=8 breakdown, Mellum2/Qwen3.6-A3B expert shape E=128, hidden=2304, inter=896, g=32,
top_k=8):

- `apply = gemm1(0.366 ms) + silu(0.007) + gemm2(0.397) ; sum ≈ apply` → no orchestration
  overhead left to cut.
- **gemm1 reads w13 = 100 MB at 273 GB/s** (already beats stock's ~248 GB/s/byte).
- **gemm2 reads w2 = 50 MB at only 126 GB/s — the entire remaining gap.**
- After the `COLS` column-tiling fix (`UPDATE — COLS column-tiling broke the latency half`),
  gemm2 went **126 → 151 GB/s**, M=1 became a win (0.94×), but M≥2 still trails because
  "gemm2 caps 151 GB/s vs gemm1's 273; COLS is exhausted."

So the standing target is: get gemm2's achieved weight bandwidth from ~126–151 GB/s up toward
gemm1's ~273 GB/s (and stock's ~250). gemm1 already proves the HW can sustain 273 GB/s on this
exact op/kernel — the question is purely *why gemm2's read pattern is worse* and whether a repack
fixes it.

---

## 1. The w2 `(E, N=hidden, K=inter//8)` layout and why gemm2's bursts are short

### 1.1 Layout (as built today)

`_run_grouped_moe` (`moe_experts.py:335`) documents the op layout:

- w1 (gemm1): `(E, 2*inter, K=hidden//8)` int32 — contraction K = **hidden** (2304).
- w2 (gemm2): `(E, K=hidden, inter//8)` int32 — contraction K = **inter** (896).

For the grouped op the kernel's `(N, K)` are: gemm1 `N=2*inter`, `K=hidden`; gemm2 `N=hidden`,
`K=inter`. Packing is `PACK_FACTOR=8` nibbles → one int32, **K-major** (the 8 nibbles in a word
are 8 consecutive K values). So per output column `n` the weight row is `ppr = K/8` contiguous
int32:

```
w2[e, n, :]  =  [ int32_0 | int32_1 | ... | int32_(ppr-1) ]      ppr = inter/8
                 ^k=0..7    ^k=8..15            ^last 8 k's
addr( w2[e,n,k8] ) = base_e + (n*ppr + k8) * 4 bytes
```

For gemm2: `ppr = inter/8 = 896/8 = 112 int32 = 448 bytes` per output column.
The whole w2 slab per expert = `hidden * ppr * 4 = 2304*112*4 ≈ 1.03 MB`.

### 1.2 How `moe_gemv_v7_kernel` reads it (the burst, exactly)

`moe_kernel.hip:608–620`, one **warp per output column** `nc` (COLS-tiled), lanes stride along K:

```cpp
const int* wq_e = w_packed + (long)e * N * ppr;             // expert slab base
for (int base = lane * 4; base < ppr_chunk; base += 32 * 4) // lane*4 int32 stride
    wv[c] = *(const v4i_t*)&wq_e[(long)nc * ppr + kw + base]; // 16-byte (b128) load
```

- Each lane issues a `v4i_t` = **16-byte (b128)** load. Within a warp the 32 lanes cover
  `32 * 4 = 128 consecutive int32 = 512 bytes` of one column's K-row in **one pass**.
- The 32 lanes' addresses are `nc*ppr*4 + lane*16` → **fully contiguous, perfectly coalesced**:
  a single 512-byte (4×128-byte cache-line) burst per warp-iteration. This is already optimal
  *intra-warp coalescing*. v7 is not mis-coalesced; the issue is the burst is **too short to
  overlap and the warps are too few / too independent.**

### 1.3 Why the burst is "short" — the three quantitative reasons

1. **The K-row is shorter than one warp pass.** gemm2 `ppr = 112 int32`. One warp pass already
   covers 128 int32 (512 B). So the `for(base...)` loop body runs **exactly once** per column
   (`112 < 128`): `lane*4 < 112` is true only for `lane < 28`. The warp issues **one** b128 load
   (4 lanes idle), then immediately drops to the dot-product. There is **no second loop iteration
   to hide the first load's HBM latency behind** — the warp issues one ~450 B read and stalls on
   it. (Contrast gemm1: `ppr = hidden/8 = 288 int32`, so the loop runs ~3 iterations/column and
   the 2nd/3rd loads pipeline behind the 1st — exactly the DIARY's "gemm1's K=2304 issues 2-3
   [reads] → pipelined" and PIECE1's COLS rationale.) This is the **latency** half. The `COLS=4`
   auto-pick (`moe_kernel.hip:797`) attacks precisely this — issue 4 columns' reads back-to-back
   — and that is the 126→151 GB/s recovered. It is "exhausted" because COLS≥4 blows the
   `acc[COLS][MMAX]` register budget and kills occupancy.

2. **Adjacent warps read non-adjacent HBM.** Warp `w` owns column `nc = w*COLS`; its base address
   is `nc*ppr*4`. Two warps that the GPU happens to co-schedule read addresses **`COLS*ppr*4 =
   COLS*448 B` apart** — i.e. different 4 KB-ish regions of the slab. There is *no spatial
   reuse / row-buffer locality across warps*: each warp opens a fresh DRAM row, reads ~450 B,
   closes it. With short rows, the **DRAM row-activate overhead is amortized over only ~450 B**
   instead of the multi-KB streams gemm1 gets. This is the **bandwidth-efficiency** half: short,
   scattered bursts pay more activate/precharge per useful byte → lower achieved GB/s even at
   full coalescing. This is the part COLS *cannot* fix (COLS reads are the same column-run, still
   one short DRAM stream per column).

3. **Decode has ~1 real row, so the read is not amortized over compute.** At decode `M_real ≈ 1`
   (PIECE1: "1-2 of block_m"). The arithmetic intensity is ~`M_real` MACs per weight byte. The
   kernel is *purely* weight-bandwidth-bound — every inefficiency in the read pattern shows up
   1:1 in wall time. gemm1 has the same `M_real` but a 2.6× longer K-row, so its reads pipeline
   and it reaches 273 GB/s.

**Net:** the gemm2 read is coalesced but **(a) one-shot per column (no intra-warp pipelining)
and (b) split into hidden-many short, spatially-scattered DRAM streams of ~450 B each.** Short K
⇒ short stream ⇒ poor row-buffer amortization ⇒ ~126 GB/s. gemm1's longer K self-pipelines and
streams longer ⇒ 273 GB/s.

---

## 2. Proposed repacks to lengthen the burst

The goal: make each *coalesced DRAM stream the kernel issues* span many KB of contiguous w2,
so the read looks like gemm1's, while keeping the dot-product math bit-identical.

### 2.1 Repack A — **N-interleave (column-interleave) within an expert**

**Idea.** Today consecutive int32 in HBM are consecutive-K of *one* column. Instead, interleave
`G` consecutive columns at int32 granularity so that one long contiguous run feeds `G` columns'
matching K-words. A warp that owns `G` columns then reads them as **one `G*ppr*4`-byte contiguous
burst** instead of `G` scattered `ppr*4`-byte bursts.

**Layout sketch (G = interleave group, e.g. G = COLS = 4):**

```
old:  w2[e, n,   k8]                 stride between k8's = 4 B   (per-column run = 448 B)
      w2[e, n+1, k8]                 +448 B jump to next column

new:  w2rep[e, ntile, k8, g]   with  n = ntile*G + g,  g in [0,G)
      addr = base_e + ((ntile*ppr + k8)*G + g) * 4
      → for fixed ntile, the run over (k8, g) is  ppr*G  contiguous int32
        = 112*4*4 = 1792 B  (4× longer DRAM stream, same coalescing)
```

The warp owning the `ntile` reads `(k8,g)` strided so that lane `L` grabs
`g = L % G`, `k8base = (L/G)*4` — every lane still 16-byte coalesced, but now the loop runs
`ppr*G / (32*4)` iterations = **`112*4/128 ≈ 3.5` iterations** instead of 1 → the 2nd–4th
loads pipeline behind the 1st (the latency half fixed *structurally*, not by spending registers
on COLS). And the DRAM stream per `ntile` is `G*` longer → better row-buffer amortization (the
bandwidth half).

**Cost:** the in-register expand must now de-interleave (`g` selects which of G columns each
nibble belongs to) — a few extra shifts, no extra loads. `acc[G][M_real]` registers — same as
today's `COLS=G` but the latency is hidden by the loop, so we can keep `M_real` small. Bit-exact
(same nibbles, same scales, just reordered in HBM; the de-interleave index is deterministic).

### 2.2 Repack B — **expert-interleave (the decode-specific one)**

**Idea (decode-specific, the genuinely new lever vs dense).** At decode, the routed tokens hit a
*sparse, data-dependent subset* of experts — but each warp/block still reads **one expert's**
short w2 column-row. The MoE structure gives a degree of freedom dense never had: we can choose
the *expert axis* placement.

Two variants:

- **B1 — expert-major-by-column ("transpose the E and N tiling").** Store
  `w2rep[e_group][n][k8][e_in_group]` so that the *same output column n* across `Eg` consecutive
  experts is contiguous. A block that is processing several experts' contributions to the **same
  token's same hidden column** (which is exactly what the final top_k reduce sums) reads them as
  one long run. This only helps if a block fans across experts for one token — which is **not**
  how `moe_align_block_size` tiles (it tiles by expert-block). So B1 needs a *kernel* change
  (gather all of one token's top_k experts into one block), not just a repack. Flagged as
  higher-risk; see §4.

- **B2 — pad/round inter up so K-row ≥ one warp pass.** The short-loop problem (§1.3.1) is
  literally `ppr=112 < 128`. If inter were padded to a multiple of `128*8 = 1024` (here 896 →
  1024) the per-column run becomes `128 int32 = 512 B` = exactly one full warp pass with zero idle
  lanes, and `inter=1024` gives `ppr=128` → still one iteration but no wasted lanes. This *alone*
  does not add pipelining (still ~1 pass) so it's marginal; only useful **combined with A** (pad
  so `ppr*G` is a clean multiple of `128`). Cheap, bit-exact (pad nibbles = 0, scale = 0 → zero
  contribution), but ~14% more w2 bytes read — likely a wash or loss on a BW-bound kernel.
  **Recommendation: do not pursue B2 standalone; only as alignment for A.**

### 2.3 Recommended design = **Repack A (N-interleave, G=4–8), expert-local, K-major preserved**

Concretely, in `process_weights_after_loading` (where `_awq_moe_to_op_layout` already runs, so
the repack is free of any extra runtime cost and happens once at load):

```python
# after building w2_op (E, hidden, inter//8) int32 in the current layout:
E, Nh, ppr = w2_op.shape            # Nh = hidden, ppr = inter//8
G = 4                               # interleave group (match kernel COLS)
assert Nh % G == 0
w2_rep = (w2_op.view(E, Nh // G, G, ppr)      # [E, ntile, g, k8]
                .permute(0, 1, 3, 2)          # [E, ntile, k8, g]
                .contiguous()                 # the long contiguous (k8,g) run
                .view(E, Nh // G, ppr * G))   # [E, ntile, ppr*G]
# scales/zeros reorder identically by (n -> ntile*G+g); store G in the layer.
```

The kernel's only change (a *new gated v7 variant*, not a rewrite): index `wq_e` by
`ntile*ppr*G + (k8base*G + g)` and de-interleave the nibble→column mapping in the expand. Same
WMMA-less GEMV math, same scale/zp fold, bit-identical output.

---

## 3. Expected gain and the mechanism

| mechanism | what it fixes | expected effect |
|---|---|---|
| longer contiguous run (`ppr*G` vs `ppr`) | DRAM row-buffer amortization (§1.3.2) | the part that separates 151→273 GB/s |
| loop now runs `ppr*G/128 ≈ 3.5×` iters | intra-warp latency hiding (§1.3.1), *structurally* (no COLS register cost) | recovers the same ~126→151 COLS already got, but without the occupancy hit, so it stacks |
| keeps perfect 16-B coalescing | no regression vs today | — |

**Optimistic ceiling:** gemm1 proves the kernel+HW sustains **273 GB/s** with a `ppr=288`
self-pipelining K-row. Repack A makes gemm2's *effective* run `ppr*G = 448` int32 (G=4) —
**longer than gemm1's 288** — so if burst length were the *only* variable, gemm2 could reach
≥ gemm1's 273 GB/s. That would take gemm2 from 0.397 ms → ~0.397*(126/273) ≈ **0.183 ms**, i.e.
roughly halve it, turning PIECE1's M=2–16 losses (1.15–1.50× vs stock) into wins/parity and the
M=32 (1.60×) close. That is the *upper-bound* if the burst-length hypothesis is the true cause.

**The mechanism is the same one gemm1 already exploits** — that is the single strongest argument
*for* this repack and the single most important difference from the dense case (see §4).

---

## 4. Honest assessment — do Act VIII's v13/v15 findings already predict the same wall?

This is the crux the task asks for. I will not soft-pedal it.

### 4.1 What Act VIII actually falsified (dense)

`PIECE2_SMALLM_DENSE_NOTES.md` (v13, v15, "FINAL NAIL", v16/v17) and DIARY Act VIII are
unambiguous for **dense small-M**: **LDS-staging, register-direct warp-shuffle (v13), Marlin-style
register-direct repack (v15), zero-LDS/zero-sync v15, fp16-WMMA (v16), true-W4A16 (v17) — ~22
variants — ALL converge to ~108 GB/s at M=16–32.** The verdict (PIECE2 v15): *"the dense M=16–32
ceiling is NOT the weight-read mechanism."* The diagnosis (v14): at M=16 the kernel is at **1.7%
of WMMA peak and 15% of HBM** — it is **latency/occupancy-bound because the GEMM is too small to
fill the GPU** (~64 blocks total), and Triton's autotuned codegen just schedules an
under-utilized GPU better. So for dense, *a weight repack is proven not to help* — the wall is the
under-filled GPU, not the read pattern.

### 4.2 Why MoE gemm2 is genuinely a different shape (the case it might escape)

The dense falsification was specifically **"the GPU is starved — too few blocks."** MoE gemm2 at
decode is **not** the same starvation:

1. **Block count is large, not tiny.** Dense M=16 launched ~64 blocks total (PIECE2 v14:
   "~64 blocks for the whole GPU"). gemm2's grid is `(N/per_block, P/block_m)` =
   `(hidden/(NWARPS*COLS), P/8)`. With E=128 experts each contributing ≥1 padded block,
   `P/block_m ≈ E = 128` rows of blocks × `hidden/(8*4) = 2304/32 = 72` column-blocks =
   **~9000 blocks**. The GPU is **not** starved — gemm2 is genuinely **bandwidth-bound at
   126 GB/s**, not latency-bound at 1.7% of compute peak. This is the decisive difference:
   dense's wall was "can't fill the GPU"; gemm2's wall (per PIECE1's own isolation:
   "NWARPS 8/16/32 all cap ~120 GB/s … intrinsic: short K → short coalesced bursts") is "the
   bytes arrive slowly," which is *exactly* a read-pattern problem a repack can attack.

2. **gemm1 is the existence proof.** On the *same kernel, same op, same expert slab structure*,
   the only thing that differs is K-row length, and gemm1 hits 273 GB/s. Dense had no such
   "identical kernel reaching 2.6× the BW with a longer K" control. For MoE we have a clean A/B
   already in the data: **lengthen the run, get the BW.** PIECE1 explicitly attributes the gap to
   K-length ("gemm2's short K=896 → short coalesced bursts → ~120 GB/s, vs gemm1's K=2304 → 273").

3. **The COLS result is a *positive partial confirmation*, not a dead-end.** Unlike dense (where
   every lever returned the *identical* 108), the gemm2 COLS lever **moved the number: 126 →
   151 GB/s**. That is a measured ~20% gain from issuing more reads per warp — i.e. the read
   pattern *is* on the critical path here, partially. Repack A is the structural version of COLS
   that doesn't pay the register/occupancy tax that capped COLS at 4.

### 4.3 Why it still might hit the same wall (the case *against*, stated fairly)

1. **v13's strided-load lesson is adjacent.** PIECE2 v13's *second* fatal reason was: B's `(N,K/8)`
   layout makes consecutive lanes jump `ppr*4` bytes — "the coalesced burst is 8 bytes with huge
   gaps." Our gemm2 is **not** in that failure mode (lanes stride *along K within one column* —
   contiguous, §1.2), so v13 doesn't directly condemn us. **But** the deeper v13/Act-VIII lesson
   — *"Triton wins by a vectorized codegen the compiler optimizes, not by the weight-read
   mechanism"* — is a warning that **even a perfect read pattern may not beat stock's fused
   Triton MoE kernel**, which does gemm1→silu→gemm2→reduce in *one* launch with the gemm2 weights
   streamed inside an already-resident pipeline. We are comparing our *standalone* gemm2 against a
   *fused* gemm2; some of stock's edge is fusion (no intermediate HBM for buf2/out2), which a
   repack does **not** address. PIECE1 measured `sum ≈ apply` (no orchestration overhead *in our
   path*), but stock's fusion means stock never *materializes* buf2 — that HBM write/read of
   `P × inter` fp16 is real traffic our path pays and a repack can't remove.

2. **Achievable-BW ceiling may be below gemm1's 273 for short total bytes.** gemm1 reads 100 MB,
   gemm2 reads 50 MB. Even with a perfect long run, fewer total bytes = less opportunity to reach
   steady-state streaming BW; the fixed launch/ramp overhead is a larger fraction. So the
   realistic target is probably **~200–230 GB/s, not the full 273** — still a large win
   (~0.397 → ~0.23 ms) but not the optimistic §3 ceiling.

3. **Repack A's de-interleave adds VALU.** PIECE1's whole breakthrough was *removing dead work*
   (the fp8 round-trip, oversized MMAX, K-chunk syncs). Repack A *adds* per-nibble de-interleave
   shifts. On a BW-bound kernel this is likely free (VALU hides under HBM latency), but it is the
   opposite direction from the lessons that actually paid out, so it must be measured, not
   assumed.

### 4.4 Verdict

**The dense v13/v15 falsification does NOT transfer to MoE gemm2, because dense's wall was a
starved GPU (1.7% compute peak, ~64 blocks) while gemm2 is a genuinely bandwidth-bound kernel
(~9000 blocks, 126 GB/s, BW-moved-by-COLS) with an in-data existence proof (gemm1 at 273 GB/s on
the same kernel).** The burst-length hypothesis was *falsified for dense* and is *live for MoE
gemm2* — they are different shapes despite the DIARY's "same disease" framing (Act VIII's framing
conflated *latency-bound dense* with *bandwidth-bound gemm2*; the isolation data in PIECE1
distinguishes them).

**However**, the *honest* expected outcome is a **partial win**: a repack can plausibly take
gemm2 from ~150 to ~200–230 GB/s (closing most of PIECE1's M=2–32 residual to parity/win), but it
**will not by itself beat stock's *fused* MoE kernel everywhere**, because part of stock's decode
edge is single-launch fusion (no buf2 HBM round-trip) that a weight repack does not touch. The
repack is necessary-but-maybe-not-sufficient: it removes the read-pattern handicap so that the
*next* lever (a fused gemm1→gemm2 decode kernel keeping buf2 in LDS — PIECE1's stated "next lever
is a FUSED decode apply") can be measured cleanly. **Recommendation: pursue Repack A; expect it to
finish the M=2–32 parity story PIECE1 left open, not to be a standalone stock-beater.**

---

## 5. Risks

- **R1 — the read is already optimally coalesced (§1.2), so the gain may be only the row-buffer
  amortization half, not the latency half** (COLS already captured most of the latency half:
  126→151). If COLS already got the bulk, Repack A's *marginal* gain over COLS=4 could be small
  (151 → ~180 rather than → 273). This is the single biggest risk and is **only resolvable on
  GPU** (the latency-vs-row-buffer split is not analytically separable here).
- **R2 — bit-exactness of the de-interleave.** Reordering nibbles in HBM + reindexing scales/zeros
  must be an exact permutation. Off-by-one in the `(ntile, k8, g)` index = silent wrong MoE
  output. Mitigated: it is a pure gather, verifiable in numpy off-HW (see §6).
- **R3 — scales/zeros must be reordered identically.** `ws_e[nc*num_groups + g]` and the packed
  `wz_e[(nc/8)*num_groups + g]` index by `nc`; the column reorder changes `nc`. The zeros are
  **N-packed 8-to-a-word** (`moe_kernel.hip:623`), so an N-permutation that isn't a multiple of 8
  re-shuffles *within* a zero word — must keep G | 8 or repack the zeros too. **Constraint: choose
  G ∈ {1,2,4,8} so the column permutation is within/aligned-to the 8-wide zero packing.** (G=4
  and G=8 both satisfy this cleanly.)
- **R4 — fusion is the real residual.** Even a perfect gemm2 BW leaves the buf2 (`P×inter` fp16)
  HBM round-trip that stock's fused kernel avoids; the repack cannot close that part. Could lead
  to "we hit 230 GB/s and *still* lose M=8 by 10%."
- **R5 — extra w2 bytes if padding (B2) is used.** Avoid B2 standalone (§2.2).
- **R6 — interaction with the gather_reduce / SCATTER epilogues.** The repack only touches the
  weight read; the SCATTER epilogue (`moe_kernel.hip:699`) and the `gather_reduce` kernel are
  untouched. Low risk but must confirm the non-scatter (`out[(long)s_pad[r]*N + nc]`) write still
  uses the *original* `nc` (output is in hidden-order, not repacked-order) — it does, since `nc`
  is reconstructed from `ntile*G+g`.

---

## 6. Experiment plan — CPU-buildable now vs needs a GPU window

### 6.1 CPU-buildable NOW (no GPU) — do these first

1. **Numpy repack + de-interleave equivalence (no HW).** Mirror `test_moe_conversion_ct_numpy.py`
   / `test_moe_conversion_numpy.py`: build a random w2 in the current `(E, hidden, inter//8)`
   layout, apply the §2.3 permutation, write a numpy reference that reads it back with the
   kernel's `(ntile, k8, g)` indexing + the de-interleave nibble map, and assert
   `max|diff| == 0.0` vs reading the original layout. This proves bit-exactness of the layout math
   *before any kernel exists*. **CPU only, runs on this host today.**
2. **Reorder scales/zeros consistency check (numpy).** Verify the §5-R3 constraint: for G∈{4,8},
   the column permutation composed with the 8-wide N-packing of `w_zeros` is the identity on the
   *unpacked* zeros. Assert in numpy.
3. **Kernel cross-compile (gfx1201), no run.** Add the new gated v7-interleave variant behind a
   `version`/env gate (NOT dispatched by default — exactly how v13/v15/v16/v17 are kept), and
   confirm it **compiles** with
   `source /home/pat/code/vllm-rocm714-gfx1250/activate-build-env.sh && cd
   …/w4a8_fp8_wmma && python setup.py build_ext --inplace`. Do **not** import or run the .so.
4. **`py_compile`** the touched `moe_experts.py` / adapter (vLLM not importable here; syntax only).

### 6.2 Needs a GPU window (ask first — shared 2× gfx1201 box)

5. **Bit-exact on-HW** vs v0 golden + v7: `test_moe_correctness.py` (the existing 20-case
   harness) with the interleave variant gated on. Expect `max ≤ 0.0078` (fp8 granularity), 0 bad
   on g=32 (the target group size) — same bar v7 met.
6. **gemm2 BW micro-bench (the decisive measurement).** Use `gemm2_probe.py` /
   `breakdown_moe_apply.py` (Mellum2 shape E=128 h=2304 inter=896 g=32 tk=8) to read achieved
   gemm2 GB/s for `{v7-current, v7-interleave G=4, G=8}` at M∈{1,2,4,8,16,32}. **Primary success
   metric: gemm2 GB/s rises from ~151 toward ≥200.** This single number settles R1 (was it
   row-buffer amortization or already-captured latency).
7. **Full-apply A/B vs stock** (`moe_microbench.py` / `e2e_moe_decode_ab.py`): the PIECE1 table
   (M=1..32, now/stock). Success = the M=2–32 residual (1.15–1.60×) closes to ≤1.0×.
8. **Only if 6/7 win:** the orthogonal **fused gemm1→gemm2 decode kernel** (PIECE1's "next
   lever") to kill the buf2 round-trip (R4) — a separate, larger effort; this doc scopes only the
   repack.

### 6.3 Decision gate

- If micro-bench 6 shows gemm2 **stays ≤160 GB/s** despite the longer run → the burst-length
  hypothesis is **falsified for MoE too** (the wall is then something COLS already saturated, or
  it's the fixed per-launch/ramp overhead on 50 MB), and we stop, documenting it as the MoE
  counterpart to PIECE2's v15 negative result.
- If it reaches **≥200 GB/s** → repack A ships (gated, then default for decode), and the residual
  M-loss (if any) is re-attributed to fusion (R4), justifying lever 8.

---

## 7. One-paragraph bottom line

Unlike the **dense** small-M wall — where Act VIII's ~22 variants (v13 register-direct shuffle,
v15 Marlin repack, v16 f16-WMMA, v17 true-W4A16) all converged to ~108 GB/s and *falsified* the
weight-repack hypothesis because the true cause was a **starved GPU** (1.7% compute peak, ~64
blocks) — the **MoE gemm2** decode wall is a genuinely **bandwidth-bound** kernel (~9000 blocks,
126 GB/s, a number that *moved* 20% under COLS) with an in-data **existence proof** that the same
kernel reaches 273 GB/s on the longer-K gemm1. A K-major-preserving **N-interleave repack
(Repack A, G=4–8)** lengthens each coalesced DRAM stream from ~450 B to ~1.8 KB, structurally
restoring intra-warp pipelining (without COLS's register tax) and improving DRAM row-buffer
amortization. It is **bit-exact** (a pure HBM permutation, numpy-verifiable today, with a
G∈{4,8} constraint so the 8-wide zero packing stays aligned) and **CPU-buildable to the
compile/equivalence stage now**, needing a GPU window only for the decisive gemm2-GB/s
micro-bench. Honest expectation: a **partial win** — gemm2 ~150→~200–230 GB/s, closing PIECE1's
open M=2–32 residual to parity/win, but **not** a standalone stock-beater, because part of stock's
decode edge is single-launch fusion (the buf2 HBM round-trip) that no weight repack can remove —
that residual is the next, separate lever.
