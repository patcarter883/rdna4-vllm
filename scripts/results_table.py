#!/usr/bin/env python3
"""
Regenerate a canonical stock-vs-W4A8 throughput table from a profiling sweep's
`results.jsonl`, so the table and the raw data can never silently diverge again
(the project's headline numbers currently disagree across DIARY / AUDIT.md /
SWEEP_FINDINGS.md — this is the single source of truth).

Usage:
    python scripts/results_table.py [results.jsonl ...]
    # default: every profiling/**/results.jsonl found

Schema consumed (one JSON object per line; extra keys ignored):
    model, pathway (stock|w4a8), kv (auto|fp8), regime (prefill|decode|mid|large),
    status (OK|FAILED|...), out_tok_s, total_tok_s

Metric per regime: decode/mid/large -> out_tok_s (decode throughput);
prefill -> total_tok_s (prefill throughput). Ratio = w4a8 / stock (>1 = win).

CAVEAT: the schema does NOT record the dispatch mode (AUTO vs VLLM_ROCM_W4A8_FORCE).
A forced sweep rams the custom kernel onto every shape, including ones AUTO routes
to Triton, so forced numbers are NOT the served-pathway numbers. Record the mode in
the sweep (or the run dir name) and pass --mode to label the table honestly.
"""

import glob
import json
import sys
from collections import defaultdict

REGIME_ORDER = ["prefill", "decode", "mid", "large"]
PATHWAYS = ["stock", "w4a8"]


def metric_for(regime: str) -> str:
    return "total_tok_s" if regime == "prefill" else "out_tok_s"


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    print(f"# WARN: skipped unparseable line in {p}", file=sys.stderr)
    return rows


def fmt(x):
    return f"{x:,.1f}" if isinstance(x, (int, float)) else str(x)


def main(argv):
    mode = None
    args = []
    for a in argv:
        if a.startswith("--mode="):
            mode = a.split("=", 1)[1]
        else:
            args.append(a)

    paths = args or sorted(glob.glob("profiling/**/results.jsonl", recursive=True))
    if not paths:
        print("no results.jsonl found (pass a path, or run from the repo root)", file=sys.stderr)
        return 2
    rows = load(paths)
    if not rows:
        print("no rows parsed", file=sys.stderr)
        return 2

    # (model, kv, regime) -> pathway -> row
    table = defaultdict(dict)
    for r in rows:
        key = (r.get("model", "?"), r.get("kv", "?"), r.get("regime", "?"))
        table[key][r.get("pathway", "?")] = r

    print(f"# Sweep results — stock vs W4A8")
    print(f"# sources: {', '.join(paths)}")
    print(f"# dispatch mode: {mode or 'UNRECORDED (see caveat — AUTO vs FORCE changes everything)'}")
    print()
    print("| model | kv | regime | metric | stock | w4a8 | ratio | note |")
    print("|---|---|---|---|--:|--:|--:|---|")

    def sort_key(k):
        model, kv, regime = k
        ri = REGIME_ORDER.index(regime) if regime in REGIME_ORDER else len(REGIME_ORDER)
        return (model, kv, ri)

    for key in sorted(table, key=sort_key):
        model, kv, regime = key
        m = metric_for(regime)
        cells = table[key]
        stock, w4a8 = cells.get("stock"), cells.get("w4a8")
        sv = stock.get(m) if stock and stock.get("status") == "OK" else None
        wv = w4a8.get(m) if w4a8 and w4a8.get("status") == "OK" else None
        notes = []
        if stock and stock.get("status") != "OK":
            notes.append(f"stock {stock.get('status')}")
        if w4a8 and w4a8.get("status") != "OK":
            notes.append(f"w4a8 {w4a8.get('status')}")
        if stock is None:
            notes.append("no stock")
        if w4a8 is None:
            notes.append("no w4a8")
        ratio = f"{wv / sv:.2f}×" if (sv and wv) else "—"
        mark = ""
        if sv and wv:
            mark = " ✅" if wv / sv > 1.02 else (" ❌" if wv / sv < 0.98 else "")
        print(f"| {model} | {kv} | {regime} | {m} | {fmt(sv) if sv else '—'} | "
              f"{fmt(wv) if wv else '—'} | {ratio}{mark} | {', '.join(notes)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
