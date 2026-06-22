#!/usr/bin/env python3
"""Compare two bitident_client.py dumps and report the first per-prompt divergence.

    python bitident_diff.py baseline.json spec.json

Exit 0 => every prompt's chosen-token sequence is identical (bit-lossless).
Exit 1 => at least one prompt diverged; prints the first differing position,
          a small window around it, and how far the two agreed.
"""
import json
import sys


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: bitident_diff.py A.json B.json", file=sys.stderr)
        return 2
    a, b = load(sys.argv[1]), load(sys.argv[2])
    la, lb = a.get("label", sys.argv[1]), b.get("label", sys.argv[2])
    prompts = a["prompts"]
    ok = True
    for i in range(len(prompts)):
        ta = a["runs"].get(str(i), {}).get("tokens", [])
        tb = b["runs"].get(str(i), {}).get("tokens", [])
        if ta == tb:
            print(f"prompt {i}: OK  ({len(ta)} tokens identical)")
            continue
        ok = False
        # find first divergence
        n = min(len(ta), len(tb))
        j = 0
        while j < n and ta[j] == tb[j]:
            j += 1
        print(f"prompt {i}: DIVERGE at token {j}/{min(len(ta), len(tb))} "
              f"(len {la}={len(ta)} {lb}={len(tb)})")
        lo = max(0, j - 3)
        print(f"    agreed prefix [...{lo}:{j}]: {ta[lo:j]!r}")
        print(f"    {la:>14} [{j}:{j+4}]: {ta[j:j+4]!r}")
        print(f"    {lb:>14} [{j}:{j+4}]: {tb[j:j+4]!r}")
    print()
    if ok:
        print(f"RESULT: BIT-IDENTICAL  ({la} == {lb})  — all {len(prompts)} prompts match")
    else:
        print(f"RESULT: DIVERGENCE  ({la} != {lb})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
