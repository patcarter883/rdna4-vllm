"""Compare two GDN A/B dumps (ab_perf_qwen35.py): coherence token-id match + prefill speedup."""
import json
import sys


def load(p):
    with open(p) as f:
        return json.load(f)


def main(a_path, b_path):
    a, b = load(a_path), load(b_path)
    wmma = a if a["prefill_path"] == "wmma" else b
    rec = b if a["prefill_path"] == "wmma" else a

    # coherence
    n = len(wmma["coherence"])
    ident = 0
    first_div = []
    for i in range(n):
        ta, tb = wmma["coherence"][i]["token_ids"], rec["coherence"][i]["token_ids"]
        if ta == tb:
            ident += 1
        else:
            j = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]), min(len(ta), len(tb)))
            first_div.append((i, j))
    print(f"=== coherence (WMMA vs recurrent greedy ids) ===")
    print(f"  identical: {ident}/{n}")
    for i, j in first_div:
        print(f"  prompt {i}: first divergence at token {j}")

    # perf
    print(f"=== prefill perf (prompt_tokens={wmma['prompt_tokens']}, iters={wmma['iters']}) ===")
    print(f"  WMMA      mean={wmma['prefill_mean_ms']:.2f}ms median={wmma['prefill_median_ms']:.2f}ms min={wmma['prefill_min_ms']:.2f}ms")
    print(f"  recurrent mean={rec['prefill_mean_ms']:.2f}ms median={rec['prefill_median_ms']:.2f}ms min={rec['prefill_min_ms']:.2f}ms")
    for k in ("mean", "median", "min"):
        sp = rec[f"prefill_{k}_ms"] / wmma[f"prefill_{k}_ms"]
        print(f"  speedup ({k}): {sp:.2f}x")

    engaged = abs(wmma["prefill_mean_ms"] - rec["prefill_mean_ms"]) / rec["prefill_mean_ms"] > 0.02
    print(f"=== engagement: paths differ by >2% wall -> {'YES (WMMA path engaged)' if engaged else 'NO — suspect silent fallback'} ===")
    ok = ident >= n - 1 and engaged
    print("RESULT:", "PASS" if ok else "REVIEW")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
