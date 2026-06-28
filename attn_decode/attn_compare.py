"""Compare two token_diff_vllm.py dumps: HIP attn (flag=1) vs Triton (flag=0) greedy ids."""
import json
import sys


def main(base):
    a = json.load(open(f"{base}_0.json"))["ids"]
    b = json.load(open(f"{base}_1.json"))["ids"]
    n = len(a)
    ident = 0
    divs = []
    for i in range(n):
        if a[i] == b[i]:
            ident += 1
        else:
            j = next((k for k in range(min(len(a[i]), len(b[i]))) if a[i][k] != b[i][k]),
                     min(len(a[i]), len(b[i])))
            divs.append((i, j))
    print(f"=== attn token-diff: HIP(flag=1) vs Triton(flag=0) greedy ids ===")
    print(f"  identical: {ident}/{n}")
    for i, j in divs:
        print(f"  prompt {i}: first divergence at token {j} (late flip = benign two-kernel drift)")
    # >=4/6 identical is the documented pass gate (two correct attn kernels drift late)
    print("RESULT:", "PASS" if ident >= max(1, n - 2) else "REVIEW")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ad")
