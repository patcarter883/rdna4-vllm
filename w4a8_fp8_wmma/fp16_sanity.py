"""Container sanity check: does this image generate coherent FP16 text on gfx1201?
Isolates container/attention health from our quantized-kernel work. No kernel needed.
"""
import sys
from vllm import LLM, SamplingParams


def main():
    mid = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    llm = LLM(model=mid, dtype="float16", enforce_eager=True,
              gpu_memory_utilization=0.55, max_model_len=2048)
    o = llm.generate(["The capital of France is", "Q: What is 2+2? A:"],
                     SamplingParams(temperature=0.0, max_tokens=40))
    print("===== FP16 SANITY GENERATION =====")
    for r in o:
        print("OUT:", repr(r.outputs[0].text))


if __name__ == "__main__":
    main()
