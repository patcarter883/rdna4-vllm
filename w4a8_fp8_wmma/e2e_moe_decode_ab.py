"""E2E decode-throughput A/B for the MoE W4A8 path on a single GPU.

Loads a small AWQ/compressed-tensors MoE (Mellum2-12B-A2.5B, g=32, the Piece-1
v7 decode kernel's shape) on GPU0, checks coherence, and times batch=1 decode
throughput. Engagement is controlled by env set OUTSIDE this script:
  STOCK:  VLLM_ROCM_USE_W4A8_FP8_WMMA=0
  v7:     VLLM_ROCM_USE_W4A8_FP8_WMMA=1 VLLM_ROCM_W4A8_FP8_WMMA_MOE_MIN_M=1
Run each in its own process (model state differs).
"""
import os
import time


def main():
    from vllm import LLM, SamplingParams
    model = os.environ.get(
        "E2E_MODEL",
        "cyankiwi/Mellum2-12B-A2.5B-Instruct-AWQ-INT4")
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_seqs=8,
        gpu_memory_utilization=0.90,
        enforce_eager=os.environ.get("E2E_EAGER", "1") == "1",
        trust_remote_code=True,
    )
    prompt = ("Write a short paragraph explaining why the sky appears blue "
              "during the day and red at sunset.")
    # coherence + warmup
    warm = llm.generate(
        [prompt], SamplingParams(temperature=0.0, max_tokens=64))
    txt = warm[0].outputs[0].text
    print("=== COHERENCE (first 200 chars) ===")
    print(txt[:200].replace("\n", " "))

    # timed batch=1 decode: long-ish gen, ignore eos so token count is fixed.
    NT = int(os.environ.get("E2E_TOKENS", "256"))
    sp = SamplingParams(temperature=0.0, max_tokens=NT, ignore_eos=True)
    # 2 runs, take the faster (cache warm).
    best = None
    for _ in range(2):
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp)
        dt = time.perf_counter() - t0
        n = len(out[0].outputs[0].token_ids)
        tps = n / dt
        best = tps if best is None else max(best, tps)
    tag = os.environ.get("E2E_TAG", "run")
    print(f"=== RESULT [{tag}] batch=1 decode: {best:.1f} tok/s "
          f"({NT} tokens) | W4A8={os.environ.get('VLLM_ROCM_USE_W4A8_FP8_WMMA','1')} "
          f"MOE_MIN_M={os.environ.get('VLLM_ROCM_W4A8_FP8_WMMA_MOE_MIN_M','default')}")


if __name__ == "__main__":
    main()
