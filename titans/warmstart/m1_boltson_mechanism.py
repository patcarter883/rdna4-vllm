"""M1 (bolt-on) — input-embeds injection + adapter-trainability mechanism check.

De-risks the foundation of the frozen-base bolt-on Titans adapter, cheaply, on 1 card:
  (i)   the frozen Qwen3.5-4B GDN-hybrid accepts inputs_embeds with a prepended K-vector "memory"
        prefix and runs (its linear_attn + full_attention layers both handle the longer sequence),
  (ii)  gradients flow to the prefix while the base stays frozen  -> the adapter IS trainable
        through the frozen base (prefix/prompt-tuning style backprop),
  (iii) records the frozen-base LM baseline (the bar gate-(a)/(b) build on).

The learnable prefix here is a stand-in for the real Titans memory adapter's output (M2).

Run via gpu-lease (1 card):
  scripts/gpu-lease.sh -n 1 --name titans-m1 -- docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 16g \
    --security-opt seccomp=unconfined --security-opt label=disable \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
    -v /home/pat/code/vllm-gfx1201-titans/titans:/work \
    -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
    --entrypoint bash titans:dev -lc 'source /app/.venv/bin/activate && python -u /work/warmstart/m1_boltson_mechanism.py'
"""
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = "Qwen/Qwen3.5-4B"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = 8  # injected memory tokens (prefix length)

# Distinct (non-repeated) text — a meaningful smoke baseline, kept short so backprop-through-the-
# frozen-4B fits 16 GB (the real M1 memory lesson). Grad checkpointing does the heavy lifting.
TEXT = (
    "Test-time memorization lets a model store information while it reads, rather than only "
    "during training. The mechanism keeps a compact state that summarizes what has been seen "
    "and is consulted when generating new tokens. Because attention can only look within a "
    "bounded window, a separate persistent memory is what carries facts across long distances. "
    "The surprise signal decides how strongly each new observation is written into that state."
)


def load_frozen_base():
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    last = None
    for loader in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            m = loader.from_pretrained(MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True)
            m = m.to(DEV).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            # backprop-through-frozen-base is activation-heavy at 16 GB -> checkpoint + no kv-cache
            m.config.use_cache = False
            try:
                m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except Exception as e:  # noqa
                print(f"[m1] grad-checkpoint enable warning: {e}")
            return m, tok
        except Exception as e:  # noqa
            last = e
    raise last


def base_logits(model, inputs_embeds):
    out = model(inputs_embeds=inputs_embeds)
    return out.logits


def main():
    print(f"[m1] device={DEV} torch={torch.__version__} K={K}")
    model, tok = load_frozen_base()
    cfg = model.config.get_text_config()
    H, V = cfg.hidden_size, cfg.vocab_size
    n_train = sum(p.requires_grad for p in model.parameters())
    print(f"[m1] frozen base loaded: hidden={H} vocab={V} trainable_base_tensors={n_train} (want 0)")

    emb = model.get_input_embeddings()
    ids = tok(TEXT, return_tensors="pt").input_ids.to(DEV)
    S = ids.shape[1]
    tok_embeds = emb(ids).detach()  # [1,S,H]
    print(f"[m1] seq_len={S} embed_dtype={tok_embeds.dtype}")

    # ---- (iii) frozen-base baseline (no memory prefix) ----
    with torch.no_grad():
        logits = base_logits(model, tok_embeds)[:, :-1].float()
        base_ce = F.cross_entropy(logits.reshape(-1, V), ids[:, 1:].reshape(-1)).item()
    print(f"[m1] BASELINE frozen base (no memory): CE={base_ce:.4f} PPL={torch.tensor(base_ce).exp():.3f}")

    # ---- (i) injection: prepend a learnable K-vector "memory" prefix as inputs_embeds ----
    prefix = nn.Parameter(torch.randn(1, K, H, device=DEV, dtype=tok_embeds.dtype) * 0.02)
    inp = torch.cat([prefix, tok_embeds], dim=1)  # [1, K+S, H]
    logits_all = base_logits(model, inp)  # [1, K+S, V]
    assert logits_all.shape[1] == K + S, logits_all.shape
    # predictions for the real tokens live at positions K..K+S-1; predict ids[1:]
    pred = logits_all[:, K:-1, :].float()
    loss = F.cross_entropy(pred.reshape(-1, V), ids[:, 1:].reshape(-1))
    print(f"[m1] INJECTION OK: logits {tuple(logits_all.shape)}; with random prefix CE={loss.item():.4f}")

    # ---- (ii) trainability: grad to prefix, none to base ----
    loss.backward()
    g = prefix.grad
    prefix_grad_ok = g is not None and torch.isfinite(g).all() and g.abs().sum().item() > 0
    base_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"[m1] TRAINABILITY: prefix.grad nonzero+finite={bool(prefix_grad_ok)} "
          f"(|grad|sum={g.abs().sum().item():.3e}); base_received_grad={base_has_grad} (want False)")

    ok = prefix_grad_ok and not base_has_grad and logits_all.shape[1] == K + S
    print(f"[m1] {'PASS' if ok else 'FAIL'}: bolt-on foundation "
          f"{'validated — injection runs + adapter is trainable through the frozen base.' if ok else 'BROKEN.'}")


if __name__ == "__main__":
    main()
