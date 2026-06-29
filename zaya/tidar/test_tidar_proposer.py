"""CPU tests for the TiDAR proposer/runner integration module (no GPU lease).

Weight-independent: exercises the orchestration (env flag, carrier set/clear discipline, β=1
verify/commit forwarding, evict-column contract) against the StubCCALM, asserting the wiring
reproduces the standalone loop's β=1 == greedy-AR losslessness through the proposer surface. The
mask/rollback math itself is pinned by test_tidar_mask.py / test_tidar_loop.py — this only proves the
proposer routes to them correctly.

Run (container venv, no lease):
  docker run --rm -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= \
    -v /home/pat/code/vllm-gfx1201-tidar-serve/zaya/tidar:/work -w /work \
    --entrypoint bash vllm22-w4a8:dflash-rxf -lc \
    'source /app/.venv/bin/activate && python -m pytest -q test_tidar_proposer.py'
"""
import os

import torch

from tidar_attn_metadata import get_active_tidar_mask
from tidar_loop import MASK_ID, StubCCALM, beta_verify
from tidar_mask import MaskDescriptor
from tidar_proposer import (
    TidarProposer,
    make_proposer_from_env,
    tidar_block_len,
    tidar_serving_enabled,
)


# --------------------------------------------------------------------------------------------- #
# env flag
# --------------------------------------------------------------------------------------------- #
def test_env_flag_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_TIDAR_BLOCK_LEN", raising=False)
    assert tidar_block_len() is None
    assert tidar_serving_enabled() is False
    assert make_proposer_from_env() is None


def test_env_flag_parsing(monkeypatch):
    for raw, want in [("4", 4), ("16", 16), ("1", 1), ("0", None), ("-3", None), ("x", None), ("", None)]:
        monkeypatch.setenv("VLLM_TIDAR_BLOCK_LEN", raw)
        assert tidar_block_len() == want, raw
    monkeypatch.setenv("VLLM_TIDAR_BLOCK_LEN", "8")
    p = make_proposer_from_env()
    assert p is not None and p.block_len == 8


# --------------------------------------------------------------------------------------------- #
# carrier set/clear discipline
# --------------------------------------------------------------------------------------------- #
def test_run_block_sets_and_clears_carrier():
    p = TidarProposer(block_len=4)
    assert get_active_tidar_mask() is None
    with p.run_block(prefix_len=16) as meta:
        active = get_active_tidar_mask()
        assert active is meta
        d = MaskDescriptor(prefix_len=16, block_len=4)
        assert meta.qq_bias.shape == (d.q_len, d.q_len)
    assert get_active_tidar_mask() is None, "carrier must be cleared on normal exit"


def test_run_block_clears_on_exception():
    p = TidarProposer(block_len=8)
    try:
        with p.run_block(prefix_len=0):
            assert get_active_tidar_mask() is not None
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert get_active_tidar_mask() is None, "carrier must be cleared even on exception"


# --------------------------------------------------------------------------------------------- #
# β=1 verify_commit forwards to beta_verify (identical result)
# --------------------------------------------------------------------------------------------- #
def test_verify_commit_forwards_to_beta_verify():
    torch.manual_seed(0)
    B, V = 4, 64
    drafts = [3, 7, 11, 2]
    p_ar = torch.randn(B + 1, V, dtype=torch.float64)
    p = TidarProposer(block_len=B)
    assert p.verify_commit(drafts, p_ar, beta=1.0) == beta_verify(drafts, p_ar, beta=1.0)


# --------------------------------------------------------------------------------------------- #
# evict column contract == cca.py read column (num_accepted-1, clamp min 1)
# --------------------------------------------------------------------------------------------- #
def test_evict_contract_column():
    p = TidarProposer(block_len=8)
    assert p.evict_contract(0) == 0     # clamp(min=1)-1 == 0
    assert p.evict_contract(1) == 0
    assert p.evict_contract(4) == 3
    assert p.evict_contract(8) == 7


# --------------------------------------------------------------------------------------------- #
# end-to-end: the proposer surface reproduces β=1 == greedy-AR (the losslessness pin), single seq.
# This mirrors tidar_loop.tidar_decode but drives it through TidarProposer.verify_commit so the
# proposer's accept/commit routing is what's exercised.
# --------------------------------------------------------------------------------------------- #
def _proposer_decode(model: StubCCALM, prompt, n_new, B):
    p = TidarProposer(block_len=B)
    committed = list(prompt)
    target = len(prompt) + n_new

    def fresh_drafts(ctx):
        d = MaskDescriptor(prefix_len=len(ctx), block_len=B)
        fwd = model.tidar_forward(ctx, [MASK_ID] * B, d)
        return model.replica_drafts(fwd, d, 0)

    drafts = fresh_drafts(committed)
    while len(committed) < target:
        L = len(committed)
        full = model.causal_forward(committed + drafts)["logits"]
        p_ar = full[L - 1: L + B]
        k, bonus = p.verify_commit(drafts, p_ar, beta=1.0)
        committed = committed + drafts[:k] + [bonus]
        if len(committed) >= target:
            break
        drafts = fresh_drafts(committed)
    return committed[len(prompt):target]


def test_proposer_beta1_equals_greedy_ar():
    for seed in (0, 1, 2):
        for B in (1, 4, 8):
            model = StubCCALM(seed=seed)
            prompt = [5, 9, 2]
            ar = model.greedy_decode(prompt, 12)[len(prompt):][:12]
            td = _proposer_decode(model, prompt, 12, B)
            assert ar == td, f"seed={seed} B={B}: TiDAR(proposer) β=1 != greedy AR"
