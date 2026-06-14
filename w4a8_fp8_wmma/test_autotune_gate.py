"""CPU unit test for the W4A8 autotune dispatch gate's NON-GPU parts (no GPU, no
real torch/vLLM needed).

Task: on a crossover-cache MISS the gate now AUTO-TUNES the shape at load (dense)
/ first-batch (MoE) and persists the result, so the kernel actually engages on new
models instead of silently running stock. The GPU A/B itself must NOT run here;
this test covers the two pieces that are pure logic:

  1. the winning-suffix / winning-interval SELECTION given a table of
     (M -> our_time, triton_time) -- i.e. _winning_suffix_crossover (dense) and
     _moe_winning_intervals (MoE);
  2. the cache READ / MERGE / WRITE path -- _persist_crossover (dense) and
     _moe_persist_crossover (MoE), incl. that a fresh key merges into an existing
     cache and round-trips through JSON, and that the in-process table updates.

vllm_adapter.py and moe_experts.py import torch / vLLM, which aren't importable on
this host, so we stub just enough in sys.modules for the import to succeed; the
functions under test never touch torch at call time.
"""
import json
import os
import sys
import tempfile
import types


# --------------------------------------------------------------------------- #
# minimal sys.modules stubs so the two adapter modules import without real
# torch / vLLM (the pure functions we test use only json + os).
# --------------------------------------------------------------------------- #
def _install_stubs():
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.float16 = "float16"
        torch.bfloat16 = "bfloat16"
        torch.Tensor = type("Tensor", (), {})

        class _NN(types.ModuleType):
            Module = type("Module", (), {})
        torch.nn = _NN("torch.nn")
        sys.modules["torch"] = torch
        sys.modules["torch.nn"] = torch.nn

    def _mod(name):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
        return sys.modules[name]

    # vllm.* used at import time by vllm_adapter / lazily by moe_experts.
    _mod("vllm")
    mk = _mod("vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel")
    mk.MPLinearKernel = type("MPLinearKernel", (), {})
    mk.MPLinearLayerConfig = type("MPLinearLayerConfig", (), {})
    plat = _mod("vllm.platforms")
    plat.current_platform = types.SimpleNamespace(is_rocm=lambda: False)
    plat.PlatformEnum = types.SimpleNamespace(ROCM="ROCM")
    st = _mod("vllm.scalar_type")
    st.scalar_types = types.SimpleNamespace(uint4b8="uint4b8", uint4="uint4")
    # ensure the intermediate packages exist for the dotted import
    for p in ("vllm.model_executor", "vllm.model_executor.kernels",
              "vllm.model_executor.kernels.linear",
              "vllm.model_executor.kernels.linear.mixed_precision"):
        _mod(p)


_install_stubs()

_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w4a8_fp8_wmma")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

import vllm_adapter as va           # noqa: E402
import moe_experts as me            # noqa: E402


# --------------------------------------------------------------------------- #
# 1a. dense winning-suffix selection
# --------------------------------------------------------------------------- #
def test_dense_winning_suffix():
    g = va._AUTOTUNE_MGRID  # (48,64,96,128,160,192,224,256)
    # all <= 1.0 -> entire grid is a winning suffix -> lowest M.
    assert va._winning_suffix_crossover({m: 0.8 for m in g}) == g[0]
    # never within margin -> None (stays _NEVER / stock).
    assert va._winning_suffix_crossover({m: 1.5 for m in g}) is None
    # NON-MONOTONIC: wins at 48, DIPS at 96, then wins for the whole >=160 suffix.
    # The safe answer is the start of the contiguous winning suffix = 160, NOT 48
    # (engaging at 48 would route the M=96 loss zone to our kernel).
    r = {48: 0.95, 64: 1.30, 96: 1.30, 128: 1.30, 160: 0.95,
         192: 0.95, 224: 0.95, 256: 0.95}
    assert va._winning_suffix_crossover(r) == 160
    # within the 2% margin (eps=1.02) counts as not-worse.
    assert va._winning_suffix_crossover({m: 1.01 for m in g}) == g[0]
    assert va._winning_suffix_crossover({m: 1.03 for m in g}) is None
    # a single missing/None data point in the suffix disqualifies that suffix.
    r2 = {m: 0.9 for m in g}
    r2[g[3]] = None
    assert va._winning_suffix_crossover(r2) == g[4]
    print("PASS dense winning-suffix selection")


# --------------------------------------------------------------------------- #
# 1b. MoE winning-interval selection
# --------------------------------------------------------------------------- #
def test_moe_winning_intervals():
    # no win anywhere -> None.
    assert me._moe_winning_intervals([(16, False), (32, False)]) is None
    # one contiguous run -> single interval [lo, hi].
    assert me._moe_winning_intervals(
        [(16, False), (32, True), (64, True), (128, False)]) == [[32, 64]]
    # NON-CONTIGUOUS win (mid-M dip) -> two separate intervals; a single [min,max]
    # would wrongly engage the loss zone between them.
    wins = [(16, False), (32, True), (64, True), (128, False),
            (256, True), (512, True)]
    assert me._moe_winning_intervals(wins) == [[32, 64], [256, 512]]
    # a win that runs to the end of the grid closes its interval at the last M.
    assert me._moe_winning_intervals(
        [(16, False), (32, True), (64, True)]) == [[32, 64]]
    # single-point win -> [m, m].
    assert me._moe_winning_intervals([(48, True)]) == [[48, 48]]
    print("PASS MoE winning-interval selection")


# --------------------------------------------------------------------------- #
# 2a. dense cache read / merge / write round-trip
# --------------------------------------------------------------------------- #
def test_dense_cache_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "crossover_cache.json")
        # seed an existing cache with one entry + a comment key.
        with open(path, "w") as f:
            json.dump({"_comment": "x", "4096,4096,128": 128}, f)
        os.environ["VLLM_ROCM_W4A8_FP8_WMMA_CACHE"] = path
        va._CROSSOVER_TABLE = None  # force a reload from our temp file
        try:
            # MISS for a NEW shape -> lookup returns _NEVER (always Triton).
            assert va._crossover_for(5120, 5120, 128) == va._NEVER
            # persist a tuned value; existing entries must survive the merge.
            va._persist_crossover(5120, 5120, 128, 96)
            assert va._crossover_for(5120, 5120, 128) == 96       # in-mem updated
            assert va._crossover_for(4096, 4096, 128) == 128      # untouched
            # persist a genuine "never wins" (None) -> cached so next load is O(1)
            # and DOES NOT re-tune (the key is present even though value is null).
            va._persist_crossover(11008, 4096, 32, None)
            assert va._crossover_for(11008, 4096, 32) == va._NEVER
            assert "11008,4096,32" in va._load_crossover_table()
            # everything round-trips through JSON on a fresh reload.
            va._CROSSOVER_TABLE = None
            t = va._load_crossover_table()
            assert t["5120,5120,128"] == 96
            assert t["11008,4096,32"] is None
            assert t["4096,4096,128"] == 128
            assert t["_comment"] == "x"
        finally:
            del os.environ["VLLM_ROCM_W4A8_FP8_WMMA_CACHE"]
            va._CROSSOVER_TABLE = None
    print("PASS dense cache read/merge/write round-trip")


# --------------------------------------------------------------------------- #
# 2b. MoE cache read / merge / write round-trip
# --------------------------------------------------------------------------- #
def test_moe_cache_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "moe_crossover_cache.json")
        with open(path, "w") as f:
            json.dump({"128,2304,448,32,8": [[48, 256]]}, f)
        os.environ["VLLM_ROCM_W4A8_FP8_WMMA_MOE_CACHE"] = path
        me._MOE_XOVER = None
        try:
            # MISS for a new shape.
            assert me._moe_crossover_for(128, 2304, 896, 32, 8) is None
            # persist a tuned window list; existing entry survives.
            me._moe_persist_crossover(128, 2304, 896, 32, 8, [[48, 512], [1024, 2048]])
            assert me._moe_crossover_for(128, 2304, 896, 32, 8) == [[48, 512], [1024, 2048]]
            assert me._moe_crossover_for(128, 2304, 448, 32, 8) == [[48, 256]]
            # persist a "never wins" (None) -> cached, present-but-null.
            me._moe_persist_crossover(64, 4096, 1024, 128, 2, None)
            assert me._moe_crossover_for(64, 4096, 1024, 128, 2) is None
            assert "64,4096,1024,128,2" in me._moe_load_cache()
            # round-trip through JSON on reload.
            me._MOE_XOVER = None
            t = me._moe_load_cache()
            assert t["128,2304,896,32,8"] == [[48, 512], [1024, 2048]]
            assert t["64,4096,1024,128,2"] is None
            assert t["128,2304,448,32,8"] == [[48, 256]]
        finally:
            del os.environ["VLLM_ROCM_W4A8_FP8_WMMA_MOE_CACHE"]
            me._MOE_XOVER = None
    print("PASS MoE cache read/merge/write round-trip")


# --------------------------------------------------------------------------- #
# 3. autotune env switch (the off-switch must default ON, honor 0/off/false)
# --------------------------------------------------------------------------- #
def test_autotune_env_switch():
    for k in list(os.environ):
        if k == "VLLM_ROCM_W4A8_AUTOTUNE":
            del os.environ[k]
    assert va._autotune_enabled() is True       # default ON
    assert me._moe_autotune_enabled() is True
    for off in ("0", "off", "false", "no", "OFF", "False"):
        os.environ["VLLM_ROCM_W4A8_AUTOTUNE"] = off
        assert va._autotune_enabled() is False, off
        assert me._moe_autotune_enabled() is False, off
    for on in ("1", "on", "true", "yes"):
        os.environ["VLLM_ROCM_W4A8_AUTOTUNE"] = on
        assert va._autotune_enabled() is True, on
        assert me._moe_autotune_enabled() is True, on
    del os.environ["VLLM_ROCM_W4A8_AUTOTUNE"]
    print("PASS autotune env switch")


if __name__ == "__main__":
    test_dense_winning_suffix()
    test_moe_winning_intervals()
    test_dense_cache_roundtrip()
    test_moe_cache_roundtrip()
    test_autotune_env_switch()
    print("\nALL PASSED")
