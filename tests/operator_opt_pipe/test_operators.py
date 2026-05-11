"""Unit tests for operator_opt_pipe.operators OPS classes (CPU-only).

Covers ``make_inputs`` reproducibility, ``reference`` correctness, and
``forward_call`` argument unpacking. No CUDA / nvcc required — uses
``device='cpu'`` throughout.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from operator_opt_pipe.operators import OPS_REGISTRY, load_ops
from operator_opt_pipe.operators.lora_matmul import LoraMatmulOps
from operator_opt_pipe.operators.plain_matmul import PlainMatmulOps


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_ops_registry_has_known_operators():
    assert "lora_matmul" in OPS_REGISTRY
    assert "plain_matmul" in OPS_REGISTRY


def test_load_ops_returns_instance():
    assert isinstance(load_ops("lora_matmul"), LoraMatmulOps)
    assert isinstance(load_ops("plain_matmul"), PlainMatmulOps)


def test_load_ops_unknown_raises():
    with pytest.raises(FileNotFoundError, match="unknown operator"):
        load_ops("does_not_exist")


def test_ops_short_name_matches_registry_key():
    for short, ops in OPS_REGISTRY.items():
        assert ops.short_name == short


# ---------------------------------------------------------------------------
# LoraMatmulOps
# ---------------------------------------------------------------------------


@pytest.fixture
def cpu_device():
    return torch.device("cpu")


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def test_lora_make_inputs_keys_and_shapes(cpu_device):
    ops = load_ops("lora_matmul")
    inp = ops.make_inputs(64, device=cpu_device, generator=_gen())
    assert set(inp) == {"W", "X", "A", "B"}
    assert inp["W"].shape == (64, 64)
    assert inp["X"].shape == (64, 64)
    assert inp["A"].shape == (64, 16)
    assert inp["B"].shape == (64, 16)
    for t in inp.values():
        assert t.dtype == torch.float32
        assert t.device == cpu_device


def test_lora_make_inputs_seed_reproducible(cpu_device):
    ops = load_ops("lora_matmul")
    a = ops.make_inputs(32, device=cpu_device, generator=_gen(seed=42))
    b = ops.make_inputs(32, device=cpu_device, generator=_gen(seed=42))
    for k in a:
        assert torch.equal(a[k], b[k])


def test_lora_make_inputs_different_seed_diverges(cpu_device):
    ops = load_ops("lora_matmul")
    a = ops.make_inputs(32, device=cpu_device, generator=_gen(seed=1))
    b = ops.make_inputs(32, device=cpu_device, generator=_gen(seed=2))
    assert not torch.equal(a["W"], b["W"])


def test_lora_reference_matches_formula(cpu_device):
    ops = load_ops("lora_matmul")
    inp = ops.make_inputs(8, device=cpu_device, generator=_gen())
    Y = ops.reference(inp)
    expected = inp["W"] @ inp["X"] + inp["A"] @ (inp["B"].transpose(0, 1).contiguous() @ inp["X"])
    assert torch.allclose(Y, expected, rtol=0, atol=0)


def test_lora_forward_call_unpacks_args(cpu_device):
    ops = load_ops("lora_matmul")
    inp = ops.make_inputs(4, device=cpu_device, generator=_gen())

    received = {}

    class FakeMod:
        def forward(self, W, X, A, B):
            received["args"] = (W, X, A, B)
            return torch.zeros((4, 4))

    Y = ops.forward_call(FakeMod(), inp)
    assert Y.shape == (4, 4)
    W, X, A, B = received["args"]
    assert torch.equal(W, inp["W"])
    assert torch.equal(X, inp["X"])
    assert torch.equal(A, inp["A"])
    assert torch.equal(B, inp["B"])


# ---------------------------------------------------------------------------
# PlainMatmulOps
# ---------------------------------------------------------------------------


def test_plain_make_inputs_keys_and_shapes(cpu_device):
    ops = load_ops("plain_matmul")
    inp = ops.make_inputs(32, device=cpu_device, generator=_gen())
    assert set(inp) == {"W", "X"}
    assert inp["W"].shape == (32, 32)
    assert inp["X"].shape == (32, 32)


def test_plain_reference_matches_formula(cpu_device):
    ops = load_ops("plain_matmul")
    inp = ops.make_inputs(16, device=cpu_device, generator=_gen())
    Y = ops.reference(inp)
    assert torch.allclose(Y, inp["W"] @ inp["X"], rtol=0, atol=0)


def test_plain_forward_call_unpacks_args(cpu_device):
    ops = load_ops("plain_matmul")
    inp = ops.make_inputs(8, device=cpu_device, generator=_gen())

    received = {}

    class FakeMod:
        def forward(self, W, X):
            received["args"] = (W, X)
            return torch.zeros((8, 8))

    ops.forward_call(FakeMod(), inp)
    W, X = received["args"]
    assert torch.equal(W, inp["W"])
    assert torch.equal(X, inp["X"])


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------


def test_save_and_load_inputs_roundtrip(tmp_path, cpu_device):
    ops = load_ops("lora_matmul")
    inp = ops.make_inputs(8, device=cpu_device, generator=_gen())
    sid = ops.shape_id(8)
    ops.save_inputs(inp, tmp_path, sid)
    loaded = ops.load_inputs(tmp_path, sid, device=cpu_device)
    assert set(loaded) == set(inp)
    for k in inp:
        assert torch.equal(loaded[k], inp[k])


def test_shape_id_uses_contract_shape_param():
    lora = load_ops("lora_matmul")
    plain = load_ops("plain_matmul")
    assert lora.shape_id(4096) == "d4096"
    assert plain.shape_id(1024) == "d1024"
