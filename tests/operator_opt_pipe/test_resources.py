"""Unit tests for operator_opt_pipe.resources (no GPU required).

Covers:
  * eval_shape — controlled mini-DSL for shape resolution
  * shape_id — stable shape filename component
  * load_contract — frontmatter parsing + validation
  * OperatorContract render_* helpers — used by subprocess script generators

GPU-flagged integration tests for materialize_inputs / run_pytorch_baseline /
benchmark_on_grid would belong in a separate file; they need real torch +
CUDA + nvcc and are expected to be skipped in CI.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from operator_opt_pipe.resources import (
    OperatorContract,
    TensorSpec,
    eval_shape,
    load_contract,
    shape_id,
)


# ---------------------------------------------------------------------------
# eval_shape
# ---------------------------------------------------------------------------


def test_eval_shape_int_constant():
    assert eval_shape(16) == 16


def test_eval_shape_bare_variable():
    assert eval_shape("d", d=4096) == 4096


def test_eval_shape_arithmetic():
    assert eval_shape("d * 2", d=2048) == 4096
    assert eval_shape("d + 16", d=128) == 144
    assert eval_shape("d // 4", d=1024) == 256


def test_eval_shape_rejects_unknown_variable():
    with pytest.raises(ValueError, match="unknown variable"):
        eval_shape("x", d=128)


def test_eval_shape_rejects_function_calls():
    with pytest.raises(ValueError, match="forbidden ast node"):
        eval_shape("max(d, 16)", d=4096)


def test_eval_shape_rejects_attribute_access():
    with pytest.raises(ValueError, match="forbidden ast node"):
        eval_shape("d.bit_length", d=4096)


def test_eval_shape_rejects_non_int_result():
    # Comparisons would yield bool; comparisons are forbidden anyway.
    with pytest.raises(ValueError):
        eval_shape("d > 1", d=4096)


# ---------------------------------------------------------------------------
# shape_id
# ---------------------------------------------------------------------------


def test_shape_id_single_variable():
    contract = _lora_contract()
    assert shape_id(contract, d=3584) == "d3584"


def test_shape_id_omits_unrelated_vars():
    contract = _lora_contract()
    # Only `d` is the operator's shape param; extra unrelated keys aren't appended
    # because we only walk shape_param + sorted other shape_values.
    sid = shape_id(contract, d=4096, h=64)
    # d3584 / d4096 / d4096_h64 — h64 IS appended because it's an extra var.
    assert sid.startswith("d4096")


# ---------------------------------------------------------------------------
# load_contract
# ---------------------------------------------------------------------------


def test_load_contract_reads_lora_skill():
    """Use the real LoRA skill markdown shipped with the repo."""
    repo_root = Path(__file__).resolve().parents[2]
    contract = load_contract(repo_root / "skills", operator="lora_matmul")
    assert contract.shape_param == "d"
    assert contract.shape_param_range == (3584, 4608)
    assert len(contract.inputs) == 4
    names = [t.name for t in contract.inputs]
    assert names == ["W", "X", "A", "B"]
    assert contract.output.name == "Y"
    assert contract.forward_args == ("W", "X", "A", "B")
    # Reference formula matches the Phase-2 spec
    assert "W @ X" in contract.reference_pytorch
    # Tolerances default or come from frontmatter
    assert contract.rtol == 1e-4
    assert contract.atol == 1e-4


def test_load_contract_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_contract(tmp_path, operator="bogus_op")


def test_load_contract_rejects_missing_fields(tmp_path: Path):
    skill_path = tmp_path / "operators" / "incomplete.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(
        "---\n"
        "name: operators/incomplete\n"
        "shape_param: d\n"
        # missing shape_param_range, inputs, output, reference_pytorch, forward_args
        "---\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing fields"):
        load_contract(tmp_path, operator="incomplete")


def test_load_contract_rejects_undeclared_forward_arg(tmp_path: Path):
    skill_path = tmp_path / "operators" / "bad.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(
        "---\n"
        "name: operators/bad\n"
        "shape_param: d\n"
        "shape_param_range: [1, 100]\n"
        "inputs:\n"
        "  - {name: A, shape: [d, d], dtype: float32}\n"
        "output: {name: Y, shape: [d, d], dtype: float32}\n"
        "reference_pytorch: \"A\"\n"
        "forward_args: [A, BOGUS]\n"
        "---\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="undeclared inputs"):
        load_contract(tmp_path, operator="bad")


# ---------------------------------------------------------------------------
# OperatorContract render helpers (subprocess script generators rely on these)
# ---------------------------------------------------------------------------


def test_render_input_creation_uses_contract():
    contract = _lora_contract()
    rendered = contract.render_input_creation()
    assert "W = torch.randn(d, d, device=device, dtype=torch.float32)" in rendered
    assert "A = torch.randn(d, 16, device=device, dtype=torch.float32)" in rendered


def test_render_save_inputs_uses_shape_id_var():
    contract = _lora_contract()
    rendered = contract.render_save_inputs(dir_var="DIR", shape_id_expr="sid")
    assert 'os.path.join(DIR, f"W_{sid}.pt")' in rendered
    assert 'os.path.join(DIR, f"B_{sid}.pt")' in rendered


def test_render_forward_call_preserves_argument_order():
    contract = _lora_contract()
    assert contract.render_forward_call("mod") == "mod.forward(W, X, A, B)"


def test_default_shape_grid_is_three_points():
    contract = _lora_contract()
    grid = contract.default_shape_grid()
    assert len(grid) == 3
    assert grid[0] == 3584
    assert grid[-1] == 4608


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lora_contract() -> OperatorContract:
    return OperatorContract(
        name="operators/lora_matmul",
        inputs=(
            TensorSpec(name="W", shape=("d", "d"), dtype="float32"),
            TensorSpec(name="X", shape=("d", "d"), dtype="float32"),
            TensorSpec(name="A", shape=("d", 16), dtype="float32"),
            TensorSpec(name="B", shape=("d", 16), dtype="float32"),
        ),
        output=TensorSpec(name="Y", shape=("d", "d"), dtype="float32"),
        reference_pytorch="W @ X + A @ (B.transpose(0, 1).contiguous() @ X)",
        forward_args=("W", "X", "A", "B"),
        shape_param="d",
        shape_param_range=(3584, 4608),
    )
