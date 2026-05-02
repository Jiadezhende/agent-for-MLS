"""tests/pipeline/test_operator_spec.py — OperatorSpec parsing and rendering."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from pipeline.operator_spec import OperatorSpec, TensorSpec


_SKILLS_ROOT = Path(__file__).resolve().parents[2] / "skills"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestOperatorSpecLoad:
    def test_load_lora_matmul(self):
        spec = OperatorSpec.load_from_skill(_SKILLS_ROOT, "lora_matmul")
        assert spec.name == "lora_matmul"
        assert len(spec.inputs) == 4
        assert [t.name for t in spec.inputs] == ["W", "X", "A", "B"]
        assert spec.output.name == "Y"
        assert spec.output.shape == ("d", "d")
        assert spec.shape_param == "d"
        assert spec.shape_param_range == (3584, 4608)
        assert spec.forward_args == ("W", "X", "A", "B")
        # Reference formula must match the existing baseline numerics (B.T, not transpose().contiguous()).
        assert "B.T" in spec.reference_pytorch

    def test_load_plain_matmul(self):
        spec = OperatorSpec.load_from_skill(_SKILLS_ROOT, "plain_matmul")
        assert spec.name == "plain_matmul"
        assert [t.name for t in spec.inputs] == ["W", "X"]
        assert spec.forward_args == ("W", "X")
        assert spec.reference_pytorch == "W @ X"

    def test_missing_skill_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            OperatorSpec.load_from_skill(tmp_path, "nonexistent_op")

    def test_missing_required_field_raises(self, tmp_path):
        skills = tmp_path / "operators"
        skills.mkdir(parents=True)
        (skills / "broken.md").write_text(
            "---\nname: x\ndescription: y\n---\nbody\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="missing fields"):
            OperatorSpec.load_from_skill(tmp_path, "broken")

    def test_forward_args_must_reference_declared_inputs(self, tmp_path):
        skills = tmp_path / "operators"
        skills.mkdir(parents=True)
        (skills / "bad.md").write_text(
            "---\n"
            "name: bad\ndescription: bad\n"
            "shape_param: d\nshape_param_range: [1, 2]\n"
            "inputs:\n  - {name: W, shape: [d, d], dtype: float32}\n"
            "output: {name: Y, shape: [d, d], dtype: float32}\n"
            'reference_pytorch: "W"\n'
            "forward_args: [W, Z]\n"
            "---\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="undeclared inputs"):
            OperatorSpec.load_from_skill(tmp_path, "bad")


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

class TestRender:
    @pytest.fixture
    def lora(self):
        return OperatorSpec.load_from_skill(_SKILLS_ROOT, "lora_matmul")

    @pytest.fixture
    def plain(self):
        return OperatorSpec.load_from_skill(_SKILLS_ROOT, "plain_matmul")

    def test_render_input_creation_lora(self, lora):
        out = lora.render_input_creation()
        lines = out.splitlines()
        assert len(lines) == 4
        assert "W = torch.randn(d, d, device=device, dtype=torch.float32)" in lines
        assert "A = torch.randn(d, 16, device=device, dtype=torch.float32)" in lines

    def test_render_input_creation_plain(self, plain):
        out = plain.render_input_creation()
        assert len(out.splitlines()) == 2
        assert "A = torch" not in out and "B = torch" not in out

    def test_render_save_inputs(self, lora):
        out = lora.render_save_inputs(dir_var="INPUT_DIR", d_var="d")
        # All four input files are written, with the expected naming convention.
        for name in ("W", "X", "A", "B"):
            assert f'"{name}_d{{d}}.pt"' in out
        assert "torch.save(W.cpu()" in out

    def test_render_load_inputs(self, lora):
        out = lora.render_load_inputs(dir_var="INPUT_DIR", d_var="d")
        for name in ("W", "X", "A", "B"):
            assert f'"{name}_d{{d}}.pt"' in out
        assert "map_location=device" in out

    def test_render_reference_compute(self, lora):
        assert lora.render_reference_compute() == "Y = W @ X + A @ (B.T @ X)"

    def test_render_save_reference(self, lora):
        out = lora.render_save_reference(dir_var="REF_DIR", d_var="d")
        assert "Y.cpu()" in out
        assert '"Y_d{d}.pt"' in out

    def test_render_forward_call_lora(self, lora):
        assert lora.render_forward_call("mod") == "mod.forward(W, X, A, B)"

    def test_render_forward_call_plain(self, plain):
        assert plain.render_forward_call("mod") == "mod.forward(W, X)"

    def test_summary_for_prompt_includes_key_facts(self, lora):
        out = lora.summary_for_prompt()
        assert "lora_matmul" in out
        assert "Y = W @ X + A @ (B.T @ X)" in out
        assert "forward(W, X, A, B)" in out
        assert "[3584, 4608]" in out


# ---------------------------------------------------------------------------
# TensorSpec edge cases
# ---------------------------------------------------------------------------

class TestTensorSpec:
    def test_constant_in_shape_preserved(self):
        t = TensorSpec.from_dict({"name": "A", "shape": ["d", 16], "dtype": "float32"})
        assert t.shape == ("d", 16)
        assert t.render_shape() == "d, 16"

    def test_invalid_shape_entry_rejected(self):
        with pytest.raises(ValueError, match="shape entries"):
            TensorSpec.from_dict({"name": "A", "shape": ["d", [1, 2]], "dtype": "float32"})

    def test_torch_dtype(self):
        t = TensorSpec(name="W", shape=("d",), dtype="bfloat16")
        assert t.torch_dtype() == "torch.bfloat16"
