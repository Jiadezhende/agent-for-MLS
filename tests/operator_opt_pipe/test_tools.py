"""Unit tests for operator_opt_pipe.tools.

Covers ReadBlackboardTool / WriteBlackboardTool / WriteCandidateTool /
SubmitCandidateTool. Performance evaluation pathways are mocked — those
deserve their own GPU-flagged integration tests in test_resources.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mls_agent import ToolErrorCode, ToolRegistry, ToolStatus

from operator_opt_pipe.operators import load_ops
from operator_opt_pipe.operators._base import OperatorOps
from operator_opt_pipe.resources import OperatorContract, TensorSpec
from operator_opt_pipe.resources.evaluation import QuickEvalResult
from operator_opt_pipe.state import (
    RunLayout,
    load_blackboard,
    save_blackboard,
)
from operator_opt_pipe.tools import (
    ReadBlackboardTool,
    SubmitCandidateTool,
    WriteBlackboardTool,
    WriteCandidateTool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def layout(tmp_path: Path) -> RunLayout:
    lay = RunLayout(workspace_root=tmp_path, run_id="r")
    lay.mkdir()
    return lay


@pytest.fixture
def contract() -> OperatorContract:
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


@pytest.fixture
def ops() -> OperatorOps:
    return load_ops("lora_matmul")


# ---------------------------------------------------------------------------
# ReadBlackboardTool
# ---------------------------------------------------------------------------


def test_read_blackboard_returns_existing_key(layout: RunLayout):
    save_blackboard(layout, {"schema_version": 1, "history": [], "best": {"speedup": 1.4}})
    tool = ReadBlackboardTool(layout)
    resp = tool.run({"key": "best"})
    assert resp.status is ToolStatus.SUCCESS
    assert resp.data["present"] is True
    assert resp.data["value"] == {"speedup": 1.4}


def test_read_blackboard_returns_default_when_missing(layout: RunLayout):
    tool = ReadBlackboardTool(layout)
    resp = tool.run({"key": "absent", "default": "fallback"})
    assert resp.data["value"] == "fallback"
    assert resp.data["present"] is False


def test_read_blackboard_schema_via_registry(layout: RunLayout):
    reg = ToolRegistry()
    reg.register(ReadBlackboardTool(layout))
    resp = reg.dispatch("read_blackboard", {})  # missing required 'key'
    assert resp.status is ToolStatus.ERROR
    assert resp.error_info["code"] == ToolErrorCode.INVALID_ARGS


# ---------------------------------------------------------------------------
# WriteBlackboardTool
# ---------------------------------------------------------------------------


def test_write_blackboard_writes_payload(layout: RunLayout):
    tool = WriteBlackboardTool(layout, {"hardware": ["metrics"]})
    resp = tool.run({"key": "hardware", "payload": {"metrics": {"sm": 30}}})
    assert resp.status is ToolStatus.SUCCESS
    # Must NOT terminate — natural-exit pattern.
    assert resp.terminate is False
    bb = load_blackboard(layout)
    assert bb["hardware"]["metrics"] == {"sm": 30}


def test_write_blackboard_emits_event(layout: RunLayout):
    tool = WriteBlackboardTool(layout, {"hardware": None})
    resp = tool.run({"key": "hardware", "payload": {"x": 1}})
    assert resp.events
    assert resp.events[0].type == "blackboard_write"


def test_write_blackboard_rejects_unknown_key(layout: RunLayout):
    tool = WriteBlackboardTool(layout, {"hardware": None})
    resp = tool.run({"key": "best", "payload": {"speedup": 1.0}})
    assert resp.status is ToolStatus.ERROR


def test_write_blackboard_rejects_non_dict_payload(layout: RunLayout):
    tool = WriteBlackboardTool(layout, {"hardware": None})
    resp = tool.run({"key": "hardware", "payload": [1, 2, 3]})
    assert resp.status is ToolStatus.ERROR


def test_write_blackboard_rejects_missing_required_field(layout: RunLayout):
    tool = WriteBlackboardTool(layout, {"hardware": ["metrics", "device_name"]})
    resp = tool.run({"key": "hardware", "payload": {"metrics": {}}})
    assert resp.status is ToolStatus.ERROR
    assert "missing required fields" in resp.text


def test_write_blackboard_per_role_isolation(layout: RunLayout):
    """Different role instances expose disjoint key sets."""
    hw_tool = WriteBlackboardTool(layout, {"hardware": None})
    diag_tool = WriteBlackboardTool(layout, {"latest_diagnosis": None})
    assert hw_tool.run({"key": "latest_diagnosis", "payload": {}}).status is ToolStatus.ERROR
    assert diag_tool.run({"key": "hardware", "payload": {}}).status is ToolStatus.ERROR


def test_write_blackboard_requires_at_least_one_key(layout: RunLayout):
    with pytest.raises(ValueError):
        WriteBlackboardTool(layout, {})


# ---------------------------------------------------------------------------
# WriteCandidateTool
# ---------------------------------------------------------------------------


class _StubExecutor:
    """Minimal stand-in for Executor — only profile_with_torch is used."""
    pass


def test_write_candidate_rejects_source_without_pybind(layout: RunLayout, ops: OperatorOps):
    tool = WriteCandidateTool(layout, ops, _StubExecutor())
    resp = tool.run({"source": "// no pybind here, no forward fn either; needs PYBIND11_MODULE and forward to pass the gate -- still missing"})
    assert resp.status is ToolStatus.ERROR


def test_write_candidate_writes_file_and_returns_quick_result(
    layout: RunLayout,
    ops: OperatorOps,
    monkeypatch,
):
    """Valid source: file allocated, compile_and_check_quick called, dict returned."""
    captured = {}

    def fake_quick(*, ops, candidate_id, candidate_cu, inputs_dir, sample_shape, build_dir, executor):
        captured["candidate_id"] = candidate_id
        captured["candidate_cu"] = candidate_cu
        captured["sample_shape"] = sample_shape
        captured["build_dir"] = build_dir
        return QuickEvalResult(
            candidate_id=candidate_id,
            compile_ok=True,
            correctness_ok=True,
            shape_id=f"d{sample_shape}",
            max_abs_err=1e-6,
            rel_l2_err=1e-7,
            compile_log="",
            diagnostics={},
        )

    import operator_opt_pipe.tools as tools_mod
    monkeypatch.setattr(tools_mod, "compile_and_check_quick", fake_quick)

    source = (
        "#include <torch/extension.h>\n"
        "torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {\n"
        "  return W;\n"
        "}\n"
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"forward\", &forward); }\n"
    )

    tool = WriteCandidateTool(layout, ops, _StubExecutor())
    resp = tool.run({"source": source})

    assert resp.status is ToolStatus.SUCCESS
    assert captured["candidate_id"] == "candidate_000"
    assert captured["candidate_cu"].is_file()
    assert captured["candidate_cu"].read_text(encoding="utf-8") == source
    # Quick eval should target the smallest shape in the default grid.
    assert captured["sample_shape"] == ops.contract.default_shape_grid()[0]
    # build_dir is per-candidate and pre-created so cpp_extension can write into it.
    assert captured["build_dir"] == layout.candidate_build_dir("candidate_000")
    assert captured["build_dir"].is_dir()
    assert resp.data["compile_ok"] is True
    assert resp.data["correctness_ok"] is True
    # Artifacts persisted
    cdir = layout.candidate_dir("candidate_000")
    assert (cdir / "compile.json").is_file()
    assert (cdir / "correctness_quick.json").is_file()


def test_write_candidate_classifies_subprocess_timeout_as_infrastructure(
    layout: RunLayout,
    ops: OperatorOps,
    monkeypatch,
):
    """A `RuntimeError("subprocess timed out before emitting ...")` is a hang,
    not a code bug — surface it as INFRASTRUCTURE_TIMEOUT so the agent stops
    iterating on a healthy kernel and the circuit breaker tracks it separately."""
    def fake_quick(**kw):
        raise RuntimeError("subprocess timed out before emitting '=== QUICK_RESULT ==='")

    import operator_opt_pipe.tools as tools_mod
    monkeypatch.setattr(tools_mod, "compile_and_check_quick", fake_quick)

    tool = WriteCandidateTool(layout, ops, _StubExecutor())
    resp = tool.run({"source": _candidate_source()})

    assert resp.status is ToolStatus.ERROR
    assert resp.error_info["code"] == "infrastructure_timeout"
    assert "NOT a bug in your candidate" in resp.error_info["message"]


def test_write_candidate_classifies_generic_exception_as_execution_error(
    layout: RunLayout,
    ops: OperatorOps,
    monkeypatch,
):
    """A non-timeout RuntimeError still maps to EXECUTION_ERROR — only the
    'timed out before emitting' marker triggers the infrastructure path."""
    def fake_quick(**kw):
        raise RuntimeError("nvcc returned non-zero exit code")

    import operator_opt_pipe.tools as tools_mod
    monkeypatch.setattr(tools_mod, "compile_and_check_quick", fake_quick)

    tool = WriteCandidateTool(layout, ops, _StubExecutor())
    resp = tool.run({"source": _candidate_source()})

    assert resp.status is ToolStatus.ERROR
    assert resp.error_info["code"] == "execution_error"


def test_write_candidate_allocates_fresh_slots(
    layout: RunLayout,
    ops: OperatorOps,
    monkeypatch,
):
    """A second write_candidate gets candidate_001."""
    def fake_quick(**kw):
        return QuickEvalResult(
            candidate_id=kw["candidate_id"], compile_ok=True, correctness_ok=True,
            shape_id="dx", max_abs_err=0.0, rel_l2_err=0.0, compile_log="", diagnostics={},
        )

    import operator_opt_pipe.tools as tools_mod
    monkeypatch.setattr(tools_mod, "compile_and_check_quick", fake_quick)

    source = (
        "#include <torch/extension.h>\n"
        "torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) { return W; }\n"
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"forward\", &forward); }\n"
    )
    tool = WriteCandidateTool(layout, ops, _StubExecutor())
    r1 = tool.run({"source": source})
    r2 = tool.run({"source": source})
    assert r1.data["candidate_id"] == "candidate_000"
    assert r2.data["candidate_id"] == "candidate_001"


def _candidate_source() -> str:
    return (
        "#include <torch/extension.h>\n"
        "torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) { return W; }\n"
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"forward\", &forward); }\n"
    )


def _patch_quick_seq(monkeypatch, errors: list[float], correctness: bool = False):
    """Patch compile_and_check_quick to return a queue of fixed max_abs_err values."""
    import operator_opt_pipe.tools as tools_mod

    queue = list(errors)

    def fake_quick(**kw):
        err = queue.pop(0)
        return QuickEvalResult(
            candidate_id=kw["candidate_id"], compile_ok=True,
            correctness_ok=correctness, shape_id=f"d{kw['sample_shape']}",
            max_abs_err=err, rel_l2_err=err / 10, compile_log="", diagnostics={},
        )

    monkeypatch.setattr(tools_mod, "compile_and_check_quick", fake_quick)


def test_write_candidate_includes_prev_delta(
    layout: RunLayout, ops: OperatorOps, monkeypatch,
):
    """The 2nd candidate's response includes a `prev_max_abs_err` delta line."""
    _patch_quick_seq(monkeypatch, [7.32e-4, 6.87e-4])
    tool = WriteCandidateTool(layout, ops, _StubExecutor())
    src = _candidate_source()
    tool.run({"source": src})
    resp = tool.run({"source": src})
    assert "prev_max_abs_err: 7.320e-04" in resp.text
    assert "delta:" in resp.text
    assert resp.data["prev_max_abs_err"] == pytest.approx(7.32e-4)
    # delta_pct should be ≈ -6.1%
    assert resp.data["delta_pct"] == pytest.approx(-6.1, abs=0.1)


def test_write_candidate_emits_invariance_hint_after_three_flat_failures(
    layout: RunLayout, ops: OperatorOps, monkeypatch,
):
    """3 consecutive correctness failures with <5% delta → emit reduction-order hint."""
    _patch_quick_seq(monkeypatch, [7.32e-4, 7.20e-4, 7.10e-4])
    tool = WriteCandidateTool(layout, ops, _StubExecutor())
    src = _candidate_source()
    tool.run({"source": src})  # candidate_000
    tool.run({"source": src})  # candidate_001 — has 1 prev, no hint yet
    resp = tool.run({"source": src})  # candidate_002 — has 2 prevs, hint fires
    assert "hint:" in resp.text
    assert "reduction-order" in resp.text
    assert resp.data.get("invariance_hint") is True


def test_write_candidate_no_hint_when_delta_large(
    layout: RunLayout, ops: OperatorOps, monkeypatch,
):
    """Big swing in max_abs_err means agent IS exploring — no hint."""
    _patch_quick_seq(monkeypatch, [7.32e-4, 1.4e-3, 2.8e-3])
    tool = WriteCandidateTool(layout, ops, _StubExecutor())
    src = _candidate_source()
    tool.run({"source": src})
    tool.run({"source": src})
    resp = tool.run({"source": src})
    assert "hint:" not in resp.text


def test_write_candidate_no_delta_when_no_prior(
    layout: RunLayout, ops: OperatorOps, monkeypatch,
):
    """First candidate has nothing to compare against — silent skip, no error."""
    _patch_quick_seq(monkeypatch, [7.32e-4])
    tool = WriteCandidateTool(layout, ops, _StubExecutor())
    resp = tool.run({"source": _candidate_source()})
    assert "prev_max_abs_err" not in resp.text
    assert "prev_max_abs_err" not in resp.data


# ---------------------------------------------------------------------------
# SubmitCandidateTool
# ---------------------------------------------------------------------------


def test_submit_candidate_terminates_with_payload(layout: RunLayout):
    # Drop a candidate file so the freeze-state guard passes.
    cu = layout.candidate_file("candidate_005", "candidate.cu")
    cu.parent.mkdir(parents=True, exist_ok=True)
    cu.write_text("// stub", encoding="utf-8")

    tool = SubmitCandidateTool(layout)
    resp = tool.run({
        "candidate_id": "candidate_005",
        "hypothesis": "fused W*X with low-rank correction",
        "experiment_type": "fused-correction",
    })
    assert resp.terminate is True
    payload = resp.terminate_payload
    # SubmitCandidateTool passes parameters through verbatim; status/stage
    # are stamped by the orchestrator boundary in agents._result_to_dict,
    # not by the tool itself, so they aren't in the raw payload.
    assert payload["candidate_id"] == "candidate_005"
    assert payload["hypothesis"] == "fused W*X with low-rank correction"
    assert "status" not in payload
    assert "stage" not in payload


def test_submit_candidate_rejects_missing_file(layout: RunLayout):
    tool = SubmitCandidateTool(layout)
    resp = tool.run({
        "candidate_id": "candidate_999",
        "hypothesis": "h",
        "experiment_type": "t",
    })
    assert resp.status is ToolStatus.ERROR
