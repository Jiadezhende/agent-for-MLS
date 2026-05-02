"""tests/pipeline/test_baseline_tools.py — GenerateBaselineTool + SubmitBaselineTool."""
from __future__ import annotations

import json

import pytest

from agents.core.types import AgentContext, MemoryStore
from agents.tools.circuit_breaker import CircuitBreaker
from agents.tools.registry import _Terminated

from pipeline.agent_loop_signal import pop_stage_result
from pipeline.state import Stage
from pipeline.tools.baseline_tools import (
    GenerateBaselineTool,
    SubmitBaselineTool,
    _BASELINE_MARKER,
    _build_baseline_script,
)
from pipeline.workspace_layout import RunLayout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layout(tmp_path) -> RunLayout:
    layout = RunLayout(tmp_path, "run_test")
    layout.mkdir()
    return layout


def _ctx() -> AgentContext:
    return AgentContext(memory=MemoryStore(), circuit_breaker=CircuitBreaker())


class _MockExecutor:
    """Records profile_with_torch calls and returns a canned output."""

    def __init__(self, output: dict):
        self.output = output
        self.calls: list[tuple] = []

    def profile_with_torch(self, code, op_name, timeout_s=120):
        self.calls.append((code, op_name, timeout_s))
        return self.output


def _baseline_stdout(per_d_records: list[dict]) -> str:
    return f"some prelude\n{_BASELINE_MARKER}\n{json.dumps(per_d_records)}\n"


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------

class TestBaselineScriptGen:
    def test_script_embeds_paths_and_d_list(self, tmp_path):
        layout = _layout(tmp_path)
        code = _build_baseline_script(
            inputs_dir=layout.baseline_inputs_dir,
            refs_dir=layout.baseline_references_dir,
            d_list=[3584, 4096],
            samples=10,
            seed=42,
        )
        # Paths are baked in absolutely.
        assert layout.baseline_inputs_dir.resolve().as_posix() in code
        assert layout.baseline_references_dir.resolve().as_posix() in code
        # Parameters embedded as literals.
        assert "[3584, 4096]" in code
        assert "SAMPLES   = 10" in code
        assert "SEED      = 42" in code
        # Marker present so parser can find the JSON.
        assert _BASELINE_MARKER in code


# ---------------------------------------------------------------------------
# GenerateBaselineTool
# ---------------------------------------------------------------------------

class TestGenerateBaselineTool:
    def test_success_parses_per_d_records(self, tmp_path):
        layout = _layout(tmp_path)
        records = [
            {"d": 3584, "torch_ms_median": 12.3, "samples": 30},
            {"d": 4096, "torch_ms_median": 14.1, "samples": 30},
        ]
        exec_ = _MockExecutor(output={"stdout": _baseline_stdout(records), "stdout_tail": _baseline_stdout(records)})
        tool = GenerateBaselineTool(executor=exec_, layout=layout)
        tool._ctx = _ctx()

        resp = tool.run({"d_list": [3584, 4096], "samples": 30})
        assert resp.status.value == "success"
        assert resp.data["per_d"] == records
        assert resp.data["d_list"] == [3584, 4096]
        assert resp.data["samples"] == 30

        # The mock executor was invoked with op_name=generate_baseline.
        assert len(exec_.calls) == 1
        _code, op_name, _timeout = exec_.calls[0]
        assert op_name == "generate_baseline"

    def test_executor_error_returns_error_response(self, tmp_path):
        class _BoomExecutor:
            def profile_with_torch(self, *args, **kwargs):
                raise RuntimeError("ncu_permission_denied")

        layout = _layout(tmp_path)
        tool = GenerateBaselineTool(executor=_BoomExecutor(), layout=layout)
        tool._ctx = _ctx()
        resp = tool.run({"d_list": [4096]})
        assert resp.status.value == "error"
        assert "subprocess failed" in resp.text.lower()

    def test_no_marker_in_stdout_returns_error(self, tmp_path):
        exec_ = _MockExecutor(output={"stdout": "just gibberish\nno marker here\n"})
        tool = GenerateBaselineTool(executor=exec_, layout=_layout(tmp_path))
        tool._ctx = _ctx()
        resp = tool.run({"d_list": [4096]})
        assert resp.status.value == "error"
        assert "no parseable json" in resp.text.lower()

    def test_subprocess_reports_cuda_unavailable(self, tmp_path):
        # The script itself emits {"error": "cuda_unavailable"} when GPU absent.
        bad = {"stdout": f"{_BASELINE_MARKER}\n{json.dumps({'error': 'cuda_unavailable'})}\n"}
        exec_ = _MockExecutor(output=bad)
        tool = GenerateBaselineTool(executor=exec_, layout=_layout(tmp_path))
        tool._ctx = _ctx()
        resp = tool.run({"d_list": [4096]})
        assert resp.status.value == "error"
        assert "cuda_unavailable" in resp.text


# ---------------------------------------------------------------------------
# SubmitBaselineTool
# ---------------------------------------------------------------------------

class TestSubmitBaselineTool:
    def test_writes_baseline_json_and_terminates(self, tmp_path):
        layout = _layout(tmp_path)
        tool = SubmitBaselineTool(layout)
        tool._ctx = _ctx()

        per_d = [
            {"d": 3584, "torch_ms_median": 12.0, "samples": 30},
            {"d": 4096, "torch_ms_median": 14.0, "samples": 30},
        ]

        with pytest.raises(_Terminated):
            tool.run({"per_d": per_d, "notes": "two d values"})

        # baseline.json on disk
        payload = json.loads(layout.baseline_path.read_text())
        assert payload["per_d"] == per_d
        assert payload["notes"] == "two d values"

        # StageResult stashed properly
        result = pop_stage_result(tool._ctx)
        assert result is not None
        assert result.stage == Stage.BASELINE_PROFILE.value
        assert result.status == "success"
        assert result.metrics["n_d"] == 2
