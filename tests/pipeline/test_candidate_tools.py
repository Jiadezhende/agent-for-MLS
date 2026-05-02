"""tests/pipeline/test_candidate_tools.py — Write/Evaluate/SubmitCandidateResult."""
from __future__ import annotations

import json

import pytest

from agents.core.types import AgentContext, MemoryStore
from agents.tools.circuit_breaker import CircuitBreaker
from agents.tools.registry import _Terminated

from pipeline.agent_loop_signal import pop_stage_result
from pipeline.state import Stage, StageResult
from pipeline.tools.candidate_tools import (
    EvaluateCandidateTool,
    SubmitCandidateResultTool,
    WriteCandidateTool,
    _EVAL_MARKER,
    _build_eval_script,
    _next_candidate_index,
)
from pipeline.workspace_layout import RunLayout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_SOURCE = """
#include <torch/extension.h>
torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {
    return W;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "lora");
}
""".strip() + "\n" + ("// padding so source is >100 chars\n" * 3)


def _layout(tmp_path) -> RunLayout:
    layout = RunLayout(tmp_path, "run_test")
    layout.mkdir()
    return layout


def _ctx() -> AgentContext:
    return AgentContext(memory=MemoryStore(), circuit_breaker=CircuitBreaker())


def _seed_baseline(layout: RunLayout, d_list):
    """Pretend BaselineAgent ran: write empty .pt-named files for each d."""
    for d in d_list:
        layout.baseline_input_path("W", d).write_bytes(b"x")
        layout.baseline_input_path("X", d).write_bytes(b"x")
        layout.baseline_input_path("A", d).write_bytes(b"x")
        layout.baseline_input_path("B", d).write_bytes(b"x")
        layout.baseline_reference_path(d).write_bytes(b"x")


class _MockExecutor:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def profile_with_torch(self, code, op_name, timeout_s=120):
        self.calls.append((code, op_name, timeout_s))
        return self.output


def _eval_stdout(parsed: dict) -> str:
    return f"some prelude\n{_EVAL_MARKER}\n{json.dumps(parsed)}\n"


# ---------------------------------------------------------------------------
# WriteCandidateTool
# ---------------------------------------------------------------------------

class TestWriteCandidateTool:
    def test_first_write_allocates_candidate_000(self, tmp_path):
        layout = _layout(tmp_path)
        tool = WriteCandidateTool(layout)
        tool._ctx = _ctx()
        resp = tool.run({"source": VALID_SOURCE})
        assert resp.status.value == "success"
        assert resp.data["candidate_id"] == "candidate_000"
        cu = layout.candidate_file("candidate_000", "candidate.cu")
        assert cu.is_file()
        assert "PYBIND11_MODULE" in cu.read_text()

    def test_second_write_increments_index(self, tmp_path):
        layout = _layout(tmp_path)
        tool = WriteCandidateTool(layout)
        tool._ctx = _ctx()
        tool.run({"source": VALID_SOURCE})
        resp2 = tool.run({"source": VALID_SOURCE})
        assert resp2.data["candidate_id"] == "candidate_001"

    def test_indexing_handles_gaps_by_continuing_from_max(self, tmp_path):
        layout = _layout(tmp_path)
        # Manually create candidate_000, candidate_004 → next should be 005
        for idx in (0, 4):
            layout.candidate_dir(f"candidate_{idx:03d}").mkdir(parents=True)
        assert _next_candidate_index(layout) == 5

    def test_rejects_source_without_pybind(self, tmp_path):
        tool = WriteCandidateTool(_layout(tmp_path))
        tool._ctx = _ctx()
        resp = tool.run({"source": "// no pybind here\n" * 10})
        assert resp.status.value == "error"


# ---------------------------------------------------------------------------
# Eval script generation
# ---------------------------------------------------------------------------

class TestEvalScriptGen:
    def test_script_embeds_paths_d_and_samples(self, tmp_path):
        layout = _layout(tmp_path)
        layout.candidate_dir("candidate_007").mkdir(parents=True)
        cu = layout.candidate_file("candidate_007", "candidate.cu")
        cu.write_text(VALID_SOURCE)

        code = _build_eval_script(
            candidate_id="candidate_007",
            candidate_cu=cu,
            inputs_dir=layout.baseline_inputs_dir,
            refs_dir=layout.baseline_references_dir,
            d_list=[3584, 4608],
            samples=5,
        )
        assert "candidate.cu" in code
        assert "[3584, 4608]" in code
        assert "SAMPLES   = 5" in code
        assert _EVAL_MARKER in code
        # Module name is sanitized (no dashes).
        assert 'CAND_NAME = "cand_candidate_007"' in code


# ---------------------------------------------------------------------------
# EvaluateCandidateTool
# ---------------------------------------------------------------------------

class TestEvaluateCandidateTool:
    def _setup(self, tmp_path, *, d_list, parsed_output):
        layout = _layout(tmp_path)
        # Pretend a candidate.cu was already written.
        layout.candidate_dir("candidate_000").mkdir(parents=True)
        layout.candidate_file("candidate_000", "candidate.cu").write_text(VALID_SOURCE)
        # And BaselineAgent ran.
        _seed_baseline(layout, d_list)
        exec_ = _MockExecutor(output={"stdout": _eval_stdout(parsed_output)})
        tool = EvaluateCandidateTool(executor=exec_, layout=layout)
        tool._ctx = _ctx()
        return layout, tool, exec_

    def test_quick_mode_parses_summary_and_writes_artifacts(self, tmp_path):
        parsed = {
            "compile_ok": True,
            "compile_error": None,
            "all_correct": True,
            "per_d": [
                {"d": 3584, "correctness_ok": True, "max_abs_diff": 1e-5,
                 "candidate_ms_median": 5.0, "ref_ms_median": 7.5, "speedup": 1.5,
                 "samples": 5, "candidate_ms_min": 4.9, "candidate_ms_max": 5.1,
                 "ref_ms_min": 7.4, "ref_ms_max": 7.6, "error": None},
                {"d": 4096, "correctness_ok": True, "max_abs_diff": 2e-5,
                 "candidate_ms_median": 6.0, "ref_ms_median": 9.0, "speedup": 1.5,
                 "samples": 5, "candidate_ms_min": 5.9, "candidate_ms_max": 6.1,
                 "ref_ms_min": 8.9, "ref_ms_max": 9.1, "error": None},
            ],
            "overall_speedup_median": 1.5,
        }
        layout, tool, exec_ = self._setup(tmp_path, d_list=[3584, 4096], parsed_output=parsed)

        resp = tool.run({"candidate_id": "candidate_000", "d_list": [3584, 4096], "mode": "quick"})
        assert resp.status.value == "success"
        s = resp.data["summary"]
        assert s["compile_ok"] is True
        assert s["all_correct"] is True
        assert s["speedup_median"] == 1.5
        assert s["samples"] == 5

        # quick_benchmark.json + compile.json + correctness.json on disk.
        cdir = layout.candidate_dir("candidate_000")
        assert (cdir / "quick_benchmark.json").is_file()
        assert (cdir / "compile.json").is_file()
        compile_payload = json.loads((cdir / "compile.json").read_text())
        assert compile_payload["ok"] is True

        # Op name reflects candidate + mode for log differentiation.
        op_name = exec_.calls[0][1]
        assert op_name == "eval_candidate_000_quick"

    def test_confirm_mode_uses_30_samples_and_writes_confirm_file(self, tmp_path):
        parsed = {
            "compile_ok": True,
            "all_correct": True,
            "per_d": [
                {"d": 4096, "correctness_ok": True, "max_abs_diff": 1e-5,
                 "candidate_ms_median": 5.0, "ref_ms_median": 6.5, "speedup": 1.3,
                 "samples": 30, "candidate_ms_min": 4.9, "candidate_ms_max": 5.1,
                 "ref_ms_min": 6.4, "ref_ms_max": 6.6, "error": None},
            ],
            "overall_speedup_median": 1.3,
        }
        layout, tool, exec_ = self._setup(tmp_path, d_list=[4096], parsed_output=parsed)

        resp = tool.run({"candidate_id": "candidate_000", "d_list": [4096], "mode": "confirm"})
        assert resp.status.value == "success"
        assert resp.data["samples"] == 30
        assert (layout.candidate_dir("candidate_000") / "confirm_benchmark.json").is_file()
        assert exec_.calls[0][1].endswith("_confirm")

    def test_compile_failure_surfaces_in_summary(self, tmp_path):
        parsed = {
            "compile_ok": False,
            "compile_error": "nvcc fatal: missing -arch flag\n",
            "all_correct": False,
            "per_d": [],
            "overall_speedup_median": None,
        }
        _, tool, _ = self._setup(tmp_path, d_list=[4096], parsed_output=parsed)
        resp = tool.run({"candidate_id": "candidate_000", "d_list": [4096], "mode": "quick"})
        assert resp.status.value == "success"  # tool ran fine; subprocess reported compile fail
        s = resp.data["summary"]
        assert s["compile_ok"] is False
        assert resp.data["compile_error"] is not None

    def test_missing_candidate_file_returns_invalid_args(self, tmp_path):
        layout = _layout(tmp_path)
        _seed_baseline(layout, [4096])
        exec_ = _MockExecutor(output={"stdout": ""})
        tool = EvaluateCandidateTool(executor=exec_, layout=layout)
        tool._ctx = _ctx()
        resp = tool.run({"candidate_id": "candidate_999", "d_list": [4096]})
        assert resp.status.value == "error"
        assert "candidate file missing" in resp.text.lower()

    def test_missing_baseline_refs_returns_invalid_args(self, tmp_path):
        layout = _layout(tmp_path)
        layout.candidate_dir("candidate_000").mkdir(parents=True)
        layout.candidate_file("candidate_000", "candidate.cu").write_text(VALID_SOURCE)
        # don't seed baseline
        exec_ = _MockExecutor(output={"stdout": ""})
        tool = EvaluateCandidateTool(executor=exec_, layout=layout)
        tool._ctx = _ctx()
        resp = tool.run({"candidate_id": "candidate_000", "d_list": [4096]})
        assert resp.status.value == "error"
        assert "baseline references missing" in resp.text.lower()


# ---------------------------------------------------------------------------
# SubmitCandidateResultTool
# ---------------------------------------------------------------------------

class TestSubmitCandidateResultTool:
    def test_best_update_record_stashes_proper_stage_result(self, tmp_path):
        layout = _layout(tmp_path)
        # Need a candidate dir so artifact path resolves cleanly.
        layout.candidate_dir("candidate_002").mkdir(parents=True)
        tool = SubmitCandidateResultTool(layout=layout, stage=Stage.TUNING_LOOP)
        tool._ctx = _ctx()

        record = {
            "candidate_id": "candidate_002",
            "compile_ok": True,
            "correctness_ok": True,
            "quick_speedup_median": 1.21,
            "quick_samples": 5,
            "quick_variance_pct": 4.0,
            "confirm_speedup_median": 1.34,
            "confirm_samples": 30,
            "confirm_variance_pct": 3.0,
            "accepted_for": "best_update",
        }
        with pytest.raises(_Terminated):
            tool.run({"record": record, "confidence": 0.92})

        result = pop_stage_result(tool._ctx)
        assert result is not None
        assert result.stage == Stage.TUNING_LOOP.value
        assert result.metrics["candidate"]["candidate_id"] == "candidate_002"
        assert result.metrics["candidate"]["accepted_for"] == "best_update"
        assert result.metrics["candidate"]["confirm_speedup_median"] == 1.34
        # Timestamp filled in
        assert "timestamp" in result.metrics["candidate"]

    def test_initial_candidate_stage_tag(self, tmp_path):
        layout = _layout(tmp_path)
        layout.candidate_dir("candidate_000").mkdir(parents=True)
        tool = SubmitCandidateResultTool(layout=layout, stage=Stage.INITIAL_CANDIDATE)
        tool._ctx = _ctx()
        record = {
            "candidate_id": "candidate_000",
            "compile_ok": True,
            "correctness_ok": True,
            "accepted_for": "best_update",
        }
        with pytest.raises(_Terminated):
            tool.run({"record": record})
        result = pop_stage_result(tool._ctx)
        assert result.stage == Stage.INITIAL_CANDIDATE.value

    def test_best_update_rejected_when_compile_failed(self, tmp_path):
        layout = _layout(tmp_path)
        layout.candidate_dir("candidate_003").mkdir(parents=True)
        tool = SubmitCandidateResultTool(layout=layout, stage=Stage.TUNING_LOOP)
        tool._ctx = _ctx()
        record = {
            "candidate_id": "candidate_003",
            "compile_ok": False,
            "correctness_ok": False,
            "accepted_for": "best_update",
        }
        resp = tool.run({"record": record})
        assert resp.status.value == "error"
        assert "best_update" in resp.text.lower()

    def test_strategy_guidance_record_passes(self, tmp_path):
        layout = _layout(tmp_path)
        layout.candidate_dir("candidate_004").mkdir(parents=True)
        tool = SubmitCandidateResultTool(layout=layout, stage=Stage.TUNING_LOOP)
        tool._ctx = _ctx()
        record = {
            "candidate_id": "candidate_004",
            "compile_ok": True,
            "correctness_ok": True,
            "quick_speedup_median": 0.85,
            "quick_samples": 5,
            "accepted_for": "strategy_guidance",
        }
        with pytest.raises(_Terminated):
            tool.run({"record": record})
        result = pop_stage_result(tool._ctx)
        assert result.metrics["candidate"]["accepted_for"] == "strategy_guidance"
