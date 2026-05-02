"""tests/pipeline/test_stage_runner.py — generic stage execution driver."""
from __future__ import annotations

import time
from typing import Sequence

import pytest

from pipeline.stage_agent import StageAgent
from pipeline.stage_runner import StageContext, run_stage
from pipeline.state import RunState, Stage, StageResult
from pipeline.workspace_layout import RunLayout


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _DummyTools:
    """Stand-in for a ToolRegistry that records construction args."""

    def __init__(self, tool_names: Sequence[str]):
        self.tool_names: tuple[str, ...] = tuple(tool_names)


def _build_tools_recording(seen: list[tuple[str, ...]]):
    def build(names: Sequence[str]):
        seen.append(tuple(names))
        return _DummyTools(names)

    return build


def _build_tools_raising(exc: Exception):
    def build(_names: Sequence[str]):
        raise exc

    return build


def _layout(tmp_path) -> RunLayout:
    layout = RunLayout(tmp_path, "run_test")
    layout.mkdir()
    return layout


def _run_state() -> RunState:
    return RunState(run_id="run_test", operator="lora_matmul")


# ---------------------------------------------------------------------------
# StageAgents used in tests
# ---------------------------------------------------------------------------

class _SuccessAgent(StageAgent):
    stage = Stage.HARDWARE_PROFILE
    allowed_tools = ("read_skill", "run_cuda_probe")

    def run(self, context: StageContext) -> StageResult:
        # Sanity: stage_runner should hand us a built tools object whose
        # tool_names exactly equals our allowed_tools declaration.
        assert isinstance(context.tools, _DummyTools)
        assert context.tools.tool_names == self.allowed_tools
        return StageResult(
            stage=self.stage.value,
            status="success",
            metrics={"dram_bandwidth_gbps": 280.0},
            confidence=0.9,
        )


class _RaisingAgent(StageAgent):
    stage = Stage.HARDWARE_PROFILE
    allowed_tools = ()

    def run(self, context: StageContext) -> StageResult:
        raise RuntimeError("boom from inside the agent")


class _WrongStageAgent(StageAgent):
    stage = Stage.HARDWARE_PROFILE
    allowed_tools = ()

    def run(self, context: StageContext) -> StageResult:
        # Returns a result tagged with a *different* stage — must be rejected.
        return StageResult(stage=Stage.BASELINE_PROFILE.value, status="success")


class _BadSchemaAgent(StageAgent):
    stage = Stage.HARDWARE_PROFILE
    allowed_tools = ()

    def run(self, context: StageContext) -> StageResult:
        bad = StageResult(stage=self.stage.value, status="success")
        bad.confidence = 5.0  # out of range — schema validation will catch it
        return bad


class _NonStageResultAgent(StageAgent):
    stage = Stage.HARDWARE_PROFILE
    allowed_tools = ()

    def run(self, context: StageContext) -> StageResult:  # type: ignore[override]
        return {"stage": "HARDWARE_PROFILE", "status": "success"}  # type: ignore[return-value]


class _SlowAgent(StageAgent):
    stage = Stage.HARDWARE_PROFILE
    allowed_tools = ()

    def run(self, context: StageContext) -> StageResult:
        time.sleep(0.05)  # well beyond the 0.001s budget below
        return StageResult(stage=self.stage.value, status="success")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestRunStageSuccess:
    def test_success_returns_validated_result(self, tmp_path):
        seen: list[tuple[str, ...]] = []
        result = run_stage(
            _SuccessAgent(),
            run_state=_run_state(),
            layout=_layout(tmp_path),
            build_tools=_build_tools_recording(seen),
            stage_budget_s=10.0,
        )
        assert result.status == "success"
        assert result.stage == Stage.HARDWARE_PROFILE.value
        assert result.metrics["dram_bandwidth_gbps"] == 280.0
        # build_tools should have been invoked with exactly allowed_tools
        assert seen == [("read_skill", "run_cuda_probe")]

    def test_within_budget_no_overrun_caveat(self, tmp_path):
        result = run_stage(
            _SuccessAgent(),
            run_state=_run_state(),
            layout=_layout(tmp_path),
            build_tools=_build_tools_recording([]),
            stage_budget_s=10.0,
        )
        assert not any("stage_overran" in c for c in result.caveats)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

class TestFailureHandling:
    def test_agent_exception_becomes_failed_result(self, tmp_path):
        result = run_stage(
            _RaisingAgent(),
            run_state=_run_state(),
            layout=_layout(tmp_path),
            build_tools=_build_tools_recording([]),
            stage_budget_s=10.0,
        )
        assert result.status == "failed"
        assert result.stage == Stage.HARDWARE_PROFILE.value
        assert result.confidence == 0.0
        assert any("agent_exception" in c for c in result.caveats)
        assert any("RuntimeError" in c for c in result.caveats)

    def test_tool_build_error_becomes_failed_result(self, tmp_path):
        result = run_stage(
            _SuccessAgent(),
            run_state=_run_state(),
            layout=_layout(tmp_path),
            build_tools=_build_tools_raising(KeyError("unknown_tool")),
            stage_budget_s=10.0,
        )
        assert result.status == "failed"
        assert any("tool_build_error" in c for c in result.caveats)

    def test_stage_tag_mismatch_becomes_failed_result(self, tmp_path):
        result = run_stage(
            _WrongStageAgent(),
            run_state=_run_state(),
            layout=_layout(tmp_path),
            build_tools=_build_tools_recording([]),
            stage_budget_s=10.0,
        )
        assert result.status == "failed"
        assert any("stage_tag_mismatch" in c for c in result.caveats)

    def test_schema_violation_becomes_failed_result(self, tmp_path):
        result = run_stage(
            _BadSchemaAgent(),
            run_state=_run_state(),
            layout=_layout(tmp_path),
            build_tools=_build_tools_recording([]),
            stage_budget_s=10.0,
        )
        assert result.status == "failed"
        assert any("schema_invalid" in c for c in result.caveats)

    def test_non_stage_result_return_becomes_failed_result(self, tmp_path):
        result = run_stage(
            _NonStageResultAgent(),
            run_state=_run_state(),
            layout=_layout(tmp_path),
            build_tools=_build_tools_recording([]),
            stage_budget_s=10.0,
        )
        assert result.status == "failed"
        assert any("agent_returned_non_StageResult" in c for c in result.caveats)


# ---------------------------------------------------------------------------
# Soft budget monitoring
# ---------------------------------------------------------------------------

class TestBudgetSoftMonitor:
    def test_overrun_adds_caveat_but_keeps_status(self, tmp_path):
        # Budget is 1ms but the agent sleeps 50ms. Result should still be
        # success (we don't hard-cancel), but with an over-run caveat.
        result = run_stage(
            _SlowAgent(),
            run_state=_run_state(),
            layout=_layout(tmp_path),
            build_tools=_build_tools_recording([]),
            stage_budget_s=0.001,
        )
        assert result.status == "success"
        assert any("stage_overran" in c for c in result.caveats)


# ---------------------------------------------------------------------------
# Tool authorization — declared tools end up in the registry, others don't
# ---------------------------------------------------------------------------

class TestToolAuthorization:
    def test_agent_only_receives_declared_tools(self, tmp_path):
        seen: list[tuple[str, ...]] = []
        run_stage(
            _SuccessAgent(),  # declares ("read_skill", "run_cuda_probe")
            run_state=_run_state(),
            layout=_layout(tmp_path),
            build_tools=_build_tools_recording(seen),
            stage_budget_s=10.0,
        )
        # Agent's allowed_tools is what gets passed to build_tools — never more.
        assert seen == [("read_skill", "run_cuda_probe")]

    def test_empty_allowed_tools_passes_empty_tuple(self, tmp_path):
        seen: list[tuple[str, ...]] = []
        run_stage(
            _RaisingAgent(),  # allowed_tools = ()
            run_state=_run_state(),
            layout=_layout(tmp_path),
            build_tools=_build_tools_recording(seen),
            stage_budget_s=10.0,
        )
        assert seen == [()]


# ---------------------------------------------------------------------------
# StageContext shape
# ---------------------------------------------------------------------------

class TestStageContextDelivery:
    def test_context_carries_layout_state_and_budget(self, tmp_path):
        captured: dict = {}

        class _Capture(StageAgent):
            stage = Stage.HARDWARE_PROFILE
            allowed_tools = ()

            def run(self, ctx: StageContext) -> StageResult:
                captured["budget"] = ctx.stage_budget_s
                captured["run_id"] = ctx.run_state.run_id
                captured["root"] = ctx.layout.root
                captured["verbose"] = ctx.verbose
                return StageResult(stage=self.stage.value, status="success")

        layout = _layout(tmp_path)
        run_stage(
            _Capture(),
            run_state=_run_state(),
            layout=layout,
            build_tools=_build_tools_recording([]),
            stage_budget_s=42.0,
            verbose=True,
        )
        assert captured["budget"] == 42.0
        assert captured["run_id"] == "run_test"
        assert captured["root"] == layout.root
        assert captured["verbose"] is True
