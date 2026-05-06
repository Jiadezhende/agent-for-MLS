"""tests/pipeline/test_spec_tools.py — SubmitBenchmarkSpecsTool + StageToolFactory."""
from __future__ import annotations

import json

import pytest

from agent.core.types import AgentContext, MemoryStore
from agent.tools.circuit_breaker import CircuitBreaker
from agent.tools.registry import _Terminated

from pipeline.agent_loop_signal import pop_stage_result, stash_stage_result
from pipeline.state import BENCHMARK_SPEC_SLOTS, Stage, StageResult
from pipeline.tool_factory import StageToolFactory
from pipeline.tools.spec_tools import SubmitBenchmarkSpecsTool
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


def _good_specs() -> dict:
    return {slot: {"slot_name": slot, "samples": 5} for slot in BENCHMARK_SPEC_SLOTS}


# ---------------------------------------------------------------------------
# SubmitBenchmarkSpecsTool
# ---------------------------------------------------------------------------

class TestSubmitBenchmarkSpecsTool:
    def test_writes_all_slots_and_terminates(self, tmp_path):
        layout = _layout(tmp_path)
        tool = SubmitBenchmarkSpecsTool(layout)
        tool._ctx = _ctx()

        with pytest.raises(_Terminated):
            tool.run({"specs": _good_specs()})

        # Each slot file exists with valid JSON.
        for slot in BENCHMARK_SPEC_SLOTS:
            p = layout.benchmark_spec_path(slot)
            assert p.is_file()
            payload = json.loads(p.read_text())
            assert payload["slot_name"] == slot

        # StageResult is stashed with all spec_versions populated.
        result = pop_stage_result(tool._ctx)
        assert result is not None
        assert result.stage == Stage.BENCHMARK_SPEC.value
        assert result.status == "success"
        assert set(result.metrics["spec_versions"].keys()) == set(BENCHMARK_SPEC_SLOTS)
        assert all(v == "v1" for v in result.metrics["spec_versions"].values())
        # Each slot artifact path is recorded as workspace-relative POSIX.
        for slot in BENCHMARK_SPEC_SLOTS:
            assert f"spec_{slot}" in result.artifacts
            assert result.artifacts[f"spec_{slot}"].endswith(f"{slot}.json")

    def test_missing_slot_returns_invalid_args(self, tmp_path):
        layout = _layout(tmp_path)
        tool = SubmitBenchmarkSpecsTool(layout)
        tool._ctx = _ctx()

        partial = {s: {"x": 1} for s in BENCHMARK_SPEC_SLOTS[:3]}  # only 3/5
        resp = tool.run({"specs": partial})

        assert resp.status.value == "error"
        assert "missing required slot" in resp.text.lower()

    def test_specs_must_be_object(self, tmp_path):
        tool = SubmitBenchmarkSpecsTool(_layout(tmp_path))
        tool._ctx = _ctx()
        resp = tool.run({"specs": "not an object"})
        assert resp.status.value == "error"

    def test_extra_slot_is_written_but_not_in_versions(self, tmp_path):
        layout = _layout(tmp_path)
        tool = SubmitBenchmarkSpecsTool(layout)
        tool._ctx = _ctx()

        specs = _good_specs()
        specs["extra_diagnostic"] = {"comment": "informational"}

        with pytest.raises(_Terminated):
            tool.run({"specs": specs})

        # Extra file written.
        assert layout.benchmark_spec_path("extra_diagnostic").is_file()
        # But not in spec_versions.
        result = pop_stage_result(tool._ctx)
        assert "extra_diagnostic" not in result.metrics["spec_versions"]
        assert set(result.metrics["spec_versions"].keys()) == set(BENCHMARK_SPEC_SLOTS)


# ---------------------------------------------------------------------------
# StageToolFactory
# ---------------------------------------------------------------------------

class TestStageToolFactory:
    def test_build_with_no_executor_offers_lite_catalogue(self, tmp_path):
        layout = _layout(tmp_path)
        factory = StageToolFactory(executor=None, layout=layout)
        # Just the no-executor tools should be available.
        reg = factory.build(["read_skill", "submit_benchmark_specs"])
        assert "read_skill" in reg._tools
        assert "submit_benchmark_specs" in reg._tools

    def test_build_rejects_unknown_tool(self, tmp_path):
        factory = StageToolFactory(executor=None, layout=_layout(tmp_path))
        with pytest.raises(ValueError, match="unknown tool"):
            factory.build(["this_tool_does_not_exist"])

    def test_build_without_executor_blocks_executor_tools(self, tmp_path):
        factory = StageToolFactory(executor=None, layout=_layout(tmp_path))
        # run_cuda_probe is executor-dependent; without executor it's absent.
        with pytest.raises(ValueError, match="unknown tool"):
            factory.build(["run_cuda_probe"])

    def test_build_with_executor_includes_executor_tools(self, tmp_path):
        # We don't need a real Executor — any non-None object suffices because
        # the executor is just held by reference and passed into tool ctors.
        # The tool ctors themselves accept an opaque executor handle.
        class _DummyExecutor:
            workspace = None  # used by some tool internals; tests don't hit them
        factory = StageToolFactory(executor=_DummyExecutor(), layout=_layout(tmp_path))
        reg = factory.build(["read_skill", "submit_benchmark_specs", "run_cuda_probe"])
        assert "run_cuda_probe" in reg._tools

    def test_each_build_creates_fresh_instances(self, tmp_path):
        factory = StageToolFactory(executor=None, layout=_layout(tmp_path))
        reg1 = factory.build(["submit_benchmark_specs"])
        reg2 = factory.build(["submit_benchmark_specs"])
        assert reg1._tools["submit_benchmark_specs"] is not reg2._tools["submit_benchmark_specs"]


# ---------------------------------------------------------------------------
# stash/pop helpers
# ---------------------------------------------------------------------------

class TestStageResultStash:
    def test_roundtrip(self):
        ctx = _ctx()
        original = StageResult(
            stage=Stage.HARDWARE_PROFILE.value,
            status="partial",
            artifacts={"a": "b"},
            metrics={"x": 1},
            confidence=0.6,
            caveats=["unstable"],
        )
        stash_stage_result(ctx, original)
        retrieved = pop_stage_result(ctx)
        assert retrieved is not None
        assert retrieved.stage == original.stage
        assert retrieved.status == original.status
        assert retrieved.artifacts == original.artifacts
        assert retrieved.metrics == original.metrics
        assert retrieved.confidence == 0.6
        assert retrieved.caveats == ["unstable"]

    def test_pop_when_nothing_stashed_returns_none(self):
        ctx = _ctx()
        assert pop_stage_result(ctx) is None
