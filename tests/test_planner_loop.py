"""
tests/test_planner_loop.py — Tests for PlannerAgent, RunSubagentTool, MarkReadyForCriticTool.

All tests run without a GPU.
"""
from __future__ import annotations

import uuid

import pytest

from agents._registry import AgentDefinition
from agents.core.config import AgentConfig
from agents.core.types import AgentContext, MemoryStore, Step, Task, WorkerOutput
from agents.tools.builtin.subagent import MarkReadyForCriticTool, RunSubagentParallelTool, RunSubagentTool
from agents.tools.circuit_breaker import CircuitBreaker
from agents.tools.registry import ToolRegistry, _Terminated
from agents.tools.response import ToolStatus


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

def _make_ctx() -> AgentContext:
    task = Task(id=str(uuid.uuid4()), type="coordination",
                description="test", payload={}, constraints={})
    return AgentContext(task=task, memory=MemoryStore(), circuit_breaker=CircuitBreaker())


def _make_agent_def(agent_class) -> AgentDefinition:
    return AgentDefinition(
        agent_type="fake_probe",
        description="fake worker",
        agent_class=agent_class,
        required_tools=[],
        critic_system_prompt="",
        critic_tool_schema={},
    )


class _FakeSuccessAgent:
    """Stub worker agent: returns a WorkerOutput with one result."""
    def __init__(self, llm, agent_cfg, verbose=False):
        pass

    def run(self, step: Step, tools):
        return WorkerOutput(
            step_id=step.id,
            results=[{"metric": step.targets[0], "value": 42, "confidence": 0.9,
                      "unit": "unit", "method": "fake", "evidence": ["stub"]}],
            success=True,
            targets_requested=step.targets,
            summary="fake result",
        )


class _FakeFailAgent:
    """Stub worker agent: returns failure."""
    def __init__(self, llm, agent_cfg, verbose=False):
        pass

    def run(self, step: Step, tools):
        return WorkerOutput(
            step_id=step.id,
            results=[],
            success=False,
            targets_requested=step.targets,
            summary="failed",
        )


class _FakeLLM:
    pass


class _FakeExecutor:
    detect_notes: list = []


# ---------------------------------------------------------------------------
# RunSubagentTool tests
# ---------------------------------------------------------------------------

class TestRunSubagentTool:

    def _make_tool(self, agent_class=_FakeSuccessAgent):
        agent_def = _make_agent_def(agent_class)
        registry = {"fake_probe": agent_def}
        tool = RunSubagentTool(
            llm=_FakeLLM(),
            executor=_FakeExecutor(),
            agent_registry=registry,
            agent_cfg=AgentConfig(max_iterations=5),
        )
        return tool

    def test_successful_run_returns_success_response(self):
        tool = self._make_tool()
        ctx = _make_ctx()
        tool._ctx = ctx

        resp = tool.run({"agent_type": "fake_probe", "targets": ["dram_bandwidth_gbps"]})

        assert resp.status == ToolStatus.SUCCESS
        assert "fake_probe" in resp.text
        assert resp.data["success"] is True
        assert "dram_bandwidth_gbps" in resp.data["targets_measured"]

    def test_appends_to_job_history(self):
        tool = self._make_tool()
        ctx = _make_ctx()
        tool._ctx = ctx

        tool.run({"agent_type": "fake_probe", "targets": ["boost_clock_mhz"]})

        assert len(ctx.job_history) == 1
        entry = ctx.job_history[0]
        assert entry["agent_type"] == "fake_probe"
        assert entry["targets_requested"] == ["boost_clock_mhz"]
        assert entry["success"] is True

    def test_multiple_calls_accumulate_history(self):
        tool = self._make_tool()
        ctx = _make_ctx()
        tool._ctx = ctx

        tool.run({"agent_type": "fake_probe", "targets": ["metric_a"]})
        tool.run({"agent_type": "fake_probe", "targets": ["metric_b"]})

        assert len(ctx.job_history) == 2
        metrics = [e["targets_requested"][0] for e in ctx.job_history]
        assert "metric_a" in metrics
        assert "metric_b" in metrics

    def test_unknown_agent_type_returns_error(self):
        tool = self._make_tool()
        ctx = _make_ctx()
        tool._ctx = ctx

        resp = tool.run({"agent_type": "nonexistent", "targets": ["x"]})

        assert resp.status == ToolStatus.ERROR
        assert len(ctx.job_history) == 0

    def test_failed_agent_still_appends_history(self):
        tool = self._make_tool(agent_class=_FakeFailAgent)
        ctx = _make_ctx()
        tool._ctx = ctx

        resp = tool.run({"agent_type": "fake_probe", "targets": ["some_metric"]})

        assert resp.status == ToolStatus.SUCCESS  # ToolResponse is success (we got a response)
        assert resp.data["success"] is False
        assert len(ctx.job_history) == 1
        assert ctx.job_history[0]["success"] is False

    def test_missing_targets_reported(self):
        tool = self._make_tool(agent_class=_FakeFailAgent)
        ctx = _make_ctx()
        tool._ctx = ctx

        resp = tool.run({"agent_type": "fake_probe", "targets": ["metric_x"]})

        assert "metric_x" in resp.data["missing_targets"]

    def test_works_without_ctx_injection(self):
        """Tool should work even if _ctx is None (no job_history accumulation)."""
        tool = self._make_tool()
        # No _ctx injection

        resp = tool.run({"agent_type": "fake_probe", "targets": ["x"]})
        assert resp.status == ToolStatus.SUCCESS


# ---------------------------------------------------------------------------
# RunSubagentParallelTool tests
# ---------------------------------------------------------------------------

class TestRunSubagentParallelTool:

    def _make_tool(self, agent_class=_FakeSuccessAgent, extra_types: dict | None = None):
        agent_def = _make_agent_def(agent_class)
        registry = {"fake_probe": agent_def}
        if extra_types:
            registry.update(extra_types)
        tool = RunSubagentParallelTool(
            llm=_FakeLLM(),
            executor=_FakeExecutor(),
            agent_registry=registry,
            agent_cfg=AgentConfig(max_iterations=5),
        )
        return tool

    def test_runs_two_calls_and_accumulates_history(self):
        tool = self._make_tool()
        ctx = _make_ctx()
        tool._ctx = ctx

        resp = tool.run({"calls": [
            {"agent_type": "fake_probe", "targets": ["metric_a"]},
            {"agent_type": "fake_probe", "targets": ["metric_b"]},
        ]})

        assert resp.status == ToolStatus.SUCCESS
        assert len(ctx.job_history) == 2
        measured_all = {
            m
            for entry in ctx.job_history
            for r in entry["results"]
            for m in [r.get("metric")]
            if m
        }
        assert "metric_a" in measured_all
        assert "metric_b" in measured_all

    def test_all_success_flag(self):
        tool = self._make_tool()
        ctx = _make_ctx()
        tool._ctx = ctx

        resp = tool.run({"calls": [
            {"agent_type": "fake_probe", "targets": ["x"]},
            {"agent_type": "fake_probe", "targets": ["y"]},
        ]})

        assert resp.data["all_success"] is True

    def test_partial_failure_reported(self):
        fail_def = _make_agent_def(_FakeFailAgent)
        fail_def2 = _make_agent_def(_FakeSuccessAgent)
        registry = {"fail_probe": fail_def, "ok_probe": fail_def2}
        tool = RunSubagentParallelTool(
            llm=_FakeLLM(), executor=_FakeExecutor(),
            agent_registry=registry, agent_cfg=AgentConfig(max_iterations=5),
        )
        ctx = _make_ctx()
        tool._ctx = ctx

        resp = tool.run({"calls": [
            {"agent_type": "fail_probe", "targets": ["m1"]},
            {"agent_type": "ok_probe",   "targets": ["m2"]},
        ]})

        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["all_success"] is False
        assert len(ctx.job_history) == 2

    def test_unknown_agent_type_returns_error_before_running(self):
        tool = self._make_tool()
        ctx = _make_ctx()
        tool._ctx = ctx

        resp = tool.run({"calls": [
            {"agent_type": "fake_probe", "targets": ["x"]},
            {"agent_type": "nonexistent", "targets": ["y"]},
        ]})

        assert resp.status == ToolStatus.ERROR
        # No subagent was run (validation failed before thread spawn)
        assert len(ctx.job_history) == 0

    def test_empty_calls_returns_error(self):
        tool = self._make_tool()
        resp = tool.run({"calls": []})
        assert resp.status == ToolStatus.ERROR

    def test_works_without_ctx(self):
        tool = self._make_tool()
        # No _ctx injection
        resp = tool.run({"calls": [
            {"agent_type": "fake_probe", "targets": ["x"]},
            {"agent_type": "fake_probe", "targets": ["y"]},
        ]})
        assert resp.status == ToolStatus.SUCCESS

    def test_schema_has_calls_array(self):
        tool = self._make_tool()
        schema = tool.to_openai_schema()
        fn = schema["function"]
        assert fn["name"] == "run_subagent_parallel"
        params = fn["parameters"]["properties"]
        assert "calls" in params
        assert params["calls"]["type"] == "array"
        assert params["calls"]["minItems"] == 2
        assert "calls" in fn["parameters"]["required"]
        items = params["calls"]["items"]
        assert "agent_type" in items["properties"]
        assert "targets" in items["properties"]


# ---------------------------------------------------------------------------
# MarkReadyForCriticTool tests
# ---------------------------------------------------------------------------

class TestMarkReadyForCriticTool:

    def test_raises_terminated(self):
        tool = MarkReadyForCriticTool()
        with pytest.raises(_Terminated) as exc_info:
            tool.run({"summary": "all done"})
        assert exc_info.value.summary == "all done"

    def test_stores_summary_in_ctx_memory(self):
        tool = MarkReadyForCriticTool()
        ctx = _make_ctx()
        tool._ctx = ctx

        with pytest.raises(_Terminated):
            tool.run({"summary": "optimization complete"})

        assert ctx.memory.get("run", "summary") == "optimization complete"

    def test_works_without_ctx(self):
        tool = MarkReadyForCriticTool()
        with pytest.raises(_Terminated):
            tool.run({"summary": "done"})


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

class TestSubagentToolSchemas:

    def test_run_subagent_schema_has_required_fields(self):
        tool = RunSubagentTool(
            llm=_FakeLLM(), executor=_FakeExecutor(),
            agent_registry={}, agent_cfg=AgentConfig(max_iterations=5),
        )
        schema = tool.to_openai_schema()
        fn = schema["function"]
        assert fn["name"] == "run_subagent"
        params = fn["parameters"]["properties"]
        assert "agent_type" in params
        assert "targets" in params
        assert params["targets"]["type"] == "array"
        assert "agent_type" in fn["parameters"]["required"]
        assert "targets" in fn["parameters"]["required"]
        # retry_context is optional
        assert "retry_context" not in fn["parameters"]["required"]

    def test_mark_ready_schema_has_summary(self):
        tool = MarkReadyForCriticTool()
        schema = tool.to_openai_schema()
        fn = schema["function"]
        assert fn["name"] == "mark_ready_for_critic"
        params = fn["parameters"]["properties"]
        assert "summary" in params
        assert "summary" in fn["parameters"]["required"]


# ---------------------------------------------------------------------------
# PlannerAgent (integration-light: mock LLM)
# ---------------------------------------------------------------------------

class TestPlannerAgent:
    """Tests for PlannerAgent with a mock LLM that returns canned responses."""

    def _make_planner(self, agent_class=_FakeSuccessAgent):
        from agents.agents.planner_agent import PlannerAgent
        agent_def = _make_agent_def(agent_class)
        planner = PlannerAgent(
            llm=_FakeLLM(),
            agent_registry={"fake_probe": agent_def},
            executor=_FakeExecutor(),
            agent_cfg=AgentConfig(max_iterations=3),
        )
        return planner

    def test_run_returns_agent_context(self):
        """PlannerAgent.run() returns an AgentContext regardless of LLM output."""
        from agents.agents.planner_agent import PlannerAgent
        planner = PlannerAgent(
            llm=_FakeLLM(),
            agent_registry={},
            executor=_FakeExecutor(),
            agent_cfg=AgentConfig(max_iterations=2),
        )

        # With a FakeLLM that has no chat method, the AgentLoop will raise quickly.
        # We expect an AgentContext back (loop's RuntimeError is caught).
        ctx = planner.run({"operator": "lora_matmul"})
        assert isinstance(ctx, AgentContext)

    def test_instance_attributes_injectable(self):
        """run_id, agent_id, shared_store can be set before run()."""
        from agents.agents.planner_agent import PlannerAgent
        planner = PlannerAgent(
            llm=_FakeLLM(),
            agent_registry={},
            executor=_FakeExecutor(),
            agent_cfg=AgentConfig(max_iterations=2),
        )
        planner.run_id = "test-run-123"
        planner.agent_id = "planner"
        planner.shared_store = None

        ctx = planner.run({"operator": "lora_matmul"})
        assert ctx.run_id == "test-run-123"
        assert ctx.agent_id == "planner"
