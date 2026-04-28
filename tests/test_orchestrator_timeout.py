from __future__ import annotations

import time

from agents._registry import AgentDefinition
from agents.core.config import AgentConfig
from agents.core.types import Result, Step, Task, WorkerOutput
from orchestrator import Orchestrator


class _FakeLLM:
    pass


class _FakeExecutor:
    def run_cuda_probe(self, **kwargs):
        return {"status": "done"}

    def profile_with_ncu(self, **kwargs):
        return {"status": "done"}

    def profile_with_nsys(self, **kwargs):
        return {"status": "done"}

    def profile_with_torch(self, **kwargs):
        return {"status": "done"}

    def probe_environment(self, **kwargs):
        return {"status": "done"}


class _SlowWorkerAgent:
    def __init__(self, llm, agent_cfg, verbose=False, worker_id=0):
        self.worker_id = worker_id

    def run(self, step, tools):
        time.sleep(0.05)
        result = Result(
            metric=step.task,
            value=7,
            unit="unit",
            confidence=0.9,
            method="slow fake",
            evidence=[f"worker_id={self.worker_id}"],
        )
        return WorkerOutput(
            step_id=step.id,
            results=[result.to_dict()],
            success=True,
            summary="late-success",
        )


def test_worker_timeout_collects_late_successful_output():
    task = Task(
        id="task_0",
        type="hardware_probe",
        description="late worker",
        payload={"targets": []},
        constraints={},
    )
    agent_def = AgentDefinition(
        agent_type="fake_probe",
        description="fake worker",
        agent_class=_SlowWorkerAgent,
        required_tools=[],
        planner_hints="",
        critic_system_prompt="",
        critic_tool_schema={},
    )
    orch = Orchestrator(
        llm=_FakeLLM(),
        executor=_FakeExecutor(),
        task=task,
        agent_cfg=AgentConfig(max_iterations=2),
        agent_registry={"fake_probe": agent_def},
    )

    outputs = orch._run_workers(
        steps=[Step(id="step_0", task="late_metric", worker="fake_probe")],
        timeout_s=0.001,
    )

    assert len(outputs) == 1
    assert outputs[0].success is True
    assert outputs[0].summary == "late-success"
    assert outputs[0].results[0]["metric"] == "late_metric"
