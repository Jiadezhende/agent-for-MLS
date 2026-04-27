from __future__ import annotations

from agents._registry import AgentDefinition
from agents.core.config import AgentConfig
from agents.core.llm import ChatResponse, ToolCall
from agents.core.types import Result, Task, WorkerOutput
from orchestrator import Orchestrator


class _FakeLLM:
    def __init__(self, decisions: list[str] | None = None) -> None:
        self.decisions = decisions or ["accept"]
        self.calls: list[dict] = []

    def chat(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        if tools:
            decision = self.decisions.pop(0) if self.decisions else "accept"
            return ChatResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="tc_0",
                        name="audit_results",
                        arguments={
                            "decisions": [
                                {
                                    "step_id": "step_0",
                                    "decision": decision,
                                    "confidence": 1.0,
                                    "reason": "test",
                                }
                            ]
                        },
                    )
                ],
            )
        return ChatResponse(
            content='{"steps":[{"id":"step_0","task":"x","worker":"fake_probe","hints":[]}]}'
        )


class _FakeWorkerAgent:
    calls = 0

    def __init__(self, llm, agent_cfg, verbose=False, worker_id=0):
        self.worker_id = worker_id

    def run(self, step, tools):
        _FakeWorkerAgent.calls += 1
        result = Result(
            metric=step.task,
            value=1,
            unit="unit",
            confidence=0.9,
            method="fake",
            evidence=[f"worker_id={self.worker_id}"],
        )
        return WorkerOutput(
            step_id=step.id,
            results=[result.to_dict()],
            success=True,
            summary="ok",
        )


class _FakeExecutor:
    def run_cuda_probe(self, **kwargs):
        return {"status": "done"}

    def profile_with_ncu(self, **kwargs):
        return {"status": "done"}

    def profile_with_nsys(self, **kwargs):
        return {"status": "done"}

    def profile_with_torch(self, **kwargs):
        return {"status": "done"}


def _fake_agent_def() -> AgentDefinition:
    return AgentDefinition(
        agent_type="fake_probe",
        description="fake worker for pipeline tests",
        agent_class=_FakeWorkerAgent,
        required_tools=[],
        planner_hints="Use fake_probe for fake targets.",
        critic_system_prompt="Accept valid fake results.",
        critic_tool_schema={
            "type": "function",
            "function": {
                "name": "audit_results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decisions": {"type": "array", "items": {"type": "object"}}
                    },
                    "required": ["decisions"],
                },
            },
        },
    )


def test_orchestrator_uses_injected_agent_registry_for_worker():
    _FakeWorkerAgent.calls = 0
    task = Task(
        id="task_0",
        type="hardware_probe",
        description="pipeline smoke",
        payload={"targets": ["fake_metric"]},
        constraints={},
    )
    orch = Orchestrator(
        llm=_FakeLLM(),
        executor=_FakeExecutor(),
        task=task,
        agent_cfg=AgentConfig(max_iterations=2),
        agent_registry={"fake_probe": _fake_agent_def()},
    )

    state = orch.run()

    assert state.done is True
    assert _FakeWorkerAgent.calls == 1
    assert state.outputs["step_0"].success is True
    assert state.outputs["step_0"].results[0]["metric"] == "fake_metric"


def test_orchestrator_retries_step_when_critic_requests_retry():
    _FakeWorkerAgent.calls = 0
    task = Task(
        id="task_0",
        type="hardware_probe",
        description="pipeline retry smoke",
        payload={"targets": ["fake_metric"]},
        constraints={},
    )
    cfg = AgentConfig(max_iterations=2)
    cfg.max_retries = 1
    orch = Orchestrator(
        llm=_FakeLLM(decisions=["retry", "accept"]),
        executor=_FakeExecutor(),
        task=task,
        agent_cfg=cfg,
        agent_registry={"fake_probe": _fake_agent_def()},
    )

    state = orch.run()

    assert state.done is True
    assert _FakeWorkerAgent.calls == 2
    assert state.retry_set == set()
