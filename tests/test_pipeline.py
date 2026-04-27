from __future__ import annotations

import threading
import time

from agents._registry import AgentDefinition
from agents.core.config import AgentConfig
from agents.core.llm import ChatResponse, ToolCall
from agents.core.types import Result, Step, Task, WorkerOutput
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


# ---------------------------------------------------------------------------
# Shared helpers for timeout tests
# ---------------------------------------------------------------------------

_AUDIT_SCHEMA = {
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
}


def _make_agent_def(agent_type: str, agent_class: type) -> AgentDefinition:
    return AgentDefinition(
        agent_type=agent_type,
        description=f"{agent_type} worker",
        agent_class=agent_class,
        required_tools=[],
        planner_hints="",
        critic_system_prompt="",
        critic_tool_schema=_AUDIT_SCHEMA,
    )


# ---------------------------------------------------------------------------
# Test: slow worker finishing after pool timeout still yields its real output
# ---------------------------------------------------------------------------

def test_slow_worker_result_preferred_over_timeout_placeholder():
    """A worker that finishes after the soft deadline still yields its real output."""

    class _SlowWorkerAgent:
        def __init__(self, llm, agent_cfg, verbose=False, worker_id=0):
            pass

        def run(self, step, tools):
            time.sleep(0.3)
            return WorkerOutput(
                step_id=step.id,
                results=[{
                    "metric": step.task, "value": 99, "unit": "cycles",
                    "confidence": 0.9, "method": "slow", "evidence": [],
                }],
                success=True,
                summary="slow_ok",
            )

    task = Task(
        id="task_0",
        type="hardware_probe",
        description="slow timeout test",
        payload={"targets": ["slow_metric"]},
        constraints={},
    )
    cfg = AgentConfig(max_iterations=1, worker_timeout_s=0.05)

    orch = Orchestrator(
        llm=_FakeLLM(),
        executor=_FakeExecutor(),
        task=task,
        agent_cfg=cfg,
        agent_registry={"slow_probe": _make_agent_def("slow_probe", _SlowWorkerAgent)},
    )

    state = orch.run()

    assert state.done is True
    out = state.outputs["step_0"]
    assert out.success is True, f"expected success but got summary={out.summary!r}"
    assert out.summary == "slow_ok"
    assert out.results[0]["value"] == 99


# ---------------------------------------------------------------------------
# Test: a future that can be cancelled returns worker_timeout
# ---------------------------------------------------------------------------

def test_cancellable_future_returns_timeout_summary(monkeypatch):
    """A queued future that is cancelled before starting returns worker_timeout."""
    from concurrent.futures import ThreadPoolExecutor as _RealTPE

    block = threading.Event()

    class _BlockingWorkerAgent:
        def __init__(self, llm, agent_cfg, verbose=False, worker_id=0):
            pass

        def run(self, step, tools):
            block.wait(timeout=5)
            return WorkerOutput(
                step_id=step.id,
                results=[],
                success=True,
                summary="block_ok",
            )

    class _SingleThreadTPE(_RealTPE):
        def __init__(self, max_workers):
            super().__init__(max_workers=1)

    monkeypatch.setattr("orchestrator.ThreadPoolExecutor", _SingleThreadTPE)

    task = Task(
        id="task_0",
        type="hardware_probe",
        description="cancel test",
        payload={"targets": []},
        constraints={},
    )
    cfg = AgentConfig(max_iterations=1, worker_timeout_s=0.05)
    orch = Orchestrator(
        llm=_FakeLLM(),
        executor=_FakeExecutor(),
        task=task,
        agent_cfg=cfg,
        agent_registry={"blocking_probe": _make_agent_def("blocking_probe", _BlockingWorkerAgent)},
    )

    steps = [
        Step(id="step_0", task="t0", worker="blocking_probe"),
        Step(id="step_1", task="t1", worker="blocking_probe"),
    ]

    timer = threading.Timer(0.15, block.set)
    timer.start()
    try:
        results = orch._run_workers(steps, timeout_s=0.05)
    finally:
        block.set()
        timer.cancel()

    by_id = {r.step_id: r for r in results}

    # step_0 was running when timeout fired; real output collected after pool drains
    assert by_id["step_0"].success is True
    assert by_id["step_0"].summary == "block_ok"

    # step_1 was queued and not started; it was cancelled → worker_timeout
    assert by_id["step_1"].success is False
    assert by_id["step_1"].summary == "worker_timeout"
