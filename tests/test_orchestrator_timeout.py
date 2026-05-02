"""
tests/test_orchestrator_timeout.py — Orchestrator state machine tests (no GPU required).

Tests the new state machine: planning → ready_for_critic → accepted / revising paths.
"""
from __future__ import annotations

from agents._registry import AgentDefinition
from agents.core.config import AgentConfig
from agents.core.types import WorkerOutput
from orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# Shared test fixture builder
# ---------------------------------------------------------------------------

def _make_worker_output(step_id: str, agent_type: str = "hardware_probe", **kwargs) -> WorkerOutput:
    defaults = dict(
        results=[],
        success=True,
        targets_requested=[],
        reasoning_log=[],
        events=[],
        summary="",
    )
    defaults.update(kwargs)
    return WorkerOutput(step_id=step_id, agent_type=agent_type, **defaults)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeLLM:
    pass


class _FakeExecutor:
    detect_notes: list = []

    def run_cuda_probe(self, **kwargs):
        return {"status": "done"}


class _FakePlannerAgent:
    """Stub for PlannerAgent: records calls, returns canned AgentContext."""

    def __init__(self, job_history_entries: list[WorkerOutput], calls: list | None = None):
        self._history = job_history_entries
        self._calls = calls if calls is not None else []
        self.run_id = None
        self.agent_id = None
        self.shared_store = None

    def run(self, spec, critic_feedback=None):
        from agents.core.types import AgentContext, MemoryStore
        from agents.tools.circuit_breaker import CircuitBreaker

        self._calls.append({"spec": spec, "critic_feedback": critic_feedback})
        ctx = AgentContext(memory=MemoryStore(), circuit_breaker=CircuitBreaker())
        ctx.job_history = list(self._history)
        ctx.memory.set("run", "summary", "stub summary")
        return ctx


class _FakeCriticAgent:
    """Stub for CriticAgent: returns canned decisions."""

    def __init__(self, decisions_per_call: list[list]):
        self._decisions = list(decisions_per_call)
        self._call_idx = 0

    def run(self, outputs, retry_counts=None, system_prompt_override=None,
            critic_tool_schema_override=None):
        from agents.core.types import CriticDecision
        if self._call_idx < len(self._decisions):
            raw = self._decisions[self._call_idx]
            self._call_idx += 1
            return [
                CriticDecision(
                    step_id=d["step_id"],
                    decision=d["decision"],
                    confidence=d.get("confidence", 1.0),
                    reason=d.get("reason", ""),
                    failing_targets=d.get("failing_targets", []),
                )
                for d in raw
            ]
        # Default: accept all
        return [
            CriticDecision(step_id=sid, decision="accept",
                           confidence=1.0, reason="default accept")
            for sid in outputs
        ]


def _make_orchestrator(job_history, critic_decisions, agent_cfg_kwargs=None):
    """Build an Orchestrator with stubbed Planner and Critic."""
    cfg_kwargs = {"max_iterations": 5, "max_critic_cycles": 3}
    if agent_cfg_kwargs:
        cfg_kwargs.update(agent_cfg_kwargs)
    agent_cfg = AgentConfig(**cfg_kwargs)

    orch = Orchestrator(
        llm=_FakeLLM(),
        executor=_FakeExecutor(),
        spec={"operator": "lora_matmul", "targets": []},
        agent_cfg=agent_cfg,
        agent_registry={},
    )
    orch.planner = _FakePlannerAgent(job_history)
    orch.critic = _FakeCriticAgent(critic_decisions)
    return orch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_orchestrator_accepts_on_first_critic_pass():
    """All steps accepted on first Critic review → state.accepted = True."""
    step_id = "step_abc"
    job_history = [_make_worker_output(
        step_id,
        results=[{"metric": "dram_bandwidth_gbps", "value": 500, "confidence": 0.9}],
        success=True,
        targets_requested=["dram_bandwidth_gbps"],
        summary="done",
    )]
    critic_decisions = [[{"step_id": step_id, "decision": "accept", "confidence": 0.95, "reason": "ok"}]]

    orch = _make_orchestrator(job_history, critic_decisions)
    state = orch.run()

    assert state.accepted is True
    assert state.phase == "accepted"
    assert state.planner_ctx is not None
    assert len(state.planner_ctx.job_history) == 1


def test_orchestrator_revises_then_accepts():
    """Critic retries once, then accepts on second cycle."""
    step_id = "step_xyz"
    job_history = [_make_worker_output(
        step_id,
        results=[{"metric": "boost_clock_mhz", "value": 1500, "confidence": 0.85}],
        success=True,
        targets_requested=["boost_clock_mhz"],
        summary="done",
    )]
    critic_decisions = [
        [{"step_id": step_id, "decision": "retry", "confidence": 0.5,
          "reason": "value seems low", "failing_targets": ["boost_clock_mhz"]}],
        [{"step_id": step_id, "decision": "accept", "confidence": 0.9, "reason": "ok now"}],
    ]
    planner_calls: list = []
    orch = _make_orchestrator(job_history, critic_decisions)
    orch.planner = _FakePlannerAgent(job_history, calls=planner_calls)
    orch.critic = _FakeCriticAgent(critic_decisions)

    state = orch.run()

    assert state.accepted is True
    assert state.phase == "accepted"
    # Planner was called twice (planning + revising)
    assert len(planner_calls) == 2
    # Second call had critic_feedback
    assert planner_calls[1]["critic_feedback"] is not None
    assert "boost_clock_mhz" in planner_calls[1]["critic_feedback"].get("failing_targets", [])


def test_orchestrator_emits_event_log():
    """run_log.jsonl contains expected event kinds."""
    step_id = "step_ev"
    job_history = [_make_worker_output(step_id, success=True)]
    critic_decisions = [[{"step_id": step_id, "decision": "accept", "confidence": 1.0, "reason": "ok"}]]

    orch = _make_orchestrator(job_history, critic_decisions)
    orch.run()

    kinds = {r["kind"] for r in orch.run_ctx.event_log.records()}
    assert "plan.start" in kinds
    assert "plan.complete" in kinds
    assert "critic.start" in kinds
    assert "critic.decision" in kinds
    assert "pipeline.done" in kinds


def test_orchestrator_no_outputs_accepts_immediately():
    """Planner returned no subagent calls → Critic skipped, state accepted."""
    orch = _make_orchestrator(job_history=[], critic_decisions=[])
    state = orch.run()

    assert state.accepted is True
    assert state.phase == "accepted"


def test_collect_outputs_reconstruction():
    """_collect_outputs returns the WorkerOutput objects from job_history directly."""
    agent_cfg = AgentConfig(max_iterations=5, max_critic_cycles=3)
    orch = Orchestrator(
        llm=_FakeLLM(), executor=_FakeExecutor(),
        spec={}, agent_cfg=agent_cfg, agent_registry={},
    )
    from agents.core.types import AgentContext, MemoryStore
    from agents.tools.circuit_breaker import CircuitBreaker
    ctx = AgentContext(memory=MemoryStore(), circuit_breaker=CircuitBreaker())
    ctx.job_history = [
        _make_worker_output(
            "s1",
            results=[{"metric": "dram_bandwidth_gbps", "value": 500}],
            success=True,
            targets_requested=["dram_bandwidth_gbps"],
            summary="ok",
        )
    ]
    outputs = orch._collect_outputs(ctx)
    assert "s1" in outputs
    assert outputs["s1"] is ctx.job_history[0]  # same object, no reconstruction
    assert outputs["s1"].success is True
    assert outputs["s1"].results[0]["metric"] == "dram_bandwidth_gbps"
