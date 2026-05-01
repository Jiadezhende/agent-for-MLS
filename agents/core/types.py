"""
agents/core/types.py — Shared data structures for the multi-agent pipeline.

Sections:
  1. Internal agent types (Task, Result, MemoryStore, AgentContext)
     — used by AgentLoop and recording tools.
  2. Pipeline types (Step, WorkerOutput, CriticDecision)
     — used by Orchestrator, Planner, Critic, and Worker agents.

CircuitBreaker has moved to agents/tools/circuit_breaker.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.tools.circuit_breaker import CircuitBreaker  # noqa: F401 (re-exported for imports)


# ===========================================================================
# 1. Internal agent types
# ===========================================================================

@dataclass(frozen=True)
class Task:
    """Immutable description of one unit of work passed to an AgentLoop."""
    id: str
    type: str
    description: str
    payload: dict
    constraints: dict


@dataclass
class Result:
    """A single measured or inferred metric value produced by a worker."""
    metric: str
    value: float | int | str | dict
    unit: str | None
    confidence: float
    method: str
    evidence: list[str]
    task_type: str = "hardware_probe"

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "confidence": self.confidence,
            "method": self.method,
            "evidence": self.evidence,
            "task_type": self.task_type,
        }


class MemoryStore:
    """Namespace-isolated key-value store. All values must be JSON-serializable."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def set(self, ns: str, key: str, value: Any) -> None:
        if not ns or not key:
            raise ValueError("namespace and key must be non-empty strings")
        json.dumps(value, default=str)
        self._data.setdefault(ns, {})[key] = value

    def get(self, ns: str, key: str, default: Any = None) -> Any:
        return self._data.get(ns, {}).get(key, default)

    def ns(self, ns: str) -> dict:
        return dict(self._data.get(ns, {}))

    def namespaces(self) -> list[str]:
        return list(self._data.keys())

    def dump(self) -> dict:
        return {ns: dict(vals) for ns, vals in self._data.items()}



@dataclass
class AgentContext:
    """Mutable blackboard shared across the agent loop, tool dispatch, and callbacks."""
    task: Task
    memory: MemoryStore
    iteration: int = 0
    results: list[Result] = field(default_factory=list)
    artifacts: dict[str, Path] = field(default_factory=dict)
    reasoning_log: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    job_history: list[dict] = field(default_factory=list)
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    def serialize(self) -> dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "reasoning_log": self.reasoning_log,
            "events": self.events,
            "job_history": self.job_history,
            "memory": self.memory.dump(),
            "circuit_breaker": self.circuit_breaker.serialize(),
        }


# ===========================================================================
# 2. Pipeline types (new — used by Orchestrator, Planner, Critic, Workers)
# ===========================================================================

@dataclass
class Step:
    """One unit of work produced by the Planner for one worker agent."""
    id: str
    worker: str                              # agent_type key in the registry
    targets: list[str] = field(default_factory=list)  # target names assigned to this worker
    task: str = ""                           # auto-generated log label, not sent to Worker LLM
    hints: list[str] = field(default_factory=list)    # environment/routing hints injected by orchestrator
    retry_context: dict | None = None        # set on retry steps; contains reason + previous bad values


@dataclass
class WorkerOutput:
    """What a worker agent delivers after completing a Step."""
    step_id: str
    results: list[dict]     # list of Result.to_dict() entries
    success: bool
    targets_requested: list[str] = field(default_factory=list)  # Step.targets forwarded for Critic coverage check
    reasoning_log: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    summary: str = ""


@dataclass
class CriticDecision:
    """Structured decision for one step from the Critic."""
    step_id: str
    decision: str       # "accept" | "retry"
    confidence: float
    reason: str
    failing_targets: list[str] = field(default_factory=list)  # which targets to re-measure
