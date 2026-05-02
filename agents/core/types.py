"""
agents/core/types.py — Shared data structures for the multi-agent pipeline.

Sections:
  1. Internal agent types (Task, Result, MemoryStore, AgentContext)
     — used by AgentLoop and recording tools.
  2. Run-level context (EventLog, SharedStore, RunContext)
     — owned by Orchestrator; not exposed to LLMs.
  3. Pipeline types (Step, WorkerOutput, CriticDecision)
     — used by Orchestrator, Planner, Critic, and Worker agents.

CircuitBreaker has moved to agents/tools/circuit_breaker.py.

Event kind taxonomy for EventLog:
  plan.start / plan.complete / plan.fallback
  worker.start / worker.complete / worker.timeout / worker.error
  critic.start / critic.decision
  retry.trigger / retry.carry_forward
  pipeline.done
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.tools.circuit_breaker import CircuitBreaker  # noqa: F401 (re-exported for imports)

if TYPE_CHECKING:
    from agents.core.log_manager import LogManager


# ===========================================================================
# 1. Internal agent types
# ===========================================================================

@dataclass
class Result:
    """A single measured or inferred metric value produced by a worker."""
    metric: str
    value: float | int | str | dict
    unit: str | None
    confidence: float
    method: str
    evidence: list[str]

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "confidence": self.confidence,
            "method": self.method,
            "evidence": self.evidence,
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
    memory: MemoryStore
    iteration: int = 0
    results: list[Result] = field(default_factory=list)
    artifacts: dict[str, Path] = field(default_factory=dict)
    reasoning_log: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    job_history: list["WorkerOutput"] = field(default_factory=list)
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    # Run-level context (optional; injected by Orchestrator for cross-agent coordination)
    run_id: str | None = None
    agent_id: str | None = None
    shared_store: SharedStore | None = None
    # Structured log writer; injected by Orchestrator when --log-dir is set.
    log_manager: "LogManager | None" = None
    # Full conversation history written by AgentLoop; empty until loop starts.
    messages: list[dict] = field(default_factory=list)

    def serialize(self, include_messages: bool = False) -> dict:
        data: dict = {
            "results": [r.to_dict() for r in self.results],
            "reasoning_log": self.reasoning_log,
            "events": self.events,
            "job_history": [e.to_dict() for e in self.job_history],
            "memory": self.memory.dump(),
            "circuit_breaker": self.circuit_breaker.serialize(),
        }
        if self.run_id is not None:
            data["run_id"] = self.run_id
        if self.agent_id is not None:
            data["agent_id"] = self.agent_id
        if self.shared_store is not None:
            data["shared_store"] = self.shared_store.dump()
        if include_messages:
            data["messages"] = self.messages
        return data


# ===========================================================================
# 2. Run-level context (owned by Orchestrator; not exposed to LLMs)
# ===========================================================================

class EventLog:
    """Append-only thread-safe event log for one pipeline run.

    Write via append(); read via records(); persist via flush().
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._records: list[dict] = []
        self._lock = threading.RLock()

    def append(self, kind: str, source: str, payload: dict) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "kind": kind,
            "source": source,
            "payload": payload,
        }
        with self._lock:
            self._records.append(record)

    def records(self) -> list[dict]:
        with self._lock:
            return list(self._records)

    def flush(self, path: Path) -> None:
        records = self.records()
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")


class SharedStore:
    """Thread-safe cross-agent namespaced KV store.

    Recommended namespaces: "metrics", "artifacts", "decisions", "facts", "candidates"
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, dict]] = {}
        self._lock = threading.RLock()

    def put(self, ns: str, key: str, record: dict) -> None:
        with self._lock:
            self._data.setdefault(ns, {})[key] = record

    def get(self, ns: str, key: str) -> dict | None:
        with self._lock:
            return self._data.get(ns, {}).get(key)

    def list_ns(self, ns: str) -> list[dict]:
        with self._lock:
            return list(self._data.get(ns, {}).values())

    def dump(self) -> dict:
        with self._lock:
            return {ns: dict(vals) for ns, vals in self._data.items()}


@dataclass
class RunContext:
    """Lightweight global run state owned by Orchestrator. Not exposed to LLMs."""
    run_id: str
    shared_store: SharedStore = field(default_factory=SharedStore)
    event_log: EventLog = field(init=False)

    def __post_init__(self) -> None:
        self.event_log = EventLog(self.run_id)


# ===========================================================================
# 3. Pipeline types (new — used by Orchestrator, Planner, Critic, Workers)
# ===========================================================================

@dataclass
class Step:
    """One unit of work produced by the Planner for one worker agent."""
    id: str
    worker: str                              # agent_type key in the registry
    targets: list[str] = field(default_factory=list)  # metric names (measurement agents) or [] (analysis agents)
    task: str = ""                           # auto-generated log label, not sent to Worker LLM
    instructions: str = ""                   # Planner-authored task description + upstream context → worker initial prompt
    retry_context: dict | None = None        # set on retry steps; contains reason + previous bad values


@dataclass
class WorkerOutput:
    """What a worker agent delivers after completing a Step."""
    step_id: str
    results: list[dict]     # list of Result.to_dict() entries
    success: bool
    agent_type: str = ""    # filled by _execute_one(); used by _group_by_type()
    targets_requested: list[str] = field(default_factory=list)  # Step.targets forwarded for Critic coverage check
    reasoning_log: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "agent_type": self.agent_type,
            "results": self.results,
            "success": self.success,
            "targets_requested": self.targets_requested,
            "reasoning_log": self.reasoning_log,
            "events": self.events,
            "summary": self.summary,
        }


@dataclass
class CriticDecision:
    """Structured decision for one step from the Critic."""
    step_id: str
    decision: str       # "accept" | "retry"
    confidence: float
    reason: str
    failing_targets: list[str] = field(default_factory=list)  # which targets to re-measure
