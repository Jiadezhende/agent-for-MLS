"""
agent/types.py — Pure data structures shared across agent, tools, and executor.
No business logic; no imports from the rest of this project.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """Immutable task description.

    type="hardware_probe" in Phase 1.
    Future phases may introduce "code_analysis", "operator_bench", etc.
    without restructuring AgentContext.
    """
    id: str
    type: str
    description: str
    payload: dict
    constraints: dict


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class Result:
    """A single measured or inferred metric value."""
    metric: str
    value: float | int | str | dict
    unit: str | None
    confidence: float        # 0.0 – 1.0
    method: str              # free-form description of how this was obtained
    evidence: list[str]      # tool-returned strings that support this value
    task_type: str = "hardware_probe"   # which task plugin produced this result

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


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class MemoryStore:
    """Namespace-isolated key-value store.

    Each agent role writes to its own namespace and reads from others.
    All values must be JSON-serializable (enforced at set-time).
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def set(self, ns: str, key: str, value: Any) -> None:
        if not ns or not key:
            raise ValueError("namespace and key must be non-empty strings")
        # Validate serializability early so bugs surface at write-time, not output-time.
        json.dumps(value, default=str)
        self._data.setdefault(ns, {})[key] = value

    def get(self, ns: str, key: str, default: Any = None) -> Any:
        return self._data.get(ns, {}).get(key, default)

    def ns(self, ns: str) -> dict:
        """Return a shallow copy of a namespace."""
        return dict(self._data.get(ns, {}))

    def namespaces(self) -> list[str]:
        return list(self._data.keys())

    def dump(self) -> dict:
        """Full snapshot for debugging / serialization."""
        return {ns: dict(vals) for ns, vals in self._data.items()}


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

@dataclass
class CircuitBreaker:
    """Tracks consecutive failures per (tool_name, error_kind) pair.

    The circuit opens after `threshold` consecutive failures of the same pair.
    Any successful result from that tool resets all its counters.
    Only executor tools participate (enforced in tool_registry.py).

    Three states per (tool, kind) pair:
      CLOSED    — normal operation
      OPEN      — blocked; opened after `threshold` failures
      HALF-OPEN — one probe allowed through after `half_open_timeout_s` seconds;
                  probe success → CLOSED, probe failure → OPEN (timer reset)
    """
    threshold: int = 3
    half_open_timeout_s: float = 60.0
    _counts: dict = field(default_factory=lambda: defaultdict(int))
    _open: set = field(default_factory=set)
    _open_since: dict = field(default_factory=dict)   # (tool, kind) → monotonic ts
    _probing: set = field(default_factory=set)         # circuits in half-open probe

    def record_failure(self, tool: str, kind: str) -> bool:
        """Record a failure. Returns True if the circuit just opened."""
        key = (tool, kind)
        if key in self._probing:
            # Probe failed: re-open, reset timer
            self._probing.discard(key)
            self._open_since[key] = time.monotonic()
            return False   # was already open; not "newly" opened
        self._counts[key] += 1
        if self._counts[key] >= self.threshold and key not in self._open:
            self._open.add(key)
            self._open_since[key] = time.monotonic()
            return True
        return False

    def record_success(self, tool: str) -> None:
        """Reset all failure counts for this tool (any error_kind)."""
        for k in [k for k in list(self._counts) if k[0] == tool]:
            self._counts[k] = 0
            self._open.discard(k)
            self._probing.discard(k)
            self._open_since.pop(k, None)

    def is_open(self, tool: str, kind: str) -> bool:
        """Return True if the circuit is blocking calls.

        When the circuit has been open longer than `half_open_timeout_s`,
        transitions to half-open by allowing one probe through (returns False).
        """
        key = (tool, kind)
        if key not in self._open:
            return False
        if key in self._probing:
            return True   # probe already in-flight; block further calls
        elapsed = time.monotonic() - self._open_since.get(key, 0.0)
        if elapsed > self.half_open_timeout_s:
            # Transition to half-open: allow one probe through
            self._probing.add(key)
            return False
        return True

    def open_circuits(self) -> list[tuple[str, str]]:
        return list(self._open)

    def failure_count(self, tool: str, kind: str) -> int:
        return self._counts.get((tool, kind), 0)

    def serialize(self) -> dict:
        """JSON-serializable snapshot (tuple keys → strings)."""
        return {
            "open_circuits": [f"{t}:{e}" for t, e in self._open],
            "probing_circuits": [f"{t}:{e}" for t, e in self._probing],
            "failure_counts": {f"{t}:{e}": v for (t, e), v in self._counts.items() if v > 0},
        }


# ---------------------------------------------------------------------------
# AgentContext
# ---------------------------------------------------------------------------

@dataclass
class AgentContext:
    """Mutable blackboard shared across the agent loop, tool dispatch, and
    Executor callback.  All agents communicate through this object, not
    through direct calls to each other.
    """
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
        """Produce the dict written to results.json and reasoning_log.json."""
        return {
            "results": [r.to_dict() for r in self.results],
            "reasoning_log": self.reasoning_log,
            "events": self.events,
            "job_history": self.job_history,
            "memory": self.memory.dump(),
            "circuit_breaker": self.circuit_breaker.serialize(),
        }


# ---------------------------------------------------------------------------
# Multi-Agent types
# ---------------------------------------------------------------------------

@dataclass
class WorkerSpec:
    """Assignment produced by the Planner for one worker."""
    worker_id: int
    targets: list[str]
    strategy_hints: list[str]
    agent_type: str = "hardware_probe"  # must match a registered TaskDefinition
    group_rationale: str = ""


@dataclass
class WorkerResult:
    """What one worker delivers to the Aggregator."""
    worker_id: int
    worker_spec: WorkerSpec
    ctx: AgentContext
    exit_code: int          # 0=success, 3=budget-exhausted, 1=error
    error: str | None = None


@dataclass
class CritiqueResult:
    """Output of the Critic LLM call."""
    confidence_adjustments: dict[str, float]   # metric → new confidence (0–1)
    anomaly_flags: list[str]
    flagged_results: list[str]                 # metrics below confidence threshold
    overall_assessment: str
