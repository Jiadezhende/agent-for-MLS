"""
agent/types.py — Pure data structures shared across agent, tools, and executor.
No business logic; no imports from the rest of this project.
"""
from __future__ import annotations

import json
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

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "confidence": self.confidence,
            "method": self.method,
            "evidence": self.evidence,
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

    def serialize(self) -> dict:
        """Produce the dict written to results.json and reasoning_log.json."""
        return {
            "results": [r.to_dict() for r in self.results],
            "reasoning_log": self.reasoning_log,
            "events": self.events,
            "job_history": self.job_history,
            "memory": self.memory.dump(),
        }
