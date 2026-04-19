"""
agents/tools/circuit_breaker.py — Resilience mechanism applied to ALL tool calls.

Three states per (tool_name, error_kind) pair: CLOSED → OPEN → HALF-OPEN.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class CircuitBreaker:
    """Tracks consecutive failures per (tool_name, error_kind) pair.

    Three states per (tool, kind): CLOSED → OPEN → HALF-OPEN → CLOSED.
    Applies uniformly to all tools registered in ToolRegistry.
    """
    threshold: int = 3
    half_open_timeout_s: float = 60.0
    _counts: dict = field(default_factory=lambda: defaultdict(int))
    _open: set = field(default_factory=set)
    _open_since: dict = field(default_factory=dict)
    _probing: set = field(default_factory=set)

    def record_failure(self, tool: str, kind: str) -> bool:
        key = (tool, kind)
        if key in self._probing:
            self._probing.discard(key)
            self._open_since[key] = time.monotonic()
            return False
        self._counts[key] += 1
        if self._counts[key] >= self.threshold and key not in self._open:
            self._open.add(key)
            self._open_since[key] = time.monotonic()
            return True
        return False

    def record_success(self, tool: str) -> None:
        for k in [k for k in list(self._counts) if k[0] == tool]:
            self._counts[k] = 0
            self._open.discard(k)
            self._probing.discard(k)
            self._open_since.pop(k, None)

    def is_open(self, tool: str, kind: str) -> bool:
        key = (tool, kind)
        if key not in self._open:
            return False
        if key in self._probing:
            return True
        elapsed = time.monotonic() - self._open_since.get(key, 0.0)
        if elapsed > self.half_open_timeout_s:
            self._probing.add(key)
            return False
        return True

    def open_circuits(self) -> list[tuple[str, str]]:
        return list(self._open)

    def failure_count(self, tool: str, kind: str) -> int:
        return self._counts.get((tool, kind), 0)

    def serialize(self) -> dict:
        return {
            "open_circuits": [f"{t}:{e}" for t, e in self._open],
            "probing_circuits": [f"{t}:{e}" for t, e in self._probing],
            "failure_counts": {f"{t}:{e}": v for (t, e), v in self._counts.items() if v > 0},
        }
