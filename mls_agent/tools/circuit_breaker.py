"""Circuit breaker for repeated tool failures.

Tracks consecutive failures per ``(tool_name, error_kind)`` pair. After
``threshold`` consecutive failures the circuit OPENs; the registry
short-circuits subsequent calls to the same (tool, kind) until
``half_open_timeout_s`` elapses, at which point one PROBE is allowed
through. A successful probe (any success on that tool) closes the
circuit; a failed probe re-opens it for another timeout window.

Adapted from the legacy implementation in
``agent/tools/circuit_breaker.py`` with stricter typing and a unit-test
seam (``_now``) so tests don't need ``time.sleep``.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CircuitBreaker:
    threshold: int = 3
    half_open_timeout_s: float = 60.0

    _counts: dict[tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _open: set[tuple[str, str]] = field(default_factory=set)
    _open_since: dict[tuple[str, str], float] = field(default_factory=dict)
    _probing: set[tuple[str, str]] = field(default_factory=set)

    # Time source for tests; defaults to time.monotonic().
    _now: Callable[[], float] = field(default=time.monotonic, repr=False)

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {self.threshold}")
        if self.half_open_timeout_s < 0:
            raise ValueError(
                f"half_open_timeout_s must be >= 0, got {self.half_open_timeout_s}"
            )

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def record_failure(self, tool: str, kind: str) -> bool:
        """Record a failure. Return True iff this transition opened the circuit."""
        key = (tool, kind)
        if key in self._probing:
            # Probe failed; stay open and reset the timeout window.
            self._probing.discard(key)
            self._open_since[key] = self._now()
            return False
        self._counts[key] += 1
        if self._counts[key] >= self.threshold and key not in self._open:
            self._open.add(key)
            self._open_since[key] = self._now()
            return True
        return False

    def record_success(self, tool: str) -> None:
        """A success on this tool closes ALL of its open circuits (any kind)."""
        for k in [k for k in list(self._counts) if k[0] == tool]:
            self._counts[k] = 0
            self._open.discard(k)
            self._probing.discard(k)
            self._open_since.pop(k, None)

    def is_open(self, tool: str, kind: str) -> bool:
        """Is the (tool, kind) circuit currently blocking calls?"""
        key = (tool, kind)
        if key not in self._open:
            return False
        if key in self._probing:
            return True
        elapsed = self._now() - self._open_since.get(key, 0.0)
        if elapsed > self.half_open_timeout_s:
            # Allow exactly one probe through.
            self._probing.add(key)
            return False
        return True

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def open_circuits(self) -> list[tuple[str, str]]:
        return list(self._open)

    def open_for(self, tool: str) -> list[str]:
        """Error kinds currently blocking calls to ``tool``."""
        return [k for (t, k) in self._open if t == tool]

    def failure_count(self, tool: str, kind: str) -> int:
        return self._counts.get((tool, kind), 0)

    def serialize(self) -> dict:
        return {
            "open_circuits": [f"{t}:{k}" for t, k in self._open],
            "probing_circuits": [f"{t}:{k}" for t, k in self._probing],
            "failure_counts": {
                f"{t}:{k}": v for (t, k), v in self._counts.items() if v > 0
            },
        }
