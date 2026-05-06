"""Unit tests for mls_agent.tools.circuit_breaker."""
from __future__ import annotations

import pytest

from mls_agent.tools.circuit_breaker import CircuitBreaker


class _Clock:
    """Mutable clock so tests don't need time.sleep."""

    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class TestCircuitBreaker:
    def test_default_threshold(self):
        cb = CircuitBreaker()
        assert cb.threshold == 3

    def test_threshold_must_be_positive(self):
        with pytest.raises(ValueError):
            CircuitBreaker(threshold=0)

    def test_negative_timeout_rejected(self):
        with pytest.raises(ValueError):
            CircuitBreaker(half_open_timeout_s=-1.0)

    def test_starts_closed(self):
        cb = CircuitBreaker(threshold=3)
        assert not cb.is_open("t", "execution_error")
        assert cb.open_circuits() == []

    def test_opens_after_threshold(self):
        clock = _Clock()
        cb = CircuitBreaker(threshold=3, _now=clock)
        assert cb.record_failure("t", "execution_error") is False
        assert cb.record_failure("t", "execution_error") is False
        assert cb.record_failure("t", "execution_error") is True
        assert cb.is_open("t", "execution_error")
        assert ("t", "execution_error") in cb.open_circuits()

    def test_independent_kinds(self):
        cb = CircuitBreaker(threshold=2)
        cb.record_failure("t", "a")
        cb.record_failure("t", "a")
        cb.record_failure("t", "b")
        assert cb.is_open("t", "a")
        assert not cb.is_open("t", "b")

    def test_independent_tools(self):
        cb = CircuitBreaker(threshold=2)
        cb.record_failure("t1", "x")
        cb.record_failure("t1", "x")
        assert cb.is_open("t1", "x")
        assert not cb.is_open("t2", "x")

    def test_success_resets_all_kinds_for_tool(self):
        cb = CircuitBreaker(threshold=2)
        cb.record_failure("t", "a")
        cb.record_failure("t", "a")
        cb.record_failure("t", "b")
        cb.record_success("t")
        assert not cb.is_open("t", "a")
        assert cb.failure_count("t", "a") == 0
        assert cb.failure_count("t", "b") == 0

    def test_half_open_probe_after_timeout(self):
        clock = _Clock(0.0)
        cb = CircuitBreaker(threshold=2, half_open_timeout_s=10.0, _now=clock)
        cb.record_failure("t", "x")
        cb.record_failure("t", "x")
        assert cb.is_open("t", "x")  # still open within window

        clock.t = 11.0  # past timeout
        # is_open() flips into probing state and returns False (probe allowed).
        assert not cb.is_open("t", "x")
        # Subsequent is_open() during probing should return True (in probe).
        assert cb.is_open("t", "x")

    def test_probe_failure_re_opens_circuit(self):
        clock = _Clock(0.0)
        cb = CircuitBreaker(threshold=2, half_open_timeout_s=10.0, _now=clock)
        cb.record_failure("t", "x")
        cb.record_failure("t", "x")
        clock.t = 11.0
        assert not cb.is_open("t", "x")  # probe allowed

        cb.record_failure("t", "x")
        # No further immediate probe — window restarted.
        clock.t = 12.0
        assert cb.is_open("t", "x")

    def test_probe_success_closes(self):
        clock = _Clock(0.0)
        cb = CircuitBreaker(threshold=2, half_open_timeout_s=10.0, _now=clock)
        cb.record_failure("t", "x")
        cb.record_failure("t", "x")
        clock.t = 11.0
        assert not cb.is_open("t", "x")  # probe allowed
        cb.record_success("t")
        assert not cb.is_open("t", "x")
        assert cb.open_circuits() == []

    def test_open_for(self):
        cb = CircuitBreaker(threshold=2)
        cb.record_failure("t", "a")
        cb.record_failure("t", "a")
        cb.record_failure("t", "b")
        cb.record_failure("t", "b")
        assert sorted(cb.open_for("t")) == ["a", "b"]
        assert cb.open_for("u") == []

    def test_serialize(self):
        cb = CircuitBreaker(threshold=2)
        cb.record_failure("t", "a")
        cb.record_failure("t", "a")
        out = cb.serialize()
        assert "t:a" in out["open_circuits"]
        assert out["failure_counts"]["t:a"] == 2
