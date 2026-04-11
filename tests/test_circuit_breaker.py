"""
tests/test_circuit_breaker.py — Unit tests for CircuitBreaker and dispatch() integration.
"""
from __future__ import annotations

import uuid

import pytest

from agent.types import AgentContext, CircuitBreaker, MemoryStore, Task
from agent.tool_registry import ToolRegistry, _EXECUTOR_TOOL_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(threshold: int = 3) -> AgentContext:
    task = Task(
        id=str(uuid.uuid4()),
        type="hardware_probe",
        description="test",
        payload={"targets": []},
        constraints={},
    )
    return AgentContext(
        task=task,
        memory=MemoryStore(),
        circuit_breaker=CircuitBreaker(threshold=threshold),
    )


def _make_registry_with_tool(tool_name: str, return_value: dict) -> ToolRegistry:
    """Build a registry with a single tool that always returns return_value."""
    reg = ToolRegistry()
    schema = {
        "type": "function",
        "function": {
            "name": tool_name,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    reg.register(tool_name, lambda: return_value, schema)
    return reg


# ---------------------------------------------------------------------------
# CircuitBreaker unit tests
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(threshold=3)
        cb.record_failure("run_cuda_probe", "compile_failed")
        cb.record_failure("run_cuda_probe", "compile_failed")
        assert not cb.is_open("run_cuda_probe", "compile_failed")
        cb.record_failure("run_cuda_probe", "compile_failed")
        assert cb.is_open("run_cuda_probe", "compile_failed")

    def test_record_failure_returns_true_on_open(self):
        cb = CircuitBreaker(threshold=2)
        cb.record_failure("run_cuda_probe", "compile_failed")
        newly_opened = cb.record_failure("run_cuda_probe", "compile_failed")
        assert newly_opened is True

    def test_record_failure_returns_false_before_threshold(self):
        cb = CircuitBreaker(threshold=3)
        result = cb.record_failure("run_cuda_probe", "compile_failed")
        assert result is False

    def test_record_failure_returns_false_after_already_open(self):
        cb = CircuitBreaker(threshold=1)
        cb.record_failure("run_cuda_probe", "compile_failed")  # opens
        result = cb.record_failure("run_cuda_probe", "compile_failed")
        assert result is False  # already open, not "newly" opened

    def test_success_resets_counts(self):
        cb = CircuitBreaker(threshold=3)
        cb.record_failure("run_cuda_probe", "compile_failed")
        cb.record_failure("run_cuda_probe", "compile_failed")
        cb.record_success("run_cuda_probe")
        assert cb.failure_count("run_cuda_probe", "compile_failed") == 0
        assert not cb.is_open("run_cuda_probe", "compile_failed")

    def test_success_closes_open_circuit(self):
        cb = CircuitBreaker(threshold=1)
        cb.record_failure("run_cuda_probe", "compile_failed")
        assert cb.is_open("run_cuda_probe", "compile_failed")
        cb.record_success("run_cuda_probe")
        assert not cb.is_open("run_cuda_probe", "compile_failed")

    def test_different_error_kinds_are_independent(self):
        cb = CircuitBreaker(threshold=2)
        cb.record_failure("run_cuda_probe", "compile_failed")
        cb.record_failure("run_cuda_probe", "compile_failed")
        assert cb.is_open("run_cuda_probe", "compile_failed")
        assert not cb.is_open("run_cuda_probe", "compile_timeout")

    def test_different_tools_are_independent(self):
        cb = CircuitBreaker(threshold=2)
        cb.record_failure("run_cuda_probe", "compile_failed")
        cb.record_failure("run_cuda_probe", "compile_failed")
        assert not cb.is_open("profile_with_ncu", "compile_failed")

    def test_open_circuits_lists_all_open(self):
        cb = CircuitBreaker(threshold=1)
        cb.record_failure("run_cuda_probe", "compile_failed")
        cb.record_failure("run_cuda_probe", "binary_not_found")
        open_set = cb.open_circuits()
        assert ("run_cuda_probe", "compile_failed") in open_set
        assert ("run_cuda_probe", "binary_not_found") in open_set

    def test_failure_count_tracks_correctly(self):
        cb = CircuitBreaker(threshold=5)
        for _ in range(3):
            cb.record_failure("run_cuda_probe", "compile_failed")
        assert cb.failure_count("run_cuda_probe", "compile_failed") == 3
        assert cb.failure_count("run_cuda_probe", "other") == 0

    def test_serialize_empty(self):
        cb = CircuitBreaker(threshold=3)
        s = cb.serialize()
        assert s["open_circuits"] == []
        assert s["failure_counts"] == {}

    def test_serialize_with_failures(self):
        cb = CircuitBreaker(threshold=1)
        cb.record_failure("run_cuda_probe", "compile_failed")
        s = cb.serialize()
        assert "run_cuda_probe:compile_failed" in s["open_circuits"]
        assert s["failure_counts"].get("run_cuda_probe:compile_failed", 0) >= 1


# ---------------------------------------------------------------------------
# dispatch() circuit breaker integration tests
# ---------------------------------------------------------------------------

class TestDispatchCircuitBreaker:
    def test_circuit_opens_after_threshold_dispatches(self):
        ctx = _make_ctx(threshold=2)
        reg = _make_registry_with_tool(
            "run_cuda_probe",
            {"status": "error", "error": "compile_failed"}
        )
        reg.dispatch("run_cuda_probe", {}, ctx)
        reg.dispatch("run_cuda_probe", {}, ctx)
        # Third call — circuit should be open, returns circuit_open
        result = reg.dispatch("run_cuda_probe", {}, ctx)
        assert result["status"] == "circuit_open"
        assert result["tool"] == "run_cuda_probe"
        assert "compile_failed" in result["open_error_kinds"]

    def test_circuit_open_message_is_informative(self):
        ctx = _make_ctx(threshold=1)
        reg = _make_registry_with_tool(
            "run_cuda_probe",
            {"status": "error", "error": "compile_failed"}
        )
        reg.dispatch("run_cuda_probe", {}, ctx)   # opens circuit
        result = reg.dispatch("run_cuda_probe", {}, ctx)
        assert "message" in result
        assert len(result["message"]) > 20

    def test_circuit_open_persists_on_subsequent_calls(self):
        ctx = _make_ctx(threshold=1)
        reg = _make_registry_with_tool(
            "run_cuda_probe",
            {"status": "error", "error": "compile_failed"}
        )
        reg.dispatch("run_cuda_probe", {}, ctx)   # opens circuit
        r1 = reg.dispatch("run_cuda_probe", {}, ctx)
        r2 = reg.dispatch("run_cuda_probe", {}, ctx)
        assert r1["status"] == "circuit_open"
        assert r2["status"] == "circuit_open"

    def test_success_resets_circuit(self):
        # Once open, the pre-call check blocks all dispatches.
        # Reset must happen via record_success() directly (e.g. after environment is fixed).
        ctx = _make_ctx(threshold=2)
        failing_reg = _make_registry_with_tool(
            "run_cuda_probe",
            {"status": "error", "error": "compile_failed"}
        )
        # Two failures → opens circuit
        failing_reg.dispatch("run_cuda_probe", {}, ctx)
        failing_reg.dispatch("run_cuda_probe", {}, ctx)
        assert ctx.circuit_breaker.is_open("run_cuda_probe", "compile_failed")

        # Simulate environment fix: manually reset the circuit
        ctx.circuit_breaker.record_success("run_cuda_probe")
        assert not ctx.circuit_breaker.is_open("run_cuda_probe", "compile_failed")

        # Now dispatch goes through again (not circuit_open)
        result = failing_reg.dispatch("run_cuda_probe", {}, ctx)
        assert result["status"] == "error"   # real error, not circuit_open

    def test_non_executor_tool_never_circuit_breaks(self):
        """flag_event and other non-executor tools must not be circuit-broken."""
        assert "flag_event" not in _EXECUTOR_TOOL_NAMES
        ctx = _make_ctx(threshold=1)
        reg = _make_registry_with_tool(
            "flag_event",
            {"error": "some_error", "status": "error"}
        )
        reg.dispatch("flag_event", {}, ctx)
        result = reg.dispatch("flag_event", {}, ctx)
        # Should NOT be circuit_open
        assert result.get("status") != "circuit_open"

    def test_timed_out_does_not_count_toward_circuit(self):
        """timed_out results should not open the circuit (kernel just slow)."""
        ctx = _make_ctx(threshold=1)
        reg = _make_registry_with_tool(
            "run_cuda_probe",
            {"status": "timed_out", "stdout": "", "timed_out": True}
        )
        reg.dispatch("run_cuda_probe", {}, ctx)
        result = reg.dispatch("run_cuda_probe", {}, ctx)
        assert result.get("status") != "circuit_open"

    def test_different_tools_have_independent_circuits(self):
        ctx = _make_ctx(threshold=1)
        cuda_reg = _make_registry_with_tool(
            "run_cuda_probe",
            {"status": "error", "error": "compile_failed"}
        )
        ncu_reg = _make_registry_with_tool(
            "profile_with_ncu",
            {"status": "done", "metrics": {}}
        )
        cuda_reg.dispatch("run_cuda_probe", {}, ctx)   # opens run_cuda_probe circuit
        result = ncu_reg.dispatch("profile_with_ncu", {}, ctx)
        assert result["status"] == "done"   # not affected


# ---------------------------------------------------------------------------
# AgentContext integration
# ---------------------------------------------------------------------------

class TestAgentContextCircuitBreaker:
    def test_context_has_circuit_breaker_field(self):
        ctx = _make_ctx()
        assert hasattr(ctx, "circuit_breaker")
        assert isinstance(ctx.circuit_breaker, CircuitBreaker)

    def test_default_threshold_is_3(self):
        ctx = _make_ctx()
        assert ctx.circuit_breaker.threshold == 3

    def test_custom_threshold_is_respected(self):
        ctx = _make_ctx(threshold=5)
        assert ctx.circuit_breaker.threshold == 5

    def test_serialize_includes_circuit_breaker(self):
        ctx = _make_ctx()
        s = ctx.serialize()
        assert "circuit_breaker" in s
        assert "open_circuits" in s["circuit_breaker"]
        assert "failure_counts" in s["circuit_breaker"]

    def test_serialize_circuit_breaker_after_failures(self):
        ctx = _make_ctx(threshold=5)
        ctx.circuit_breaker.record_failure("run_cuda_probe", "compile_failed")
        ctx.circuit_breaker.record_failure("run_cuda_probe", "compile_failed")
        s = ctx.serialize()
        counts = s["circuit_breaker"]["failure_counts"]
        assert counts.get("run_cuda_probe:compile_failed", 0) == 2
