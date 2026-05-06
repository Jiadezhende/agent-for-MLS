"""Unit tests for mls_agent.tools.registry."""
from __future__ import annotations

import pytest

from mls_agent.tools.base import Tool, ToolParameter
from mls_agent.tools.circuit_breaker import CircuitBreaker
from mls_agent.tools.registry import ToolRegistry
from mls_agent.tools.response import ToolErrorCode, ToolResponse, ToolStatus


# ---------------------------------------------------------------------------
# Sample tools used by the tests
# ---------------------------------------------------------------------------


class _Echo(Tool):
    NAME = "echo"
    DESCRIPTION = "Echo input."

    def parameters_schema(self):
        return Tool.schema_from_parameters([
            ToolParameter(name="text", type="string", description="text"),
        ])

    def run(self, parameters):
        return ToolResponse.success(parameters["text"])


class _AlwaysFail(Tool):
    NAME = "fail"
    DESCRIPTION = "Always fails with execution_error."

    def parameters_schema(self):
        return {"type": "object", "properties": {}}

    def run(self, parameters):
        raise RuntimeError("boom")


class _AlwaysReturnsError(Tool):
    NAME = "soft_fail"
    DESCRIPTION = "Returns ERROR ToolResponse."

    def parameters_schema(self):
        return {"type": "object", "properties": {}}

    def run(self, parameters):
        return ToolResponse.error(code="my_code", message="nope")


class _ReturnsNonResponse(Tool):
    NAME = "rogue"
    DESCRIPTION = "Returns the wrong type."

    def parameters_schema(self):
        return {"type": "object", "properties": {}}

    def run(self, parameters):
        return "not a response"  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Registration / introspection
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_and_list(self):
        reg = ToolRegistry()
        reg.register(_Echo())
        assert reg.names() == ["echo"]

    def test_duplicate_register_rejected(self):
        reg = ToolRegistry()
        reg.register(_Echo())
        with pytest.raises(ValueError, match="already"):
            reg.register(_Echo())

    def test_register_non_tool_rejected(self):
        reg = ToolRegistry()
        with pytest.raises(TypeError):
            reg.register("not a tool")  # type: ignore[arg-type]

    def test_schemas_returns_one_per_tool(self):
        reg = ToolRegistry()
        reg.register(_Echo())
        reg.register(_AlwaysFail())
        schemas = reg.schemas()
        assert {s["function"]["name"] for s in schemas} == {"echo", "fail"}


# ---------------------------------------------------------------------------
# Dispatch — happy paths and errors
# ---------------------------------------------------------------------------


class TestDispatchHappyPath:
    def test_returns_tool_response(self):
        reg = ToolRegistry()
        reg.register(_Echo())
        r = reg.dispatch("echo", {"text": "hi"})
        assert r.status == ToolStatus.SUCCESS
        assert r.text == "hi"

    def test_none_args_treated_as_empty(self):
        reg = ToolRegistry()
        reg.register(_AlwaysFail())  # accepts {}
        r = reg.dispatch("fail", None)
        # Dispatch turns the raised RuntimeError into EXECUTION_ERROR.
        assert r.status == ToolStatus.ERROR
        assert r.error_info["code"] == ToolErrorCode.EXECUTION_ERROR


class TestDispatchErrors:
    def test_unknown_tool(self):
        reg = ToolRegistry()
        reg.register(_Echo())
        r = reg.dispatch("nope", {})
        assert r.status == ToolStatus.ERROR
        assert r.error_info["code"] == ToolErrorCode.UNKNOWN_TOOL

    def test_invalid_args(self):
        reg = ToolRegistry()
        reg.register(_Echo())
        r = reg.dispatch("echo", {})  # missing required "text"
        assert r.status == ToolStatus.ERROR
        assert r.error_info["code"] == ToolErrorCode.INVALID_ARGS

    def test_args_must_be_dict(self):
        reg = ToolRegistry()
        reg.register(_Echo())
        r = reg.dispatch("echo", "oops")  # type: ignore[arg-type]
        assert r.status == ToolStatus.ERROR
        assert r.error_info["code"] == ToolErrorCode.INVALID_ARGS

    def test_execution_error(self):
        reg = ToolRegistry()
        reg.register(_AlwaysFail())
        r = reg.dispatch("fail", {})
        assert r.status == ToolStatus.ERROR
        assert r.error_info["code"] == ToolErrorCode.EXECUTION_ERROR
        assert "boom" in r.error_info["message"]

    def test_returning_non_response_yields_internal_error(self):
        reg = ToolRegistry()
        reg.register(_ReturnsNonResponse())
        r = reg.dispatch("rogue", {})
        assert r.status == ToolStatus.ERROR
        assert r.error_info["code"] == ToolErrorCode.INTERNAL_ERROR


# ---------------------------------------------------------------------------
# Circuit breaker integration
# ---------------------------------------------------------------------------


class TestCircuitIntegration:
    def test_repeated_execution_errors_open_circuit(self):
        reg = ToolRegistry(breaker=CircuitBreaker(threshold=3))
        reg.register(_AlwaysFail())
        for _ in range(3):
            assert reg.dispatch("fail", {}).error_info["code"] == ToolErrorCode.EXECUTION_ERROR
        # 4th call short-circuits.
        r = reg.dispatch("fail", {})
        assert r.error_info["code"] == ToolErrorCode.CIRCUIT_OPEN

    def test_invalid_args_counted_separately(self):
        reg = ToolRegistry(breaker=CircuitBreaker(threshold=2))
        reg.register(_Echo())
        # Two invalid_args failures open the circuit on that kind.
        reg.dispatch("echo", {})
        reg.dispatch("echo", {})
        r = reg.dispatch("echo", {})
        assert r.error_info["code"] == ToolErrorCode.CIRCUIT_OPEN

    def test_soft_error_uses_response_code(self):
        reg = ToolRegistry(breaker=CircuitBreaker(threshold=2))
        reg.register(_AlwaysReturnsError())
        for _ in range(2):
            r = reg.dispatch("soft_fail", {})
            assert r.error_info["code"] == "my_code"
        r = reg.dispatch("soft_fail", {})
        assert r.error_info["code"] == ToolErrorCode.CIRCUIT_OPEN
        assert "my_code" in str(r.stats["open_error_kinds"])

    def test_open_circuit_blocks_all_calls_until_probe(self):
        """Contract: an open circuit blocks ALL calls to that tool — even
        valid ones — until the half-open timeout elapses and a probe is
        allowed through. A successful probe then closes the circuit."""
        from mls_agent.tests.test_circuit_breaker import _Clock
        clock = _Clock(0.0)
        breaker = CircuitBreaker(threshold=2, half_open_timeout_s=10.0, _now=clock)
        reg = ToolRegistry(breaker=breaker)
        reg.register(_Echo())

        # Open the circuit on invalid_args.
        reg.dispatch("echo", {})
        reg.dispatch("echo", {})

        # Even a syntactically valid call is blocked while open.
        r = reg.dispatch("echo", {"text": "hi"})
        assert r.error_info["code"] == ToolErrorCode.CIRCUIT_OPEN

        # After the timeout, a probe is allowed; a valid call goes through
        # and the success closes the circuit.
        clock.t = 11.0
        r = reg.dispatch("echo", {"text": "hi"})
        assert r.status == ToolStatus.SUCCESS

        # Subsequent valid calls flow normally.
        r = reg.dispatch("echo", {"text": "hi again"})
        assert r.status == ToolStatus.SUCCESS

    def test_unknown_tool_does_not_affect_breaker(self):
        reg = ToolRegistry(breaker=CircuitBreaker(threshold=2))
        # No tools registered.
        for _ in range(5):
            r = reg.dispatch("nope", {})
            assert r.error_info["code"] == ToolErrorCode.UNKNOWN_TOOL
        # No circuits should be open.
        assert reg.breaker.open_circuits() == []
