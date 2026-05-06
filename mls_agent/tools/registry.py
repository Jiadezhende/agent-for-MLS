"""Tool registry — register, schema-export, and dispatch tool calls.

dispatch() never raises: every error path is converted to a
``ToolResponse`` with ``status=ERROR`` and an appropriate code, so the
runtime loop can treat tool dispatch as a total function.

The registry holds its own ``CircuitBreaker`` so each agent run can
construct a fresh registry and not share failure state with sibling runs.
"""
from __future__ import annotations

from typing import Any

import jsonschema

from mls_agent.tools.base import Tool
from mls_agent.tools.circuit_breaker import CircuitBreaker
from mls_agent.tools.response import ToolErrorCode, ToolResponse, ToolStatus


class ToolRegistry:
    def __init__(self, breaker: CircuitBreaker | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._breaker = breaker or CircuitBreaker()

    # ------------------------------------------------------------------
    # Registration / introspection
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise TypeError(
                f"register() expected a Tool, got {type(tool).__name__}"
            )
        if tool.NAME in self._tools:
            raise ValueError(f"Tool {tool.NAME!r} is already registered")
        self._tools[tool.NAME] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, name: str, args: dict[str, Any] | None) -> ToolResponse:
        """Execute the named tool. Always returns a ToolResponse."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResponse.error(
                code=ToolErrorCode.UNKNOWN_TOOL,
                message=(
                    f"Unknown tool {name!r}. "
                    f"Valid tools: {sorted(self._tools)}"
                ),
            )

        # Circuit-breaker pre-check.
        open_kinds = self._breaker.open_for(name)
        # ``is_open`` may flip a circuit into the probing state, so iterate
        # via is_open() rather than a flat membership test.
        blocking = [k for k in open_kinds if self._breaker.is_open(name, k)]
        if blocking:
            counts = {k: self._breaker.failure_count(name, k) for k in blocking}
            return ToolResponse.error(
                code=ToolErrorCode.CIRCUIT_OPEN,
                message=(
                    f"Tool {name!r} has failed {self._breaker.threshold}+ times "
                    f"with errors {blocking}. Circuit is open — try a different "
                    "approach or terminate."
                ),
                stats={
                    "tool": name,
                    "open_error_kinds": blocking,
                    "failure_counts": counts,
                },
            )

        args = args or {}
        if not isinstance(args, dict):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message=f"arguments must be an object, got {type(args).__name__}",
            )

        # JSON Schema validation.
        param_schema = tool.parameters_schema()
        if param_schema:
            validator = jsonschema.Draft202012Validator(param_schema)
            errors = sorted(validator.iter_errors(args), key=lambda e: e.path)
            if errors:
                first = errors[0]
                path = list(first.absolute_path)
                msg = f"{first.message} (path: {path})" if path else first.message
                err_resp = ToolResponse.error(
                    code=ToolErrorCode.INVALID_ARGS,
                    message=msg,
                )
                self._breaker.record_failure(name, ToolErrorCode.INVALID_ARGS)
                return err_resp

        # Execute.
        try:
            result = tool.run(args)
        except Exception as exc:  # noqa: BLE001 — dispatch is a total function
            err_resp = ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"{type(exc).__name__}: {exc}",
            )
            self._breaker.record_failure(name, ToolErrorCode.EXECUTION_ERROR)
            return err_resp

        if not isinstance(result, ToolResponse):
            err_resp = ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=(
                    f"Tool {name!r} returned {type(result).__name__}, "
                    "expected ToolResponse"
                ),
            )
            self._breaker.record_failure(name, ToolErrorCode.INTERNAL_ERROR)
            return err_resp

        # Update breaker based on outcome.
        if result.status == ToolStatus.ERROR:
            code = (result.error_info or {}).get("code", "unknown")
            self._breaker.record_failure(name, code)
        else:
            self._breaker.record_success(name)

        return result
