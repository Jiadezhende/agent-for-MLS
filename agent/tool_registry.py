"""
agent/tool_registry.py — Maps tool names to callables + JSON schemas.

The registry is the single place where the LLM's tool-call string name is
translated into a Python function call. It validates arguments, injects
context where needed, and wraps all tool errors into structured dicts so the
loop can feed them back to the LLM without crashing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import jsonschema

from agent.types import AgentContext, CircuitBreaker


# ---------------------------------------------------------------------------
# Internal exception for clean termination
# ---------------------------------------------------------------------------

class _Terminated(Exception):
    """Raised by submit_results to unwind the agent loop cleanly."""
    def __init__(self, summary: str) -> None:
        self.summary = summary
        super().__init__(summary)


# ---------------------------------------------------------------------------
# Circuit breaker — only applies to executor tools
# ---------------------------------------------------------------------------

_EXECUTOR_TOOL_NAMES: frozenset[str] = frozenset({
    "run_cuda_probe",
    "profile_with_ncu",
    "profile_with_nsys",
    "profile_with_torch",
})


def _update_circuit_breaker(cb: CircuitBreaker, tool: str, result: dict) -> None:
    """Update circuit breaker state based on a tool result dict."""
    status = result.get("status", "")
    kind = result.get("error", "")
    if status == "error" and kind:
        cb.record_failure(tool, kind)
    elif status == "done":
        cb.record_success(tool)
    # timed_out does not count toward the circuit — it may just be a slow kernel


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

@dataclass
class ToolEntry:
    fn: Callable
    schema: dict          # full OpenAI-format tool dict (with "type" and "function")
    needs_ctx: bool = False


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, ToolEntry] = {}

    def register(
        self,
        name: str,
        fn: Callable,
        schema: dict,
        needs_ctx: bool = False,
    ) -> None:
        if name in self._entries:
            raise ValueError(f"Tool '{name}' is already registered.")
        self._entries[name] = ToolEntry(fn=fn, schema=schema, needs_ctx=needs_ctx)

    def schemas(self) -> list[dict]:
        """Return the list of tool dicts fed to the LLM."""
        return [entry.schema for entry in self._entries.values()]

    def dispatch(self, name: str, args_dict: Any, ctx: AgentContext) -> dict:
        """Execute a tool call.

        Returns a JSON-serializable dict always — tool errors become
        {"error": ..., "detail": ...} so the LLM sees them as data.
        The special _Terminated exception is re-raised so the loop can break.
        """
        # 1. Look up
        entry = self._entries.get(name)
        if entry is None:
            return {"error": "unknown_tool", "name": name,
                    "hint": f"Valid tools: {list(self._entries.keys())}"}

        # 2. Circuit breaker pre-call check (executor tools only)
        if name in _EXECUTOR_TOOL_NAMES:
            open_for_tool = [(t, ek) for (t, ek) in ctx.circuit_breaker.open_circuits() if t == name]
            if open_for_tool:
                open_kinds = [ek for _, ek in open_for_tool]
                return {
                    "status": "circuit_open",
                    "tool": name,
                    "open_error_kinds": open_kinds,
                    "failure_counts": {
                        ek: ctx.circuit_breaker.failure_count(name, ek) for ek in open_kinds
                    },
                    "message": (
                        f"Tool '{name}' has failed {ctx.circuit_breaker.threshold}+ times "
                        f"with errors {open_kinds}. Circuit is open — stop retrying this approach. "
                        f"Use a different tool or call submit_results with current findings."
                    ),
                }

        # 3. Ensure args_dict is a dict (LLM sometimes sends null for no-arg tools)
        if args_dict is None:
            args_dict = {}

        # 4. Schema validation
        param_schema = entry.schema.get("function", {}).get("parameters", {})
        if param_schema:
            validator = jsonschema.Draft202012Validator(
                param_schema,
                format_checker=None,   # disable format checks — too strict for LLM output
            )
            errors = list(validator.iter_errors(args_dict))
            if errors:
                first = errors[0]
                return {
                    "error": "invalid_args",
                    "detail": first.message,
                    "path": list(first.absolute_path),
                }

        # 5. Inject ctx if needed
        try:
            if entry.needs_ctx:
                result = entry.fn(ctx, **args_dict)
            else:
                result = entry.fn(**args_dict)
        except _Terminated:
            raise   # let the loop handle termination
        except Exception as exc:  # noqa: BLE001
            return {"error": exc.__class__.__name__, "detail": str(exc)}

        # 6. Ensure the result is JSON-serializable
        if not isinstance(result, dict):
            result = {"result": result}

        # 7. Circuit breaker post-call update (executor tools only)
        if name in _EXECUTOR_TOOL_NAMES:
            _update_circuit_breaker(ctx.circuit_breaker, name, result)

        return result


