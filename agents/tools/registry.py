"""
agents/tools/registry.py — Maps tool names to callables + JSON schemas.

Circuit breaker now applies uniformly to ALL registered tools.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import jsonschema

from agents.core.types import AgentContext
from agents.tools.circuit_breaker import CircuitBreaker


# ---------------------------------------------------------------------------
# Internal exception for clean termination
# ---------------------------------------------------------------------------

class _Terminated(Exception):
    """Raised by submit_results to unwind the agent loop cleanly."""
    def __init__(self, summary: str) -> None:
        self.summary = summary
        super().__init__(summary)


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

@dataclass
class ToolEntry:
    fn: Callable
    schema: dict
    needs_ctx: bool = False


# ---------------------------------------------------------------------------
# ToolRegistry — circuit breaker applies to ALL tools
# ---------------------------------------------------------------------------

def _update_circuit_breaker(cb: CircuitBreaker, tool: str, result: dict) -> None:
    status = result.get("status", "")
    kind = result.get("error", "")
    if kind and (status == "error" or ("status" not in result and "error" in result)):
        cb.record_failure(tool, kind)
    elif status == "done" or result.get("ok"):
        cb.record_success(tool)


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
        return [entry.schema for entry in self._entries.values()]

    def dispatch(self, name: str, args_dict: Any, ctx: AgentContext) -> dict:
        """Execute a tool call. Returns a JSON-serializable dict always.

        Circuit breaker check applies to all registered tools.
        """
        entry = self._entries.get(name)
        if entry is None:
            return {"error": "unknown_tool", "name": name,
                    "hint": f"Valid tools: {list(self._entries.keys())}"}

        # Universal circuit breaker check
        open_for_tool = [
            (t, ek) for (t, ek) in ctx.circuit_breaker.open_circuits() if t == name
        ]
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

        if args_dict is None:
            args_dict = {}

        param_schema = entry.schema.get("function", {}).get("parameters", {})
        if param_schema:
            validator = jsonschema.Draft202012Validator(param_schema, format_checker=None)
            errors = list(validator.iter_errors(args_dict))
            if errors:
                first = errors[0]
                return {
                    "error": "invalid_args",
                    "detail": first.message,
                    "path": list(first.absolute_path),
                }

        try:
            if entry.needs_ctx:
                result = entry.fn(ctx, **args_dict)
            else:
                result = entry.fn(**args_dict)
        except _Terminated:
            raise
        except Exception as exc:  # noqa: BLE001
            return {"error": exc.__class__.__name__, "detail": str(exc)}

        if not isinstance(result, dict):
            result = {"result": result}

        # Universal circuit breaker update
        _update_circuit_breaker(ctx.circuit_breaker, name, result)

        return result


# ---------------------------------------------------------------------------
# ToolFactory — builds ToolRegistry by name, injecting executor
# ---------------------------------------------------------------------------

class ToolFactory:
    """Creates ToolRegistry instances from a tool-name list.

    The executor is injected once at construction; its methods are bound
    into the registry so agents never reference the executor directly.
    """

    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def build(self, tool_names: list[str]) -> ToolRegistry:
        from agents.tools.builtin.recording import flag_event, record_measurement, submit_results
        from agents.tools.builtin.skills import list_skills, read_skill
        from agents.tools.schemas import TOOL_SCHEMAS

        schema_map = {s["function"]["name"]: s for s in TOOL_SCHEMAS}
        all_tools: dict[str, tuple[Callable, bool]] = {
            "list_skills":        (list_skills,                           False),
            "read_skill":         (read_skill,                            False),
            "run_cuda_probe":     (self._executor.run_cuda_probe,         False),
            "profile_with_ncu":   (self._executor.profile_with_ncu,       False),
            "profile_with_nsys":  (self._executor.profile_with_nsys,      False),
            "profile_with_torch": (self._executor.profile_with_torch,     False),
            "record_measurement": (record_measurement,                    True),
            "flag_event":         (flag_event,                            True),
            "submit_results":     (submit_results,                        True),
            "find_binary":        (self._executor.find_binary,            False),
        }

        reg = ToolRegistry()
        for name in tool_names:
            if name not in all_tools:
                raise ValueError(f"Unknown tool '{name}'. Available: {list(all_tools)}")
            fn, needs_ctx = all_tools[name]
            reg.register(name, fn, schema_map[name], needs_ctx=needs_ctx)
        return reg
