"""
agents/tools/registry.py — Maps tool names to Tool objects.

Circuit breaker applies uniformly to every registered tool.
Context injection for recording tools uses duck-typing: if a Tool has a _ctx
attribute the registry sets it before dispatch (no ABC required).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import jsonschema

from agents.core.types import AgentContext
from agents.tools.base import Tool
from agents.tools.circuit_breaker import CircuitBreaker
from agents.tools.response import ToolErrorCode, ToolResponse, ToolStatus


# ---------------------------------------------------------------------------
# Internal exception for clean termination
# ---------------------------------------------------------------------------

class _Terminated(Exception):
    """Raised by submit_results to unwind the agent loop cleanly."""
    def __init__(self, summary: str) -> None:
        self.summary = summary
        super().__init__(summary)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _update_circuit_breaker(cb: CircuitBreaker, tool: str, response: ToolResponse) -> None:
    if response.status == ToolStatus.ERROR:
        code = (response.error_info or {}).get("code", "unknown")
        cb.record_failure(tool, code)
    else:
        cb.record_success(tool)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def dispatch(self, name: str, args_dict: Any, ctx: AgentContext) -> ToolResponse:
        """Execute a tool call. Returns ToolResponse always.

        Circuit breaker check and update apply to all registered tools.
        Context is injected into tools that carry a _ctx attribute.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResponse.error(
                code=ToolErrorCode.UNKNOWN_TOOL,
                message=(
                    f"Unknown tool '{name}'. "
                    f"Valid tools: {list(self._tools.keys())}"
                ),
            )

        # Universal circuit breaker check
        open_for_tool = [
            (t, ek) for (t, ek) in ctx.circuit_breaker.open_circuits() if t == name
        ]
        if open_for_tool:
            open_kinds = [ek for _, ek in open_for_tool]
            counts = {ek: ctx.circuit_breaker.failure_count(name, ek) for ek in open_kinds}
            return ToolResponse.error(
                code=ToolErrorCode.CIRCUIT_OPEN,
                message=(
                    f"Tool '{name}' has failed {ctx.circuit_breaker.threshold}+ times "
                    f"with errors {open_kinds}. Circuit is open — stop retrying this approach. "
                    "Use a different tool or call submit_results with current findings."
                ),
                stats={
                    "tool": name,
                    "open_error_kinds": open_kinds,
                    "failure_counts": counts,
                },
            )

        if args_dict is None:
            args_dict = {}

        # JSON schema validation
        param_schema = (
            tool.to_openai_schema().get("function", {}).get("parameters", {})
        )
        if param_schema:
            validator = jsonschema.Draft202012Validator(param_schema, format_checker=None)
            errors = list(validator.iter_errors(args_dict))
            if errors:
                first = errors[0]
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_ARGS,
                    message=f"{first.message} (path: {list(first.absolute_path)})",
                )

        # Context injection (duck-typing — no ABC needed)
        if hasattr(tool, "_ctx"):
            tool._ctx = ctx  # type: ignore[attr-defined]

        try:
            result = tool.run(args_dict)
        except _Terminated:
            raise
        except Exception as exc:  # noqa: BLE001
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"{exc.__class__.__name__}: {exc}",
            )

        if not isinstance(result, ToolResponse):
            # Defensive: wrap any stray dict/value returned by a tool
            result = ToolResponse.success(
                text=str(result),
                data=result if isinstance(result, dict) else {"result": result},
            )

        # Universal circuit breaker update
        _update_circuit_breaker(ctx.circuit_breaker, name, result)

        return result


# ---------------------------------------------------------------------------
# ToolFactory — builds ToolRegistry by name, injecting executor
# ---------------------------------------------------------------------------

class ToolFactory:
    """Creates ToolRegistry instances from a tool-name list.

    The executor is injected once at construction; Tool objects are created
    fresh for each registry build so each worker gets isolated instances.
    """

    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def build(self, tool_names: list[str]) -> ToolRegistry:
        from agents.tools.builtin.recording import (
            FlagEventTool,
            RecordMeasurementTool,
            SubmitResultsTool,
        )
        from agents.tools.builtin.skills import ListSkillsTool, ReadSkillTool
        from agents.tools.executor_tools import (
            ProfileWithNcuTool,
            ProfileWithNsysTool,
            ProfileWithTorchTool,
            ProbeEnvironmentTool,
            RunCudaProbeTool,
        )

        all_tools: dict[str, Tool] = {
            "list_skills":        ListSkillsTool(),
            "read_skill":         ReadSkillTool(),
            "run_cuda_probe":     RunCudaProbeTool(self._executor),
            "profile_with_ncu":   ProfileWithNcuTool(self._executor),
            "profile_with_nsys":  ProfileWithNsysTool(self._executor),
            "profile_with_torch": ProfileWithTorchTool(self._executor),
            "probe_environment":  ProbeEnvironmentTool(self._executor),
            "record_measurement": RecordMeasurementTool(),
            "flag_event":         FlagEventTool(),
            "submit_results":     SubmitResultsTool(),
        }

        reg = ToolRegistry()
        for name in tool_names:
            if name not in all_tools:
                raise ValueError(
                    f"Unknown tool '{name}'. Available: {list(all_tools)}"
                )
            reg.register(all_tools[name])
        return reg
