"""
agents/tools/builtin/recording.py — Recording tools that write to AgentContext.

Context is injected by the registry via duck-typing: the registry sets _ctx
on any tool that carries that attribute before calling run().
"""
from __future__ import annotations

from typing import Any, Dict, List

from agent.core.types import AgentContext, Result
from agent.tools.base import Tool, ToolParameter
from agent.tools.registry import _Terminated
from agent.tools.response import ToolErrorCode, ToolResponse


class RecordMeasurementTool(Tool):
    """Record a confirmed hardware measurement in the results."""

    # Complex schema: value accepts number | string | object; unit is nullable;
    # confidence has min/max; evidence requires at least one item.
    # These constraints cannot be expressed via plain ToolParameter, so we
    # override to_openai_schema() and keep get_parameters() for documentation.

    _ctx: AgentContext | None = None  # set by registry before dispatch

    def __init__(self) -> None:
        super().__init__(
            name="record_measurement",
            description=(
                "Record a confirmed hardware measurement in the results. "
                "Every call MUST include at least one evidence string — "
                "a direct quote or path from a tool output earlier in this "
                "conversation. Do not call this with invented values."
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="metric",     type="string",  description="The metric name, matching the target_spec key."),
            ToolParameter(name="value",      type="string",  description="The measured value (number, string, or object)."),
            ToolParameter(name="unit",       type="string",  description="Unit of measurement (e.g. 'cycles', 'GB/s', 'MHz'), or null."),
            ToolParameter(name="confidence", type="number",  description="Confidence in this measurement (0–1)."),
            ToolParameter(name="method",     type="string",  description="Brief description of how this was measured."),
            ToolParameter(name="evidence",   type="array",   description="At least one string from a previous tool output."),
        ]

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {
                            "type": "string",
                            "description": "The metric name, matching the target_spec key.",
                        },
                        "value": {
                            "description": "The measured value (number, string, or object).",
                            "oneOf": [
                                {"type": "number"},
                                {"type": "string"},
                                {"type": "object"},
                            ],
                        },
                        "unit": {
                            "type": ["string", "null"],
                            "description": "Unit of measurement (e.g. 'cycles', 'GB/s', 'MHz').",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence in this measurement (0–1).",
                        },
                        "method": {
                            "type": "string",
                            "description": (
                                "Brief description of how this was measured "
                                "(e.g. 'pointer-chasing kernel with 256MB array > L2')."
                            ),
                        },
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": (
                                "At least one string from a previous tool output that "
                                "supports this measurement."
                            ),
                        },
                    },
                    "required": ["metric", "value", "unit", "confidence", "method", "evidence"],
                },
            },
        }

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        ctx = self._ctx
        evidence: list = parameters.get("evidence", [])
        if not evidence:
            return ToolResponse.error(
                code=ToolErrorCode.EMPTY_EVIDENCE,
                message=(
                    "record_measurement requires at least one evidence string. "
                    "Include a direct quote or reference from a previous tool output."
                ),
            )

        metric     = parameters["metric"]
        value      = parameters["value"]
        unit       = parameters.get("unit")
        confidence = float(parameters["confidence"])
        method     = parameters["method"]

        result = Result(
            metric=metric,
            value=value,
            unit=unit,
            confidence=confidence,
            method=method,
            evidence=list(evidence),
        )
        ctx.results.append(result)
        count = len(ctx.results)
        return ToolResponse.success(
            text=f"Recorded '{metric}' (total recorded: {count}).",
            data={"ok": True, "count": count, "metric": metric},
        )


class FlagEventTool(Tool):
    """Record an anomaly, decision, or observation for the engineering reasoning log."""

    _ctx: AgentContext | None = None

    def __init__(self) -> None:
        super().__init__(
            name="flag_event",
            description=(
                "Record an anomaly, decision, or observation for the engineering "
                "reasoning log. Use this when you detect a non-standard environment "
                "(e.g. clock throttling, SM masking, API spoofing) or make a "
                "significant methodological decision."
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="type",     type="string", description="Event category (e.g. 'clock_locked', 'strategy_switch')."),
            ToolParameter(name="severity", type="string", description="One of: info | warn | error."),
            ToolParameter(name="detail",   type="string", description="Human-readable explanation."),
        ]

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": (
                                "Event category. Examples: 'clock_locked', 'sm_masked', "
                                "'api_spoofed', 'retry_measurement', 'strategy_switch'."
                            ),
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["info", "warn", "error"],
                        },
                        "detail": {
                            "type": "string",
                            "description": "Human-readable explanation for the LLM-as-Judge.",
                        },
                    },
                    "required": ["type", "severity", "detail"],
                },
            },
        }

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        ctx = self._ctx
        event = {
            "iteration": ctx.iteration,
            "type":      parameters["type"],
            "severity":  parameters["severity"],
            "detail":    parameters["detail"],
        }
        ctx.events.append(event)
        return ToolResponse.success(text="Event recorded.", data={"ok": True})


class SubmitResultsTool(Tool):
    """Finalize and terminate the agent loop."""

    _ctx: AgentContext | None = None

    def __init__(self) -> None:
        super().__init__(
            name="submit_results",
            description=(
                "Finalize and submit all measured results. Call this exactly once "
                "when you have recorded all target metrics with acceptable confidence. "
                "After this call the agent loop terminates."
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="summary",
                type="string",
                description=(
                    "2–5 sentence summary of methodology, anomalies found, "
                    "and overall confidence. This is read by the LLM-as-Judge."
                ),
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        ctx = self._ctx
        summary = parameters["summary"]
        ctx.memory.set("run", "summary", summary)
        raise _Terminated(summary)
