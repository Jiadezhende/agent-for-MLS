"""Side-effect recording tools.

These tools never reach into ``AgentContext`` directly — they describe the
side effect they want via ``ToolResponse.events`` / ``ToolResponse.measurements``,
and the runtime loop appends them to the context after dispatch. This keeps
the framework's "tools are stateless w.r.t. context" invariant.
"""
from __future__ import annotations

from typing import Any

from mls_agent.tools.base import Tool
from mls_agent.tools.response import Event, Measurement, ToolErrorCode, ToolResponse


_VALID_SEVERITIES = ("info", "warn", "error")


class RecordMeasurementTool(Tool):
    NAME = "record_measurement"
    DESCRIPTION = (
        "Record a confirmed hardware/runtime measurement. Every call MUST "
        "include at least one evidence string — a direct quote, file path, or "
        "stdout line from a prior tool output. Do not record invented values."
    )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "description": "Metric name (e.g. 'dram_bw_gbps', 'sm_count').",
                },
                "value": {
                    "description": "Measured value — number, string, or object.",
                    "oneOf": [
                        {"type": "number"},
                        {"type": "string"},
                        {"type": "object"},
                    ],
                },
                "unit": {
                    "type": ["string", "null"],
                    "description": "Unit of measurement (e.g. 'GB/s', 'MHz'); null when dimensionless.",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in this measurement, in [0, 1].",
                },
                "method": {
                    "type": "string",
                    "description": (
                        "How the value was obtained, e.g. "
                        "'pointer-chasing kernel with 256MB array > L2'."
                    ),
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "≥1 strings from prior tool outputs that justify this measurement.",
                },
            },
            "required": ["metric", "value", "unit", "confidence", "method", "evidence"],
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        try:
            measurement = Measurement(
                metric=parameters["metric"],
                value=parameters["value"],
                unit=parameters.get("unit"),
                confidence=float(parameters["confidence"]),
                method=parameters["method"],
                evidence=tuple(parameters["evidence"]),
            )
        except (TypeError, ValueError) as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message=f"invalid measurement: {exc}",
            )
        return ToolResponse.success(
            text=f"recorded measurement {measurement.metric}={measurement.value}",
            data={"metric": measurement.metric},
            measurements=(measurement,),
        )


class FlagEventTool(Tool):
    NAME = "flag_event"
    DESCRIPTION = (
        "Record an anomaly, decision, or observation for the engineering "
        "reasoning log. Use when detecting non-standard environment behavior "
        "(clock throttling, SM masking, profiler permission issues) or when "
        "making a significant methodological choice."
    )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": (
                        "Event category. Examples: 'clock_locked', 'sm_masked', "
                        "'retry_measurement', 'strategy_switch'."
                    ),
                },
                "severity": {
                    "type": "string",
                    "enum": list(_VALID_SEVERITIES),
                },
                "detail": {
                    "type": "string",
                    "description": "Human-readable explanation.",
                },
            },
            "required": ["type", "severity", "detail"],
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        try:
            event = Event(
                type=parameters["type"],
                severity=parameters["severity"],
                detail=parameters["detail"],
            )
        except (TypeError, ValueError) as exc:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message=f"invalid event: {exc}",
            )
        return ToolResponse.success(
            text=f"flagged event {event.type} ({event.severity})",
            data={"type": event.type},
            events=(event,),
        )


def make_side_effect_tools() -> tuple[Tool, ...]:
    """Construct fresh instances of every side-effect tool."""
    return (RecordMeasurementTool(), FlagEventTool())
