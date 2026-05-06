"""Tool response protocol + side-effect payloads.

ToolResponse is the *only* channel a tool uses to talk back to the runtime.
Beyond the textual/structured payload that gets shown to the LLM, it can
carry three kinds of side-effects that the loop applies to AgentContext:

  * ``events`` — append-only audit entries (anomalies, decisions)
  * ``measurements`` — append-only measurement records
  * ``terminate`` (+ summary, payload) — signals the loop to stop and return

Tools NEVER reach into AgentContext directly; the loop is the only writer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Enums / error codes
# ---------------------------------------------------------------------------


class ToolStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"   # result with caveats (timeout, truncation)
    ERROR = "error"


class ToolErrorCode:
    """String constants for the ``error_info["code"]`` field."""

    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGS = "invalid_args"
    CIRCUIT_OPEN = "circuit_open"
    EXECUTION_ERROR = "execution_error"
    INTERNAL_ERROR = "internal_error"


# ---------------------------------------------------------------------------
# Side-effect payloads
# ---------------------------------------------------------------------------


_VALID_SEVERITIES = ("info", "warn", "error")


@dataclass(frozen=True)
class Event:
    """A single event the tool wants the runtime to record on ctx.events."""

    type: str
    severity: Literal["info", "warn", "error"]
    detail: str

    def __post_init__(self) -> None:
        if not self.type:
            raise ValueError("Event.type must be non-empty")
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"Event.severity must be one of {_VALID_SEVERITIES}, "
                f"got {self.severity!r}"
            )


@dataclass(frozen=True)
class Measurement:
    """A measurement to be appended to ctx.measurements.

    ``confidence`` is constrained to [0, 1]. ``evidence`` must be non-empty:
    every measurement must point back to at least one observation
    (raw stdout line, profiler row, …) so a downstream judge can verify it.
    """

    metric: str
    value: float | int | str | dict
    unit: str | None
    confidence: float
    method: str
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("Measurement.metric must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Measurement.confidence must be in [0, 1], got {self.confidence}"
            )
        if not isinstance(self.evidence, tuple):
            raise TypeError(
                f"Measurement.evidence must be tuple, "
                f"got {type(self.evidence).__name__}"
            )
        if not self.evidence:
            raise ValueError("Measurement.evidence must be non-empty")
        if not self.method:
            raise ValueError("Measurement.method must be non-empty")


# ---------------------------------------------------------------------------
# ToolResponse
# ---------------------------------------------------------------------------


@dataclass
class ToolResponse:
    """Standardized tool response."""

    status: ToolStatus
    text: str
    data: dict[str, Any] = field(default_factory=dict)
    error_info: dict[str, str] | None = None
    stats: dict[str, Any] | None = None

    # Side effects applied by the runtime loop after dispatch
    terminate: bool = False
    terminate_summary: str | None = None
    terminate_payload: dict | None = None
    events: tuple[Event, ...] = ()
    measurements: tuple[Measurement, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, ToolStatus):
            raise TypeError(
                f"ToolResponse.status must be ToolStatus, "
                f"got {type(self.status).__name__}"
            )
        if self.status == ToolStatus.ERROR and not self.error_info:
            raise ValueError("ERROR status requires error_info")
        if self.status != ToolStatus.ERROR and self.error_info:
            raise ValueError("error_info only allowed when status is ERROR")
        if self.terminate and self.status == ToolStatus.ERROR:
            raise ValueError("terminate=True is incompatible with status=ERROR")
        if (self.terminate_summary is not None or self.terminate_payload is not None) and not self.terminate:
            raise ValueError(
                "terminate_summary / terminate_payload set but terminate is False"
            )
        if not isinstance(self.events, tuple):
            raise TypeError(
                f"ToolResponse.events must be tuple, "
                f"got {type(self.events).__name__}"
            )
        if not isinstance(self.measurements, tuple):
            raise TypeError(
                f"ToolResponse.measurements must be tuple, "
                f"got {type(self.measurements).__name__}"
            )
        if self.terminate_payload is not None:
            # Strict JSON: payloads are persisted as the final agent output;
            # callers should convert non-JSON values (Path, datetime) to str
            # themselves rather than relying on a permissive default.
            try:
                json.dumps(self.terminate_payload)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"terminate_payload must be JSON-serializable: {e}"
                ) from None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status.value,
            "text": self.text,
            "data": self.data,
        }
        if self.error_info:
            result["error"] = self.error_info
        if self.stats:
            result["stats"] = self.stats
        return result

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def success(
        cls,
        text: str,
        data: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
        events: tuple[Event, ...] = (),
        measurements: tuple[Measurement, ...] = (),
    ) -> "ToolResponse":
        return cls(
            status=ToolStatus.SUCCESS,
            text=text,
            data=data or {},
            stats=stats,
            events=events,
            measurements=measurements,
        )

    @classmethod
    def partial(
        cls,
        text: str,
        data: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
        events: tuple[Event, ...] = (),
        measurements: tuple[Measurement, ...] = (),
    ) -> "ToolResponse":
        return cls(
            status=ToolStatus.PARTIAL,
            text=text,
            data=data or {},
            stats=stats,
            events=events,
            measurements=measurements,
        )

    @classmethod
    def error(
        cls,
        code: str,
        message: str,
        data: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
        events: tuple[Event, ...] = (),
    ) -> "ToolResponse":
        return cls(
            status=ToolStatus.ERROR,
            text=message,
            data=data or {},
            error_info={"code": code, "message": message},
            stats=stats,
            events=events,
        )

    @classmethod
    def terminate_with(
        cls,
        summary: str,
        payload: dict | None = None,
        text: str | None = None,
        data: dict[str, Any] | None = None,
        events: tuple[Event, ...] = (),
        measurements: tuple[Measurement, ...] = (),
    ) -> "ToolResponse":
        return cls(
            status=ToolStatus.SUCCESS,
            text=text if text is not None else summary,
            data=data or {},
            terminate=True,
            terminate_summary=summary,
            terminate_payload=payload,
            events=events,
            measurements=measurements,
        )
