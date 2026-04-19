"""
agents/tools/builtin/recording.py — Recording tools that write to AgentContext.

These tools need ctx injection (needs_ctx=True in the registry).
"""
from __future__ import annotations

from agents.tools.registry import _Terminated
from agents.core.types import AgentContext, Result


def record_measurement(
    ctx: AgentContext,
    metric: str,
    value: float | int | str | dict,
    unit: str | None,
    confidence: float,
    method: str,
    evidence: list[str],
) -> dict:
    """Append a confirmed measurement to ctx.results."""
    if not evidence:
        return {
            "error": "empty_evidence",
            "detail": (
                "record_measurement requires at least one evidence string. "
                "Include a direct quote or reference from a previous tool output."
            ),
        }

    result = Result(
        metric=metric,
        value=value,
        unit=unit,
        confidence=float(confidence),
        method=method,
        evidence=list(evidence),
        task_type=ctx.task.type,
    )
    ctx.results.append(result)
    return {"ok": True, "count": len(ctx.results), "metric": metric}


def flag_event(
    ctx: AgentContext,
    type: str,    # noqa: A002
    severity: str,
    detail: str,
) -> dict:
    """Append an anomaly / decision / observation to ctx.events."""
    event = {
        "iteration": ctx.iteration,
        "type": type,
        "severity": severity,
        "detail": detail,
    }
    ctx.events.append(event)
    return {"ok": True}


def submit_results(ctx: AgentContext, summary: str) -> None:
    """Finalize and terminate the agent loop. Never returns normally."""
    ctx.memory.set("run", "summary", summary)
    raise _Terminated(summary)
