"""
tools/recording.py — Recording tools that write to AgentContext.

These tools need ctx injection (needs_ctx=True in the registry).
The _Terminated exception is defined here and re-raised by submit_results
so the agent loop can break out cleanly.
"""
from __future__ import annotations

from agent.tool_registry import _Terminated  # shared signal class
from agent.types import AgentContext, Result


# ---------------------------------------------------------------------------
# Recording functions
# ---------------------------------------------------------------------------

def record_measurement(
    ctx: AgentContext,
    metric: str,
    value: float | int | str | dict,
    unit: str | None,
    confidence: float,
    method: str,
    evidence: list[str],
) -> dict:
    """Append a confirmed measurement to ctx.results.

    Returns an error dict (not an exception) if evidence is empty — so the
    LLM receives it as structured feedback.
    """
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
    """Finalize and terminate the agent loop.

    Raises _Terminated, which the loop catches to exit cleanly.
    This function never returns normally.
    """
    # Store summary in memory so main.py can serialize it
    ctx.memory.set("run", "summary", summary)
    raise _Terminated(summary)
