"""
agents/tools/builtin/audit.py — AuditResultsTool used by CriticAgent.

Parses the Critic LLM's audit_results call, stores decisions as dicts in
AgentContext memory under ("audit", "decisions"), then raises _Terminated
to exit the AgentLoop cleanly.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from agents.core.types import AgentContext, CriticDecision
from agents.tools.base import Tool, ToolParameter
from agents.tools.registry import _Terminated
from agents.tools.response import ToolResponse


AUDIT_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "audit_results",
        "description": "Return per-step accept/retry decisions for all worker outputs.",
        "parameters": {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "description": "One decision per step_id.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_id": {
                                "type": "string",
                                "description": "The step_id from the worker output.",
                            },
                            "decision": {
                                "type": "string",
                                "enum": ["accept", "retry"],
                                "description": "'accept' if results are valid; 'retry' if suspicious.",
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Your confidence in the measurement quality (0–1).",
                            },
                            "reason": {
                                "type": "string",
                                "description": "1–2 sentences explaining the decision.",
                            },
                            "failing_targets": {
                                "type": "array",
                                "description": (
                                    "Names of targets that need re-measurement "
                                    "(subset of the step's targets). "
                                    "Leave empty only if ALL targets need retry."
                                ),
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "step_id", "decision", "confidence", "reason", "failing_targets"
                        ],
                    },
                },
            },
            "required": ["decisions"],
        },
    },
}


class AuditResultsTool(Tool):
    """Critic-side tool: validates audit decisions, stores them in memory, exits the loop."""

    _ctx: AgentContext | None = None  # injected by ToolRegistry before dispatch

    def __init__(self, outputs: dict[str, Any]) -> None:
        super().__init__(
            name="audit_results",
            description="Return per-step accept/retry decisions for all worker outputs.",
        )
        self._outputs = outputs  # step_id → WorkerOutput; used for step_id validation

    def get_parameters(self) -> List[ToolParameter]:
        return []  # schema fully defined by to_openai_schema override

    def to_openai_schema(self) -> Dict[str, Any]:
        return AUDIT_TOOL_SCHEMA

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        raw_decisions: list = parameters.get("decisions", [])
        decisions: list[CriticDecision] = []
        seen_ids: set[str] = set()

        for d in raw_decisions:
            sid = d.get("step_id", "")
            if sid not in self._outputs:
                continue
            seen_ids.add(sid)
            try:
                conf = max(0.0, min(1.0, float(d.get("confidence", 1.0))))
            except (TypeError, ValueError):
                conf = 1.0
            dec = d.get("decision", "accept")
            if dec not in ("accept", "retry"):
                dec = "accept"
            raw_failing = d.get("failing_targets", [])
            failing = (
                [m for m in raw_failing if isinstance(m, str)]
                if isinstance(raw_failing, list)
                else []
            )
            decisions.append(CriticDecision(
                step_id=sid,
                decision=dec,
                confidence=conf,
                reason=str(d.get("reason", "")),
                failing_targets=failing,
            ))

        # Fill in any step_ids the LLM omitted with an accept decision
        for sid in self._outputs:
            if sid not in seen_ids:
                decisions.append(CriticDecision(
                    step_id=sid, decision="accept",
                    confidence=1.0, reason="not reviewed",
                ))

        self._ctx.memory.set("audit", "decisions", [asdict(d) for d in decisions])
        raise _Terminated("audit complete")
