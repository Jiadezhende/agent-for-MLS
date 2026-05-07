"""Explicit task-completion signal.

A pure-semantic tool: calling it ends the ReAct loop with
``AgentResult.reason="completed"`` and no payload. Use this when the
agent has finished its work via prior tool calls and there is nothing
left to do — separating "done on purpose" from the
``no_tool_call`` streak that fires when the LLM silently stops calling
tools.
"""
from __future__ import annotations

from typing import Any

from mls_agent.tools.base import Tool
from mls_agent.tools.response import ToolResponse


class TerminateTool(Tool):
    NAME = "terminate"
    DESCRIPTION = (
        "Call this when your task is fully done and you have nothing more "
        "to do. The loop ends immediately after this returns. No payload "
        "is needed — your prior tool calls are the work product."
    )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One-line description of what you accomplished.",
                },
            },
            "required": [],
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        summary = parameters.get("summary") or "task completed"
        return ToolResponse.terminate_with(summary=summary, payload=None)


def make_terminate_tool() -> TerminateTool:
    return TerminateTool()
