"""
agents/core/exceptions.py — Unified exception hierarchy for the agent pipeline.
"""
from __future__ import annotations

from typing import Any


class AgentError(Exception):
    """Base exception for all agent pipeline errors."""


class LLMError(AgentError):
    """LLM API call failed after all retries."""


class ToolError(AgentError):
    """A tool invocation failed."""


class ExecutorError(ToolError):
    """Raised for expected executor-level failures (compile error, bad path, …).

    error_class values:
      "user_code"      — the CUDA source or tool arguments are wrong; LLM should fix code.
      "infrastructure" — a binary is missing or env is misconfigured; LLM should NOT retry.
      "timeout"        — execution exceeded the time limit; LLM may reduce workload.
    """

    def __init__(
        self,
        kind: str,
        error_class: str = "infrastructure",
        hint: str | None = None,
        **details: Any,
    ) -> None:
        self.kind = kind
        self.error_class = error_class
        self.hint = hint
        self.details = details
        super().__init__(f"ExecutorError({kind}): {details}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "error",
            "error": self.kind,
            "error_class": self.error_class,
            "hint": self.hint,
            **self.details,
        }


class CircuitOpenError(ToolError):
    """Raised when a circuit breaker blocks a tool call."""

    def __init__(self, tool: str, kind: str) -> None:
        super().__init__(f"circuit open for ({tool}, {kind})")
        self.tool = tool
        self.kind = kind
