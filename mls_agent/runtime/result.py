"""Final result returned by Agent.run()."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mls_agent.runtime.context import AgentContext

TerminationReason = Literal[
    "completed",        # tool explicitly requested termination
    "max_iterations",   # hit the iteration ceiling without termination
    "no_tool_call",     # LLM stopped calling tools after the configured streak
    "llm_error",        # backend.chat raised after retries exhausted
]


@dataclass(frozen=True)
class AgentResult:
    reason: TerminationReason
    summary: str | None
    payload: dict | None
    iterations: int
    context: AgentContext

    def __post_init__(self) -> None:
        valid = ("completed", "max_iterations", "no_tool_call", "llm_error")
        if self.reason not in valid:
            raise ValueError(
                f"AgentResult.reason must be one of {valid}, got {self.reason!r}"
            )
        if self.iterations < 0:
            raise ValueError(
                f"AgentResult.iterations must be >= 0, got {self.iterations}"
            )
