"""Mutable per-run agent state.

The conversation history (``messages``) IS the agent's session memory.
We deliberately do not provide a separate generic KV store: structured
output flows out via ``terminate_payload`` (set on a tool's
``ToolResponse``); intermediate side-effects go to ``events`` /
``measurements``. Tools never read or write this object directly — the
loop is the only writer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mls_agent.llm.types import Message
from mls_agent.tools.response import Event, Measurement


@dataclass
class AgentContext:
    messages: list[Message] = field(default_factory=list)
    iteration: int = 0
    events: list[Event] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)

    def snapshot(self) -> dict:
        """A debug-friendly view; does not include raw provider blobs."""
        return {
            "iteration": self.iteration,
            "n_messages": len(self.messages),
            "n_events": len(self.events),
            "n_measurements": len(self.measurements),
        }
