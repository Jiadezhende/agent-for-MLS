"""ReAct phases — explicit state for one iteration of the loop."""
from __future__ import annotations

from enum import Enum


class ReActPhase(Enum):
    """The discrete phases of one ReAct iteration.

    THOUGHT  → call the LLM, get back a ChatResponse
    VALIDATE → check finish_reason / tool_calls structure
    ACT      → dispatch each tool_call sequentially
    OBSERVE  → fold the ToolResponse into the conversation history
    APPLY    → land side-effects (events, measurements) on the context
    DECIDE   → terminate? max_iter? continue?
    """

    THOUGHT = "thought"
    VALIDATE = "validate"
    ACT = "act"
    OBSERVE = "observe"
    APPLY = "apply"
    DECIDE = "decide"
