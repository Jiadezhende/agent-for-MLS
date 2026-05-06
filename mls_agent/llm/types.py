"""Provider-agnostic message / tool-call / response types.

These dataclasses are the *only* types that cross the LLM-backend boundary.
Backends are responsible for translating to/from provider-specific formats
(OpenAI dicts, Anthropic blocks, …) inside their implementation; callers
above the backend layer must never see provider SDK objects.

Invariants are enforced in ``__post_init__`` so any malformed Message /
ToolCall raised at construction time, not later when it surfaces in a
serialization call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
FinishReason = Literal[
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "other",
]


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the assistant.

    ``arguments`` is always a dict — even if the model returned malformed
    JSON. In that case, ``arguments`` is ``{}`` and ``parse_error`` carries
    the JSON decode message; callers can show it back to the model so it
    can self-correct on the next turn.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    arguments_raw: str = ""
    parse_error: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ToolCall.id must be non-empty")
        if not self.name:
            raise ValueError("ToolCall.name must be non-empty")
        if not isinstance(self.arguments, dict):
            raise TypeError(
                f"ToolCall.arguments must be dict, got {type(self.arguments).__name__}"
            )


@dataclass(frozen=True)
class Message:
    """One chat message in normalized form.

    Mutual-exclusion invariants:
      - ``tool_calls`` is non-empty only when ``role == 'assistant'``
      - ``tool_call_id`` is set only when ``role == 'tool'``
      - tool messages must carry a string content (the tool's textual result)
    """

    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None

    def __post_init__(self) -> None:
        valid_roles = ("system", "user", "assistant", "tool")
        if self.role not in valid_roles:
            raise ValueError(
                f"Message.role must be one of {valid_roles}, got {self.role!r}"
            )
        if self.tool_calls and self.role != "assistant":
            raise ValueError(
                f"tool_calls allowed only on assistant role, got role={self.role!r}"
            )
        if self.tool_call_id is not None and self.role != "tool":
            raise ValueError(
                f"tool_call_id allowed only on tool role, got role={self.role!r}"
            )
        if self.role == "tool":
            if self.tool_call_id is None:
                raise ValueError("tool message requires tool_call_id")
            if self.content is None:
                raise ValueError("tool message requires non-None content")
        if not isinstance(self.tool_calls, tuple):
            raise TypeError(
                f"Message.tool_calls must be tuple, got {type(self.tool_calls).__name__}"
            )

    # -- factory helpers -------------------------------------------------

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: str | None = None,
        tool_calls: tuple[ToolCall, ...] = (),
        reasoning_content: str | None = None,
    ) -> "Message":
        return cls(
            role="assistant",
            content=content,
            tool_calls=tuple(tool_calls),
            reasoning_content=reasoning_content,
        )

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


@dataclass(frozen=True)
class ChatResponse:
    """A single LLM chat response.

    Container for an assistant ``Message`` plus call-level metadata
    (finish reason, token usage). Callers typically do
    ``ctx.messages.append(response.message)`` and inspect
    ``response.finish_reason`` for control-flow decisions.
    """

    message: Message
    finish_reason: FinishReason
    usage: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.message.role != "assistant":
            raise ValueError(
                "ChatResponse.message must have role='assistant', "
                f"got {self.message.role!r}"
            )
        valid_reasons = ("stop", "length", "tool_calls", "content_filter", "other")
        if self.finish_reason not in valid_reasons:
            raise ValueError(
                f"finish_reason must be one of {valid_reasons}, "
                f"got {self.finish_reason!r}"
            )
