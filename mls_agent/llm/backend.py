"""Provider-agnostic LLM backend protocol.

Backends accept normalized ``Message`` sequences and return a
``ChatResponse`` whose ``.message`` is also normalized. They are
responsible for absorbing every provider-specific wrinkle:

* parameter shape differences (``temperature`` vs not; ``max_tokens`` vs
  ``max_completion_tokens``)
* retries / backoff (rate limits, timeouts, connection errors)
* streaming vs non-streaming
* vendor quirks (GLM SSE artifact leakage, DeepSeek ``reasoning_content``)
* parsing the raw provider response into ``Message`` + ``ToolCall``

Callers above this layer never see provider SDK types or raw dicts.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from mls_agent.llm.types import ChatResponse, Message


@runtime_checkable
class LLMBackend(Protocol):
    """Send chat requests, get back a normalized response."""

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> ChatResponse: ...
