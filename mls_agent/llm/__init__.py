"""LLM adapter layer.

Exposes the provider-agnostic chat interface and message types. The only
file in this package allowed to import provider SDKs is ``openai_backend``.
"""
from mls_agent.llm.backend import LLMBackend
from mls_agent.llm.config import LLMConfig
from mls_agent.llm.types import (
    ChatResponse,
    FinishReason,
    Message,
    Role,
    ToolCall,
)

__all__ = [
    "ChatResponse",
    "FinishReason",
    "LLMBackend",
    "LLMConfig",
    "Message",
    "Role",
    "ToolCall",
]
