"""mls_agent — Standardized ReAct agent framework.

Quick start::

    from mls_agent import (
        Agent, AgentConfig, ToolRegistry, OpenAIBackend, LLMConfig,
        StdoutObserver,
    )

The framework has three layers, each importable independently:

* ``mls_agent.llm`` — provider-agnostic Message types + LLMBackend protocol
* ``mls_agent.tools`` — Tool ABC + ToolResponse + ToolRegistry + CircuitBreaker
* ``mls_agent.runtime`` — Agent + ReActLoop + AgentContext + observers

Tools are stateless w.r.t. the agent's session state but may carry their
own external dependencies (executors, HTTP clients, MCP handles).
"""
from mls_agent.llm import (
    ChatResponse,
    FinishReason,
    LLMBackend,
    LLMConfig,
    Message,
    Role,
    ToolCall,
)
from mls_agent.llm.openai_backend import OpenAIBackend
from mls_agent.runtime import (
    Agent,
    AgentBackendError,
    AgentConfig,
    AgentContext,
    AgentObserver,
    AgentResult,
    MlsAgentError,
    NullObserver,
    ReActLoop,
    ReActPhase,
    StdoutObserver,
    TerminationReason,
)
from mls_agent.tools import (
    CircuitBreaker,
    Event,
    Measurement,
    Tool,
    ToolErrorCode,
    ToolParameter,
    ToolRegistry,
    ToolResponse,
    ToolStatus,
)

__all__ = [
    # llm
    "ChatResponse",
    "FinishReason",
    "LLMBackend",
    "LLMConfig",
    "Message",
    "OpenAIBackend",
    "Role",
    "ToolCall",
    # tools
    "CircuitBreaker",
    "Event",
    "Measurement",
    "Tool",
    "ToolErrorCode",
    "ToolParameter",
    "ToolRegistry",
    "ToolResponse",
    "ToolStatus",
    # runtime
    "Agent",
    "AgentBackendError",
    "AgentConfig",
    "AgentContext",
    "AgentObserver",
    "AgentResult",
    "MlsAgentError",
    "NullObserver",
    "ReActLoop",
    "ReActPhase",
    "StdoutObserver",
    "TerminationReason",
]
