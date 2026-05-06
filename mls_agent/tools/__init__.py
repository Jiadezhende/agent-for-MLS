"""Tool layer.

Tools are stateless with respect to ``AgentContext``: they MUST NOT read
or write the agent's session state. Side-effect requests (events,
measurements, termination) flow back through ``ToolResponse`` fields and
are applied to the context by the runtime loop.

Tool *instances* may hold external dependencies (executors, HTTP clients,
MCP handles) — those are constructor-injected, not framework-managed.
"""
from mls_agent.tools.base import Tool, ToolParameter
from mls_agent.tools.circuit_breaker import CircuitBreaker
from mls_agent.tools.registry import ToolRegistry
from mls_agent.tools.response import (
    Event,
    Measurement,
    ToolErrorCode,
    ToolResponse,
    ToolStatus,
)

__all__ = [
    "CircuitBreaker",
    "Event",
    "Measurement",
    "Tool",
    "ToolErrorCode",
    "ToolParameter",
    "ToolRegistry",
    "ToolResponse",
    "ToolStatus",
]
