"""
agents/core/agent.py — Abstract base classes for all agent implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.core.types import Step, WorkerOutput
    from agent.tools.registry import ToolRegistry


class Agent(ABC):
    """Marker base. Planner and Critic are direct subclasses."""

    @abstractmethod
    def run(self, *args, **kwargs): ...

    def reset(self) -> None:
        """Reset any internal state between runs. No-op by default."""


class SubAgent(Agent):
    """Base for all subagents (workers) routed via run_subagent tool.

    _execute_one() calls run(step, tools) polymorphically — all subagents
    must satisfy this typed contract. Class-level attributes are injection
    points set by _execute_one() before run() is called.
    """

    run_id: str | None = None
    agent_id: str | None = None
    shared_store: Any = None

    @abstractmethod
    def run(self, step: "Step", tools: "ToolRegistry") -> "WorkerOutput": ...
