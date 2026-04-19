"""
agents/core/agent.py — Abstract base class for all agent implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Agent(ABC):
    """Base class for Planner, Critic, and Worker agents."""

    @abstractmethod
    def run(self, *args, **kwargs):
        """Execute the agent's primary task. Subclasses define the signature."""

    def reset(self) -> None:
        """Reset any internal state between runs. No-op by default."""
