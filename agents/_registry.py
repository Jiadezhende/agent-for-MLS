"""
agents/_registry.py — AgentDefinition dataclass and global agent registry.

Each agent type (hardware_probe, …) registers itself by calling register()
from its own __init__.py. The Orchestrator, Planner, and Critic look up
definitions here at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.core.agent import SubAgent


@dataclass
class AgentDefinition:
    """Everything the framework needs to run one agent type.

    agent_type           — unique string key (matches Step.worker)
    description          — capability + routing/grouping rules shown to the Planner LLM
    agent_class          — concrete SubAgent class instantiated per worker by _execute_one()
    required_tools       — tool names this agent needs; ToolFactory injects them
    critic_system_prompt — per-type system prompt for the Critic LLM call (fallback when
                           no task-level system_prompt_override is provided)
    max_tokens           — per-agent LLM output token budget; None = inherit global config
    """
    agent_type: str
    description: str
    agent_class: "type[SubAgent]"
    required_tools: list[str]
    critic_system_prompt: str
    max_tokens: int | None = None


_REGISTRY: dict[str, AgentDefinition] = {}


def register(defn: AgentDefinition) -> None:
    """Register an agent definition. Called from each agent plugin's __init__.py."""
    if defn.agent_type in _REGISTRY:
        raise ValueError(f"Agent type '{defn.agent_type}' is already registered.")
    _REGISTRY[defn.agent_type] = defn


def get(agent_type: str) -> AgentDefinition | None:
    """Return the AgentDefinition for a given agent type, or None if not found."""
    return _REGISTRY.get(agent_type)


def all_definitions() -> dict[str, AgentDefinition]:
    """Return a snapshot of all registered agent definitions."""
    return dict(_REGISTRY)
