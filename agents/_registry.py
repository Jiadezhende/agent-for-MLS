"""
agents/_registry.py — AgentDefinition dataclass and global agent registry.

Each agent type (hardware_probe, …) registers itself by calling register()
from its own __init__.py. The Orchestrator, Planner, and Critic look up
definitions here at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentDefinition:
    """Everything the framework needs to run one agent type.

    agent_type         — unique string key (matches Step.worker)
    description        — one-line description shown to the Planner LLM
    agent_class        — concrete class instantiated per worker by the Orchestrator
    required_tools     — tool names this agent needs; ToolFactory injects them
    planner_hints      — grouping/routing rules shown to the Planner LLM
    critic_system_prompt — system prompt for the Critic LLM call
    critic_tool_schema   — forced-tool JSON schema for the Critic call
    """
    agent_type: str
    description: str
    agent_class: type
    required_tools: list[str]
    planner_hints: str
    critic_system_prompt: str
    critic_tool_schema: dict


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
