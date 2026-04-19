"""
agent/tasks/_registry.py — TaskDefinition dataclass and global plugin registry.

Each task type (hardware_probe, op_profiler, …) registers itself by calling
register() from its own __init__.py. The Orchestrator, Planner, and Critic
look up definitions here at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class TaskDefinition:
    """Everything the framework needs to run one task type.

    task_type         — unique string key, matches WorkerSpec.agent_type
    description       — one-line description shown to the Planner LLM
    system_prompt     — system prompt injected into each worker's AgentLoop
    build_registry    — (executor) -> ToolRegistry; called per worker
    planner_hints     — grouping/routing rules shown to the Planner LLM
    critic_system_prompt — system prompt for the Critic LLM call
    critic_tool_schema   — forced-tool JSON schema for the Critic call
    """
    task_type: str
    description: str
    system_prompt: str
    build_registry: Callable[[Any], Any]
    planner_hints: str
    critic_system_prompt: str
    critic_tool_schema: dict


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, TaskDefinition] = {}


def register(defn: TaskDefinition) -> None:
    """Register a task definition. Called from each task plugin's __init__.py."""
    if defn.task_type in _REGISTRY:
        raise ValueError(f"Task type '{defn.task_type}' is already registered.")
    _REGISTRY[defn.task_type] = defn


def get(task_type: str) -> TaskDefinition | None:
    """Return the TaskDefinition for a given task type, or None if not found."""
    return _REGISTRY.get(task_type)


def all_definitions() -> dict[str, TaskDefinition]:
    """Return a snapshot of all registered task definitions."""
    return dict(_REGISTRY)
