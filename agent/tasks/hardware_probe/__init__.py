"""
agent/tasks/hardware_probe — Task plugin for GPU hardware parameter measurement.

Importing this module registers the 'hardware_probe' TaskDefinition.
"""
from agent.tasks._registry import TaskDefinition, register
from agent.tasks.hardware_probe.critic_rules import AUDIT_SCHEMA, CRITIC_SYSTEM_PROMPT
from agent.tasks.hardware_probe.prompt import PLANNER_HINTS, SYSTEM_PROMPT
from agent.tasks.hardware_probe.tools import build_registry

register(TaskDefinition(
    task_type="hardware_probe",
    description=(
        "Measure GPU hardware parameters (latency, bandwidth, clock frequency) "
        "via CUDA C microbenchmarks and Nsight counters."
    ),
    system_prompt=SYSTEM_PROMPT,
    build_registry=build_registry,
    planner_hints=PLANNER_HINTS,
    critic_system_prompt=CRITIC_SYSTEM_PROMPT,
    critic_tool_schema=AUDIT_SCHEMA,
))
