"""
agent/tasks/hardware_probe/tools.py — ToolRegistry factory for hardware_probe workers.

Moved from agent/tool_registry.build_default_registry().
"""
from __future__ import annotations

from typing import Any


def build_registry(executor: Any):
    """Build and return a ToolRegistry wired for hardware_probe workers.

    executor — a live Executor instance (its public methods become tools).
    """
    from agent.tool_registry import ToolRegistry
    from agent.tool_schemas import TOOL_SCHEMAS
    from tools.recording import flag_event, record_measurement, submit_results
    from tools.skills import list_skills, read_skill

    schema_by_name = {s["function"]["name"]: s for s in TOOL_SCHEMAS}

    reg = ToolRegistry()

    # Knowledge tools
    reg.register("list_skills", list_skills, schema_by_name["list_skills"])
    reg.register("read_skill",  read_skill,  schema_by_name["read_skill"])

    # Executor tools
    reg.register("run_cuda_probe",     executor.run_cuda_probe,     schema_by_name["run_cuda_probe"])
    reg.register("profile_with_ncu",   executor.profile_with_ncu,   schema_by_name["profile_with_ncu"])
    reg.register("profile_with_nsys",  executor.profile_with_nsys,  schema_by_name["profile_with_nsys"])
    reg.register("profile_with_torch", executor.profile_with_torch, schema_by_name["profile_with_torch"])

    # Recording tools (need ctx injection)
    reg.register("record_measurement", record_measurement, schema_by_name["record_measurement"], needs_ctx=True)
    reg.register("flag_event",         flag_event,         schema_by_name["flag_event"],         needs_ctx=True)
    reg.register("submit_results",     submit_results,     schema_by_name["submit_results"],     needs_ctx=True)

    return reg
