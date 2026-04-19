"""
agent/planner.py — Lightweight LLM call that decomposes targets into WorkerSpecs.

This is NOT a full agent loop. It makes one LLM call with a single forced
tool (assign_workers) and returns structured WorkerSpec objects.
Falls back to 1:1 mapping (one WorkerSpec per target) on any failure.
"""
from __future__ import annotations

import sys

from agent.types import WorkerSpec
from llm.client import LLMClient

_PLANNER_BASE_PROMPT = """\
You are a multi-agent task planner. Given a list of targets, assign each to
the most appropriate agent type and group related targets into workers.
Workers run in parallel, so targets that share measurement infrastructure
may be grouped to reduce overhead. Independent targets should go to separate
workers.

Available agent types:
{agent_types_block}

Call assign_workers exactly once with your groupings.\
"""

_FALLBACK_PROMPT = """\
You are a GPU benchmark task planner. Given a list of target metrics,
group them into worker assignments. Workers run in parallel so targets
that share measurement infrastructure (same CUDA kernel pattern) may be
grouped together to reduce compilation overhead. Targets that are
independent should be given to separate workers.

Grouping guidelines:
- dram_latency and dram_bandwidth can share a worker (same pointer-chase kernel)
- clock measurements should be isolated (they briefly alter GPU state)
- L1/L2 cache measurements can share a worker
- Default: 1 target per worker when no grouping rationale exists
- Maximum 8 workers total regardless of target count

Call assign_workers exactly once with your groupings.\
"""


def _build_system_prompt(task_registry: dict) -> str:
    """Build the Planner system prompt from registered task definitions."""
    if not task_registry:
        return _FALLBACK_PROMPT
    lines = []
    for defn in task_registry.values():
        lines.append(f"### {defn.task_type}")
        lines.append(f"  {defn.description}")
        lines.append(defn.planner_hints)
        lines.append("")
    agent_types_block = "\n".join(lines).rstrip()
    return _PLANNER_BASE_PROMPT.format(agent_types_block=agent_types_block)


def _build_schema(task_registry: dict) -> dict:
    """Build the assign_workers tool schema, including valid agent_type values."""
    valid_types = list(task_registry.keys()) if task_registry else ["hardware_probe"]
    return {
        "type": "function",
        "function": {
            "name": "assign_workers",
            "description": "Assign benchmark targets to parallel workers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "assignments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "agent_type": {
                                    "type": "string",
                                    "enum": valid_types,
                                    "description": "Which task agent type handles this group.",
                                },
                                "targets": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Target metric names for this worker.",
                                },
                                "strategy_hints": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Short hints for the worker.",
                                },
                                "group_rationale": {
                                    "type": "string",
                                    "description": "Why these targets are grouped together.",
                                },
                            },
                            "required": ["agent_type", "targets"],
                        },
                        "minItems": 1,
                    }
                },
                "required": ["assignments"],
            },
        },
    }


def plan_tasks(
    llm: LLMClient,
    targets: list,
    task_registry: dict | None = None,
    verbose: bool = False,
) -> list[WorkerSpec]:
    """Decompose targets into WorkerSpecs via one LLM call.

    Falls back to 1:1 mapping on any failure so the Orchestrator always
    has something to work with.
    """
    task_registry = task_registry or {}
    target_names = _normalize_target_names(targets)

    if not target_names:
        return []

    default_agent_type = next(iter(task_registry), "hardware_probe")

    if len(target_names) == 1:
        return [WorkerSpec(
            worker_id=0,
            targets=target_names,
            strategy_hints=[],
            agent_type=default_agent_type,
        )]

    system_prompt = _build_system_prompt(task_registry)
    schema = _build_schema(task_registry)

    user_msg = (
        "Plan parallel workers for these targets:\n"
        + "\n".join(f"  - {t}" for t in target_names)
        + "\n\nCall assign_workers with your groupings."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_msg},
    ]

    try:
        resp = llm.chat(messages, tools=[schema])
    except Exception as exc:
        if verbose:
            print(f"[planner] LLM call failed ({exc}); falling back to 1:1.", file=sys.stderr)
        return _fallback_plan(target_names, default_agent_type)

    if not resp.tool_calls or resp.tool_calls[0].name != "assign_workers":
        if verbose:
            print("[planner] No assign_workers call received; falling back to 1:1.", file=sys.stderr)
        return _fallback_plan(target_names, default_agent_type)

    args = resp.tool_calls[0].arguments or {}
    assignments = args.get("assignments", [])
    if not assignments:
        return _fallback_plan(target_names, default_agent_type)

    specs: list[WorkerSpec] = []
    for assignment in assignments:
        assigned_targets = assignment.get("targets", [])
        if not assigned_targets:
            continue
        agent_type = assignment.get("agent_type", default_agent_type)
        # Validate agent_type against registry
        if task_registry and agent_type not in task_registry:
            if verbose:
                print(
                    f"[planner] Unknown agent_type '{agent_type}'; "
                    f"falling back to '{default_agent_type}'.",
                    file=sys.stderr,
                )
            agent_type = default_agent_type
        specs.append(WorkerSpec(
            worker_id=0,   # renumbered below
            targets=assigned_targets,
            strategy_hints=assignment.get("strategy_hints", []),
            agent_type=agent_type,
            group_rationale=assignment.get("group_rationale", ""),
        ))

    if not specs:
        return _fallback_plan(target_names, default_agent_type)

    # Safety: ensure every target appears in at least one spec
    planned = {t for spec in specs for t in spec.targets}
    for t in target_names:
        if t not in planned:
            specs.append(WorkerSpec(
                worker_id=0,
                targets=[t],
                strategy_hints=[],
                agent_type=default_agent_type,
                group_rationale="fallback: missing from planner output",
            ))

    # Assign sequential worker_ids
    for i, spec in enumerate(specs):
        spec.worker_id = i

    if verbose:
        print(
            f"[planner] {len(specs)} worker(s): "
            + ", ".join(f"W{s.worker_id}({s.agent_type})={s.targets}" for s in specs),
            file=sys.stderr,
        )

    return specs


def _fallback_plan(target_names: list[str], default_agent_type: str) -> list[WorkerSpec]:
    return [
        WorkerSpec(
            worker_id=i,
            targets=[t],
            strategy_hints=[],
            agent_type=default_agent_type,
        )
        for i, t in enumerate(target_names)
    ]


def _normalize_target_names(targets: list) -> list[str]:
    """Convert targets list (str or dict) to a flat list of name strings."""
    names: list[str] = []
    for t in targets:
        if isinstance(t, dict):
            names.append(t.get("name", str(t)))
        else:
            names.append(str(t))
    return names
