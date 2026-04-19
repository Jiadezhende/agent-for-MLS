"""
agent/prompts.py — Generic user message builder.

Task-specific system prompts live in agent/tasks/<task_type>/prompt.py.
This module provides build_user_message(), which is generic across all
task types (renders a list of targets + optional strategy hints).
"""
from __future__ import annotations


def build_user_message(target_spec: dict) -> str:
    """Render the target spec into the first user message.

    Keeping the spec here (not in the system prompt) lets the system prompt
    be cached across multiple runs with different specs.
    """
    targets = target_spec.get("targets", [])
    if not targets:
        return "No targets specified. Call submit_results with an empty summary."

    lines = [
        "Measure the following GPU hardware parameters. "
        "For each, produce a record_measurement call with confidence ≥ 0.75.\n",
    ]
    for t in targets:
        if isinstance(t, dict):
            name = t.get("name", str(t))
            desc = t.get("description", "")
            unit = t.get("unit", "")
            line = f"  • {name}"
            if unit:
                line += f"  [{unit}]"
            if desc:
                line += f"  — {desc}"
            lines.append(line)
        else:
            lines.append(f"  • {t}")

    strategy_hints = target_spec.get("strategy_hints", [])
    if strategy_hints:
        lines.append("\n## Planner hints (follow these):")
        for h in strategy_hints:
            lines.append(f"  - {h}")

    lines.append(
        "\nStart by calling list_skills to discover available measurement "
        "strategies, then proceed metric by metric."
    )
    return "\n".join(lines)
