"""
agents/core/prompts.py — Generic user message builder for worker agents.

Task-specific system prompts live in each agent plugin's prompt.py.
"""
from __future__ import annotations


def build_user_message(target_spec: dict) -> str:
    """Render the target spec into the first user message for a worker agent."""
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
    env_hints  = [h[6:] for h in strategy_hints if h.startswith("[env] ")]
    plan_hints = [h     for h in strategy_hints if not h.startswith("[env] ")]

    if env_hints:
        lines.append("\n## Environment status (check before calling executor tools):")
        for h in env_hints:
            lines.append(f"  - {h}")

    if plan_hints:
        lines.append("\n## Planner hints (follow these):")
        for h in plan_hints:
            lines.append(f"  - {h}")

    lines.append(
        "\nStart by calling list_skills to discover available measurement "
        "strategies, then proceed metric by metric."
    )
    return "\n".join(lines)
