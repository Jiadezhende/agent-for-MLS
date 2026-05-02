"""
agents/core/prompts.py — Generic user message builder for worker agents.

Task-specific system prompts live in each agent plugin's prompt.py.
"""
from __future__ import annotations


def build_user_message(targets: list[str], retry_context: dict | None = None) -> str:
    """Render targets into the first user message for a worker agent."""
    if not targets:
        return "No targets specified. Call submit_results with an empty summary."

    if retry_context:
        return _build_retry_message(targets, retry_context)
    return _build_initial_message(targets)


def _build_initial_message(targets: list) -> str:
    lines = [
        "Process the following targets. "
        "For each, produce a record_measurement call with confidence ≥ 0.75.\n",
    ]
    for t in targets:
        lines.append(f"  • {t}" if isinstance(t, str) else f"  • {t.get('name', str(t))}")
    lines.append(
        "\nStart by calling list_skills to discover available strategies, "
        "then proceed target by target."
    )
    return "\n".join(lines)


def _build_retry_message(targets: list, retry_context: dict) -> str:
    reason = retry_context.get("reason", "")
    prev_bad: dict = retry_context.get("previous_bad_values", {})

    lines = [
        "RETRY: The Critic rejected your previous measurement(s). "
        "Re-measure ONLY the targets below using a DIFFERENT approach than before.\n",
        f"Critic feedback: {reason}\n",
        "Targets to re-measure (with rejected previous values):",
    ]
    for t in targets:
        name = t if isinstance(t, str) else t.get("name", str(t))
        bad = prev_bad.get(name)
        line = f"  • {name}"
        if bad:
            val = bad.get("value", "?")
            unit = bad.get("unit", "") or ""
            method = bad.get("method", "") or ""
            unit_str = f" {unit}" if unit else ""
            method_str = f" via {method}" if method else ""
            line += f"  [previous: {val}{unit_str}{method_str} — REJECTED]"
        lines.append(line)

    lines.append(
        "\nDo NOT repeat the same measurement method. "
        "Use read_skill to find an alternative approach, then re-measure and record."
    )
    return "\n".join(lines)
