"""
agent/critic.py — Cross-validation LLM call that reviews all worker results.

One LLM call per task type (not a full agent loop). Groups results by
task_type, runs each task's Critic with its own system prompt and schema,
then merges all CritiqueResults into one.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

from agent.types import CritiqueResult, Result
from llm.client import LLMClient

_CONFIDENCE_FLAG_THRESHOLD = 0.5


def critique_results(
    llm: LLMClient,
    all_results: list[Result],
    task_registry: dict | None = None,
    verbose: bool = False,
) -> CritiqueResult:
    """Cross-validate all worker results, grouped by task_type.

    For each task type present in the results, one LLM call is made using
    that task's critic_system_prompt and critic_tool_schema. Confidence
    values are mutated in-place. Returns a merged CritiqueResult.
    Returns a no-op CritiqueResult on complete failure.
    """
    if not all_results:
        return _noop_critique("No results to critique.")

    task_registry = task_registry or {}

    # Group results by task_type
    by_type: dict[str, list[Result]] = defaultdict(list)
    for r in all_results:
        by_type[r.task_type].append(r)

    merged_adjustments: dict[str, float] = {}
    merged_anomalies: list[str] = []
    merged_assessments: list[str] = []

    for task_type, results in by_type.items():
        task_def = task_registry.get(task_type)
        if task_def is None:
            if verbose:
                print(
                    f"[critic] No TaskDefinition for '{task_type}'; skipping critique.",
                    file=sys.stderr,
                )
            merged_assessments.append(f"[{task_type}] No critic available.")
            continue

        cr = _critique_one_type(
            llm=llm,
            results=results,
            task_type=task_type,
            critic_system_prompt=task_def.critic_system_prompt,
            critic_tool_schema=task_def.critic_tool_schema,
            verbose=verbose,
        )
        merged_adjustments.update(cr.confidence_adjustments)
        merged_anomalies.extend(cr.anomaly_flags)
        merged_assessments.append(f"[{task_type}] {cr.overall_assessment}")

    flagged = [r.metric for r in all_results if r.confidence < _CONFIDENCE_FLAG_THRESHOLD]

    return CritiqueResult(
        confidence_adjustments=merged_adjustments,
        anomaly_flags=merged_anomalies,
        flagged_results=flagged,
        overall_assessment=" | ".join(merged_assessments) or "No critique performed.",
    )


def _critique_one_type(
    llm: LLMClient,
    results: list[Result],
    task_type: str,
    critic_system_prompt: str,
    critic_tool_schema: dict,
    verbose: bool = False,
) -> CritiqueResult:
    """Run one LLM critic call for a single task type's results."""
    results_json = json.dumps(
        [r.to_dict() for r in results],
        indent=2,
        default=str,
    )
    tool_name = critic_tool_schema.get("function", {}).get("name", "audit_results")
    user_msg = (
        f"Review these {task_type} results from parallel workers:\n\n"
        f"```json\n{results_json}\n```\n\n"
        f"Call {tool_name} with your cross-validation findings."
    )
    messages = [
        {"role": "system", "content": critic_system_prompt},
        {"role": "user",   "content": user_msg},
    ]

    try:
        resp = llm.chat(messages, tools=[critic_tool_schema])
    except Exception as exc:
        if verbose:
            print(
                f"[critic/{task_type}] LLM call failed ({exc}); returning no-op.",
                file=sys.stderr,
            )
        return _noop_critique()

    if not resp.tool_calls or resp.tool_calls[0].name != tool_name:
        if verbose:
            print(
                f"[critic/{task_type}] No {tool_name} call received; returning no-op.",
                file=sys.stderr,
            )
        return _noop_critique()

    args = resp.tool_calls[0].arguments or {}
    raw_adjustments: dict = args.get("confidence_adjustments", {})
    anomalies: list[str] = args.get("anomaly_flags", [])
    assessment: str = args.get("overall_assessment", "")

    adj_map: dict[str, float] = {}
    for metric, val in raw_adjustments.items():
        try:
            adj_map[metric] = max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            continue

    for r in results:
        if r.metric in adj_map:
            r.confidence = adj_map[r.metric]

    if verbose:
        print(
            f"[critic/{task_type}] {len(anomalies)} anomaly flag(s), "
            f"{len(adj_map)} adjustment(s).",
            file=sys.stderr,
        )

    return CritiqueResult(
        confidence_adjustments=adj_map,
        anomaly_flags=anomalies,
        flagged_results=[],   # merged by caller
        overall_assessment=assessment,
    )


def _noop_critique(assessment: str = "Critic unavailable; no adjustments applied.") -> CritiqueResult:
    return CritiqueResult(
        confidence_adjustments={},
        anomaly_flags=[],
        flagged_results=[],
        overall_assessment=assessment,
    )
