"""
agents/agents/critic_agent.py — CriticAgent: cross-validates worker outputs and
returns structured accept/retry decisions.

One LLM call per agent type present in the outputs. Falls back to all-accept
on any failure so the pipeline can always continue.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

from agents.core.agent import Agent
from agents.core.llm import LLMClient
from agents.core.types import CriticDecision, WorkerOutput


# Shared audit tool schema used by all agent types and task-level evaluation.
# The audit_results format (decisions array with step_id/decision/confidence/reason/
# failing_targets) is identical regardless of which agent type produced the outputs.
AUDIT_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "audit_results",
        "description": "Return per-step accept/retry decisions for all worker outputs.",
        "parameters": {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "description": "One decision per step_id.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_id": {
                                "type": "string",
                                "description": "The step_id from the worker output.",
                            },
                            "decision": {
                                "type": "string",
                                "enum": ["accept", "retry"],
                                "description": "'accept' if results are valid; 'retry' if suspicious.",
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Your confidence in the measurement quality (0–1).",
                            },
                            "reason": {
                                "type": "string",
                                "description": "1–2 sentences explaining the decision.",
                            },
                            "failing_targets": {
                                "type": "array",
                                "description": (
                                    "Names of targets that need re-measurement "
                                    "(subset of the step's targets). "
                                    "Leave empty only if ALL targets need retry."
                                ),
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["step_id", "decision", "confidence", "reason", "failing_targets"],
                    },
                },
            },
            "required": ["decisions"],
        },
    },
}


class CriticAgent(Agent):
    """Wraps per-type critic LLM calls as a standard agent with a run() interface.

    Groups WorkerOutputs by their agent_type, then runs one LLM call per type
    using that type's critic_system_prompt and critic_tool_schema.
    Returns list[CriticDecision] with accept/retry decisions.
    """

    def __init__(
        self,
        llm: LLMClient,
        agent_registry: dict,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.agent_registry = agent_registry
        self.verbose = verbose

    def run(
        self,
        outputs: dict[str, WorkerOutput],
        retry_counts: dict[str, int] | None = None,
        system_prompt_override: str | None = None,
        critic_tool_schema_override: dict | None = None,
    ) -> list[CriticDecision]:
        """Cross-validate all worker outputs.

        outputs:                  step_id → WorkerOutput
        retry_counts:             step_id → retries already performed (0 = first attempt)
        system_prompt_override:   if set, use this prompt instead of per-agent-type prompts
                                  (used by Orchestrator for task-level evaluation)
        critic_tool_schema_override: if set, use this tool schema for all types
        Returns list[CriticDecision] with one decision per step.
        """
        if not outputs:
            return _accept_decisions({})

        counts = retry_counts or {}
        all_decisions: list[CriticDecision] = []

        if system_prompt_override is not None:
            # Task-level evaluation: one call for all outputs using the provided prompt
            schema = critic_tool_schema_override or AUDIT_TOOL_SCHEMA
            decisions = self._critique_one_type(
                agent_type="task",
                outputs=outputs,
                retry_counts=counts,
                critic_system_prompt=system_prompt_override,
                critic_tool_schema=schema,
            )
            return decisions

        by_type = self._group_by_type(outputs)
        for agent_type, typed_outputs in by_type.items():
            defn = self.agent_registry.get(agent_type)
            if defn is None:
                if self.verbose:
                    print(f"[critic] No definition for '{agent_type}'; accepting all.", file=sys.stderr)
                all_decisions.extend(_accept_decisions(typed_outputs))
                continue

            decisions = self._critique_one_type(
                agent_type=agent_type,
                outputs=typed_outputs,
                retry_counts=counts,
                critic_system_prompt=defn.critic_system_prompt,
                critic_tool_schema=AUDIT_TOOL_SCHEMA,
            )
            all_decisions.extend(decisions)

        return all_decisions

    def _group_by_type(
        self, outputs: dict[str, WorkerOutput]
    ) -> dict[str, dict[str, WorkerOutput]]:
        fallback = next(iter(self.agent_registry), "hardware_probe")
        grouped: dict[str, dict[str, WorkerOutput]] = defaultdict(dict)
        for step_id, out in outputs.items():
            grouped[out.agent_type or fallback][step_id] = out
        return dict(grouped)

    def _critique_one_type(
        self,
        agent_type: str,
        outputs: dict[str, WorkerOutput],
        retry_counts: dict[str, int],
        critic_system_prompt: str,
        critic_tool_schema: dict,
    ) -> list[CriticDecision]:
        results_json = json.dumps(
            {
                sid: {
                    "retry_count": retry_counts.get(sid, 0),
                    "results": out.results,
                    "success": out.success,
                    "targets_requested": out.targets_requested,
                    "targets_measured": [r["metric"] for r in out.results],
                    "summary": out.summary,
                }
                for sid, out in outputs.items()
            },
            indent=2,
            default=str,
        )
        tool_name = critic_tool_schema.get("function", {}).get("name", "audit_results")
        step_ids = list(outputs.keys())
        # Skip targets coverage check in pipeline-level (system_prompt_override) mode.
        # When agent_type == "task", the system prompt's Success Criteria guide evaluation.
        coverage_instruction = "" if agent_type == "task" else (
            "IMPORTANT: For each step, compare 'targets_requested' against "
            "'targets_measured'. Any target present in 'targets_requested' but "
            "absent from 'targets_measured' is MISSING and requires retry.\n\n"
        )
        user_msg = (
            f"Review these {agent_type} worker outputs:\n\n"
            f"```json\n{results_json}\n```\n\n"
            f"Step IDs to evaluate: {step_ids}\n\n"
            f"{coverage_instruction}"
            f"Call {tool_name} with your decisions for each step_id."
        )
        messages = [
            {"role": "system", "content": critic_system_prompt},
            {"role": "user",   "content": user_msg},
        ]

        try:
            resp = self.llm.chat(messages, tools=[critic_tool_schema])
        except Exception as exc:
            if self.verbose:
                print(f"[critic/{agent_type}] LLM call failed ({exc}); accepting all.", file=sys.stderr)
            return _accept_decisions(outputs)

        if not resp.tool_calls or resp.tool_calls[0].name != tool_name:
            if self.verbose:
                print(f"[critic/{agent_type}] No {tool_name} call; accepting all.", file=sys.stderr)
            return _accept_decisions(outputs)

        args = resp.tool_calls[0].arguments or {}
        raw_decisions = args.get("decisions", [])

        decisions: list[CriticDecision] = []
        seen_ids: set[str] = set()
        for d in raw_decisions:
            sid = d.get("step_id", "")
            if sid not in outputs:
                continue
            seen_ids.add(sid)
            try:
                conf = max(0.0, min(1.0, float(d.get("confidence", 1.0))))
            except (TypeError, ValueError):
                conf = 1.0
            dec = d.get("decision", "accept")
            if dec not in ("accept", "retry"):
                dec = "accept"
            raw_failing = d.get("failing_targets", [])
            failing = [m for m in raw_failing if isinstance(m, str)] if isinstance(raw_failing, list) else []
            decisions.append(CriticDecision(
                step_id=sid,
                decision=dec,
                confidence=conf,
                reason=str(d.get("reason", "")),
                failing_targets=failing,
            ))

        for sid in outputs:
            if sid not in seen_ids:
                decisions.append(CriticDecision(step_id=sid, decision="accept",
                                                confidence=1.0, reason="not reviewed"))

        if self.verbose:
            retries = sum(1 for d in decisions if d.decision == "retry")
            print(f"[critic/{agent_type}] {retries} retry decision(s) out of {len(decisions)}.",
                  file=sys.stderr, flush=True)

        return decisions


def _accept_decisions(outputs: dict[str, WorkerOutput]) -> list[CriticDecision]:
    return [
        CriticDecision(step_id=sid, decision="accept", confidence=1.0, reason="no critic available")
        for sid in outputs
    ]
