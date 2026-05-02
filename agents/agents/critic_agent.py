"""
agents/agents/critic_agent.py — CriticAgent: cross-validates worker outputs and
returns structured accept/retry decisions.

Uses AgentLoop (the same infrastructure as worker agents) — one loop per agent type.
Falls back to all-accept on any failure so the pipeline can always continue.

Public re-export: AUDIT_TOOL_SCHEMA (used by orchestrator / tests that reference it here).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

from agents.core.agent import Agent
from agents.core.llm import LLMClient
from agents.core.loop import AgentLoop
from agents.core.types import AgentContext, CriticDecision, MemoryStore, WorkerOutput
from agents.tools.builtin.audit import AUDIT_TOOL_SCHEMA, AuditResultsTool  # noqa: F401 (AUDIT_TOOL_SCHEMA re-exported)
from agents.tools.registry import ToolRegistry


class CriticAgent(Agent):
    """Cross-validates worker outputs via AgentLoop, returning accept/retry decisions.

    Groups WorkerOutputs by agent_type, then runs one AgentLoop per type using that
    type's critic_system_prompt. Each loop has a single tool (AuditResultsTool) which
    parses decisions and terminates via _Terminated. The external run() interface is
    identical to the previous implementation — Orchestrator sees no change.
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
        critic_tool_schema_override: dict | None = None,  # accepted but unused; AuditResultsTool owns the schema
    ) -> list[CriticDecision]:
        """Cross-validate all worker outputs.

        outputs:                step_id → WorkerOutput
        retry_counts:           step_id → retries already performed (0 = first attempt)
        system_prompt_override: if set, use this prompt for all outputs in one call
                                (used by Orchestrator for task-level evaluation)
        """
        if not outputs:
            return _accept_decisions({})

        counts = retry_counts or {}

        if system_prompt_override is not None:
            return self._critique_one_type(
                agent_type="task",
                outputs=outputs,
                retry_counts=counts,
                critic_system_prompt=system_prompt_override,
            )

        by_type = self._group_by_type(outputs)
        all_decisions: list[CriticDecision] = []
        for agent_type, typed_outputs in by_type.items():
            defn = self.agent_registry.get(agent_type)
            if defn is None:
                if self.verbose:
                    print(
                        f"[critic] No definition for '{agent_type}'; accepting all.",
                        file=sys.stderr,
                    )
                all_decisions.extend(_accept_decisions(typed_outputs))
                continue

            decisions = self._critique_one_type(
                agent_type=agent_type,
                outputs=typed_outputs,
                retry_counts=counts,
                critic_system_prompt=defn.critic_system_prompt,
            )
            all_decisions.extend(decisions)

        return all_decisions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        step_ids = list(outputs.keys())
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
            f"Call audit_results with your decisions for each step_id."
        )

        ctx = AgentContext(memory=MemoryStore())
        registry = ToolRegistry()
        registry.register(AuditResultsTool(outputs))

        try:
            AgentLoop(
                llm=self.llm,
                registry=registry,
                ctx=ctx,
                max_iterations=3,
                verbose=self.verbose,
                worker_id=f"critic/{agent_type}",
                system_prompt=critic_system_prompt,
                user_message=user_msg,
            ).run()
        except Exception as exc:
            if self.verbose:
                print(
                    f"[critic/{agent_type}] AgentLoop failed ({exc}); accepting all.",
                    file=sys.stderr,
                    flush=True,
                )
            return _accept_decisions(outputs)

        raw = ctx.memory.get("audit", "decisions", [])
        if not raw:
            if self.verbose:
                print(
                    f"[critic/{agent_type}] No decisions in memory; accepting all.",
                    file=sys.stderr,
                    flush=True,
                )
            return _accept_decisions(outputs)

        decisions = [CriticDecision(**d) for d in raw]

        if self.verbose:
            retries = sum(1 for d in decisions if d.decision == "retry")
            print(
                f"[critic/{agent_type}] {retries} retry decision(s) out of {len(decisions)}.",
                file=sys.stderr,
                flush=True,
            )

        return decisions


def _accept_decisions(outputs: dict[str, WorkerOutput]) -> list[CriticDecision]:
    return [
        CriticDecision(step_id=sid, decision="accept", confidence=1.0, reason="no critic available")
        for sid in outputs
    ]
