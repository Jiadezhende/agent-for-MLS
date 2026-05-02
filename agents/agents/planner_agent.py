"""
agents/agents/planner_agent.py — PlannerAgent: optimization coordinator running as a ReAct AgentLoop.

Replaces the old single-LLM-call Planner. Coordinates the full operator optimization workflow:
  1. Reads the operator skill to understand the target and success criteria
  2. Calls run_subagent to delegate hardware probing, profiling, analysis
  3. Integrates results across tool calls
  4. Calls mark_ready_for_critic when all success criteria are satisfied

Instance attributes run_id, agent_id, shared_store are injected by Orchestrator before run().
"""
from __future__ import annotations

import sys
from typing import Any

from agents.core.agent import Agent
from agents.core.llm import LLMClient
from agents.core.loop import AgentLoop
from agents.core.types import AgentContext, MemoryStore
from agents.tools.builtin.recording import FlagEventTool
from agents.tools.builtin.skills import ListSkillsTool, ReadSkillTool
from agents.tools.builtin.subagent import MarkReadyForCriticTool, RunSubagentParallelTool, RunSubagentTool
from agents.tools.circuit_breaker import CircuitBreaker
from agents.tools.registry import ToolRegistry


_COORDINATOR_PROMPT = """\
You are an autonomous GPU kernel optimization coordinator.

## Your Workflow

Step 0 — Understand the target operator:
  Call read_skill("operators/<operator>") where <operator> comes from the task spec.
  This skill file gives you: the formula, variable ranges, optimization goal,
  required hardware measurements, and success criteria.

Step 1 & 2 — Hardware characterization + Baseline profiling:
  These two steps are independent — run them concurrently when op_profiler is available:
    run_subagent_parallel([
      {{"agent_type": "hardware_probe", "targets": [...]}},
      {{"agent_type": "op_profiler",    "targets": [...]}}
    ])
  If op_profiler is NOT available, use run_subagent("hardware_probe", [...]) alone.
  Max 4 targets per agent entry; split large target lists across multiple calls.

Step 3 — Bottleneck analysis (only if 'bottleneck_analyst' appears in Available Agent Types):
  Call run_subagent("bottleneck_analyst", [...])

Step 4 — Optimization & validation:
  Based on the skill's "Potential Strategies" and your analysis results, reason about
  which optimizations to apply. If a kernel_optimizer agent is available, delegate
  implementation; otherwise document your optimization recommendation with CUDA pseudocode
  or a concrete implementation plan.

Step 5 — Submit for review:
  Only call mark_ready_for_critic when ALL success criteria in the operator skill are met.
  Provide a comprehensive summary covering every criterion.

## If Critic Feedback Is Provided
You are in REVISING mode. Address only the failing items listed. Do not redo accepted work.

## Available Agent Types
{agent_type_block}

## Rules
- Always start with read_skill to understand the operator before doing anything else.
- Max 4 targets per agent entry in any subagent call; split large target lists.
- Use run_subagent_parallel for independent tasks (no result dependency); use run_subagent for dependent tasks.
- You are fully autonomous — never ask for human input.
- Use flag_event for strategy decisions and anomalies.
- Do NOT call mark_ready_for_critic until all success criteria from the skill are met.
"""


def _build_system_prompt(agent_registry: dict) -> str:
    lines = []
    for defn in agent_registry.values():
        lines.append(f"### {defn.agent_type}")
        lines.append(defn.description)
        lines.append("")
    agent_type_block = "\n".join(lines).rstrip() or "(no agent types registered)"
    return _COORDINATOR_PROMPT.format(agent_type_block=agent_type_block)


def _build_initial_user_message(spec: dict, critic_feedback: dict | None) -> str:
    operator = spec.get("operator", "unknown")
    targets = spec.get("targets", [])

    lines = [f"Optimize the '{operator}' operator on this GPU."]

    if targets:
        lines.append(f"\nRequired measurements listed in spec: {targets}")

    if critic_feedback:
        lines.append("\n## REVISING MODE — Critic Feedback")
        failing = critic_feedback.get("failing_targets", [])
        reason = critic_feedback.get("reason", "")
        lines.append(f"Failing items: {failing}")
        if reason:
            lines.append(f"Reason: {reason}")
        lines.append(
            "\nRe-address only the failing items listed above. "
            "Do not redo work that was already accepted."
        )
    else:
        lines.append(
            f"\nStart by calling read_skill('operators/{operator}') "
            "to understand the task requirements and success criteria."
        )

    return "\n".join(lines)


class PlannerAgent(Agent):
    """Optimization coordinator running as a ReAct AgentLoop.

    Replaces the old single-LLM-call Planner. Available tools:
      run_subagent, mark_ready_for_critic, flag_event, read_skill, list_skills.

    Instance attributes run_id, agent_id, shared_store are injected by Orchestrator.
    """

    def __init__(
        self,
        llm: LLMClient,
        agent_registry: dict,
        executor: Any,
        agent_cfg: Any,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.agent_registry = agent_registry
        self.executor = executor
        self.agent_cfg = agent_cfg
        self.verbose = verbose
        # Injected by Orchestrator before run():
        self.run_id: str | None = None
        self.agent_id: str | None = None
        self.shared_store: Any = None
        self.log_manager: Any = None  # LogManager | None

    def run(self, spec: dict, critic_feedback: dict | None = None) -> AgentContext:  # type: ignore[override]
        """Run the coordinator loop.

        spec: task payload (may include 'operator', 'targets', etc.)
        critic_feedback: set when in REVISING mode; contains 'failing_targets' and 'reason'.
        Returns the AgentContext with ctx.job_history populated by run_subagent calls.
        """
        ctx = AgentContext(
            memory=MemoryStore(),
            circuit_breaker=CircuitBreaker(
                threshold=getattr(self.agent_cfg, "circuit_breaker_threshold", 3),
                half_open_timeout_s=getattr(self.agent_cfg, "half_open_timeout_s", 60),
            ),
            run_id=self.run_id,
            agent_id=self.agent_id or "planner",
            shared_store=self.shared_store,
            log_manager=self.log_manager,
        )
        self._last_ctx = ctx  # always up-to-date; read by orchestrator on interrupt

        registry = ToolRegistry()
        _subagent_kwargs = dict(
            llm=self.llm,
            executor=self.executor,
            agent_registry=self.agent_registry,
            agent_cfg=self.agent_cfg,
            verbose=self.verbose,
        )
        registry.register(RunSubagentTool(**_subagent_kwargs))
        registry.register(RunSubagentParallelTool(**_subagent_kwargs))
        registry.register(MarkReadyForCriticTool())
        registry.register(FlagEventTool())
        registry.register(ReadSkillTool())
        registry.register(ListSkillsTool())

        system_prompt = _build_system_prompt(self.agent_registry)
        initial_msg = _build_initial_user_message(spec, critic_feedback)

        try:
            loop = AgentLoop(
                llm=self.llm,
                registry=registry,
                ctx=ctx,
                max_iterations=self.agent_cfg.max_iterations,
                verbose=self.verbose,
                worker_id="planner",
                system_prompt=system_prompt,
                user_message=initial_msg,
            )
            loop.run()
        except RuntimeError as exc:
            if self.verbose:
                print(f"[planner] Loop exhausted without mark_ready_for_critic: {exc}",
                      file=sys.stderr, flush=True)
            ctx.memory.set("run", "error", str(exc))
        except Exception as exc:
            if self.verbose:
                print(f"[planner] Unexpected error: {exc}", file=sys.stderr, flush=True)

        return ctx
