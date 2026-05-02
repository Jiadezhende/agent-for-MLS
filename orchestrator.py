"""
orchestrator.py — State machine driver for the multi-agent optimization pipeline.

State machine:
  planning / revising  → PlannerAgent (ReAct loop, calls subagent tools)
  ready_for_critic     → CriticAgent (reviews collected outputs)
  accepted             → done
  failed               → done (partial results)

Replaces the old fixed Planner → ThreadPoolExecutor → Critic pipeline.
"""
from __future__ import annotations

import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents._registry import all_definitions
from agents.agents.critic_agent import CriticAgent
from agents.agents.planner_agent import PlannerAgent
from agents.core.llm import LLMClient
from agents.core.log_manager import LogManager
from agents.core.types import (
    AgentContext,
    CriticDecision,
    RunContext,
    WorkerOutput,
)


# ---------------------------------------------------------------------------
# Task-level Critic prompt template (operator skill content embedded at runtime)
# ---------------------------------------------------------------------------

_TASK_CRITIC_PROMPT_TEMPLATE = """\
You are a GPU kernel optimization auditor.

## Target Operator Specification
{skill_content}

## Your Job
Review the coordinator's submitted work against the "Success Criteria" section above.

For each step in the output:
  1. Coverage check (MANDATORY — check first):
     Are ALL success criteria items present in the output?
     Any missing item → decision="retry", failing_targets = list of missing criterion names.

  2. Quality check (only if coverage is complete):
     - Is the bottleneck diagnosis consistent with the measured hardware parameters?
     - Is the optimization strategy targeting the actual bottleneck?
     - Are performance numbers plausible (check for suspiciously large speedups)?
     - Is correctness verification rigorous (≥ 3 distinct input sizes)?

decision="accept"  — ALL criteria met and quality is sound.
decision="retry"   — any criterion missing, OR strategy contradicts measured hardware.

Call audit_results exactly once with your findings.
"""

_TASK_CRITIC_PROMPT_NO_SKILL = """\
You are a GPU kernel optimization auditor.

Review all worker outputs for completeness and physical plausibility:
  1. Coverage check: are all requested targets measured? Missing targets → retry.
  2. Quality check: are measured values physically plausible?

Call audit_results exactly once with your findings.
"""


# ---------------------------------------------------------------------------
# Execution state
# ---------------------------------------------------------------------------

@dataclass
class _ExecutionState:
    planner_ctx: AgentContext | None = None
    accepted: bool = False
    phase: str = "planning"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Drives the PlannerAgent (ReAct loop) ↔ CriticAgent state machine.

    The Orchestrator is pure-code scheduling — no LLM calls of its own.
    LLM reasoning happens inside PlannerAgent (coordinator) and CriticAgent.
    """

    def __init__(
        self,
        llm: LLMClient,
        executor: Any,
        spec: dict,
        agent_cfg: Any,       # config.AgentConfig
        agent_registry: dict | None = None,
        verbose: bool = False,
        log_dir: Path | None = None,
    ) -> None:
        self.llm = llm
        self.executor = executor
        self.spec = spec
        self.agent_cfg = agent_cfg
        self.agent_registry = agent_registry or all_definitions()
        self.verbose = verbose
        self.log_dir = log_dir
        self._print_lock = threading.Lock()

        self.planner = PlannerAgent(
            llm=llm,
            agent_registry=self.agent_registry,
            executor=executor,
            agent_cfg=agent_cfg,
            verbose=verbose,
        )
        self.critic = CriticAgent(llm, self.agent_registry, verbose)
        self.run_ctx = RunContext(run_id=str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> _ExecutionState:
        """Execute the full optimization pipeline.

        Returns the final _ExecutionState (planner_ctx with job_history).
        """
        spec = self.spec
        state = _ExecutionState()
        max_critic_cycles: int = self.agent_cfg.max_critic_cycles
        retry_counts: dict[str, int] = {}
        critic_feedback: dict | None = None

        # Build the task-level Critic prompt (reads operator skill if available)
        critic_prompt = self._build_critic_prompt(spec)

        # Create LogManager if a log directory was configured
        log_mgr: LogManager | None = None
        if self.log_dir is not None:
            log_mgr = LogManager(run_id=self.run_ctx.run_id, log_root=self.log_dir)

        self.run_ctx.event_log.append("plan.start", "orchestrator", {"spec": spec})

        try:
            for cycle in range(max_critic_cycles):
                # --- Planner phase ---
                self._emit(
                    f"[orchestrator] Cycle {cycle + 1}/{max_critic_cycles} "
                    f"phase={state.phase}"
                )
                if log_mgr is not None:
                    log_mgr.begin_cycle(cycle + 1)

                planner_ctx = self._run_planner_loop(spec, critic_feedback, log_mgr=log_mgr)
                state.planner_ctx = planner_ctx
                state.phase = "ready_for_critic"

                n_subagent_calls = len(planner_ctx.job_history)
                self.run_ctx.event_log.append("plan.complete", "planner", {
                    "n_subagent_calls": n_subagent_calls,
                    "cycle": cycle,
                })
                self._emit(
                    f"[orchestrator] Planner done: {n_subagent_calls} subagent call(s). "
                    "Running Critic."
                )

                # Write planner reasoning log for this cycle
                if log_mgr is not None:
                    log_mgr.write_planner_log(planner_ctx.reasoning_log, planner_ctx.events)

                # --- Critic phase ---
                worker_outputs = self._collect_outputs(planner_ctx)
                if not worker_outputs:
                    self._emit("[orchestrator] No subagent outputs to review; accepting.")
                    state.accepted = True
                    state.phase = "accepted"
                    break

                self.run_ctx.event_log.append("critic.start", "critic", {
                    "n_outputs": len(worker_outputs),
                    "cycle": cycle,
                })
                decisions: list[CriticDecision] = self.critic.run(
                    worker_outputs,
                    retry_counts,
                    system_prompt_override=critic_prompt,
                )
                for dec in decisions:
                    self.run_ctx.event_log.append("critic.decision", "critic", {
                        "step_id": dec.step_id,
                        "decision": dec.decision,
                        "confidence": dec.confidence,
                        "reason": dec.reason,
                        "failing_targets": dec.failing_targets,
                    })
                    # Track retry counts per step
                    if dec.decision == "retry":
                        retry_counts[dec.step_id] = retry_counts.get(dec.step_id, 0) + 1

                self._emit(
                    f"[orchestrator] Critic decisions: "
                    + ", ".join(f"{d.step_id}={d.decision}" for d in decisions)
                )

                # Write critic log for this cycle
                if log_mgr is not None:
                    log_mgr.write_critic_log(decisions, self.critic.last_reasoning_traces)

                failing = [d for d in decisions if d.decision == "retry"]
                if not failing:
                    state.accepted = True
                    state.phase = "accepted"
                    self._emit("[orchestrator] All accepted.")
                    break

                # --- Revising phase ---
                critic_feedback = self._build_feedback(failing, worker_outputs)
                state.phase = "revising"
                self._emit(
                    f"[orchestrator] Retry requested for: "
                    f"{critic_feedback.get('failing_targets')}. Re-entering Planner."
                )
                self.run_ctx.event_log.append("retry.trigger", "orchestrator", {
                    "cycle": cycle,
                    "failing_targets": critic_feedback.get("failing_targets"),
                    "reason": critic_feedback.get("reason"),
                })
            else:
                self._emit(
                    f"[orchestrator] Hard limit of {max_critic_cycles} critic cycles reached; "
                    "accepting current results."
                )
                state.accepted = True
                state.phase = "accepted"

        except KeyboardInterrupt:
            self._emit("[orchestrator] Interrupted by user. Collecting partial results.")
            state.phase = "failed"
            if state.planner_ctx is None:
                state.planner_ctx = getattr(self.planner, "_last_ctx", None)
        except Exception as exc:
            self._emit(f"[orchestrator] Unexpected error: {exc}. Collecting partial results.")
            state.phase = "failed"

        self.run_ctx.event_log.append("pipeline.done", "orchestrator", {
            "phase": state.phase,
            "accepted": state.accepted,
            "n_subagent_calls": len(state.planner_ctx.job_history) if state.planner_ctx else 0,
        })

        # Flush structured logs to disk
        if log_mgr is not None:
            log_mgr.flush_orchestrator_events(self.run_ctx.event_log.records())
            log_mgr.write_manifest(
                operator=spec.get("operator", "unknown"),
                accepted=state.accepted,
            )

        return state

    # ------------------------------------------------------------------
    # Planner delegation
    # ------------------------------------------------------------------

    def _run_planner_loop(
        self,
        spec: dict,
        critic_feedback: dict | None,
        log_mgr: "LogManager | None" = None,
    ) -> AgentContext:
        """Run the PlannerAgent ReAct loop and return its AgentContext."""
        self.planner.run_id = self.run_ctx.run_id
        self.planner.agent_id = "planner"
        self.planner.shared_store = self.run_ctx.shared_store
        self.planner.log_manager = log_mgr

        ctx = self.planner.run(spec, critic_feedback)

        if self.verbose:
            summary = ctx.memory.get("run", "summary") or ""
            self._emit(
                f"[orchestrator] Planner summary: {summary[:120]}"
                f"{'...' if len(summary) > 120 else ''}"
            )
        return ctx

    # ------------------------------------------------------------------
    # Output collection for Critic
    # ------------------------------------------------------------------

    def _collect_outputs(self, planner_ctx: AgentContext) -> dict[str, WorkerOutput]:
        """Build dict[step_id → WorkerOutput] from Planner's job_history."""
        return {out.step_id: out for out in planner_ctx.job_history}

    # ------------------------------------------------------------------
    # Feedback for REVISING phase
    # ------------------------------------------------------------------

    def _build_feedback(
        self, failing_decisions: list[CriticDecision], worker_outputs: dict[str, WorkerOutput]
    ) -> dict:
        """Build critic_feedback dict to pass to Planner in revising mode."""
        all_failing: list[str] = []
        reasons: list[str] = []
        for dec in failing_decisions:
            if dec.failing_targets:
                all_failing.extend(dec.failing_targets)
            else:
                # No specific targets listed → flag the whole step's targets
                out = worker_outputs.get(dec.step_id)
                if out:
                    all_failing.extend(out.targets_requested)
            if dec.reason:
                reasons.append(dec.reason)
        return {
            "failing_targets": list(dict.fromkeys(all_failing)) or ["all"],
            "reason": " | ".join(reasons),
        }

    # ------------------------------------------------------------------
    # Critic prompt construction
    # ------------------------------------------------------------------

    def _build_critic_prompt(self, spec: dict) -> str:
        """Build task-level Critic prompt, embedding operator skill if available."""
        operator = spec.get("operator")
        if not operator:
            return _TASK_CRITIC_PROMPT_NO_SKILL

        skill_path = Path("skills/operators") / f"{operator}.md"
        if not skill_path.exists():
            self._emit(
                f"[orchestrator] Operator skill not found: {skill_path}; "
                "using generic Critic prompt."
            )
            return _TASK_CRITIC_PROMPT_NO_SKILL

        skill_content = skill_path.read_text(encoding="utf-8")
        return _TASK_CRITIC_PROMPT_TEMPLATE.format(skill_content=skill_content)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _emit(self, msg: str) -> None:
        if self.verbose:
            with self._print_lock:
                print(msg, file=sys.stderr, flush=True)
