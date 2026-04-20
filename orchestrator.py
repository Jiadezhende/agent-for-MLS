"""
orchestrator.py — Top-level coordinator for the multi-agent pipeline.

State machine:
  Phase 1  Planner     → list[Step]
  Phase 2+ Execute loop → workers run pending Steps in parallel
                        → Critic reviews outputs → accept / retry
                        → repeat until all steps accepted or retries exhausted
"""
from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any

from agents._registry import all_definitions, get as _get_agent_def
from agents.agents.critic_agent import CriticAgent
from agents.agents.planner_agent import PlannerAgent
from agents.core.llm import LLMClient
from agents.core.types import (
    CriticDecision,
    Step,
    WorkerOutput,
)
from agents.tools.registry import ToolFactory


@dataclass
class _ExecutionState:
    steps: list[Step] = field(default_factory=list)
    outputs: dict[str, WorkerOutput] = field(default_factory=dict)
    retry_set: set[str] = field(default_factory=set)
    carry_forward: dict[str, list[dict]] = field(default_factory=dict)  # step_id → accepted results
    done: bool = False


class Orchestrator:
    """Drives Planner → Worker pool (with Critic retry loop) → done.

    The Orchestrator is pure-code scheduling — no LLM calls of its own.
    LLM reasoning happens inside Planner, Worker agents, and Critic.
    """

    def __init__(
        self,
        llm: LLMClient,
        executor: Any,
        task: Any,            # agents.core.types.Task
        agent_cfg: Any,       # config.AgentConfig
        agent_registry: dict | None = None,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.executor = executor
        self.task = task
        self.agent_cfg = agent_cfg
        self.agent_registry = agent_registry or all_definitions()
        self.verbose = verbose
        self._print_lock = threading.Lock()

        self.tool_factory = ToolFactory(executor)
        self.planner = PlannerAgent(llm, self.agent_registry, verbose)
        self.critic  = CriticAgent(llm, self.agent_registry, verbose)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> _ExecutionState:
        """Execute the full multi-agent pipeline.

        Returns the final _ExecutionState (steps + outputs).
        """
        targets = self.task.payload.get("targets", [])
        state = _ExecutionState()

        # Phase 1: Planner
        self._trace("plan", "input", {"targets": targets})
        plan: list[Step] = self.planner.run(targets)
        state.steps = plan
        self._trace("plan", "output", {"steps": [
            {"id": s.id, "task": s.task, "worker": s.worker} for s in plan
        ]})

        max_worker_retries: int = self.agent_cfg.max_worker_retries
        max_critic_cycles: int = self.agent_cfg.max_critic_cycles
        retry_counts: dict[str, int] = {}
        timeout_s: float = self.agent_cfg.worker_timeout_s

        # Phase 2+: Bounded execute loop with critic feedback
        try:
            for _cycle in range(max_critic_cycles):
                pending = [
                    s for s in state.steps
                    if s.id not in state.outputs
                    or s.id in state.retry_set
                ]

                if not pending:
                    state.done = True
                    break

                self._trace("workers", "start", {"pending": [s.id for s in pending]})
                outputs = self._run_workers(pending, timeout_s)
                for out in outputs:
                    # Merge carried-forward accepted results with new measurements
                    if out.step_id in state.carry_forward:
                        carried = state.carry_forward.pop(out.step_id)
                        new_metrics = {r.get("metric") for r in out.results}
                        merged = [r for r in carried if r.get("metric") not in new_metrics] + out.results
                        out = WorkerOutput(
                            step_id=out.step_id,
                            results=merged,
                            success=out.success,
                            reasoning_log=out.reasoning_log,
                            events=out.events,
                            summary=out.summary,
                        )
                    state.outputs[out.step_id] = out
                    state.retry_set.discard(out.step_id)
                self._trace("workers", "output", {
                    sid: {"success": out.success, "n_results": len(out.results)}
                    for sid, out in state.outputs.items()
                })

                # Critic reviews current outputs
                self._trace("critic", "start", {})
                decisions: list[CriticDecision] = self.critic.run(state.outputs, retry_counts)
                self._trace("critic", "output", {
                    "decisions": [
                        {"step_id": d.step_id, "decision": d.decision,
                         "confidence": d.confidence, "reason": d.reason}
                        for d in decisions
                    ]
                })

                # Apply decisions: mark retries or accept
                for dec in decisions:
                    if dec.step_id not in state.outputs:
                        continue
                    if dec.decision == "retry":
                        n = retry_counts.get(dec.step_id, 0)
                        if n < max_worker_retries:
                            original_step = next(s for s in state.steps if s.id == dec.step_id)
                            prev_output = state.outputs[dec.step_id]

                            # Narrow to only the failing targets if Critic specified them
                            retry_targets = (
                                [t for t in dec.failing_targets if t in original_step.targets]
                                if dec.failing_targets else original_step.targets
                            ) or original_step.targets

                            # Extract previous bad values for the failing targets
                            prev_bad: dict = {}
                            for r in prev_output.results:
                                if r.get("metric") in retry_targets:
                                    prev_bad[r["metric"]] = {
                                        "value": r.get("value"),
                                        "unit": r.get("unit"),
                                        "method": r.get("method"),
                                    }

                            # Carry forward accepted results (targets NOT being retried)
                            retry_target_set = set(retry_targets)
                            accepted_results = [
                                r for r in prev_output.results
                                if r.get("metric") not in retry_target_set
                            ]
                            if accepted_results:
                                state.carry_forward[dec.step_id] = accepted_results

                            # Replace step with narrowed retry step
                            retry_step = Step(
                                id=dec.step_id,
                                worker=original_step.worker,
                                targets=retry_targets,
                                task=", ".join(retry_targets),
                                retry_context={
                                    "reason": dec.reason,
                                    "previous_bad_values": prev_bad,
                                },
                            )
                            state.steps = [retry_step if s.id == dec.step_id else s for s in state.steps]

                            state.retry_set.add(dec.step_id)
                            retry_counts[dec.step_id] = n + 1
                            self._emit(
                                f"[orchestrator] Retry {n + 1}/{max_worker_retries} "
                                f"for {dec.step_id} (targets={retry_targets}): {dec.reason}"
                            )
                        else:
                            self._emit(
                                f"[orchestrator] Max retries reached for {dec.step_id}; accepting."
                            )
            else:
                self._emit(
                    f"[orchestrator] Hard limit of {max_critic_cycles} critic cycles reached; "
                    "terminating with current results."
                )
                state.done = True

        except KeyboardInterrupt:
            self._emit("[orchestrator] Interrupted by user. Collecting partial results.")
            state.done = True

        return state

    # ------------------------------------------------------------------
    # Worker pool
    # ------------------------------------------------------------------

    def _run_workers(self, steps: list[Step], timeout_s: float) -> list[WorkerOutput]:
        max_threads = min(len(steps), 8)
        future_to_step: dict = {}
        results: list[WorkerOutput] = []

        pool = ThreadPoolExecutor(max_workers=max_threads)
        try:
            for i, step in enumerate(steps):
                future = pool.submit(self._run_single_step, step, i)
                future_to_step[future] = step

            try:
                for future in as_completed(future_to_step, timeout=timeout_s):
                    step = future_to_step[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        self._emit(f"[orchestrator] Step {step.id} raised: {exc}")
                        results.append(WorkerOutput(
                            step_id=step.id, results=[], success=False
                        ))

            except FuturesTimeout:
                self._emit("[orchestrator] Worker pool timed out. Collecting partial results.")
                for future, step in future_to_step.items():
                    if future.done():
                        try:
                            results.append(future.result())
                        except Exception:
                            results.append(WorkerOutput(
                                step_id=step.id, results=[], success=False
                            ))
                    else:
                        future.cancel()
                        results.append(WorkerOutput(
                            step_id=step.id, results=[], success=False,
                            summary="worker_timeout",
                        ))

        except KeyboardInterrupt:
            self._emit("[orchestrator] Interrupted — cancelling pending workers.")
            for future, step in future_to_step.items():
                if future.cancel():
                    results.append(WorkerOutput(
                        step_id=step.id, results=[], success=False,
                        summary="cancelled",
                    ))
                elif future.done():
                    try:
                        results.append(future.result())
                    except Exception:
                        results.append(WorkerOutput(
                            step_id=step.id, results=[], success=False
                        ))
            pool.shutdown(wait=False, cancel_futures=True)
            raise  # re-raise so run()'s KeyboardInterrupt handler fires
        finally:
            pool.shutdown(wait=False)

        return results

    def _run_single_step(self, step: Step, worker_id: int) -> WorkerOutput:
        """Instantiate and run the agent for one Step. Runs on a thread."""
        self._emit(f"[W{worker_id}] Starting step={step.id} task='{step.task}'")

        agent_def = _get_agent_def(step.worker)
        if agent_def is None:
            msg = f"Unknown agent_type '{step.worker}'"
            self._emit(f"[W{worker_id}] Error: {msg}")
            return WorkerOutput(step_id=step.id, results=[], success=False, summary=msg)

        tools = self.tool_factory.build(agent_def.required_tools)
        agent = agent_def.agent_class(
            llm=self.llm,
            agent_cfg=self.agent_cfg,
            verbose=self.verbose,
            worker_id=worker_id,
        )

        try:
            out = agent.run(step, tools)
            self._emit(
                f"[W{worker_id}] Done step={step.id} "
                f"success={out.success} n_results={len(out.results)}"
            )
            return out
        except Exception as exc:
            self._emit(f"[W{worker_id}] Unhandled exception in step={step.id}: {exc}")
            return WorkerOutput(step_id=step.id, results=[], success=False, summary=str(exc))

    # ------------------------------------------------------------------
    # Trace logging
    # ------------------------------------------------------------------

    def _trace(self, agent: str, phase: str, data: dict) -> None:
        """Structured trace log: one line per agent / phase transition."""
        if self.verbose:
            import json as _json
            line = f"[orchestrator] [{agent}] {phase}: {_json.dumps(data, default=str)}"
            with self._print_lock:
                print(line, file=sys.stderr, flush=True)

    def _emit(self, msg: str) -> None:
        if self.verbose:
            with self._print_lock:
                print(msg, file=sys.stderr, flush=True)
