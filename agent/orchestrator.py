"""
agent/orchestrator.py — Coordinates Planner, parallel Worker pool, and Critic.

Pipeline:
  Phase 1 — Planner: one LLM call decomposes targets into WorkerSpecs
  Phase 2 — Workers: N concurrent AgentLoops (ThreadPoolExecutor)
  Phase 3 — Aggregate: collect all results from worker contexts
  Phase 4 — Critic: one LLM call cross-validates and adjusts confidence

The Executor is shared across workers (thread-safe after executor.py changes).
The LLMClient is shared (stateless API wrapper).
Each worker gets its own AgentContext and CircuitBreaker.
"""
from __future__ import annotations

import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from typing import Any

from agent.critic import critique_results
from agent.loop import AgentLoop
from agent.planner import plan_tasks
from agent.tasks._registry import get as _get_task_def
from agent.types import (
    AgentContext,
    CircuitBreaker,
    CritiqueResult,
    MemoryStore,
    Result,
    Task,
    WorkerResult,
    WorkerSpec,
)
from llm.client import LLMClient


class Orchestrator:
    """Top-level coordinator for multi-agent GPU benchmarking.

    Accepts the same LLMClient and Executor that main.py already builds.
    """

    def __init__(
        self,
        llm: LLMClient,
        executor: Any,      # executor.Executor — avoid circular import
        task: Task,
        agent_cfg: Any,     # config.AgentConfig
        task_registry: dict | None = None,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.executor = executor
        self.task = task
        self.agent_cfg = agent_cfg
        self.task_registry = task_registry or {}
        self.verbose = verbose
        self._print_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> tuple[list[Result], CritiqueResult, list[WorkerResult]]:
        """Execute the full multi-agent pipeline.

        Returns:
            all_results   — flat list of Result objects (post-critique)
            critique      — CritiqueResult from the Critic
            worker_results — one WorkerResult per worker (for logging)
        """
        targets = self.task.payload.get("targets", [])

        # Phase 1: Planner
        self._emit("[orchestrator] Phase 1: Planning worker assignments …")
        specs = plan_tasks(self.llm, targets, task_registry=self.task_registry, verbose=self.verbose)
        self._emit(
            f"[orchestrator] Planner produced {len(specs)} worker(s): "
            + ", ".join(f"W{s.worker_id}={s.targets}" for s in specs)
        )

        # Phase 2: Workers
        self._emit("[orchestrator] Phase 2: Starting worker pool …")
        worker_results = self._run_workers(specs)

        # Phase 3: Aggregate
        all_results: list[Result] = []
        for wr in worker_results:
            all_results.extend(wr.ctx.results)
        self._emit(
            f"[orchestrator] Aggregated {len(all_results)} result(s) "
            f"from {len(worker_results)} worker(s)."
        )

        # Phase 4: Critic
        self._emit("[orchestrator] Phase 4: Running Critic …")
        critique = critique_results(
            self.llm, all_results,
            task_registry=self.task_registry,
            verbose=self.verbose,
        )
        self._emit(
            f"[orchestrator] Critic: {len(critique.anomaly_flags)} anomaly flag(s), "
            f"{len(critique.flagged_results)} result(s) below confidence threshold."
        )

        return all_results, critique, worker_results

    # ------------------------------------------------------------------
    # Worker pool
    # ------------------------------------------------------------------

    def _run_workers(self, specs: list[WorkerSpec]) -> list[WorkerResult]:
        """Run all workers in a ThreadPoolExecutor and collect results."""
        max_threads = min(len(specs), 8)
        timeout_s = self.agent_cfg.worker_timeout_s

        future_to_spec: dict = {}
        results: list[WorkerResult] = []

        with ThreadPoolExecutor(max_workers=max_threads) as pool:
            for spec in specs:
                future = pool.submit(self._run_single_worker, spec)
                future_to_spec[future] = spec

            try:
                for future in as_completed(future_to_spec, timeout=timeout_s):
                    spec = future_to_spec[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        self._emit(f"[W{spec.worker_id}] Unhandled exception: {exc}")
                        results.append(self._error_worker_result(spec, str(exc)))

            except FuturesTimeout:
                self._emit(
                    "[orchestrator] Worker pool timed out. Collecting partial results."
                )
                for future, spec in future_to_spec.items():
                    if future.done():
                        try:
                            results.append(future.result())
                        except Exception as exc:
                            results.append(self._error_worker_result(spec, str(exc)))
                    else:
                        future.cancel()
                        results.append(self._error_worker_result(spec, "worker_timeout"))

        results.sort(key=lambda wr: wr.worker_id)
        return results

    def _run_single_worker(self, spec: WorkerSpec) -> WorkerResult:
        """Build and run one AgentLoop for the given WorkerSpec.

        Runs on a thread-pool thread. Never raises — all exceptions are caught.
        """
        self._emit(f"[W{spec.worker_id}] Starting: targets={spec.targets}")

        worker_task = self._make_worker_task(spec)
        ctx = AgentContext(
            task=worker_task,
            memory=MemoryStore(),
            circuit_breaker=CircuitBreaker(
                threshold=self.agent_cfg.circuit_breaker_threshold,
                half_open_timeout_s=self.agent_cfg.half_open_timeout_s,
            ),
        )

        # Register a per-worker job-history listener on the shared executor
        def _job_cb(r: Any) -> None:
            ctx.job_history.append(r.to_log_dict())

        self.executor.add_job_listener(_job_cb)
        exit_code = 0
        error_msg: str | None = None

        try:
            task_def = _get_task_def(spec.agent_type)
            if task_def is None:
                raise RuntimeError(
                    f"Unknown agent_type '{spec.agent_type}' — "
                    "no TaskDefinition registered for it."
                )
            registry = task_def.build_registry(self.executor)
            loop = AgentLoop(
                llm=self.llm,
                registry=registry,
                ctx=ctx,
                max_iterations=self.agent_cfg.max_iterations,
                verbose=self.verbose,
                worker_id=spec.worker_id,
                system_prompt=task_def.system_prompt,
            )
            loop.run()
            self._emit(f"[W{spec.worker_id}] Completed successfully.")
        except RuntimeError as exc:
            # Budget exhausted — partial results are still usable
            exit_code = 3
            error_msg = str(exc)
            self._emit(f"[W{spec.worker_id}] Budget exhausted: {exc}")
        except Exception as exc:
            exit_code = 1
            error_msg = str(exc)
            self._emit(f"[W{spec.worker_id}] Error: {exc}")
        finally:
            self.executor.remove_job_listener(_job_cb)

        return WorkerResult(
            worker_id=spec.worker_id,
            worker_spec=spec,
            ctx=ctx,
            exit_code=exit_code,
            error=error_msg,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_worker_task(self, spec: WorkerSpec) -> Task:
        """Build a Task for one worker, scoped to its assigned targets."""
        original_targets = self.task.payload.get("targets", [])
        assigned = set(spec.targets)

        if original_targets and isinstance(original_targets[0], dict):
            worker_targets: list = [
                t for t in original_targets if t.get("name") in assigned
            ]
        else:
            worker_targets = [t for t in original_targets if t in assigned]

        # Fallback: if filtering produced nothing (type mismatch), use raw names
        if not worker_targets:
            worker_targets = list(spec.targets)

        sub_payload = dict(self.task.payload)
        sub_payload["targets"] = worker_targets
        if spec.strategy_hints:
            sub_payload["strategy_hints"] = spec.strategy_hints

        return Task(
            id=str(uuid.uuid4()),
            type=spec.agent_type,
            description=f"Worker {spec.worker_id} ({spec.agent_type}): {', '.join(spec.targets)}",
            payload=sub_payload,
            constraints=self.task.constraints,
        )

    def _error_worker_result(self, spec: WorkerSpec, error: str) -> WorkerResult:
        """Create a minimal WorkerResult representing a failed worker."""
        empty_ctx = AgentContext(
            task=self._make_worker_task(spec),
            memory=MemoryStore(),
        )
        return WorkerResult(
            worker_id=spec.worker_id,
            worker_spec=spec,
            ctx=empty_ctx,
            exit_code=1,
            error=error,
        )

    def _emit(self, msg: str) -> None:
        """Thread-safe stderr print for orchestrator-level messages."""
        if self.verbose:
            with self._print_lock:
                print(msg, file=sys.stderr, flush=True)
