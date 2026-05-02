"""pipeline/orchestrator.py — PipelineOrchestrator state-machine driver.

Replaces the old planner+critic-retry orchestrator. Pure code: drives the
stage state machine, persists state.json, syncs the root-level
``optimized_lora.cu`` whenever best updates, and finalizes deterministically
on timeout.
"""
from __future__ import annotations

import json
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from typing import Callable, Sequence

from .stage_agent import StageAgent
from .stage_runner import run_stage
from .state import (
    CandidateRecord,
    RunState,
    Stage,
    StageResult,
)
from .transitions import next_stage
from .workspace_layout import RunLayout, make_run_id


# ---------------------------------------------------------------------------
# Stage budgets — default per-stage seconds. Orchestrator clamps each stage
# to min(default_for_stage, remaining_total_budget) so a runaway agent can
# never blow past the user-provided time_budget_s.
# ---------------------------------------------------------------------------

DEFAULT_STAGE_BUDGETS_S: dict[Stage, float] = {
    Stage.BENCHMARK_SPEC: 120.0,
    Stage.HARDWARE_PROFILE: 180.0,
    Stage.BASELINE_PROFILE: 180.0,
    Stage.INITIAL_CANDIDATE: 240.0,
    Stage.TUNING_LOOP: 240.0,
    Stage.OPTIONAL_PROFILE: 180.0,
    Stage.FINALIZE: 60.0,
}


# Hard guard: even if state.json + agents disagree, the main loop won't run
# more than this many iterations. Prevents pathological infinite loops in
# tests / regression. The number is generous; real runs are bounded by
# time_budget_s long before this trips.
MAX_LOOP_ITERATIONS: int = 200


# ---------------------------------------------------------------------------
# PipelineOrchestrator
# ---------------------------------------------------------------------------

class PipelineOrchestrator:
    """Drives the pipeline through Stage transitions until FINALIZE.

    Construction parameters:
      spec: operator spec dict (must contain ``operator``)
      time_budget_s: total wall-clock budget for the run
      workspace_root: directory under which ``runs/<run_id>/`` lives
      output_path: ``./optimized_lora.cu`` (synced from best/best.cu)
      stage_agents: dict[Stage, StageAgent] — concrete agents per stage
      build_tools: callable ``(allowed_tools, current_stage) → ToolRegistry``;
        the stage parameter lets tools like submit_candidate_result tag the
        StageResult with the right Stage enum value (INITIAL_CANDIDATE vs
        TUNING_LOOP). Production code passes a closure over StageToolFactory.
      run_id: pass an existing run_id to resume; ``None`` mints a fresh one
      stage_budgets_s: optional override for per-stage budgets
    """

    def __init__(
        self,
        *,
        spec: dict,
        time_budget_s: float,
        workspace_root: str | Path,
        output_path: str | Path,
        stage_agents: dict[Stage, StageAgent],
        build_tools: Callable[[Sequence[str], Stage], object],
        run_id: str | None = None,
        log_manager: Any = None,
        verbose: bool = False,
        stage_budgets_s: dict[Stage, float] | None = None,
    ):
        self.spec = spec
        self.time_budget_s = float(time_budget_s)
        self.workspace_root = Path(workspace_root).resolve()
        self.output_path = Path(output_path).resolve()
        self.stage_agents = stage_agents
        self.build_tools = build_tools
        self.log_manager = log_manager
        self.verbose = verbose
        self.stage_budgets_s = {**DEFAULT_STAGE_BUDGETS_S, **(stage_budgets_s or {})}

        # Wall clock origin — set in _init_or_resume so resumed runs use the
        # already-accumulated elapsed_s as offset.
        self._wall_start: float = 0.0
        self._elapsed_at_start: float = 0.0

        # Populated by _init_or_resume.
        self.run_state: RunState
        self.layout: RunLayout
        self._init_or_resume(run_id=run_id)

    # ---- public API -----------------------------------------------------

    def run(self) -> dict:
        """Drive the state machine to FINALIZE and return a summary dict."""
        self._save_state()

        for _ in range(MAX_LOOP_ITERATIONS):
            stage = self._compute_next_stage()
            if stage is Stage.FINALIZE:
                self._finalize()
                break

            agent = self.stage_agents.get(stage)
            if agent is None:
                self._append_event(
                    stage,
                    status="missing_agent",
                    detail=f"no StageAgent registered for {stage.value}",
                )
                # Without an agent we cannot make progress on this stage.
                # Setup stages are blocking, so we have to finalize.
                if stage in (
                    Stage.BENCHMARK_SPEC,
                    Stage.HARDWARE_PROFILE,
                    Stage.BASELINE_PROFILE,
                    Stage.INITIAL_CANDIDATE,
                ):
                    self._finalize()
                    break
                # Optional / tuning stages: skip and let next_stage decide.
                # If TUNING_LOOP has no agent we'd loop forever — just finalize.
                self._finalize()
                break

            self.run_state.set_current_stage(stage)
            self._save_state()

            budget = self._stage_budget(stage)
            self._vprint(f"[orch] stage={stage.value} budget={budget:.1f}s elapsed={self._tick():.1f}s")
            # Wrap the stage-aware build_tools into the unary form run_stage expects.
            stage_for_closure = stage  # avoid late-binding gotcha
            single_arg_builder = lambda allowed, _s=stage_for_closure: self.build_tools(allowed, _s)
            result = run_stage(
                agent,
                run_state=self.run_state,
                layout=self.layout,
                build_tools=single_arg_builder,
                stage_budget_s=budget,
                log_manager=self.log_manager,
                verbose=self.verbose,
            )
            self._tick()  # refresh elapsed
            self._handle_result(stage, result)
            self._save_state()
        else:
            # Loop guard tripped — shouldn't happen in practice.
            self._append_event(Stage.FINALIZE, status="loop_guard_tripped", detail=str(MAX_LOOP_ITERATIONS))
            self._finalize()

        return {
            "run_id": self.run_state.run_id,
            "best_candidate_id": self.run_state.best_candidate_id,
            "best_speedup": self.run_state.best_speedup,
            "output_path": str(self.output_path),
            "completed_stages": list(self.run_state.completed_stages),
            "elapsed_s": self.run_state.elapsed_s,
        }

    # ---- internals ------------------------------------------------------

    def _init_or_resume(self, run_id: str | None) -> None:
        """Either resume an existing run or mint a fresh one.

        Resume condition: run_id given AND state.json exists for that run.
        Otherwise we create a new run. ``self._wall_start`` is set so that
        the first ``_tick()`` returns the accumulated elapsed_s of the run.
        """
        candidate_layout = (
            RunLayout(self.workspace_root, run_id) if run_id else None
        )
        if candidate_layout is not None and candidate_layout.state_path.is_file():
            text = candidate_layout.state_path.read_text(encoding="utf-8")
            self.run_state = RunState.from_json(text)
            self.layout = candidate_layout
            self.layout.mkdir()  # idempotent — ensure subdirs exist
            # Carry forward time budget if caller passed a different value.
            self.run_state.time_budget_s = self.time_budget_s
            self._elapsed_at_start = self.run_state.elapsed_s
            self._wall_start = time.monotonic()
            self._vprint(f"[orch] resumed run_id={run_id} elapsed_s={self._elapsed_at_start:.1f}")
        else:
            new_run_id = run_id or make_run_id()
            self.run_state = RunState(
                run_id=new_run_id,
                operator=self.spec.get("operator", "unknown"),
                time_budget_s=self.time_budget_s,
            )
            self.layout = RunLayout(self.workspace_root, new_run_id)
            self.layout.mkdir()
            self.run_state.mark_completed(Stage.INIT)
            self._elapsed_at_start = 0.0
            self._wall_start = time.monotonic()
            self._vprint(f"[orch] new run_id={new_run_id} budget_s={self.time_budget_s:.0f}")

    def _tick(self) -> float:
        """Update run_state.elapsed_s from wall clock and return it."""
        self.run_state.elapsed_s = self._elapsed_at_start + (time.monotonic() - self._wall_start)
        return self.run_state.elapsed_s

    def _compute_next_stage(self) -> Stage:
        self._tick()
        return next_stage(
            self.run_state,
            has_hardware_profile=self.layout.has_hardware_profile,
            has_baseline=self.layout.has_baseline,
            has_candidate_profile=self.layout.has_candidate_profile,
        )

    def _stage_budget(self, stage: Stage) -> float:
        """min(per-stage default, remaining total budget)."""
        per_stage = self.stage_budgets_s.get(stage, 60.0)
        remaining = max(0.0, self.run_state.remaining_budget_s())
        return min(per_stage, remaining if remaining > 0 else per_stage)

    # ---- result handling ------------------------------------------------

    def _handle_result(self, stage: Stage, result: StageResult) -> None:
        self._append_event(
            stage,
            status=result.status,
            confidence=result.confidence,
            artifacts=dict(result.artifacts),
            caveats=list(result.caveats),
        )

        if result.status == "failed":
            self._handle_failure(stage, result)
            return

        # Success or partial — apply stage-specific side effects on RunState.
        if stage is Stage.BENCHMARK_SPEC:
            self._absorb_spec_versions(result)
            self.run_state.mark_completed(stage)
        elif stage is Stage.HARDWARE_PROFILE:
            self.run_state.mark_completed(stage)
        elif stage is Stage.BASELINE_PROFILE:
            self.run_state.mark_completed(stage)
        elif stage is Stage.INITIAL_CANDIDATE:
            self.run_state.current_iteration += 1
            self._absorb_candidate(result, is_initial=True)
            self.run_state.mark_completed(stage)
        elif stage is Stage.TUNING_LOOP:
            self.run_state.current_iteration += 1
            self._absorb_candidate(result, is_initial=False)
            # TUNING_LOOP isn't "completed" in the one-shot sense — leave it
            # off completed_stages so next_stage keeps re-entering it.
        elif stage is Stage.OPTIONAL_PROFILE:
            self.run_state.mark_completed(stage)

    def _handle_failure(self, stage: Stage, result: StageResult) -> None:
        # For setup stages, a failure is a hard blocker — orchestrator will
        # finalize on the next loop iteration because the artifact predicate
        # will still return False. We still record the failure here.
        if stage in (Stage.INITIAL_CANDIDATE, Stage.TUNING_LOOP):
            # candidate failures: bump compile/correctness counters from
            # caveats so we can detect chronic failure patterns.
            counters = self.run_state.failure_counts
            for c in result.caveats:
                if "compile" in c.lower():
                    counters["compile"] = counters.get("compile", 0) + 1
                if "correctness" in c.lower():
                    counters["correctness"] = counters.get("correctness", 0) + 1
                if "unstable" in c.lower():
                    counters["benchmark_unstable"] = counters.get("benchmark_unstable", 0) + 1
            # iterate even on failure so we don't loop on the same id
            self.run_state.current_iteration += 1

    # ---- BENCHMARK_SPEC absorption -------------------------------------

    def _absorb_spec_versions(self, result: StageResult) -> None:
        spec_versions = result.metrics.get("spec_versions") or {}
        if isinstance(spec_versions, dict):
            for slot, ver in spec_versions.items():
                if isinstance(slot, str) and isinstance(ver, str):
                    self.run_state.benchmark_spec_versions[slot] = ver

    # ---- candidate result absorption + best update ---------------------

    def _absorb_candidate(self, result: StageResult, *, is_initial: bool) -> None:
        cand = result.metrics.get("candidate")
        if not isinstance(cand, dict):
            return

        # Append to leaderboard exactly once per candidate result.
        self._append_leaderboard(cand)

        accepted_for = cand.get("accepted_for")
        candidate_id = cand.get("candidate_id")
        speedup = cand.get("confirm_speedup_median") or cand.get("quick_speedup_median")

        if accepted_for == "best_update" and candidate_id:
            self._promote_to_best(candidate_id, speedup)
            self._sync_output()
        elif is_initial and candidate_id and self.run_state.best_candidate_id is None:
            # Initial candidate that didn't formally claim best_update: accept
            # it as the floor so the root file is never absent. Caveat
            # documents the demotion.
            if cand.get("compile_ok") and cand.get("correctness_ok"):
                self._promote_to_best(candidate_id, speedup)
                self._sync_output()

    def _promote_to_best(self, candidate_id: str, speedup: float | None) -> None:
        cand_cu = self.layout.candidate_file(candidate_id, "candidate.cu")
        if not cand_cu.is_file():
            self._append_event(
                Stage.TUNING_LOOP,
                status="best_update_failed",
                detail=f"candidate {candidate_id}/candidate.cu missing",
            )
            return
        self.layout.best_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cand_cu, self.layout.best_cu_path)
        # Persist a small best_result.json next to it for quick inspection.
        self.layout.best_result_path.write_text(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "speedup": speedup,
                    "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.run_state.best_candidate_id = candidate_id
        if isinstance(speedup, (int, float)):
            self.run_state.best_speedup = float(speedup)
        self._vprint(f"[orch] best -> {candidate_id} speedup={speedup}")

    def _sync_output(self) -> None:
        """Copy best/best.cu to the root-level output_path. Idempotent."""
        if not self.layout.has_best():
            return
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.layout.best_cu_path, self.output_path)
        except OSError as e:
            self._append_event(
                Stage.TUNING_LOOP,
                status="output_sync_failed",
                detail=f"{type(e).__name__}: {e}",
            )

    # ---- finalize -------------------------------------------------------

    def _finalize(self) -> None:
        """Make sure ``optimized_lora.cu`` exists at the root, then mark done.

        SummaryAgent (Step 5) will fill in final/final_report.json + summary.md;
        the orchestrator only guarantees the submission contract here.
        """
        self.run_state.set_current_stage(Stage.FINALIZE.value)
        self._sync_output()
        if not self.output_path.is_file():
            self._append_event(
                Stage.FINALIZE,
                status="no_output_at_finalize",
                detail="best/best.cu absent — submission would fail",
            )
        self.run_state.mark_completed(Stage.FINALIZE)
        self._save_state()

    # ---- persistence ----------------------------------------------------

    def _save_state(self) -> None:
        self._tick()
        self.layout.state_path.write_text(self.run_state.to_json(), encoding="utf-8")

    def _append_leaderboard(self, cand: dict) -> None:
        line = json.dumps(cand, ensure_ascii=False)
        with self.layout.leaderboard_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _append_event(self, stage: Stage | str, **payload: Any) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        s = stage.value if isinstance(stage, Stage) else stage
        record = {"ts": ts, "stage": s, **payload}
        try:
            with self.layout.events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            # Don't let logging failures crash the pipeline.
            pass

    # ---- misc -----------------------------------------------------------

    def _vprint(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
