"""PipelineOrchestrator + RoundRunner.

Owns the pipeline state machine, blackboard persistence, multi-shape
benchmark on every accepted candidate, best promotion, and
``./optimized_lora.cu`` synchronization. Each LLM stage is a thin call
into ``operator_opt_pipe.agents``; performance evaluation is done by
``operator_opt_pipe.resources`` (deterministic) and is never reachable
from inside an agent.
"""
from __future__ import annotations

import json
import shutil
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import mls_agent
from mls_agent import AgentConfig, NullObserver, StdoutObserver

from operator_opt_pipe import agents as agents_mod
from operator_opt_pipe.operators._base import OperatorOps
from operator_opt_pipe.resources import OperatorContract
from operator_opt_pipe.resources.baseline import (
    build_correctness_fixtures,
    measure_pytorch_latency,
)
from operator_opt_pipe.resources.benchmark import BenchmarkSpec
from operator_opt_pipe.resources.evaluation import benchmark_on_grid
from operator_opt_pipe.state import (
    ROUND_STEP_ANALYZE,
    ROUND_STEP_EVALUATE,
    ROUND_STEP_OPTIMIZE,
    RunLayout,
    RunState,
    Stage,
    append_history,
    load_blackboard,
    make_run_id,
    save_blackboard,
)
from operator_opt_pipe.transitions import next_stage


MAX_LOOP_ITERATIONS = 200


# ---------------------------------------------------------------------------
# Helper types
# ---------------------------------------------------------------------------


@dataclass
class RoundResult:
    round_index: int
    candidate_id: str | None
    speedup: float | None
    promoted: bool
    diagnosis_status: str | None
    optimizer_status: str | None
    eval_status: str | None
    caveats: list[str] = field(default_factory=list)


# Type aliases
BenchmarkRunner = Callable[..., Any]
"""(ops, spec, candidate_id, candidate_cu, inputs_dir, oracle_dir,
   baseline_per_shape, build_dir, executor) -> BenchmarkResult-like."""

BaselineLatencyRunner = Callable[..., Any]
"""(ops, spec, inputs_dir) -> BaselineResult-like (in-process; no executor)."""

FixturesRunner = Callable[..., Any]
"""(ops, spec, inputs_dir, oracle_dir) -> dict (in-process; no executor)."""

PromoteCallback = Callable[[str, float | None], None]
AgentRunner = Callable[..., dict]


# ---------------------------------------------------------------------------
# RoundRunner
# ---------------------------------------------------------------------------


class RoundRunner:
    """One Analyst → Optimizer → multi-shape-benchmark → maybe-promote round.

    Constructed fresh per round so per-round state (round_index, latest
    candidate) does not leak across rounds.
    """

    def __init__(
        self,
        *,
        layout: RunLayout,
        contract: OperatorContract,
        ops: OperatorOps,
        backend: mls_agent.LLMBackend,
        agent_cfg: AgentConfig,
        executor: Any,
        skills_dir: Any | None,
        benchmark_runner: BenchmarkRunner,
        promote_callback: PromoteCallback,
        observer: mls_agent.AgentObserver,
        round_index: int,
        analyst_runner: AgentRunner | None = None,
        optimizer_runner: AgentRunner | None = None,
    ) -> None:
        self.layout = layout
        self.contract = contract
        self.ops = ops
        self.backend = backend
        self.agent_cfg = agent_cfg
        self.executor = executor
        self.skills_dir = skills_dir
        self.benchmark_runner = benchmark_runner
        self.promote_callback = promote_callback
        self.observer = observer
        self.round_index = round_index
        self._analyst_runner = analyst_runner or agents_mod.run_analyst
        self._optimizer_runner = optimizer_runner or agents_mod.run_optimizer

    # ------------------------------------------------------------------

    def run_one_round(self, remaining_budget_s: float) -> RoundResult:
        # Per-round metadata lives in events.jsonl + history entries — no
        # need to clobber a blackboard["round"] field that gets overwritten
        # every round and was never load-bearing.
        result = RoundResult(
            round_index=self.round_index,
            candidate_id=None, speedup=None, promoted=False,
            diagnosis_status=None, optimizer_status=None, eval_status=None,
        )

        # 1. Analyst
        analyst_payload = self._analyst_runner(
            backend=self.backend,
            registry=agents_mod.build_registry(
                "analyst", layout=self.layout, contract=self.contract,
                ops=self.ops,
                executor=self.executor, skills_dir=self.skills_dir,
            ),
            layout=self.layout, contract=self.contract,
            agent_cfg=self.agent_cfg, observer=self.observer,
        )
        result.diagnosis_status = analyst_payload.get("status")
        append_history(
            self.layout,
            {
                "step": ROUND_STEP_ANALYZE,
                "round_index": self.round_index,
                "ts": _utcnow_iso(),
                "status": analyst_payload.get("status"),
            },
        )

        # 2. Optimizer
        optimizer_payload = self._optimizer_runner(
            backend=self.backend,
            registry=agents_mod.build_registry(
                "optimizer", layout=self.layout, contract=self.contract,
                ops=self.ops,
                executor=self.executor, skills_dir=self.skills_dir,
            ),
            layout=self.layout, contract=self.contract,
            agent_cfg=self.agent_cfg, observer=self.observer,
        )
        result.optimizer_status = optimizer_payload.get("status")
        candidate_id = optimizer_payload.get("candidate_id")
        result.candidate_id = candidate_id
        append_history(
            self.layout,
            {
                "step": ROUND_STEP_OPTIMIZE,
                "round_index": self.round_index,
                "ts": _utcnow_iso(),
                "status": optimizer_payload.get("status"),
                "candidate_id": candidate_id,
                "hypothesis": optimizer_payload.get("hypothesis"),
                "experiment_type": optimizer_payload.get("experiment_type"),
            },
        )
        if optimizer_payload.get("status") != "success" or not candidate_id:
            result.caveats.append(
                f"optimizer did not submit a candidate ({optimizer_payload.get('status')})"
            )
            return result

        # 3. Multi-shape benchmark — orchestrator-owned, agent never sees this
        bench_dict = _run_candidate_benchmark(
            self.layout, self.ops, self.executor,
            self.benchmark_runner, candidate_id,
        )
        compile_ok = bool(bench_dict.get("compile_ok"))
        all_correct = bool(bench_dict.get("all_correct"))
        speedup = _maybe_float(bench_dict.get("speedup_geomean"))
        result.speedup = speedup
        result.eval_status = "ok" if (compile_ok and all_correct) else "rejected"

        _append_leaderboard(self.layout, {
            "round_index": self.round_index,
            "candidate_id": candidate_id,
            "compile_ok": compile_ok,
            "all_correct": all_correct,
            "speedup_geomean": speedup,
            "speedup_worst": _maybe_float(bench_dict.get("speedup_worst")),
            "ts": _utcnow_iso(),
        })

        append_history(
            self.layout,
            {
                "step": ROUND_STEP_EVALUATE,
                "round_index": self.round_index,
                "ts": _utcnow_iso(),
                "candidate_id": candidate_id,
                "status": result.eval_status,
                "speedup_geomean": speedup,
                "compile_ok": compile_ok,
                "all_correct": all_correct,
            },
        )

        # 4. Maybe promote — orchestrator owns the actual best/best.cu copy
        if compile_ok and all_correct and speedup is not None:
            current_best = _read_current_best_speedup(self.layout)
            if current_best is None or speedup > current_best:
                self.promote_callback(candidate_id, speedup)
                result.promoted = True

        return result


# ---------------------------------------------------------------------------
# PipelineOrchestrator
# ---------------------------------------------------------------------------


class PipelineOrchestrator:
    """Top-level state machine driver.

    Routing per stage:
      INIT                 — pure code (mkdir + state init + write benchmark/spec.json)
      HARDWARE_PROFILE     — agent (agents.run_hardware_profiler)
      BENCHMARK_BASELINE   — pure code (resources.benchmark + resources.baseline)
      INITIAL_CANDIDATE    — agent (agents.run_optimizer_cold) + multi-shape benchmark
      TUNING_LOOP          — RoundRunner: analyst → optimizer → benchmark → promote
      FINALIZE             — agent (agents.run_summary), then read blackboard["final_summary"]
    """

    def __init__(
        self,
        *,
        operator: str,
        time_budget_s: float,
        workspace_root: str | Path,
        output_path: str | Path,
        backend: mls_agent.LLMBackend,
        agent_cfg: AgentConfig,
        executor: Any,
        contract: OperatorContract,
        ops: OperatorOps,
        skills_dir: str | Path | None = None,
        run_id: str | None = None,
        verbose: bool = False,
        # Injection points — tests override; production uses resources.*
        baseline_runner: BaselineLatencyRunner | None = None,
        benchmark_runner: BenchmarkRunner | None = None,
        fixtures_runner: FixturesRunner | None = None,
        hardware_profiler: AgentRunner | None = None,
        initial_candidate_runner: AgentRunner | None = None,
        analyst_runner: AgentRunner | None = None,
        optimizer_runner: AgentRunner | None = None,
        summary_runner: AgentRunner | None = None,
    ) -> None:
        self.operator = operator
        self.time_budget_s = float(time_budget_s)
        self.workspace_root = Path(workspace_root).resolve()
        self.output_path = Path(output_path).resolve()
        self.backend = backend
        self.agent_cfg = agent_cfg
        self.executor = executor
        self.contract = contract
        self.ops = ops
        self.skills_dir = skills_dir
        self.run_id = run_id or make_run_id()
        self.layout = RunLayout(workspace_root=self.workspace_root, run_id=self.run_id)
        self.observer = StdoutObserver(prefix=f"[{self.run_id[:8]}] ") if verbose else NullObserver()
        self.verbose = verbose

        # Default deterministic resource runners
        self.baseline_runner = baseline_runner or measure_pytorch_latency
        self.benchmark_runner = benchmark_runner or benchmark_on_grid
        self.fixtures_runner = fixtures_runner or build_correctness_fixtures

        # Default agent runners
        self._hardware_profiler = hardware_profiler or agents_mod.run_hardware_profiler
        self._initial_candidate_runner = initial_candidate_runner or agents_mod.run_optimizer_cold
        self._analyst_runner = analyst_runner or agents_mod.run_analyst
        self._optimizer_runner = optimizer_runner or agents_mod.run_optimizer
        self._summary_runner = summary_runner or agents_mod.run_summary

        self.run_state = self._init_or_resume()
        self._wall_started: float = 0.0
        self._elapsed_at_start: float = self.run_state.elapsed_s
        self._final_summary: dict | None = None

    # ------------------------------------------------------------------
    # Setup / persistence
    # ------------------------------------------------------------------

    def _init_or_resume(self) -> RunState:
        self.layout.mkdir()
        if self.layout.state_path.is_file():
            try:
                state = RunState.from_json(self.layout.state_path.read_text(encoding="utf-8"))
                if state.run_id != self.run_id:
                    raise ValueError(
                        f"resumed state.run_id={state.run_id!r} != requested {self.run_id!r}"
                    )
                state.time_budget_s = self.time_budget_s
                self._seed_blackboard()
                return state
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"failed to resume from {self.layout.state_path}: {exc}"
                ) from exc

        state = RunState(
            run_id=self.run_id,
            operator=self.operator,
            time_budget_s=self.time_budget_s,
        )
        self._seed_blackboard()
        return state

    def _seed_blackboard(self) -> None:
        """Seed blackboard with the operator snapshot. Idempotent.

        The benchmark spec is NOT persisted to the blackboard — it is
        derived from the contract on demand (``BenchmarkSpec.for_contract``)
        so there is one source of truth.
        """
        bb = load_blackboard(self.layout)
        bb["operator"] = {
            "name": self.contract.name,
            "shape_param": self.contract.shape_param,
            "shape_param_range": list(self.contract.shape_param_range),
            "rtol": self.contract.rtol,
            "atol": self.contract.atol,
            "forward_signature": self.contract.forward_signature_text(),
            "reference_pytorch": self.contract.reference_pytorch,
        }
        save_blackboard(self.layout, bb)

    def _save_state(self) -> None:
        self._tick_elapsed()
        self.layout.state_path.write_text(self.run_state.to_json(), encoding="utf-8")

    def _tick_elapsed(self) -> None:
        wall = time.monotonic() - self._wall_started if self._wall_started else 0.0
        self.run_state.elapsed_s = self._elapsed_at_start + wall

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        self._wall_started = time.monotonic()
        for _ in range(MAX_LOOP_ITERATIONS):
            self._tick_elapsed()
            stage = next_stage(self.run_state, self.layout)
            self.run_state.current_stage = stage.value
            _emit_event(self.layout, {"type": "stage_enter", "stage": stage.value, "ts": _utcnow_iso()})
            if self.verbose:
                print(
                    f"[orch] stage={stage.value} elapsed={self.run_state.elapsed_s:.1f}s "
                    f"remaining={self.run_state.remaining_budget_s():.1f}s",
                    flush=True,
                )

            if stage is Stage.FINALIZE:
                self._run_finalize()
                self._save_state()
                break
            if stage is Stage.HARDWARE_PROFILE:
                self._run_hardware_profile()
            elif stage is Stage.BENCHMARK_BASELINE:
                self._run_benchmark_baseline()
            elif stage is Stage.INITIAL_CANDIDATE:
                self._run_initial_candidate()
            elif stage is Stage.TUNING_LOOP:
                self._run_tuning_loop()
            else:
                raise RuntimeError(f"unhandled stage {stage!r} from next_stage")
            self._save_state()
        else:
            _emit_event(self.layout, {"type": "loop_guard_tripped", "ts": _utcnow_iso()})
            self._run_finalize()
            self._save_state()
        return self._collect_summary()

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    def _run_hardware_profile(self) -> None:
        try:
            registry = agents_mod.build_registry(
                "hardware_profiler",
                layout=self.layout, contract=self.contract,
                ops=self.ops,
                executor=self.executor, skills_dir=self.skills_dir,
            )
        except ValueError as exc:
            self._record_stage_failure(Stage.HARDWARE_PROFILE, f"build_registry: {exc}")
            return
        payload = self._hardware_profiler(
            backend=self.backend, registry=registry,
            layout=self.layout, contract=self.contract,
            agent_cfg=self.agent_cfg, observer=self.observer,
        )
        if payload.get("status") in ("success", "partial"):
            # Mirror the blackboard["hardware"] payload onto disk for resume
            bb = load_blackboard(self.layout)
            hw = bb.get("hardware") or payload.get("payload") or payload
            self.layout.hardware_path.write_text(
                json.dumps(hw, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.run_state.mark_stage_complete(Stage.HARDWARE_PROFILE)
        else:
            self._record_stage_failure(Stage.HARDWARE_PROFILE, payload.get("caveats", []))

    def _run_benchmark_baseline(self) -> None:
        spec = BenchmarkSpec.for_contract(self.contract)
        try:
            self.fixtures_runner(
                ops=self.ops, spec=spec,
                inputs_dir=self.layout.inputs_dir,
                oracle_dir=self.layout.oracle_dir,
            )
            baseline = self.baseline_runner(
                ops=self.ops, spec=spec,
                inputs_dir=self.layout.inputs_dir,
            )
        except Exception as exc:  # noqa: BLE001
            self._record_stage_failure(
                Stage.BENCHMARK_BASELINE,
                f"baseline path raised: {type(exc).__name__}: {exc}",
            )
            return

        baseline_dict = _coerce_dict(baseline)
        bb = load_blackboard(self.layout)
        bb["baseline"] = baseline_dict
        save_blackboard(self.layout, bb)
        self.layout.baseline_path.write_text(
            json.dumps(baseline_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.run_state.mark_stage_complete(Stage.BENCHMARK_BASELINE)

    def _run_initial_candidate(self) -> None:
        try:
            registry = agents_mod.build_registry(
                "optimizer_cold",
                layout=self.layout, contract=self.contract,
                ops=self.ops,
                executor=self.executor, skills_dir=self.skills_dir,
            )
        except ValueError as exc:
            self._record_stage_failure(Stage.INITIAL_CANDIDATE, f"build_registry: {exc}")
            return
        payload = self._initial_candidate_runner(
            backend=self.backend, registry=registry,
            layout=self.layout, contract=self.contract,
            agent_cfg=self.agent_cfg, observer=self.observer,
        )
        candidate_id = payload.get("candidate_id")
        if payload.get("status") != "success" or not candidate_id:
            self._record_stage_failure(
                Stage.INITIAL_CANDIDATE,
                payload.get("caveats") or [f"no candidate_id in payload: {payload}"],
            )
            return

        bench = _run_candidate_benchmark(
            self.layout, self.ops, self.executor,
            self.benchmark_runner, candidate_id,
        )
        compile_ok = bool(bench.get("compile_ok"))
        all_correct = bool(bench.get("all_correct"))
        speedup = _maybe_float(bench.get("speedup_geomean"))

        if compile_ok and all_correct:
            # Initial candidate is not a "tuning round". The promotion
            # itself + the best_promoted event in events.jsonl are the
            # canonical record; leaderboard.jsonl is reserved for
            # tuning-round candidates so its line numbers align with rounds.
            self._promote_to_best(candidate_id, speedup)
            self.run_state.mark_stage_complete(Stage.INITIAL_CANDIDATE)
        else:
            self._record_stage_failure(
                Stage.INITIAL_CANDIDATE,
                f"initial candidate failed compile/correctness: "
                f"compile_ok={compile_ok}, all_correct={all_correct}",
            )

    def _run_tuning_loop(self) -> None:
        self.run_state.round_index += 1
        runner = RoundRunner(
            layout=self.layout, contract=self.contract, ops=self.ops,
            backend=self.backend, agent_cfg=self.agent_cfg,
            executor=self.executor, skills_dir=self.skills_dir,
            benchmark_runner=self.benchmark_runner,
            promote_callback=self._promote_to_best,
            observer=self.observer,
            round_index=self.run_state.round_index,
            analyst_runner=self._analyst_runner,
            optimizer_runner=self._optimizer_runner,
        )
        runner.run_one_round(self.run_state.remaining_budget_s())

    def _run_finalize(self) -> None:
        # Pre-fill final_metrics so the summary agent only writes narrative
        # and never (re-)fabricates speedup numbers.
        self._write_final_metrics()

        try:
            registry = agents_mod.build_registry(
                "summary",
                layout=self.layout, contract=self.contract,
                ops=self.ops,
                executor=self.executor, skills_dir=self.skills_dir,
            )
        except ValueError as exc:
            payload = {
                "status": "failed",
                "stage": Stage.FINALIZE.value,
                "caveats": [f"build_registry: {exc}"],
            }
        else:
            try:
                payload = self._summary_runner(
                    backend=self.backend, registry=registry,
                    layout=self.layout, contract=self.contract,
                    agent_cfg=self.agent_cfg, observer=self.observer,
                )
            except Exception as exc:  # noqa: BLE001
                payload = {
                    "status": "failed",
                    "stage": Stage.FINALIZE.value,
                    "caveats": [f"summary agent raised: {type(exc).__name__}: {exc}"],
                    "trace": traceback.format_exc(),
                }
        # Render final artifacts using whatever the agent wrote into the
        # blackboard (preferred) plus the orchestrator's own state.
        bb = load_blackboard(self.layout)
        narrative = bb.get("final_summary") or {}
        final_payload = {
            "status": payload.get("status", "failed"),
            "agent_summary": payload,
            "narrative": narrative,
            "metrics": bb.get("final_metrics") or {},
            "best": bb.get("best") or {},
            "history_count": len(bb.get("history") or []),
        }
        self._write_final_artifacts(final_payload)
        self.run_state.mark_stage_complete(Stage.FINALIZE)
        self._final_summary = final_payload

    def _write_final_metrics(self) -> None:
        """Snapshot orchestrator-owned performance numerics for the summary
        agent to read. The summary agent is forbidden from inventing
        speedup numbers — these are authoritative."""
        bb = load_blackboard(self.layout)
        best = bb.get("best") or {}
        baseline = bb.get("baseline") or {}
        bb["final_metrics"] = {
            "best_candidate_id": self.run_state.best_candidate_id,
            "best_speedup_geomean": self.run_state.best_speedup,
            "baseline_ms_median_overall": baseline.get("ms_median_overall"),
            "baseline_per_shape": baseline.get("per_shape") or {},
            "best_promoted_at": best.get("promoted_at"),
        }
        save_blackboard(self.layout, bb)

    # ------------------------------------------------------------------
    # Best promotion + output sync
    # ------------------------------------------------------------------

    def _promote_to_best(self, candidate_id: str, speedup: float | None) -> None:
        cand_cu = self.layout.candidate_file(candidate_id, "candidate.cu")
        if not cand_cu.is_file():
            # SubmitCandidateTool already validates this — if we reach the
            # promote path without a real .cu, that's a bug, not a state
            # we silently paper over with a placeholder source file.
            raise FileNotFoundError(
                f"_promote_to_best: candidate {candidate_id} has no candidate.cu at {cand_cu}"
            )
        self.layout.best_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cand_cu, self.layout.best_cu_path)
        self.layout.best_result_path.write_text(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "speedup": speedup,
                    "promoted_at": _utcnow_iso(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.run_state.best_candidate_id = candidate_id
        self.run_state.best_speedup = float(speedup) if speedup is not None else None
        bb = load_blackboard(self.layout)
        bb["best"] = {
            "candidate_id": candidate_id,
            "speedup": speedup,
            "promoted_at": _utcnow_iso(),
        }
        save_blackboard(self.layout, bb)
        self._sync_output()
        _emit_event(self.layout, {
            "type": "best_promoted", "candidate_id": candidate_id,
            "speedup": speedup, "ts": _utcnow_iso(),
        })

    def _sync_output(self) -> None:
        if self.run_state.best_candidate_id is None or not self.layout.best_cu_path.is_file():
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.layout.best_cu_path, self.output_path)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _record_stage_failure(self, stage: Stage, detail: Any) -> None:
        if isinstance(detail, list):
            detail_text = "; ".join(str(x) for x in detail)
        else:
            detail_text = str(detail)
        _emit_event(self.layout, {
            "type": "stage_failed",
            "stage": stage.value,
            "detail": detail_text,
            "ts": _utcnow_iso(),
        })

    def _write_final_artifacts(self, payload: dict) -> None:
        self.layout.final_report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_md = _format_summary_md(self.run_state, payload)
        self.layout.summary_path.write_text(summary_md, encoding="utf-8")
        # Final best-effort sync — preserve whatever optimized_lora.cu we have.
        self._sync_output()

    def _collect_summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "operator": self.run_state.operator,
            "elapsed_s": self.run_state.elapsed_s,
            "completed_stages": list(self.run_state.completed_stages),
            "best_candidate_id": self.run_state.best_candidate_id,
            "best_speedup": self.run_state.best_speedup,
            "output_path": str(self.output_path),
            "workspace": str(self.layout.run_dir),
            "final": self._final_summary,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _emit_event(layout: RunLayout, event: dict) -> None:
    layout.events_path.parent.mkdir(parents=True, exist_ok=True)
    with layout.events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _append_leaderboard(layout: RunLayout, entry: dict) -> None:
    layout.leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    with layout.leaderboard_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_current_best_speedup(layout: RunLayout) -> float | None:
    bb = load_blackboard(layout)
    best = bb.get("best") or {}
    sp = best.get("speedup")
    return float(sp) if isinstance(sp, (int, float)) else None


def _maybe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _coerce_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(
        f"expected dict or object with .to_dict(); got {type(obj).__name__}"
    )


def _run_candidate_benchmark(
    layout: RunLayout,
    ops: OperatorOps,
    executor: Any,
    benchmark_runner: BenchmarkRunner,
    candidate_id: str,
) -> dict:
    """Run the multi-shape benchmark for a candidate and persist the result."""
    cand_cu = layout.candidate_file(candidate_id, "candidate.cu")
    spec = BenchmarkSpec.for_contract(ops.contract)
    bb = load_blackboard(layout)
    baseline_per_shape = (bb.get("baseline") or {}).get("per_shape") or {}

    try:
        bench = benchmark_runner(
            ops=ops,
            spec=spec,
            candidate_id=candidate_id,
            candidate_cu=cand_cu,
            inputs_dir=layout.inputs_dir,
            oracle_dir=layout.oracle_dir,
            baseline_per_shape=baseline_per_shape,
            build_dir=layout.build_dir,
            executor=executor,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "candidate_id": candidate_id,
            "compile_ok": False,
            "all_correct": False,
            "diagnostics": {"error": f"{type(exc).__name__}: {exc}"},
        }

    bench_dict = _coerce_dict(bench)
    layout.benchmark_dir.mkdir(parents=True, exist_ok=True)
    layout.benchmark_result_path(candidate_id).write_text(
        json.dumps(bench_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return bench_dict


def _format_summary_md(state: RunState, final_payload: dict) -> str:
    lines = [
        f"# Run {state.run_id} — {state.operator}",
        "",
        f"- Elapsed: {state.elapsed_s:.1f}s / {state.time_budget_s:.0f}s budget",
        f"- Completed stages: {', '.join(state.completed_stages) or '(none)'}",
        f"- Best candidate: {state.best_candidate_id or '(none)'}",
        f"- Best speedup (geomean): {state.best_speedup if state.best_speedup is not None else '(none)'}",
        "",
        "## Final summary payload",
        "",
        "```json",
        json.dumps(final_payload, ensure_ascii=False, indent=2),
        "```",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "MAX_LOOP_ITERATIONS",
    "PipelineOrchestrator",
    "RoundRunner",
    "RoundResult",
]
