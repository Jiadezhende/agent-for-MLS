"""PipelineOrchestrator + RoundRunner + role registry construction.

All orchestration is consolidated in this module. The orchestrator owns the
state machine, blackboard persistence, candidate-best promotion, and
``./optimized_lora.cu`` synchronization. The ``RoundRunner`` owns one tuning
round (Analyst → Optimizer → evaluation → maybe-promote) and is the only
code allowed to invoke the ``lora_resources.evaluation.evaluate_candidate``
function.
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
from mls_agent import AgentConfig, NullObserver, StdoutObserver, Tool, ToolRegistry
from mls_agent.tools.builtin import make_side_effect_tools, make_skill_tools
from mls_agent.tools.cuda.profile_tools import make_profile_tools

from operator_opt_pipe import agents
from operator_opt_pipe.lora_resources import benchmark as lr_benchmark
from operator_opt_pipe.lora_resources import baseline as lr_baseline
from operator_opt_pipe.lora_resources import evaluation as lr_evaluation
from operator_opt_pipe.lora_resources.contract import LoRAContract
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
from operator_opt_pipe.tools import (
    EditCandidateTool,
    ReadBlackboardTool,
    SubmitCandidateTool,
    SubmitTool,
    VerifyCandidateTool,
    WriteCandidateTool,
)
from operator_opt_pipe.transitions import next_stage


MAX_LOOP_ITERATIONS = 200


# ---------------------------------------------------------------------------
# Role → tool whitelist
# ---------------------------------------------------------------------------


ROLE_TOOLS: dict[str, tuple[str, ...]] = {
    "hardware_profiler": (
        "read_skill", "list_skills",
        "run_cuda_probe", "profile_with_ncu", "profile_with_nsys",
        "probe_environment",
        "record_measurement", "flag_event",
        "submit_hardware_profile",
    ),
    "optimizer_cold": (
        "read_skill",
        "read_blackboard",
        "write_candidate", "edit_candidate", "verify_candidate", "submit_candidate",
        "flag_event",
    ),
    "analyst": (
        "read_skill",
        "read_blackboard",
        "profile_with_ncu", "profile_with_nsys", "profile_with_torch",
        "record_measurement", "flag_event",
        "submit_diagnosis",
    ),
    "optimizer": (
        "read_skill",
        "read_blackboard",
        "write_candidate", "edit_candidate", "verify_candidate", "submit_candidate",
        "flag_event",
    ),
    "summary": (
        "read_skill",
        "read_blackboard",
        "submit_summary", "flag_event",
    ),
}


def build_registry(role: str, tools: dict[str, Tool]) -> ToolRegistry:
    """Return a fresh ``ToolRegistry`` populated with the role's whitelist.

    Missing tool names raise ``ValueError`` — production must supply the
    required ``mls_agent.tools.builtin`` and ``mls_agent.tools.cuda.profile_tools``
    factories before invoking the agent; tests can mock them.
    """
    if role not in ROLE_TOOLS:
        raise ValueError(f"unknown role {role!r}; valid: {sorted(ROLE_TOOLS)}")
    allowed = ROLE_TOOLS[role]
    missing = [n for n in allowed if n not in tools]
    if missing:
        raise ValueError(
            f"role {role!r} requires tools that are not provided: {missing}. "
            "Supply them via make_default_tools(builtin_factory=..., profile_factory=...) "
            "or via the orchestrator's `tools` injection point."
        )
    reg = ToolRegistry()
    for name in allowed:
        reg.register(tools[name])
    return reg


# ---------------------------------------------------------------------------
# Tool-bag construction
# ---------------------------------------------------------------------------


def make_default_tools(
    *,
    layout: RunLayout,
    executor: Any | None = None,
    skills_dir: str | Path | None = None,
) -> dict[str, Tool]:
    """Build the default ``name → Tool`` mapping for a run.

    Always populated:
      * operator-pipe internal tools (read_blackboard, submit_*, candidate stubs)
      * side-effect tools (record_measurement, flag_event)

    Conditionally populated:
      * skill tools (list_skills, read_skill) — when ``skills_dir`` is provided
      * profile tools (run_cuda_probe, profile_with_*, write_workspace_file,
        probe_environment) — when ``executor`` is provided

    Skipping the optional groups is useful for unit tests that bypass the LLM
    layer entirely; production callers (``main.py``) supply both.
    """
    tools: dict[str, Tool] = {
        "read_blackboard": ReadBlackboardTool(layout),
        "submit_hardware_profile": SubmitTool(
            name="submit_hardware_profile",
            layout=layout,
            blackboard_key="hardware",
            expected_stage=Stage.HARDWARE_PROFILE,
        ),
        "submit_diagnosis": SubmitTool(
            name="submit_diagnosis",
            layout=layout,
            blackboard_key="latest_diagnosis",
            expected_stage=Stage.TUNING_LOOP,
        ),
        "submit_summary": SubmitTool(
            name="submit_summary",
            layout=layout,
            blackboard_key="final_summary",
            expected_stage=Stage.FINALIZE,
        ),
        "write_candidate": WriteCandidateTool(layout),
        "edit_candidate": EditCandidateTool(layout),
        "verify_candidate": VerifyCandidateTool(layout),
        "submit_candidate": SubmitCandidateTool(layout),
    }

    # Side-effect tools — no external deps.
    for tool in make_side_effect_tools():
        tools[tool.NAME] = tool

    # Skill tools — need a directory to scan.
    if skills_dir is not None:
        for tool in make_skill_tools(skills_dir):
            tools[tool.NAME] = tool

    # CUDA profile / probe tools — need a live Executor.
    if executor is not None:
        for tool in make_profile_tools(executor=executor):
            tools[tool.NAME] = tool

    return tools


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


# Type aliases for injection points
EvaluatorFn = Callable[..., Any]
"""(layout, executor, candidate_id, baseline_ms_median, **kw) -> EvalResult-like."""

PromoteCallback = Callable[[str, float | None], None]
"""(candidate_id, speedup) -> None — orchestrator's _promote_to_best."""


# ---------------------------------------------------------------------------
# RoundRunner
# ---------------------------------------------------------------------------


class RoundRunner:
    """One Analyst → Optimizer → evaluate → maybe-promote round.

    The class is constructed fresh per round by the orchestrator so its state
    (round_index, latest candidate_id) does not leak across rounds.
    """

    def __init__(
        self,
        *,
        layout: RunLayout,
        contract: LoRAContract,
        backend: mls_agent.LLMBackend,
        agent_cfg: AgentConfig,
        executor: Any,
        tools: dict[str, Tool],
        evaluator: EvaluatorFn,
        promote_callback: PromoteCallback,
        observer: mls_agent.AgentObserver,
        round_index: int,
        analyst_runner: Callable[..., dict] | None = None,
        optimizer_runner: Callable[..., dict] | None = None,
    ) -> None:
        self.layout = layout
        self.contract = contract
        self.backend = backend
        self.agent_cfg = agent_cfg
        self.executor = executor
        self.tools = tools
        self.evaluator = evaluator
        self.promote_callback = promote_callback
        self.observer = observer
        self.round_index = round_index
        self._analyst_runner = analyst_runner or agents.run_analyst
        self._optimizer_runner = optimizer_runner or agents.run_optimizer

    # ------------------------------------------------------------------

    def run_one_round(self, remaining_budget_s: float) -> RoundResult:
        # 1. Round bookkeeping
        bb = load_blackboard(self.layout)
        bb["round"] = {
            "index": self.round_index,
            "started_at": _utcnow_iso(),
            "budget_s": remaining_budget_s,
        }
        save_blackboard(self.layout, bb)

        result = RoundResult(
            round_index=self.round_index,
            candidate_id=None,
            speedup=None,
            promoted=False,
            diagnosis_status=None,
            optimizer_status=None,
            eval_status=None,
        )

        # 2. Analyst
        analyst_payload = self._analyst_runner(
            backend=self.backend,
            registry=build_registry("analyst", self.tools),
            layout=self.layout,
            contract=self.contract,
            agent_cfg=self.agent_cfg,
            observer=self.observer,
        )
        result.diagnosis_status = analyst_payload.get("status")
        append_history(
            self.layout,
            {
                "step": ROUND_STEP_ANALYZE,
                "round_index": self.round_index,
                "ts": _utcnow_iso(),
                "status": analyst_payload.get("status"),
                "summary": analyst_payload.get("metrics", {}).get("summary")
                or analyst_payload.get("next_recommendation"),
            },
        )

        # 3. Optimizer
        optimizer_payload = self._optimizer_runner(
            backend=self.backend,
            registry=build_registry("optimizer", self.tools),
            layout=self.layout,
            contract=self.contract,
            agent_cfg=self.agent_cfg,
            observer=self.observer,
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

        # 4. Evaluation
        baseline_ms = _read_baseline_median(self.layout)
        try:
            eval_result = self.evaluator(
                layout=self.layout,
                executor=self.executor,
                candidate_id=candidate_id,
                baseline_ms_median=baseline_ms,
            )
        except Exception as exc:  # noqa: BLE001 — evaluator failures must not crash the round
            result.eval_status = "error"
            result.caveats.append(f"evaluator raised: {type(exc).__name__}: {exc}")
            append_history(
                self.layout,
                {
                    "step": ROUND_STEP_EVALUATE,
                    "round_index": self.round_index,
                    "ts": _utcnow_iso(),
                    "candidate_id": candidate_id,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return result

        eval_dict = _eval_to_dict(eval_result)
        speedup = eval_dict.get("speedup")
        compile_ok = bool(eval_dict.get("compile_ok"))
        correctness_ok = bool(eval_dict.get("correctness_ok"))
        result.speedup = speedup
        result.eval_status = "ok" if (compile_ok and correctness_ok) else "rejected"

        # leaderboard append (jsonl)
        _append_leaderboard(self.layout, {
            "round_index": self.round_index,
            "candidate_id": candidate_id,
            "compile_ok": compile_ok,
            "correctness_ok": correctness_ok,
            "speedup": speedup,
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
                "speedup": speedup,
                "compile_ok": compile_ok,
                "correctness_ok": correctness_ok,
            },
        )

        # 5. Maybe promote — orchestrator owns the actual best/best.cu copy
        if compile_ok and correctness_ok and speedup is not None:
            current_best_speedup = _read_current_best_speedup(self.layout)
            if current_best_speedup is None or speedup > current_best_speedup:
                self.promote_callback(candidate_id, speedup)
                result.promoted = True

        return result


# ---------------------------------------------------------------------------
# PipelineOrchestrator
# ---------------------------------------------------------------------------


class PipelineOrchestrator:
    """Top-level state machine driver.

    The orchestrator does not reach into ``mls_agent`` directly; it routes
    each stage to either a deterministic ``lora_resources`` call (for
    BENCHMARK_BASELINE) or a function in ``operator_opt_pipe.agents`` (for
    LLM stages). All best-promotion and ``./optimized_lora.cu`` syncing
    flows through this class.
    """

    def __init__(
        self,
        *,
        spec: dict,
        time_budget_s: float,
        workspace_root: str | Path,
        output_path: str | Path,
        backend: mls_agent.LLMBackend,
        agent_cfg: AgentConfig,
        executor: Any,
        contract: LoRAContract,
        tools: dict[str, Tool] | None = None,
        skills_dir: str | Path | None = None,
        run_id: str | None = None,
        verbose: bool = False,
        # Injection points — tests override; production uses lora_resources.*
        evaluator: EvaluatorFn | None = None,
        baseline_runner: Callable[..., Any] | None = None,
        hardware_profiler: Callable[..., dict] | None = None,
        initial_candidate_runner: Callable[..., dict] | None = None,
        analyst_runner: Callable[..., dict] | None = None,
        optimizer_runner: Callable[..., dict] | None = None,
        summary_runner: Callable[..., dict] | None = None,
    ) -> None:
        self.spec = spec
        self.time_budget_s = float(time_budget_s)
        self.workspace_root = Path(workspace_root).resolve()
        self.output_path = Path(output_path).resolve()
        self.backend = backend
        self.agent_cfg = agent_cfg
        self.executor = executor
        self.contract = contract
        self.run_id = run_id or make_run_id()
        self.layout = RunLayout(workspace_root=self.workspace_root, run_id=self.run_id)
        self.observer = StdoutObserver(prefix=f"[{self.run_id[:8]}] ") if verbose else NullObserver()
        self.verbose = verbose

        # Default tool bag includes side-effect tools, plus skill / profile
        # tools when their dependencies (skills_dir / executor) are provided.
        if tools is not None:
            self.tools = tools
        else:
            self.tools = make_default_tools(
                layout=self.layout,
                executor=executor,
                skills_dir=skills_dir,
            )

        # Evaluator / baseline runner default to the lora_resources stubs.
        self.evaluator = evaluator or lr_evaluation.evaluate_candidate
        self.baseline_runner = baseline_runner or _default_baseline_runner

        # Agent-stage injection points: default to the real agents.run_*
        # functions; tests pass mocks.
        self._hardware_profiler = hardware_profiler or agents.run_hardware_profiler
        self._initial_candidate_runner = initial_candidate_runner or agents.run_optimizer_cold
        self._analyst_runner = analyst_runner or agents.run_analyst
        self._optimizer_runner = optimizer_runner or agents.run_optimizer
        self._summary_runner = summary_runner or agents.run_summary

        self.run_state = self._init_or_resume()
        self._wall_started: float = 0.0  # set in run()
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
                # Caller may have changed budget on resume — respect the new value.
                state.time_budget_s = self.time_budget_s
                return state
            except Exception as exc:  # noqa: BLE001 — corrupt state.json should fail loudly
                raise RuntimeError(
                    f"failed to resume from {self.layout.state_path}: {exc}"
                ) from exc
        return RunState(
            run_id=self.run_id,
            operator=str(self.spec.get("operator", "lora_matmul")),
            time_budget_s=self.time_budget_s,
        )

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
                # Compact stage banner to stderr — the agent observer handles
                # per-iteration output once we enter an LLM stage.
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
            registry = build_registry("hardware_profiler", self.tools)
        except ValueError as exc:
            self._record_stage_failure(Stage.HARDWARE_PROFILE, f"build_registry: {exc}")
            return
        payload = self._hardware_profiler(
            backend=self.backend,
            registry=registry,
            layout=self.layout,
            contract=self.contract,
            agent_cfg=self.agent_cfg,
            observer=self.observer,
        )
        if payload.get("status") in ("success", "partial"):
            self._persist_hardware_artifact(payload)
            self.run_state.mark_stage_complete(Stage.HARDWARE_PROFILE)
        else:
            self._record_stage_failure(Stage.HARDWARE_PROFILE, payload.get("caveats", []))

    def _run_benchmark_baseline(self) -> None:
        try:
            spec = lr_benchmark.generate_benchmark_spec(self.contract)
            result = self.baseline_runner(
                layout=self.layout,
                executor=self.executor,
                contract=self.contract,
                spec=spec,
            )
        except NotImplementedError as exc:
            self._record_stage_failure(
                Stage.BENCHMARK_BASELINE,
                f"lora_resources stub: {exc}",
            )
            raise
        except Exception as exc:  # noqa: BLE001
            self._record_stage_failure(
                Stage.BENCHMARK_BASELINE,
                f"baseline_runner raised: {type(exc).__name__}: {exc}",
            )
            return

        # Persist benchmark + baseline to blackboard and disk
        bb = load_blackboard(self.layout)
        bb["benchmark"] = spec.to_dict() if hasattr(spec, "to_dict") else spec
        bb["baseline"] = result.to_dict() if hasattr(result, "to_dict") else result
        save_blackboard(self.layout, bb)
        self.layout.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self.layout.baseline_path.write_text(
            json.dumps(bb["baseline"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.run_state.mark_stage_complete(Stage.BENCHMARK_BASELINE)

    def _run_initial_candidate(self) -> None:
        try:
            registry = build_registry("optimizer_cold", self.tools)
        except ValueError as exc:
            self._record_stage_failure(Stage.INITIAL_CANDIDATE, f"build_registry: {exc}")
            return
        payload = self._initial_candidate_runner(
            backend=self.backend,
            registry=registry,
            layout=self.layout,
            contract=self.contract,
            agent_cfg=self.agent_cfg,
            observer=self.observer,
        )
        candidate_id = payload.get("candidate_id")
        if payload.get("status") != "success" or not candidate_id:
            self._record_stage_failure(
                Stage.INITIAL_CANDIDATE,
                payload.get("caveats") or [f"no candidate_id in payload: {payload}"],
            )
            return

        # Run evaluation on the initial candidate so the floor guarantee
        # (./optimized_lora.cu always exists once we have a working kernel)
        # holds even before the tuning loop starts.
        baseline_ms = _read_baseline_median(self.layout)
        try:
            eval_result = self.evaluator(
                layout=self.layout,
                executor=self.executor,
                candidate_id=candidate_id,
                baseline_ms_median=baseline_ms,
            )
        except Exception as exc:  # noqa: BLE001
            self._record_stage_failure(
                Stage.INITIAL_CANDIDATE,
                f"evaluator raised: {type(exc).__name__}: {exc}",
            )
            return

        eval_dict = _eval_to_dict(eval_result)
        if eval_dict.get("compile_ok") and eval_dict.get("correctness_ok"):
            speedup = eval_dict.get("speedup")
            self._promote_to_best(candidate_id, speedup)
            self.run_state.mark_stage_complete(Stage.INITIAL_CANDIDATE)
        else:
            self._record_stage_failure(
                Stage.INITIAL_CANDIDATE,
                f"initial candidate failed compile/correctness: {eval_dict}",
            )

    def _run_tuning_loop(self) -> None:
        self.run_state.round_index += 1
        runner = RoundRunner(
            layout=self.layout,
            contract=self.contract,
            backend=self.backend,
            agent_cfg=self.agent_cfg,
            executor=self.executor,
            tools=self.tools,
            evaluator=self.evaluator,
            promote_callback=self._promote_to_best,
            observer=self.observer,
            round_index=self.run_state.round_index,
            analyst_runner=self._analyst_runner,
            optimizer_runner=self._optimizer_runner,
        )
        runner.run_one_round(self.run_state.remaining_budget_s())

    def _run_finalize(self) -> None:
        try:
            registry = build_registry("summary", self.tools)
        except ValueError as exc:
            payload = {
                "status": "failed",
                "stage": Stage.FINALIZE.value,
                "caveats": [f"build_registry: {exc}"],
            }
        else:
            try:
                payload = self._summary_runner(
                    backend=self.backend,
                    registry=registry,
                    layout=self.layout,
                    contract=self.contract,
                    agent_cfg=self.agent_cfg,
                    observer=self.observer,
                )
            except Exception as exc:  # noqa: BLE001 — finalize must never crash the run
                payload = {
                    "status": "failed",
                    "stage": Stage.FINALIZE.value,
                    "caveats": [f"summary agent raised: {type(exc).__name__}: {exc}"],
                    "trace": traceback.format_exc(),
                }
        self._write_final_artifacts(payload)
        self.run_state.mark_stage_complete(Stage.FINALIZE)
        self._final_summary = payload

    # ------------------------------------------------------------------
    # Best promotion + output sync
    # ------------------------------------------------------------------

    def _promote_to_best(self, candidate_id: str, speedup: float | None) -> None:
        cand_cu = self.layout.candidate_file(candidate_id, "candidate.cu")
        self.layout.best_dir.mkdir(parents=True, exist_ok=True)
        if cand_cu.is_file():
            shutil.copyfile(cand_cu, self.layout.best_cu_path)
        else:
            # Skeleton phase: candidate file may not exist on disk because
            # write_candidate is stubbed. Drop a marker so downstream code
            # (and tests) can tell that promotion was logically requested.
            self.layout.best_cu_path.write_text(
                f"// placeholder for {candidate_id} — write_candidate stub\n",
                encoding="utf-8",
            )
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
        if not self.layout.has_best():
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.layout.best_cu_path, self.output_path)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_hardware_artifact(self, payload: dict) -> None:
        self.layout.hardware_path.parent.mkdir(parents=True, exist_ok=True)
        self.layout.hardware_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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
        self.layout.final_dir.mkdir(parents=True, exist_ok=True)
        self.layout.final_report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Brief markdown summary for human consumption
        summary_md = _format_summary_md(self.run_state, payload)
        self.layout.summary_path.write_text(summary_md, encoding="utf-8")
        # Always re-sync the output one last time so a finalize-without-best
        # run still yields whatever best.cu we managed to produce.
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


def _read_baseline_median(layout: RunLayout) -> float:
    """Best-effort: read pytorch_ms_median from the blackboard.

    Returns 0.0 if absent — the evaluator stub doesn't actually use this
    value, but the production evaluator will.
    """
    bb = load_blackboard(layout)
    baseline = bb.get("baseline") or {}
    return float(baseline.get("ms_median_overall", 0.0) or 0.0)


def _read_current_best_speedup(layout: RunLayout) -> float | None:
    bb = load_blackboard(layout)
    best = bb.get("best") or {}
    sp = best.get("speedup")
    return float(sp) if isinstance(sp, (int, float)) else None


def _eval_to_dict(eval_result: Any) -> dict:
    if isinstance(eval_result, dict):
        return eval_result
    if hasattr(eval_result, "to_dict"):
        return eval_result.to_dict()
    raise TypeError(
        f"evaluator must return a dict or an object with .to_dict(); "
        f"got {type(eval_result).__name__}"
    )


def _default_baseline_runner(*, layout, executor, contract, spec):
    """Default baseline runner — delegates to the lora_resources stub."""
    return lr_baseline.run_pytorch_baseline(layout, executor, contract, spec)


def _format_summary_md(state: RunState, final_payload: dict) -> str:
    lines = [
        f"# Run {state.run_id} — {state.operator}",
        "",
        f"- Elapsed: {state.elapsed_s:.1f}s / {state.time_budget_s:.0f}s budget",
        f"- Completed stages: {', '.join(state.completed_stages) or '(none)'}",
        f"- Best candidate: {state.best_candidate_id or '(none)'}",
        f"- Best speedup: {state.best_speedup if state.best_speedup is not None else '(none)'}",
        "",
        "## Final agent payload",
        "",
        "```json",
        json.dumps(final_payload, ensure_ascii=False, indent=2),
        "```",
    ]
    return "\n".join(lines) + "\n"


