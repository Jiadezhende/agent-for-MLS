"""Orchestrator + RoundRunner integration tests with injected mocks.

Bypasses the LLM layer by passing per-stage runner functions; bypasses GPU
work by passing fake baseline / materialize / benchmark runners. The
agent-glue tests live in test_agents.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mls_agent import AgentConfig

from operator_opt_pipe.operators import load_ops
from operator_opt_pipe.operators._base import OperatorOps
from operator_opt_pipe.orchestrator import (
    PipelineOrchestrator,
    RoundRunner,
)
from operator_opt_pipe.resources import OperatorContract, TensorSpec
from operator_opt_pipe.state import (
    RunLayout,
    Stage,
    load_blackboard,
    save_blackboard,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def contract() -> OperatorContract:
    return OperatorContract(
        name="operators/lora_matmul",
        inputs=(
            TensorSpec(name="W", shape=("d", "d"), dtype="float32"),
            TensorSpec(name="X", shape=("d", "d"), dtype="float32"),
            TensorSpec(name="A", shape=("d", 16), dtype="float32"),
            TensorSpec(name="B", shape=("d", 16), dtype="float32"),
        ),
        output=TensorSpec(name="Y", shape=("d", "d"), dtype="float32"),
        reference_pytorch="W @ X + A @ (B.transpose(0, 1).contiguous() @ X)",
        forward_args=("W", "X", "A", "B"),
        shape_param="d",
        shape_param_range=(3584, 4608),
    )


@pytest.fixture
def ops() -> OperatorOps:
    return load_ops("lora_matmul")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


class _NoopExecutor:
    def __getattr__(self, name):
        def _err(*a, **kw):
            raise AssertionError(f"_NoopExecutor.{name} should not be called in tests")
        return _err


class _NullBackend:
    """Minimal LLMBackend stub — tests use injected runners instead."""

    def chat(self, messages, tools):
        raise AssertionError("LLM backend must not be invoked when runners are injected")


# Canned agent runners --------------------------------------------------------


def _canned_hardware(**kw):
    layout = kw["layout"]
    bb = load_blackboard(layout)
    bb["hardware"] = {"metrics": {"sm": 30, "dram_bw_gbps": 384.0}}
    save_blackboard(layout, bb)
    return {
        "status": "success",
        "stage": Stage.HARDWARE_PROFILE.value,
        "blackboard_key": "hardware",
        "payload": bb["hardware"],
    }


def _canned_initial_candidate(cid: str = "candidate_000"):
    def _runner(**kw):
        layout = kw["layout"]
        target = layout.candidate_file(cid, "candidate.cu")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("// initial candidate\n", encoding="utf-8")
        return {
            "status": "success",
            "stage": Stage.TUNING_LOOP.value,
            "candidate_id": cid,
            "hypothesis": "naive baseline kernel",
            "experiment_type": "baseline",
        }
    return _runner


def _canned_failing_initial(**kw):
    return {
        "status": "failed",
        "stage": Stage.INITIAL_CANDIDATE.value,
        "caveats": ["test forces failure"],
    }


def _canned_analyst(**kw):
    layout = kw["layout"]
    bb = load_blackboard(layout)
    bb["latest_diagnosis"] = {
        "bottleneck": "dram_bound",
        "evidence": ["test"],
    }
    save_blackboard(layout, bb)
    return {"status": "success", "stage": Stage.TUNING_LOOP.value,
            "blackboard_key": "latest_diagnosis",
            "payload": bb["latest_diagnosis"]}


def _canned_optimizer(cid: str = "candidate_111"):
    def _runner(**kw):
        layout = kw["layout"]
        target = layout.candidate_file(cid, "candidate.cu")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("// canned\n", encoding="utf-8")
        return {
            "status": "success", "stage": Stage.TUNING_LOOP.value,
            "candidate_id": cid, "hypothesis": "test",
            "experiment_type": "canned",
        }
    return _runner


def _canned_summary(**kw):
    layout = kw["layout"]
    bb = load_blackboard(layout)
    bb["final_summary"] = {
        "best_speedup": bb.get("best", {}).get("speedup"),
        "narrative": "test summary",
    }
    save_blackboard(layout, bb)
    return {"status": "success", "stage": Stage.FINALIZE.value,
            "blackboard_key": "final_summary",
            "payload": bb["final_summary"]}


def _fake_baseline_runner(**kw):
    return {
        "spec": kw["spec"].to_dict(),
        "per_shape": {f"d{d}": {"ms_median": 100.0, "ms_min": 90.0,
                                "ms_max": 110.0, "samples": 30}
                      for d in kw["spec"].shape_grid},
        "ms_median_overall": 100.0,
        "reference_pytorch": kw["ops"].reference_doc(),
    }


def _fake_fixtures(**kw):
    return {"shape_ids": [f"d{d}" for d in kw["spec"].shape_grid]}


def _fake_benchmark_runner(*, speedup: float, all_correct: bool = True):
    def _runner(**kw):
        spec = kw["spec"]
        cid = kw["candidate_id"]
        per_shape = {}
        for d in spec.shape_grid:
            sid = f"d{d}"
            per_shape[sid] = {
                "ms_median": 50.0, "ms_min": 45.0, "ms_max": 55.0,
                "samples": 30, "max_abs_err": 1e-6, "rel_l2_err": 1e-7,
                "speedup": speedup,
            }
        return {
            "candidate_id": cid,
            "compile_ok": True,
            "per_shape": per_shape,
            "speedup_geomean": speedup,
            "speedup_worst": speedup,
            "speedup_best": speedup,
            "correctness_per_shape": {sid: all_correct for sid in per_shape},
            "all_correct": all_correct,
            "diagnostics": {},
        }
    return _runner


# ---------------------------------------------------------------------------
# RoundRunner: promotion only when speedup > current best
# ---------------------------------------------------------------------------


def test_round_runner_promotes_only_on_better_speedup(workspace: Path, contract: OperatorContract, ops: OperatorOps):
    layout = RunLayout(workspace_root=workspace, run_id="r")
    layout.mkdir()
    save_blackboard(layout, {
        "schema_version": 1, "history": [],
        "best": {"candidate_id": "candidate_000", "speedup": 1.5},
        # Synthetic benchmark spec mirrors what orchestrator seeds.
        "benchmark": {"shape_grid": [3584, 4096, 4608], "samples": 30,
                      "warmup": 5, "seed": 0},
    })
    promotions: list[tuple[str, float | None]] = []

    runner = RoundRunner(
        layout=layout, contract=contract, ops=ops,
        backend=_NullBackend(), agent_cfg=AgentConfig(max_iterations=2),
        executor=_NoopExecutor(), skills_dir=None,
        benchmark_runner=_fake_benchmark_runner(speedup=1.2),
        promote_callback=lambda cid, sp: promotions.append((cid, sp)),
        observer=__import__("mls_agent").NullObserver(),
        round_index=1,
        analyst_runner=_canned_analyst,
        optimizer_runner=_canned_optimizer(),
    )
    result = runner.run_one_round(remaining_budget_s=300.0)
    assert result.candidate_id == "candidate_111"
    assert result.eval_status == "ok"
    assert result.promoted is False
    assert promotions == []


def test_round_runner_promotes_when_speedup_better(workspace: Path, contract: OperatorContract, ops: OperatorOps):
    layout = RunLayout(workspace_root=workspace, run_id="r")
    layout.mkdir()
    save_blackboard(layout, {
        "schema_version": 1, "history": [],
        "benchmark": {"shape_grid": [3584, 4096, 4608], "samples": 30,
                      "warmup": 5, "seed": 0},
    })
    promotions: list[tuple[str, float | None]] = []

    runner = RoundRunner(
        layout=layout, contract=contract, ops=ops,
        backend=_NullBackend(), agent_cfg=AgentConfig(max_iterations=2),
        executor=_NoopExecutor(), skills_dir=None,
        benchmark_runner=_fake_benchmark_runner(speedup=2.3),
        promote_callback=lambda cid, sp: promotions.append((cid, sp)),
        observer=__import__("mls_agent").NullObserver(),
        round_index=1,
        analyst_runner=_canned_analyst,
        optimizer_runner=_canned_optimizer(),
    )
    result = runner.run_one_round(remaining_budget_s=300.0)
    assert result.promoted is True
    assert promotions == [("candidate_111", 2.3)]
    # leaderboard.jsonl exists with one line
    assert layout.leaderboard_path.is_file()
    lines = layout.leaderboard_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["candidate_id"] == "candidate_111"
    assert entry["speedup_geomean"] == 2.3
    # benchmark/<cid>.json persisted
    assert layout.benchmark_result_path("candidate_111").is_file()


def test_round_runner_writes_history_steps(workspace: Path, contract: OperatorContract, ops: OperatorOps):
    layout = RunLayout(workspace_root=workspace, run_id="r")
    layout.mkdir()
    save_blackboard(layout, {
        "schema_version": 1, "history": [],
        "benchmark": {"shape_grid": [3584, 4096, 4608], "samples": 30,
                      "warmup": 5, "seed": 0},
    })
    runner = RoundRunner(
        layout=layout, contract=contract, ops=ops,
        backend=_NullBackend(), agent_cfg=AgentConfig(max_iterations=2),
        executor=_NoopExecutor(), skills_dir=None,
        benchmark_runner=_fake_benchmark_runner(speedup=1.7),
        promote_callback=lambda cid, sp: None,
        observer=__import__("mls_agent").NullObserver(),
        round_index=2,
        analyst_runner=_canned_analyst,
        optimizer_runner=_canned_optimizer(),
    )
    runner.run_one_round(remaining_budget_s=120.0)
    bb = load_blackboard(layout)
    history = bb["history"]
    steps = [entry["step"] for entry in history]
    assert steps == ["ANALYZE", "OPTIMIZE", "EVALUATE"]
    assert all(entry["round_index"] == 2 for entry in history)


# ---------------------------------------------------------------------------
# PipelineOrchestrator end-to-end with injected mocks
# ---------------------------------------------------------------------------


def _build_orch(workspace: Path, contract: OperatorContract, ops: OperatorOps, **overrides) -> PipelineOrchestrator:
    """Default orchestrator wiring with safe stubs everywhere."""
    output_path = workspace / "optimized_lora.cu"
    defaults = dict(
        operator="lora_matmul",
        time_budget_s=1.0,
        workspace_root=workspace,
        output_path=output_path,
        backend=_NullBackend(),
        agent_cfg=AgentConfig(max_iterations=2),
        executor=_NoopExecutor(),
        contract=contract,
        ops=ops,
        run_id=overrides.pop("run_id", "run_test"),
        verbose=False,
        fixtures_runner=_fake_fixtures,
        baseline_runner=_fake_baseline_runner,
        benchmark_runner=_fake_benchmark_runner(speedup=2.0),
        hardware_profiler=_canned_hardware,
        initial_candidate_runner=_canned_initial_candidate(),
        analyst_runner=_canned_analyst,
        optimizer_runner=_canned_optimizer(),
        summary_runner=_canned_summary,
    )
    defaults.update(overrides)
    return PipelineOrchestrator(**defaults)


def test_orchestrator_runs_full_pipeline_with_injected_mocks(
    workspace: Path, contract: OperatorContract, ops: OperatorOps,
):
    output_path = workspace / "optimized_lora.cu"
    orch = _build_orch(workspace, contract, ops, output_path=output_path)
    summary = orch.run()

    state_blob = json.loads(orch.layout.state_path.read_text(encoding="utf-8"))
    assert state_blob["best_candidate_id"] == "candidate_000"
    assert state_blob["best_speedup"] == 2.0
    assert Stage.HARDWARE_PROFILE.value in state_blob["completed_stages"]
    assert Stage.BENCHMARK_BASELINE.value in state_blob["completed_stages"]
    assert Stage.INITIAL_CANDIDATE.value in state_blob["completed_stages"]
    assert Stage.FINALIZE.value in state_blob["completed_stages"]

    # ./optimized_lora.cu mirrors best/best.cu after promotion.
    assert output_path.is_file()
    assert output_path.read_text(encoding="utf-8") == "// initial candidate\n"

    assert summary["best_candidate_id"] == "candidate_000"
    assert summary["best_speedup"] == 2.0
    assert summary["final"]["status"] == "success"
    assert orch.layout.final_report_path.is_file()
    assert orch.layout.summary_path.is_file()


def test_orchestrator_finalize_reads_blackboard_summary(
    workspace: Path, contract: OperatorContract, ops: OperatorOps,
):
    """FINALIZE renders narrative from blackboard["final_summary"], not from
    the summary agent's payload directly."""
    orch = _build_orch(workspace, contract, ops, run_id="run_finalize")
    orch.run()
    final_payload = json.loads(orch.layout.final_report_path.read_text(encoding="utf-8"))
    assert "narrative" in final_payload
    assert final_payload["narrative"]["narrative"] == "test summary"


def test_orchestrator_resume_picks_existing_state(workspace: Path, contract: OperatorContract, ops: OperatorOps):
    """First run completes hardware + baseline, fails INITIAL_CANDIDATE → finalize.
    Second orchestrator with same run_id picks up completed_stages."""
    orch1 = _build_orch(
        workspace, contract, ops,
        run_id="run_resume",
        initial_candidate_runner=_canned_failing_initial,
    )
    orch1.run()
    assert orch1.layout.has_hardware_profile()
    assert orch1.layout.has_baseline()

    orch2 = _build_orch(
        workspace, contract, ops,
        run_id="run_resume",
        initial_candidate_runner=_canned_failing_initial,
    )
    assert Stage.HARDWARE_PROFILE.value in orch2.run_state.completed_stages
    assert Stage.BENCHMARK_BASELINE.value in orch2.run_state.completed_stages


def test_orchestrator_benchmark_baseline_skips_llm(
    workspace: Path, contract: OperatorContract, ops: OperatorOps,
):
    """BENCHMARK_BASELINE must call the deterministic runners, not any agent."""
    fixtures_calls: list[Any] = []
    baseline_calls: list[Any] = []

    def fake_fix(**kw):
        fixtures_calls.append(kw["spec"])
        return {"shape_ids": []}

    def fake_baseline(**kw):
        baseline_calls.append(kw["spec"])
        return {"spec": kw["spec"].to_dict(), "per_shape": {},
                "ms_median_overall": 42.0,
                "reference_pytorch": kw["ops"].reference_doc()}

    orch = _build_orch(
        workspace, contract, ops,
        run_id="run_bb",
        fixtures_runner=fake_fix,
        baseline_runner=fake_baseline,
        # Make initial candidate fail so the run terminates after baseline.
        initial_candidate_runner=_canned_failing_initial,
    )
    orch.run()
    assert len(fixtures_calls) == 1
    assert len(baseline_calls) == 1
    assert orch.layout.baseline_path.is_file()
    bb = load_blackboard(orch.layout)
    assert "baseline" in bb
    # benchmark spec is no longer pre-seeded into the blackboard — it is
    # derived from the contract on demand.
    assert "benchmark" not in bb


# ---------------------------------------------------------------------------
# agent_trace.log + output.md (Phase-2 reasoning artifacts)
# ---------------------------------------------------------------------------


def test_agent_trace_log_always_on(workspace: Path, contract: OperatorContract, ops: OperatorOps):
    """agent_trace.log must be produced regardless of --verbose so that
    output.md can be derived from it after the run."""
    orch = _build_orch(workspace, contract, ops, run_id="run_trace", verbose=False)
    orch.run()
    assert orch.layout.trace_path.is_file()
    # File should contain at least the orchestrator stage transitions.
    body = orch.layout.trace_path.read_text(encoding="utf-8")
    assert "[orch]" in body
    assert "stage=HARDWARE_PROFILE" in body


def test_output_md_emitted_and_mirrored(workspace: Path, contract: OperatorContract, ops: OperatorOps):
    """FINALIZE produces output.md inside run_dir and mirrors it next to
    optimized_lora.cu so the Phase-2 harness finds both artifacts."""
    output_path = workspace / "optimized_lora.cu"
    orch = _build_orch(workspace, contract, ops,
                       run_id="run_output", output_path=output_path)
    orch.run()

    canonical = orch.layout.output_log_path
    mirror = output_path.parent / "output.md"
    assert canonical.is_file()
    assert mirror.is_file()
    assert canonical.read_text(encoding="utf-8") == mirror.read_text(encoding="utf-8")

    body = canonical.read_text(encoding="utf-8")
    for header in (
        "# Phase-2 Run Report",
        "## Run directory",
        "## Agent trace",
    ):
        assert header in body, f"missing section: {header}"
    # Header references the run id + operator.
    assert "run_output" in body
    assert "lora_matmul" in body
    # Layout legend describes the run dir contents.
    assert "candidates/candidate_NNN" in body
    assert "agent_trace.log" in body
