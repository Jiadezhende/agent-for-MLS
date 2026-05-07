"""Orchestrator + RoundRunner + build_registry tests.

We bypass the LLM layer entirely by injecting per-stage runners that return
canned dicts. The agent-glue tests live in test_agents.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mls_agent import AgentConfig, NullObserver, ToolRegistry

from operator_opt_pipe.lora_resources.contract import LoRAContract
from operator_opt_pipe.orchestrator import (
    PipelineOrchestrator,
    ROLE_TOOLS,
    RoundRunner,
    build_registry,
    make_default_tools,
)
from operator_opt_pipe.state import (
    RunLayout,
    Stage,
    load_blackboard,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def contract() -> LoRAContract:
    return LoRAContract(
        operator="lora_matmul",
        d_range=(3584, 4608),
        r=16,
        dtype="float32",
        device="cuda",
        forward_args=("W", "X", "A", "B"),
        reference_pytorch="W @ X + A @ (B.T @ X)",
        output_name="Y",
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


class _NoopExecutor:
    """Stand-in for ``mls_agent.tools.cuda.cuda_executor.Executor``.

    Profile tools are constructed against this object so they appear in the
    role whitelists, but no test actually dispatches a CUDA tool — the
    LLM-stage runners are mocked instead. Methods raise to make accidental
    use loud.
    """

    def __getattr__(self, name):
        def _err(*args, **kwargs):
            raise AssertionError(f"_NoopExecutor.{name} should not be called in tests")
        return _err


def _full_tool_bag(layout: RunLayout, tmp_path: Path | None = None) -> dict:
    """Build the production tool set with safe stand-ins for external deps."""
    skills_dir = (tmp_path / "skills") if tmp_path is not None else None
    return make_default_tools(layout=layout, executor=_NoopExecutor(), skills_dir=skills_dir)


# ---------------------------------------------------------------------------
# build_registry / ROLE_TOOLS
# ---------------------------------------------------------------------------


def test_role_tools_optimizer_excludes_evaluate_and_run_python():
    """The Optimizer must never see performance-evaluation tools."""
    optimizer_tools = ROLE_TOOLS["optimizer"]
    assert "evaluate_candidate" not in optimizer_tools
    assert "benchmark_candidate" not in optimizer_tools
    assert "run_python" not in optimizer_tools
    assert "edit_file" not in optimizer_tools
    # Same for cold-start optimizer
    assert "evaluate_candidate" not in ROLE_TOOLS["optimizer_cold"]
    assert "run_python" not in ROLE_TOOLS["optimizer_cold"]


def test_build_registry_missing_tool_raises(workspace: Path):
    layout = RunLayout(workspace_root=workspace, run_id="r")
    layout.mkdir()
    # Operator-pipe internal tools only — builtin/profile factories not supplied.
    tools = make_default_tools(layout=layout)
    with pytest.raises(ValueError, match="not provided"):
        build_registry("hardware_profiler", tools)


def test_build_registry_with_complete_tools(workspace: Path):
    layout = RunLayout(workspace_root=workspace, run_id="r")
    layout.mkdir()
    tools = _full_tool_bag(layout, workspace)
    reg = build_registry("hardware_profiler", tools)
    assert isinstance(reg, ToolRegistry)
    assert "submit_hardware_profile" in reg.names()
    assert "run_cuda_probe" in reg.names()


def test_build_registry_unknown_role():
    with pytest.raises(ValueError, match="unknown role"):
        build_registry("nope", {})


# ---------------------------------------------------------------------------
# RoundRunner — best promotion happens only via promote_callback
# ---------------------------------------------------------------------------


def test_round_runner_promotes_only_on_better_speedup(workspace: Path, contract: LoRAContract):
    layout = RunLayout(workspace_root=workspace, run_id="r")
    layout.mkdir()
    # Pre-existing best speedup.
    from operator_opt_pipe.state import save_blackboard
    save_blackboard(layout, {
        "schema_version": 1, "history": [],
        "best": {"candidate_id": "candidate_000", "speedup": 1.5},
    })
    tools = _full_tool_bag(layout, workspace)
    promotions: list[tuple[str, float | None]] = []

    def promote(cid: str, sp: float | None) -> None:
        promotions.append((cid, sp))

    def fake_evaluator(*, layout, executor, candidate_id, baseline_ms_median):
        # Worse speedup than current best — must NOT promote.
        return {
            "candidate_id": candidate_id,
            "compile_ok": True,
            "correctness_ok": True,
            "candidate_ms_median": 10.0,
            "speedup": 1.2,
            "samples": 30,
            "diagnostics": {},
        }

    runner = RoundRunner(
        layout=layout, contract=contract,
        backend=_NullBackend(), agent_cfg=AgentConfig(max_iterations=2),
        executor=None, tools=tools,
        evaluator=fake_evaluator,
        promote_callback=promote,
        observer=NullObserver(),
        round_index=1,
        analyst_runner=_canned_analyst,
        optimizer_runner=_canned_optimizer,
    )
    result = runner.run_one_round(remaining_budget_s=300.0)
    assert result.candidate_id == "candidate_111"
    assert result.eval_status == "ok"
    assert result.promoted is False
    assert promotions == []


def test_round_runner_promotes_when_speedup_better(workspace: Path, contract: LoRAContract):
    layout = RunLayout(workspace_root=workspace, run_id="r")
    layout.mkdir()
    from operator_opt_pipe.state import save_blackboard
    save_blackboard(layout, {"schema_version": 1, "history": []})  # no current best
    tools = _full_tool_bag(layout, workspace)
    promotions: list[tuple[str, float | None]] = []

    runner = RoundRunner(
        layout=layout, contract=contract,
        backend=_NullBackend(), agent_cfg=AgentConfig(max_iterations=2),
        executor=None, tools=tools,
        evaluator=lambda *, layout, executor, candidate_id, baseline_ms_median: {
            "candidate_id": candidate_id, "compile_ok": True, "correctness_ok": True,
            "candidate_ms_median": 8.0, "speedup": 2.3, "samples": 30, "diagnostics": {},
        },
        promote_callback=lambda cid, sp: promotions.append((cid, sp)),
        observer=NullObserver(),
        round_index=1,
        analyst_runner=_canned_analyst,
        optimizer_runner=_canned_optimizer,
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
    assert entry["speedup"] == 2.3


def test_round_runner_writes_history_steps(workspace: Path, contract: LoRAContract):
    layout = RunLayout(workspace_root=workspace, run_id="r")
    layout.mkdir()
    tools = _full_tool_bag(layout, workspace)
    runner = RoundRunner(
        layout=layout, contract=contract,
        backend=_NullBackend(), agent_cfg=AgentConfig(max_iterations=2),
        executor=None, tools=tools,
        evaluator=lambda **kw: {"candidate_id": "candidate_111", "compile_ok": True,
                                  "correctness_ok": True, "candidate_ms_median": 9.0,
                                  "speedup": 1.7, "samples": 30, "diagnostics": {}},
        promote_callback=lambda cid, sp: None,
        observer=NullObserver(),
        round_index=2,
        analyst_runner=_canned_analyst,
        optimizer_runner=_canned_optimizer,
    )
    runner.run_one_round(remaining_budget_s=120.0)
    bb = load_blackboard(layout)
    history = bb["history"]
    steps = [entry["step"] for entry in history]
    assert steps == ["ANALYZE", "OPTIMIZE", "EVALUATE"]
    assert all(entry["round_index"] == 2 for entry in history)


# ---------------------------------------------------------------------------
# PipelineOrchestrator — full-run smoke
# ---------------------------------------------------------------------------


def test_orchestrator_runs_full_pipeline_with_injected_mocks(
    workspace: Path, contract: LoRAContract
):
    output_path = workspace / "optimized_lora.cu"

    # Inject a baseline_runner that doesn't hit the lora_resources stub.
    def fake_baseline_runner(*, layout, executor, contract, spec):
        return {"ms_median_overall": 100.0, "per_d": {}, "spec": spec.to_dict()}

    # Initial candidate runner: returns a payload with candidate_id=... and pretends
    # write_candidate already produced a file on disk so promote can copy it.
    cand_id_initial = "candidate_000"

    def fake_initial(*, backend, registry, layout, contract, agent_cfg, observer):
        # Pretend write_candidate dropped a file already.
        target = layout.candidate_file(cand_id_initial, "candidate.cu")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("// initial candidate\n", encoding="utf-8")
        return {
            "status": "success",
            "stage": Stage.INITIAL_CANDIDATE.value,
            "candidate_id": cand_id_initial,
            "hypothesis": "naive baseline kernel",
            "experiment_type": "baseline",
        }

    def fake_evaluator(*, layout, executor, candidate_id, baseline_ms_median):
        return {
            "candidate_id": candidate_id,
            "compile_ok": True, "correctness_ok": True,
            "candidate_ms_median": 50.0, "speedup": 2.0,
            "samples": 30, "diagnostics": {},
        }

    def fake_hardware(*, backend, registry, layout, contract, agent_cfg, observer):
        return {
            "status": "success", "stage": Stage.HARDWARE_PROFILE.value,
            "metrics": {"sm": 30, "dram_bw_gbps": 384.0},
        }

    def fake_summary(*, backend, registry, layout, contract, agent_cfg, observer):
        return {
            "status": "success", "stage": Stage.FINALIZE.value,
            "metrics": {"final_speedup": 2.0},
            "next_recommendation": None,
        }

    def fake_analyst(*, backend, registry, layout, contract, agent_cfg, observer):
        return {"status": "success", "stage": Stage.TUNING_LOOP.value,
                "metrics": {"summary": "DRAM-bound on low-rank correction"}}

    def fake_optimizer(*, backend, registry, layout, contract, agent_cfg, observer):
        cid = "candidate_001"
        target = layout.candidate_file(cid, "candidate.cu")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("// fused kernel\n", encoding="utf-8")
        return {
            "status": "success", "stage": Stage.TUNING_LOOP.value,
            "candidate_id": cid, "hypothesis": "fuse W*X with low-rank correction",
            "experiment_type": "fused-correction",
        }

    # Use a tiny budget so the tuning loop runs once and FINALIZE follows.
    layout = RunLayout(workspace_root=workspace, run_id="run_test")
    layout.mkdir()
    orch = PipelineOrchestrator(
        spec={"operator": "lora_matmul"},
        time_budget_s=1.0,                       # below MIN_TUNING_SLICE_S → finalize after init candidate
        workspace_root=workspace,
        output_path=output_path,
        backend=_NullBackend(), agent_cfg=AgentConfig(max_iterations=2),
        executor=None,
        contract=contract,
        run_id="run_test",
        verbose=False,
        tools=_full_tool_bag(layout, workspace),
        evaluator=fake_evaluator,
        baseline_runner=fake_baseline_runner,
        hardware_profiler=fake_hardware,
        initial_candidate_runner=fake_initial,
        analyst_runner=fake_analyst,
        optimizer_runner=fake_optimizer,
        summary_runner=fake_summary,
    )
    summary = orch.run()

    # state.json is on disk and reflects the completed stages
    state_text = orch.layout.state_path.read_text(encoding="utf-8")
    state_blob = json.loads(state_text)
    assert state_blob["best_candidate_id"] == cand_id_initial
    assert state_blob["best_speedup"] == 2.0
    assert Stage.HARDWARE_PROFILE.value in state_blob["completed_stages"]
    assert Stage.BENCHMARK_BASELINE.value in state_blob["completed_stages"]
    assert Stage.INITIAL_CANDIDATE.value in state_blob["completed_stages"]
    assert Stage.FINALIZE.value in state_blob["completed_stages"]

    # ./optimized_lora.cu mirrors best/best.cu after promotion.
    assert output_path.is_file()
    assert output_path.read_text(encoding="utf-8") == "// initial candidate\n"

    # Final summary surfaced through orchestrator.run() return value
    assert summary["best_candidate_id"] == cand_id_initial
    assert summary["best_speedup"] == 2.0
    assert summary["final"]["status"] == "success"

    # Final report markdown + json on disk
    assert orch.layout.final_report_path.is_file()
    assert orch.layout.summary_path.is_file()


def test_orchestrator_resume_picks_existing_state(workspace: Path, contract: LoRAContract):
    output_path = workspace / "optimized_lora.cu"
    # First run: only complete hardware + baseline before "stopping".
    def hw_runner(**kw):
        return {"status": "success", "stage": Stage.HARDWARE_PROFILE.value, "metrics": {}}

    def baseline_runner(*, layout, executor, contract, spec):
        return {"ms_median_overall": 100.0}

    # Initial candidate fails so the run finalizes without a best.
    def initial_failing(**kw):
        return {"status": "failed", "stage": Stage.INITIAL_CANDIDATE.value,
                "caveats": ["test forces failure"]}

    def evaluator_fail(**kw):
        raise AssertionError("must not be called for failing initial candidate")

    def summary_runner(**kw):
        return {"status": "success", "stage": Stage.FINALIZE.value, "metrics": {}}

    layout1 = RunLayout(workspace_root=workspace, run_id="run_resume")
    layout1.mkdir()
    orch1 = PipelineOrchestrator(
        spec={"operator": "lora_matmul"}, time_budget_s=1.0,
        workspace_root=workspace, output_path=output_path,
        backend=_NullBackend(), agent_cfg=AgentConfig(max_iterations=2),
        executor=None, contract=contract, run_id="run_resume", verbose=False,
        tools=_full_tool_bag(layout1, workspace),
        evaluator=evaluator_fail,
        baseline_runner=baseline_runner,
        hardware_profiler=hw_runner,
        initial_candidate_runner=initial_failing,
        analyst_runner=lambda **kw: {"status": "success", "stage": Stage.TUNING_LOOP.value},
        optimizer_runner=lambda **kw: {"status": "failed", "stage": Stage.TUNING_LOOP.value},
        summary_runner=summary_runner,
    )
    orch1.run()
    # Hardware + baseline artifacts on disk
    assert orch1.layout.has_hardware_profile()
    assert orch1.layout.has_baseline()

    # Second orchestrator with same run_id should pick up state.json
    orch2 = PipelineOrchestrator(
        spec={"operator": "lora_matmul"}, time_budget_s=1.0,
        workspace_root=workspace, output_path=output_path,
        backend=_NullBackend(), agent_cfg=AgentConfig(max_iterations=2),
        executor=None, contract=contract, run_id="run_resume", verbose=False,
        tools=_full_tool_bag(layout1, workspace),
        evaluator=evaluator_fail,
        baseline_runner=baseline_runner,
        hardware_profiler=hw_runner,
        initial_candidate_runner=initial_failing,
        analyst_runner=lambda **kw: {"status": "success", "stage": Stage.TUNING_LOOP.value},
        optimizer_runner=lambda **kw: {"status": "failed", "stage": Stage.TUNING_LOOP.value},
        summary_runner=summary_runner,
    )
    # The completed_stages from the first run must be preserved
    assert Stage.HARDWARE_PROFILE.value in orch2.run_state.completed_stages
    assert Stage.BENCHMARK_BASELINE.value in orch2.run_state.completed_stages


def test_orchestrator_benchmark_baseline_skips_llm(workspace: Path, contract: LoRAContract):
    """BENCHMARK_BASELINE must call the deterministic runner, not any agent."""
    output_path = workspace / "optimized_lora.cu"

    def hw_runner(**kw):
        return {"status": "success", "stage": Stage.HARDWARE_PROFILE.value, "metrics": {}}

    baseline_calls: list[Any] = []

    def baseline_runner(*, layout, executor, contract, spec):
        baseline_calls.append(spec)
        return {"ms_median_overall": 42.0}

    # Force the run to terminate after BENCHMARK_BASELINE by making initial candidate fail.
    def init_fail(**kw):
        return {"status": "failed", "stage": Stage.INITIAL_CANDIDATE.value,
                "caveats": ["test"]}

    layout = RunLayout(workspace_root=workspace, run_id="run_bb")
    layout.mkdir()
    orch = PipelineOrchestrator(
        spec={"operator": "lora_matmul"}, time_budget_s=1.0,
        workspace_root=workspace, output_path=output_path,
        backend=_NullBackend(), agent_cfg=AgentConfig(max_iterations=2),
        executor=None, contract=contract, run_id="run_bb", verbose=False,
        tools=_full_tool_bag(layout, workspace),
        evaluator=lambda **kw: pytest.fail("evaluator should not be called"),
        baseline_runner=baseline_runner,
        hardware_profiler=hw_runner,
        initial_candidate_runner=init_fail,
        summary_runner=lambda **kw: {"status": "success", "stage": Stage.FINALIZE.value},
    )
    orch.run()
    assert len(baseline_calls) == 1
    assert orch.layout.baseline_path.is_file()
    bb = load_blackboard(orch.layout)
    assert "benchmark" in bb and "baseline" in bb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _NullBackend:
    """Minimal LLMBackend stub — tests use injected runners instead."""
    def chat(self, messages, tools):
        raise AssertionError("LLM backend must not be invoked when runners are injected")


def _canned_analyst(*, backend, registry, layout, contract, agent_cfg, observer):
    return {"status": "success", "stage": Stage.TUNING_LOOP.value,
            "metrics": {"summary": "test"}}


def _canned_optimizer(*, backend, registry, layout, contract, agent_cfg, observer):
    cid = "candidate_111"
    target = layout.candidate_file(cid, "candidate.cu")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("// canned\n", encoding="utf-8")
    return {"status": "success", "stage": Stage.TUNING_LOOP.value,
            "candidate_id": cid, "hypothesis": "test",
            "experiment_type": "canned"}


