"""tests/pipeline/test_orchestrator.py — PipelineOrchestrator state machine + persistence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.orchestrator import PipelineOrchestrator
from pipeline.stage_agent import StageAgent
from pipeline.stage_runner import StageContext
from pipeline.state import (
    BENCHMARK_SPEC_SLOTS,
    Stage,
    StageResult,
)


# ---------------------------------------------------------------------------
# Fake StageAgents — minimal behaviour just enough to drive the state machine
# ---------------------------------------------------------------------------

class _BenchmarkSpecFake(StageAgent):
    stage = Stage.BENCHMARK_SPEC
    allowed_tools = ()

    def run(self, context: StageContext) -> StageResult:
        # Write 5 spec files + report versions in metrics.
        versions: dict[str, str] = {}
        for slot in BENCHMARK_SPEC_SLOTS:
            p = context.layout.benchmark_spec_path(slot)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"slot": slot}))
            versions[slot] = "v1"
        return StageResult(
            stage=self.stage.value,
            status="success",
            artifacts={f"spec_{s}": context.layout.relpath(context.layout.benchmark_spec_path(s)) for s in BENCHMARK_SPEC_SLOTS},
            metrics={"spec_versions": versions},
            confidence=1.0,
        )


class _HardwareFake(StageAgent):
    stage = Stage.HARDWARE_PROFILE
    allowed_tools = ()

    def run(self, context: StageContext) -> StageResult:
        p = context.layout.hardware_profile_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"dram_bandwidth_gbps": 280.0}))
        return StageResult(
            stage=self.stage.value,
            status="success",
            artifacts={"hardware_profile": context.layout.relpath(p)},
            metrics={"dram_bandwidth_gbps": 280.0},
            confidence=0.85,
        )


class _BaselineFake(StageAgent):
    stage = Stage.BASELINE_PROFILE
    allowed_tools = ()

    def run(self, context: StageContext) -> StageResult:
        p = context.layout.baseline_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"d3584": {"torch_ms": 12.0}}))
        return StageResult(
            stage=self.stage.value,
            status="success",
            artifacts={"baseline": context.layout.relpath(p)},
            confidence=1.0,
        )


def _emit_candidate(context: StageContext, *, idx: int, accepted_for: str | None = "best_update", speedup: float = 1.2):
    """Helper: write a candidate.cu to candidate_NNN and return a candidate dict."""
    cid = f"candidate_{idx:03d}"
    cdir = context.layout.candidate_dir(cid)
    cdir.mkdir(parents=True, exist_ok=True)
    cu_path = context.layout.candidate_file(cid, "candidate.cu")
    cu_path.write_text(f"// candidate {cid} placeholder\n")
    return cid, {
        "candidate_id": cid,
        "compile_ok": True,
        "correctness_ok": True,
        "quick_speedup_median": speedup,
        "quick_samples": 5,
        "quick_variance_pct": 4.0,
        "confirm_speedup_median": speedup if accepted_for == "best_update" else None,
        "confirm_samples": 30 if accepted_for == "best_update" else None,
        "confirm_variance_pct": 3.0 if accepted_for == "best_update" else None,
        "accepted_for": accepted_for,
        "timestamp": "2026-05-02T00:00:00+00:00",
    }


class _InitialCandidateFake(StageAgent):
    stage = Stage.INITIAL_CANDIDATE
    allowed_tools = ()

    def run(self, context: StageContext) -> StageResult:
        cid, cand = _emit_candidate(context, idx=0, accepted_for="best_update", speedup=1.10)
        return StageResult(
            stage=self.stage.value,
            status="success",
            artifacts={"candidate": context.layout.relpath(context.layout.candidate_dir(cid))},
            metrics={"candidate": cand},
            confidence=0.9,
        )


class _TuningCounterFake(StageAgent):
    """Each invocation produces a new candidate, alternating best_update vs strategy_guidance."""
    stage = Stage.TUNING_LOOP
    allowed_tools = ()

    def __init__(self):
        self._n = 0

    def run(self, context: StageContext) -> StageResult:
        self._n += 1
        idx = context.run_state.current_iteration  # use orchestrator-tracked iter as id
        accept = "best_update" if self._n % 2 == 1 else "strategy_guidance"
        speedup = 1.20 + 0.05 * self._n
        cid, cand = _emit_candidate(context, idx=idx, accepted_for=accept, speedup=speedup)
        return StageResult(
            stage=self.stage.value,
            status="success",
            artifacts={"candidate": context.layout.relpath(context.layout.candidate_dir(cid))},
            metrics={"candidate": cand},
            confidence=0.85,
        )


class _ProfileFake(StageAgent):
    stage = Stage.OPTIONAL_PROFILE
    allowed_tools = ()

    def run(self, context: StageContext) -> StageResult:
        cid = context.run_state.best_candidate_id
        assert cid, "OPTIONAL_PROFILE shouldn't run without a best_candidate_id"
        p = context.layout.candidate_file(cid, "profile.json")
        p.write_text(json.dumps({"compute_throughput_pct": 78.0}))
        return StageResult(
            stage=self.stage.value,
            status="success",
            artifacts={"profile": context.layout.relpath(p)},
        )


def _make_agents(*, with_tuning: bool = True, with_profile: bool = True) -> dict[Stage, StageAgent]:
    agents: dict[Stage, StageAgent] = {
        Stage.BENCHMARK_SPEC: _BenchmarkSpecFake(),
        Stage.HARDWARE_PROFILE: _HardwareFake(),
        Stage.BASELINE_PROFILE: _BaselineFake(),
        Stage.INITIAL_CANDIDATE: _InitialCandidateFake(),
    }
    if with_tuning:
        agents[Stage.TUNING_LOOP] = _TuningCounterFake()
    if with_profile:
        agents[Stage.OPTIONAL_PROFILE] = _ProfileFake()
    return agents


def _no_tools(_names, _stage):
    """build_tools stub — none of the fake agents call tools."""
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFreshRunHappyPath:
    def test_full_run_sets_best_and_syncs_output(self, tmp_path):
        out = tmp_path / "optimized_lora.cu"
        orch = PipelineOrchestrator(
            spec={"operator": "lora_matmul"},
            time_budget_s=600.0,
            workspace_root=tmp_path / "workspace",
            output_path=out,
            stage_agents=_make_agents(),
            build_tools=_no_tools,
            stage_budgets_s={s: 30.0 for s in Stage},
        )
        summary = orch.run()

        # Output file exists and is a copy of best/best.cu
        assert out.is_file()
        best_cu = orch.layout.best_cu_path
        assert best_cu.is_file()
        assert out.read_text() == best_cu.read_text()

        # State machine reached FINALIZE
        assert summary["best_candidate_id"] is not None
        assert summary["best_speedup"] is not None
        assert Stage.FINALIZE.value in orch.run_state.completed_stages

        # All setup stages were marked complete
        for s in (Stage.BENCHMARK_SPEC, Stage.HARDWARE_PROFILE, Stage.BASELINE_PROFILE, Stage.INITIAL_CANDIDATE):
            assert s.value in orch.run_state.completed_stages

    def test_state_json_persisted_at_root(self, tmp_path):
        orch = PipelineOrchestrator(
            spec={"operator": "lora_matmul"},
            time_budget_s=600.0,
            workspace_root=tmp_path / "ws",
            output_path=tmp_path / "optimized_lora.cu",
            stage_agents=_make_agents(),
            build_tools=_no_tools,
            stage_budgets_s={s: 30.0 for s in Stage},
        )
        orch.run()

        state_path = orch.layout.state_path
        assert state_path.is_file()
        loaded = json.loads(state_path.read_text())
        assert loaded["operator"] == "lora_matmul"
        assert loaded["best_candidate_id"] is not None

    def test_leaderboard_append_only(self, tmp_path):
        orch = PipelineOrchestrator(
            spec={"operator": "lora_matmul"},
            time_budget_s=600.0,
            workspace_root=tmp_path / "ws",
            output_path=tmp_path / "optimized_lora.cu",
            stage_agents=_make_agents(),
            build_tools=_no_tools,
            stage_budgets_s={s: 30.0 for s in Stage},
        )
        orch.run()

        lines = [
            json.loads(line)
            for line in orch.layout.leaderboard_path.read_text().splitlines()
            if line.strip()
        ]
        # >= 1 candidate from INITIAL_CANDIDATE; tuning loop adds more depending on budget
        assert len(lines) >= 1
        assert all("candidate_id" in r for r in lines)

    def test_events_jsonl_records_each_stage(self, tmp_path):
        orch = PipelineOrchestrator(
            spec={"operator": "lora_matmul"},
            time_budget_s=600.0,
            workspace_root=tmp_path / "ws",
            output_path=tmp_path / "optimized_lora.cu",
            stage_agents=_make_agents(),
            build_tools=_no_tools,
            stage_budgets_s={s: 30.0 for s in Stage},
        )
        orch.run()

        events = [json.loads(l) for l in orch.layout.events_path.read_text().splitlines() if l.strip()]
        stage_names = {e["stage"] for e in events}
        # Each setup stage produces at least one event
        for s in (Stage.BENCHMARK_SPEC, Stage.HARDWARE_PROFILE, Stage.BASELINE_PROFILE, Stage.INITIAL_CANDIDATE):
            assert s.value in stage_names


class TestResume:
    def test_resume_skips_setup_when_artifacts_present(self, tmp_path):
        ws = tmp_path / "ws"
        out = tmp_path / "optimized_lora.cu"

        # First run completes everything.
        orch1 = PipelineOrchestrator(
            spec={"operator": "lora_matmul"},
            time_budget_s=600.0,
            workspace_root=ws,
            output_path=out,
            stage_agents=_make_agents(),
            build_tools=_no_tools,
            stage_budgets_s={s: 30.0 for s in Stage},
        )
        orch1.run()
        run_id = orch1.run_state.run_id

        # Now create a tracking agent set that *errors* on setup stages so we
        # can prove resume skipped them.
        class _Boom(StageAgent):
            allowed_tools = ()
            def run(self, ctx):
                raise AssertionError(f"resume should not re-enter {self.stage.value}")

        class _BoomBenchmark(_Boom):  stage = Stage.BENCHMARK_SPEC
        class _BoomHardware(_Boom):   stage = Stage.HARDWARE_PROFILE
        class _BoomBaseline(_Boom):   stage = Stage.BASELINE_PROFILE

        orch2 = PipelineOrchestrator(
            spec={"operator": "lora_matmul"},
            time_budget_s=600.0,
            workspace_root=ws,
            output_path=out,
            stage_agents={
                Stage.BENCHMARK_SPEC: _BoomBenchmark(),
                Stage.HARDWARE_PROFILE: _BoomHardware(),
                Stage.BASELINE_PROFILE: _BoomBaseline(),
                Stage.INITIAL_CANDIDATE: _InitialCandidateFake(),
                Stage.TUNING_LOOP: _TuningCounterFake(),
                Stage.OPTIONAL_PROFILE: _ProfileFake(),
            },
            build_tools=_no_tools,
            run_id=run_id,
            stage_budgets_s={s: 30.0 for s in Stage},
        )
        # Should run to FINALIZE without ever re-entering setup stages.
        # (orchestrator never re-runs INITIAL_CANDIDATE either since
        # best_candidate_id is set in resumed state.)
        orch2.run()


class TestTimeoutFinalize:
    def test_timeout_at_construction_goes_straight_to_finalize(self, tmp_path):
        # time_budget_s=0 means we're already over budget on the first tick.
        out = tmp_path / "optimized_lora.cu"
        orch = PipelineOrchestrator(
            spec={"operator": "lora_matmul"},
            time_budget_s=0.0,
            workspace_root=tmp_path / "ws",
            output_path=out,
            stage_agents=_make_agents(),
            build_tools=_no_tools,
        )
        summary = orch.run()
        # No best produced because we never ran a candidate stage.
        assert summary["best_candidate_id"] is None
        # State machine still reached FINALIZE deterministically.
        assert Stage.FINALIZE.value in orch.run_state.completed_stages
        # No optimized_lora.cu (no best to copy) — but also no crash.
        assert not out.is_file()


class TestOutputSync:
    def test_output_synced_after_initial_candidate(self, tmp_path):
        # Build a minimal pipeline: only setup + initial candidate.
        out = tmp_path / "optimized_lora.cu"
        orch = PipelineOrchestrator(
            spec={"operator": "lora_matmul"},
            time_budget_s=600.0,
            workspace_root=tmp_path / "ws",
            output_path=out,
            stage_agents=_make_agents(with_tuning=False, with_profile=False),
            build_tools=_no_tools,
            stage_budgets_s={s: 30.0 for s in Stage},
        )
        orch.run()
        assert out.is_file()
        # Content matches whatever InitialCandidateFake wrote.
        assert "candidate_000" in out.read_text()

    def test_best_update_overwrites_output(self, tmp_path):
        """Tuning loop produces multiple candidates; the latest best_update wins."""
        out = tmp_path / "optimized_lora.cu"

        class _AlwaysBest(StageAgent):
            stage = Stage.TUNING_LOOP
            allowed_tools = ()
            def __init__(self):
                self._n = 0
            def run(self, ctx):
                self._n += 1
                idx = ctx.run_state.current_iteration
                cid, cand = _emit_candidate(ctx, idx=idx, accepted_for="best_update", speedup=1.0 + 0.1 * self._n)
                return StageResult(stage=self.stage.value, status="success", metrics={"candidate": cand})

        agents = _make_agents(with_profile=False)
        agents[Stage.TUNING_LOOP] = _AlwaysBest()

        # Tight time budget but generous enough to do a couple of tuning iterations.
        orch = PipelineOrchestrator(
            spec={"operator": "lora_matmul"},
            time_budget_s=600.0,
            workspace_root=tmp_path / "ws",
            output_path=out,
            stage_agents=agents,
            build_tools=_no_tools,
            stage_budgets_s={s: 0.001 for s in Stage},  # makes elapsed grow fast via overrun
        )
        orch.run()

        # Final output reflects the best candidate
        best_id = orch.run_state.best_candidate_id
        assert best_id is not None
        assert out.read_text() == orch.layout.best_cu_path.read_text()
        # Best id should match a candidate file we actually wrote
        assert orch.layout.candidate_file(best_id, "candidate.cu").is_file()


class TestMissingAgent:
    def test_missing_setup_agent_finalizes_without_crash(self, tmp_path):
        # Drop BENCHMARK_SPEC entirely.
        agents = _make_agents()
        del agents[Stage.BENCHMARK_SPEC]

        orch = PipelineOrchestrator(
            spec={"operator": "lora_matmul"},
            time_budget_s=600.0,
            workspace_root=tmp_path / "ws",
            output_path=tmp_path / "optimized_lora.cu",
            stage_agents=agents,
            build_tools=_no_tools,
            stage_budgets_s={s: 30.0 for s in Stage},
        )
        summary = orch.run()
        # Must terminate even though we couldn't make progress.
        assert Stage.FINALIZE.value in orch.run_state.completed_stages
        # No best because we never reached candidate stage.
        assert summary["best_candidate_id"] is None
