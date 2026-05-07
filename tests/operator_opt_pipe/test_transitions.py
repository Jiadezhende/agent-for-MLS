"""Unit tests for operator_opt_pipe.transitions."""
from __future__ import annotations

from pathlib import Path

import pytest

from operator_opt_pipe.state import RunLayout, RunState, Stage
from operator_opt_pipe.transitions import MIN_TUNING_SLICE_S, next_stage


@pytest.fixture
def layout(tmp_path: Path) -> RunLayout:
    layout = RunLayout(workspace_root=tmp_path, run_id="r")
    layout.mkdir()
    return layout


def _state(**kw) -> RunState:
    base = dict(run_id="r", operator="lora_matmul", time_budget_s=600.0, elapsed_s=0.0)
    base.update(kw)
    return RunState(**base)


def test_budget_exceeded_goes_to_finalize(layout: RunLayout):
    layout.hardware_path.write_text("{}", encoding="utf-8")
    layout.baseline_path.write_text("{}", encoding="utf-8")
    state = _state(elapsed_s=601.0, best_candidate_id="candidate_000")
    assert next_stage(state, layout) is Stage.FINALIZE


def test_no_hardware_profile(layout: RunLayout):
    state = _state()
    assert next_stage(state, layout) is Stage.HARDWARE_PROFILE


def test_no_baseline(layout: RunLayout):
    layout.hardware_path.write_text("{}", encoding="utf-8")
    state = _state()
    assert next_stage(state, layout) is Stage.BENCHMARK_BASELINE


def test_no_best_candidate(layout: RunLayout):
    layout.hardware_path.write_text("{}", encoding="utf-8")
    layout.baseline_path.write_text("{}", encoding="utf-8")
    state = _state()
    assert next_stage(state, layout) is Stage.INITIAL_CANDIDATE


def test_has_budget_for_tuning(layout: RunLayout):
    layout.hardware_path.write_text("{}", encoding="utf-8")
    layout.baseline_path.write_text("{}", encoding="utf-8")
    state = _state(best_candidate_id="candidate_000", elapsed_s=10.0)
    # remaining_budget_s = 590s >> 60s threshold
    assert next_stage(state, layout) is Stage.TUNING_LOOP


def test_no_budget_left_after_best_goes_to_finalize(layout: RunLayout):
    layout.hardware_path.write_text("{}", encoding="utf-8")
    layout.baseline_path.write_text("{}", encoding="utf-8")
    # remaining_budget_s == 30s < MIN_TUNING_SLICE_S (60s), but still under
    # time_budget_s overall — orchestrator should finalize cleanly rather
    # than start another round.
    elapsed = 600.0 - (MIN_TUNING_SLICE_S - 30.0)
    state = _state(best_candidate_id="candidate_000", elapsed_s=elapsed)
    assert next_stage(state, layout) is Stage.FINALIZE
