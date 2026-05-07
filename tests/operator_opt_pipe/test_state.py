"""Unit tests for operator_opt_pipe.state."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from operator_opt_pipe.state import (
    ROUND_STEP_ANALYZE,
    ROUND_STEP_EVALUATE,
    ROUND_STEP_OPTIMIZE,
    RunLayout,
    RunState,
    SCHEMA_VERSION,
    Stage,
    append_history,
    load_blackboard,
    make_run_id,
    save_blackboard,
)


# ---------------------------------------------------------------------------
# Stage enum + round-step markers
# ---------------------------------------------------------------------------


def test_stage_values_match_string_form():
    assert Stage.INIT.value == "INIT"
    assert Stage.HARDWARE_PROFILE.value == "HARDWARE_PROFILE"
    assert Stage.BENCHMARK_BASELINE.value == "BENCHMARK_BASELINE"
    assert Stage.INITIAL_CANDIDATE.value == "INITIAL_CANDIDATE"
    assert Stage.TUNING_LOOP.value == "TUNING_LOOP"
    assert Stage.FINALIZE.value == "FINALIZE"


def test_round_step_markers_are_strings():
    assert ROUND_STEP_ANALYZE == "ANALYZE"
    assert ROUND_STEP_OPTIMIZE == "OPTIMIZE"
    assert ROUND_STEP_EVALUATE == "EVALUATE"


def test_make_run_id_is_unique_and_sortable():
    a = make_run_id()
    b = make_run_id()
    assert a != b
    assert a <= b or a >= b


# ---------------------------------------------------------------------------
# RunState
# ---------------------------------------------------------------------------


def test_run_state_round_trip():
    s = RunState(run_id="r1", operator="lora_matmul", time_budget_s=600.0)
    j = s.to_json()
    s2 = RunState.from_json(j)
    assert s2.run_id == "r1"
    assert s2.operator == "lora_matmul"
    assert s2.time_budget_s == 600.0
    assert s2.completed_stages == []


def test_remaining_budget_clamped_to_zero():
    s = RunState(run_id="r", operator="op", time_budget_s=10.0, elapsed_s=20.0)
    assert s.remaining_budget_s() == 0.0


def test_mark_stage_complete_dedupes():
    s = RunState(run_id="r", operator="op")
    s.mark_stage_complete(Stage.HARDWARE_PROFILE)
    s.mark_stage_complete(Stage.HARDWARE_PROFILE)
    assert s.completed_stages == [Stage.HARDWARE_PROFILE.value]


# ---------------------------------------------------------------------------
# RunLayout
# ---------------------------------------------------------------------------


def test_runlayout_paths_under_run_dir(tmp_path: Path):
    layout = RunLayout(workspace_root=tmp_path, run_id="run_abc")
    assert layout.run_dir == tmp_path / "runs" / "run_abc"
    assert layout.state_path == layout.run_dir / "state.json"
    assert layout.blackboard_path == layout.run_dir / "blackboard.json"
    assert layout.best_cu_path == layout.run_dir / "best" / "best.cu"
    assert layout.candidate_dir("candidate_001") == layout.run_dir / "candidates" / "candidate_001"
    # Single-file artifacts live directly under run_dir (not in their own subdir).
    assert layout.hardware_path == layout.run_dir / "hardware_profile.json"
    assert layout.baseline_path == layout.run_dir / "baseline.json"
    assert layout.final_report_path == layout.run_dir / "final_report.json"
    assert layout.summary_path == layout.run_dir / "summary.md"
    # Multi-file dirs use semantic names.
    assert layout.inputs_dir == layout.run_dir / "inputs"
    assert layout.oracle_dir == layout.run_dir / "oracle"
    # Per-run build / exec roots so artifacts don't escape the run.
    assert layout.build_dir == layout.run_dir / "build"
    assert layout.exec_dir == layout.run_dir / "exec"


def test_runlayout_input_and_oracle_paths(tmp_path: Path):
    layout = RunLayout(workspace_root=tmp_path, run_id="r")
    assert layout.input_path("W", "d3584") == layout.inputs_dir / "W_d3584.pt"
    assert layout.oracle_path("Y", "d3584") == layout.oracle_dir / "Y_d3584.pt"


def test_runlayout_predicates_react_to_filesystem(tmp_path: Path):
    layout = RunLayout(workspace_root=tmp_path, run_id="run_abc")
    layout.mkdir()
    assert not layout.has_hardware_profile()
    assert not layout.has_baseline()

    layout.hardware_path.write_text("{}", encoding="utf-8")
    layout.baseline_path.write_text("{}", encoding="utf-8")
    assert layout.has_hardware_profile()
    assert layout.has_baseline()


def test_runlayout_mkdir_is_idempotent(tmp_path: Path):
    layout = RunLayout(workspace_root=tmp_path, run_id="run_abc")
    layout.mkdir()
    layout.mkdir()
    assert layout.run_dir.is_dir()
    assert layout.build_dir.is_dir()
    assert layout.exec_dir.is_dir()


def test_candidate_id_formatting():
    assert RunLayout.candidate_id(0) == "candidate_000"
    assert RunLayout.candidate_id(42) == "candidate_042"
    assert RunLayout.candidate_id(999) == "candidate_999"


# ---------------------------------------------------------------------------
# Blackboard
# ---------------------------------------------------------------------------


def test_blackboard_missing_file_returns_empty(tmp_path: Path):
    layout = RunLayout(workspace_root=tmp_path, run_id="run_abc")
    layout.mkdir()
    bb = load_blackboard(layout)
    assert bb["schema_version"] == SCHEMA_VERSION
    assert bb["history"] == []


def test_blackboard_round_trip(tmp_path: Path):
    layout = RunLayout(workspace_root=tmp_path, run_id="run_abc")
    layout.mkdir()
    save_blackboard(layout, {"schema_version": SCHEMA_VERSION, "history": [], "hardware": {"sm": 30}})
    bb = load_blackboard(layout)
    assert bb["hardware"] == {"sm": 30}


def test_blackboard_fills_defaults_when_loading(tmp_path: Path):
    layout = RunLayout(workspace_root=tmp_path, run_id="run_abc")
    layout.mkdir()
    layout.blackboard_path.write_text(json.dumps({"hardware": {}}), encoding="utf-8")
    bb = load_blackboard(layout)
    assert bb["schema_version"] == SCHEMA_VERSION
    assert bb["history"] == []


def test_blackboard_rejects_non_object(tmp_path: Path):
    layout = RunLayout(workspace_root=tmp_path, run_id="run_abc")
    layout.mkdir()
    layout.blackboard_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_blackboard(layout)


def test_append_history_preserves_other_keys(tmp_path: Path):
    layout = RunLayout(workspace_root=tmp_path, run_id="run_abc")
    layout.mkdir()
    save_blackboard(layout, {"schema_version": 1, "history": [], "best": {"speedup": 1.5}})
    append_history(layout, {"step": "ANALYZE", "ts": "now"})
    bb = load_blackboard(layout)
    assert bb["best"] == {"speedup": 1.5}
    assert bb["history"] == [{"step": "ANALYZE", "ts": "now"}]
