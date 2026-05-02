"""tests/pipeline/test_state_machine.py — RunState + next_stage rules (no GPU)."""
from __future__ import annotations

import pytest

from pipeline.state import (
    BENCHMARK_SPEC_SLOTS,
    RunState,
    Stage,
    StageResult,
    validate_stage_result,
)
from pipeline.transitions import MIN_TUNING_SLICE_S, next_stage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**overrides) -> RunState:
    base = RunState(run_id="test_run", operator="lora_matmul", time_budget_s=600.0, elapsed_s=0.0)
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _stub_predicates(*, hw=False, baseline=False, candidate_profile=False):
    """Return artifact-existence callables for next_stage()."""
    return dict(
        has_hardware_profile=lambda: hw,
        has_baseline=lambda: baseline,
        has_candidate_profile=lambda _cid: candidate_profile,
    )


# ---------------------------------------------------------------------------
# next_stage — happy path advancement
# ---------------------------------------------------------------------------

class TestNextStageAdvancement:
    def test_starts_with_benchmark_spec(self):
        st = _state()
        assert next_stage(st, **_stub_predicates()) == Stage.BENCHMARK_SPEC

    def test_advances_to_hardware_profile_after_specs(self):
        st = _state(benchmark_spec_versions={s: "v1" for s in BENCHMARK_SPEC_SLOTS})
        assert next_stage(st, **_stub_predicates()) == Stage.HARDWARE_PROFILE

    def test_advances_to_baseline_after_hardware(self):
        st = _state(benchmark_spec_versions={s: "v1" for s in BENCHMARK_SPEC_SLOTS})
        assert next_stage(st, **_stub_predicates(hw=True)) == Stage.BASELINE_PROFILE

    def test_advances_to_initial_candidate_after_baseline(self):
        st = _state(benchmark_spec_versions={s: "v1" for s in BENCHMARK_SPEC_SLOTS})
        assert next_stage(st, **_stub_predicates(hw=True, baseline=True)) == Stage.INITIAL_CANDIDATE

    def test_enters_tuning_loop_when_best_exists_and_budget_remains(self):
        st = _state(
            benchmark_spec_versions={s: "v1" for s in BENCHMARK_SPEC_SLOTS},
            best_candidate_id="candidate_000",
            elapsed_s=100.0,  # plenty left of 600
        )
        assert next_stage(st, **_stub_predicates(hw=True, baseline=True)) == Stage.TUNING_LOOP

    def test_optional_profile_when_budget_thin_and_no_profile_yet(self):
        st = _state(
            benchmark_spec_versions={s: "v1" for s in BENCHMARK_SPEC_SLOTS},
            best_candidate_id="candidate_000",
            elapsed_s=600.0 - MIN_TUNING_SLICE_S + 1,  # < MIN_TUNING_SLICE_S left
        )
        assert next_stage(st, **_stub_predicates(hw=True, baseline=True)) == Stage.OPTIONAL_PROFILE

    def test_finalize_when_budget_thin_and_profile_exists(self):
        st = _state(
            benchmark_spec_versions={s: "v1" for s in BENCHMARK_SPEC_SLOTS},
            best_candidate_id="candidate_000",
            elapsed_s=600.0 - MIN_TUNING_SLICE_S + 1,
        )
        assert next_stage(
            st, **_stub_predicates(hw=True, baseline=True, candidate_profile=True)
        ) == Stage.FINALIZE


# ---------------------------------------------------------------------------
# next_stage — timeout always wins
# ---------------------------------------------------------------------------

class TestTimeoutOverride:
    @pytest.mark.parametrize("predicates", [
        dict(),
        dict(hw=True),
        dict(hw=True, baseline=True),
        dict(hw=True, baseline=True, candidate_profile=True),
    ])
    def test_timeout_skips_to_finalize(self, predicates):
        st = _state(elapsed_s=600.0)  # >= time_budget_s
        assert next_stage(st, **_stub_predicates(**predicates)) == Stage.FINALIZE

    def test_overshooting_budget_still_finalize(self):
        st = _state(elapsed_s=999.0, best_candidate_id=None)
        # Even though we have nothing — timeout wins.
        assert next_stage(st, **_stub_predicates()) == Stage.FINALIZE


# ---------------------------------------------------------------------------
# Resume behaviour — setup stages should be skipped if artifacts exist.
# ---------------------------------------------------------------------------

class TestResumeBehaviour:
    def test_skip_hardware_profile_when_artifact_exists(self):
        # Specs done + hardware artifact present → must NOT re-enter HARDWARE_PROFILE.
        st = _state(benchmark_spec_versions={s: "v1" for s in BENCHMARK_SPEC_SLOTS})
        assert next_stage(st, **_stub_predicates(hw=True)) == Stage.BASELINE_PROFILE

    def test_skip_baseline_when_artifact_exists(self):
        st = _state(benchmark_spec_versions={s: "v1" for s in BENCHMARK_SPEC_SLOTS})
        assert next_stage(st, **_stub_predicates(hw=True, baseline=True)) == Stage.INITIAL_CANDIDATE

    def test_partial_specs_block_progression(self):
        # Only 3/5 slots filled — must redo BENCHMARK_SPEC (or, in practice, fill the rest).
        partial = {s: "v1" for s in BENCHMARK_SPEC_SLOTS[:3]}
        st = _state(benchmark_spec_versions=partial)
        assert next_stage(st, **_stub_predicates(hw=True, baseline=True)) == Stage.BENCHMARK_SPEC


# ---------------------------------------------------------------------------
# RunState (de)serialization roundtrip
# ---------------------------------------------------------------------------

class TestRunStateSerialization:
    def test_roundtrip_preserves_fields(self):
        st = _state(
            benchmark_spec_versions={s: "v1" for s in BENCHMARK_SPEC_SLOTS},
            best_candidate_id="candidate_002",
            best_speedup=1.34,
            current_iteration=5,
            failure_counts={"compile": 2, "correctness": 1, "benchmark_unstable": 0},
        )
        st.mark_completed(Stage.BENCHMARK_SPEC)
        st.set_current_stage(Stage.TUNING_LOOP)

        roundtripped = RunState.from_json(st.to_json())
        assert roundtripped.run_id == st.run_id
        assert roundtripped.operator == st.operator
        assert roundtripped.benchmark_spec_versions == st.benchmark_spec_versions
        assert roundtripped.best_candidate_id == "candidate_002"
        assert roundtripped.best_speedup == pytest.approx(1.34)
        assert roundtripped.current_iteration == 5
        assert roundtripped.failure_counts["compile"] == 2
        assert roundtripped.completed_stages == [Stage.BENCHMARK_SPEC.value]
        assert roundtripped.current_stage == Stage.TUNING_LOOP.value

    def test_from_dict_ignores_unknown_keys(self):
        # Forward compat: future versions might add keys we don't yet know about.
        d = RunState(run_id="x", operator="lora_matmul").to_dict()
        d["future_field"] = "ignored"
        assert RunState.from_dict(d).run_id == "x"

    def test_specs_complete_only_when_all_slots_populated(self):
        st = _state()
        assert not st.specs_complete()
        for slot in BENCHMARK_SPEC_SLOTS[:-1]:
            st.benchmark_spec_versions[slot] = "v1"
        assert not st.specs_complete()
        st.benchmark_spec_versions[BENCHMARK_SPEC_SLOTS[-1]] = "v1"
        assert st.specs_complete()

    def test_remaining_budget_clamped_at_zero(self):
        st = _state(elapsed_s=999.0)  # over budget
        assert st.remaining_budget_s() == 0.0

    def test_mark_completed_idempotent(self):
        st = _state()
        st.mark_completed(Stage.BENCHMARK_SPEC)
        st.mark_completed(Stage.BENCHMARK_SPEC)
        assert st.completed_stages == [Stage.BENCHMARK_SPEC.value]


# ---------------------------------------------------------------------------
# StageResult schema validation
# ---------------------------------------------------------------------------

class TestStageResultSchema:
    def _good(self, **overrides) -> dict:
        base = StageResult(
            stage=Stage.HARDWARE_PROFILE.value,
            status="success",
            artifacts={"hardware_profile": "runs/x/hardware/hardware_profile.json"},
            metrics={"dram_bandwidth_gbps": 280.0},
            confidence=0.85,
            caveats=[],
        ).to_dict()
        base.update(overrides)
        return base

    def test_valid_result_passes(self):
        ok, errs = validate_stage_result(self._good())
        assert ok and errs == []

    def test_unknown_stage_rejected(self):
        ok, errs = validate_stage_result(self._good(stage="NOT_A_REAL_STAGE"))
        assert not ok
        assert any("stage" in e for e in errs)

    def test_invalid_status_rejected(self):
        ok, errs = validate_stage_result(self._good(status="ok"))
        assert not ok
        assert any("status" in e for e in errs)

    @pytest.mark.parametrize("bad_conf", [-0.1, 1.5, "high", None])
    def test_invalid_confidence_rejected(self, bad_conf):
        ok, errs = validate_stage_result(self._good(confidence=bad_conf))
        assert not ok
        assert any("confidence" in e for e in errs)

    def test_artifacts_must_be_string_pairs(self):
        ok, errs = validate_stage_result(self._good(artifacts={"k": 123}))
        assert not ok
        assert any("artifacts" in e for e in errs)

    def test_caveats_must_be_string_list(self):
        ok, errs = validate_stage_result(self._good(caveats=[1, 2]))
        assert not ok
        assert any("caveats" in e for e in errs)

    def test_non_dict_input_rejected(self):
        ok, errs = validate_stage_result(["not", "a", "dict"])
        assert not ok and errs

    def test_partial_status_accepted(self):
        ok, _ = validate_stage_result(self._good(status="partial"))
        assert ok

    def test_failed_status_accepted(self):
        ok, _ = validate_stage_result(self._good(status="failed"))
        assert ok
