"""Stage state machine — pure ``next_stage`` predicate.

Decouples the rules (this file) from the orchestrator's main loop so the
transitions can be unit-tested with cheap fakes.
"""
from __future__ import annotations

from operator_opt_pipe.state import RunLayout, RunState, Stage


# Don't bother starting another tuning round if the budget is already shorter
# than this — go straight to FINALIZE so the run ends cleanly.
MIN_TUNING_SLICE_S = 60.0


def next_stage(state: RunState, layout: RunLayout) -> Stage:
    """Decide which stage runs next.

    Pure function: it inspects the current ``RunState`` (in-memory) and the
    on-disk artifact predicates exposed by ``RunLayout``. Setup stages gate on
    the artifact predicates so a resumed run automatically skips work that's
    already on disk.
    """
    if state.elapsed_s >= state.time_budget_s:
        return Stage.FINALIZE
    if not layout.has_hardware_profile():
        return Stage.HARDWARE_PROFILE
    if not layout.has_baseline():
        return Stage.BENCHMARK_BASELINE
    if state.best_candidate_id is None:
        return Stage.INITIAL_CANDIDATE
    if state.remaining_budget_s() > MIN_TUNING_SLICE_S:
        return Stage.TUNING_LOOP
    return Stage.FINALIZE
