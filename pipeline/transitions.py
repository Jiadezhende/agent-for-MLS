"""pipeline/transitions.py — pure stage-advance rules.

Decoupled from filesystem so it can be unit-tested without touching disk.
The orchestrator wires `RunLayout`-backed callables into `next_stage()`.
"""
from __future__ import annotations

from typing import Callable

from .state import RunState, Stage


# Minimum remaining budget below which TUNING_LOOP gives up and goes
# straight to OPTIONAL_PROFILE / FINALIZE. Tuned for LoRA: a single
# candidate compile+correctness+quick+confirm round on a ~30s budget.
MIN_TUNING_SLICE_S: float = 60.0


def next_stage(
    state: RunState,
    *,
    has_hardware_profile: Callable[[], bool],
    has_baseline: Callable[[], bool],
    has_candidate_profile: Callable[[str], bool],
) -> Stage:
    """Decide which stage to run next based on RunState + artifact existence.

    Pure rules; no I/O — callers inject the artifact predicates so this
    function stays unit-testable.
    """
    # Hard timeout always wins.
    if state.elapsed_s >= state.time_budget_s:
        return Stage.FINALIZE

    if not state.specs_complete():
        return Stage.BENCHMARK_SPEC

    if not has_hardware_profile():
        return Stage.HARDWARE_PROFILE

    if not has_baseline():
        return Stage.BASELINE_PROFILE

    if state.best_candidate_id is None:
        return Stage.INITIAL_CANDIDATE

    # We have a best. Either keep tuning, or — if budget is too thin —
    # finish up with optional profile + finalize.
    if state.remaining_budget_s() > MIN_TUNING_SLICE_S:
        return Stage.TUNING_LOOP

    if not has_candidate_profile(state.best_candidate_id):
        return Stage.OPTIONAL_PROFILE

    return Stage.FINALIZE
