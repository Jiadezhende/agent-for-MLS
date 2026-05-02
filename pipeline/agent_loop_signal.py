"""pipeline/agent_loop_signal.py — StageAgent ↔ AgentLoop coupling helpers.

A submit_* tool inside a StageAgent's tool registry should:
  1. Build a StageResult dict matching the stage's contract.
  2. Call ``stash_stage_result(ctx, result)``.
  3. Raise ``_Terminated(summary)`` (already imported from registry).

The StageAgent's ``run()`` then:
  - calls ``AgentLoop(...).run()``  (which internally catches _Terminated)
  - calls ``pop_stage_result(ctx)`` to retrieve the structured result.

This avoids modifying agents.core.loop.py while still keeping all stage
information flowing through StageResult instead of free-form summaries.
"""
from __future__ import annotations

from typing import Any

from .state import StageResult


# AgentContext namespace + key. Underscore-prefixed to discourage accidental
# collision with arbitrary tool memory.
_NS = "_stage"
_KEY = "result"


def stash_stage_result(ctx: Any, result: StageResult) -> None:
    """Called by submit_* tools right before raising _Terminated."""
    ctx.memory.set(_NS, _KEY, result.to_dict())


def pop_stage_result(ctx: Any) -> StageResult | None:
    """Called by StageAgent.run() after AgentLoop.run() returns.

    Returns None if no submit_* was ever called (e.g. agent hit max_iterations).
    """
    d = ctx.memory.get(_NS, _KEY)
    if not isinstance(d, dict):
        return None
    try:
        return StageResult.from_dict(d)
    except (KeyError, TypeError, ValueError):
        return None
