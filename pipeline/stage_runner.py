"""pipeline/stage_runner.py — generic Stage execution driver.

Responsibilities:
  1. Build a per-stage ToolRegistry restricted to ``agent.allowed_tools``.
  2. Invoke ``agent.run(StageContext) -> StageResult``.
  3. Validate the result (correct stage tag + schema).
  4. Convert any exception or schema violation to a ``failed`` StageResult so
     the orchestrator can advance deterministically.
  5. Soft-monitor wall-clock against ``stage_budget_s`` (over-runs add a caveat
     but do not hard-cancel — the orchestrator's total budget enforces hard cuts).

The driver itself is LLM-agnostic and GPU-agnostic; it only knows about
StageAgent + StageResult. LLM agents drive AgentLoop inside their ``run()``.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .stage_agent import StageAgent
from .state import Stage, StageResult, validate_stage_result
from .workspace_layout import RunLayout


# A factory that builds a ToolRegistry from a list of tool names.
# We use a Callable instead of importing ToolFactory directly so the runner
# can be unit-tested without dragging in cuda_executor / Executor.
ToolBuilder = Callable[[Sequence[str]], Any]


@dataclass
class StageContext:
    """Everything a StageAgent receives at run time.

    Use ``layout`` for path construction, ``tools`` for tool calls, and
    ``run_state`` as a *read-only* snapshot of pipeline progress (mutating it
    inside an agent is forbidden — return a StageResult and let the
    orchestrator update state).
    """
    run_state: Any            # pipeline.state.RunState — typed as Any to avoid cycle
    layout: RunLayout
    tools: Any                # ToolRegistry-shaped object
    stage_budget_s: float
    log_manager: Any = None
    verbose: bool = False

    def relpath(self, p: Path) -> str:
        return self.layout.relpath(p)


def run_stage(
    agent: StageAgent,
    *,
    run_state: Any,
    layout: RunLayout,
    build_tools: ToolBuilder,
    stage_budget_s: float,
    log_manager: Any = None,
    verbose: bool = False,
) -> StageResult:
    """Run a single stage and return a validated StageResult.

    Never raises — failure modes are surfaced as ``status="failed"`` results
    with diagnostic ``caveats``. This is by design: the orchestrator should be
    able to advance (or finalize) regardless of how a stage went wrong.
    """
    expected_stage = agent.stage

    # 1. Build the tool registry restricted to agent.allowed_tools.
    try:
        tools = build_tools(tuple(agent.allowed_tools))
    except Exception as e:
        return _failed(expected_stage, f"tool_build_error: {type(e).__name__}: {e}")

    # 2. Invoke the agent.
    ctx = StageContext(
        run_state=run_state,
        layout=layout,
        tools=tools,
        stage_budget_s=stage_budget_s,
        log_manager=log_manager,
        verbose=verbose,
    )

    started = time.monotonic()
    try:
        result = agent.run(ctx)
    except Exception as e:
        tb = traceback.format_exc(limit=5)
        return _failed(
            expected_stage,
            f"agent_exception: {type(e).__name__}: {e}",
            extra_caveat=tb.strip().splitlines()[-1] if tb else "",
        )
    elapsed = time.monotonic() - started

    # 3. Validate return type.
    if not isinstance(result, StageResult):
        return _failed(
            expected_stage,
            f"agent_returned_non_StageResult: {type(result).__name__}",
        )

    # 4. Validate stage tag matches the agent's declared stage.
    if result.stage != expected_stage.value:
        return _failed(
            expected_stage,
            f"stage_tag_mismatch: agent={expected_stage.value} result={result.stage}",
        )

    # 5. Validate StageResult schema.
    ok, errs = validate_stage_result(result.to_dict())
    if not ok:
        return _failed(
            expected_stage,
            "schema_invalid: " + "; ".join(errs),
        )

    # 6. Soft over-run warning.
    if elapsed > stage_budget_s:
        result.caveats.append(
            f"stage_overran=true elapsed_s={elapsed:.1f} budget_s={stage_budget_s:.1f}"
        )

    return result


def _failed(stage: Stage, reason: str, *, extra_caveat: str = "") -> StageResult:
    caveats = [reason]
    if extra_caveat:
        caveats.append(extra_caveat)
    return StageResult(
        stage=stage.value,
        status="failed",
        artifacts={},
        metrics={},
        confidence=0.0,
        caveats=caveats,
    )
