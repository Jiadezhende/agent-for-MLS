"""
agents/tools/builtin/subagent.py — Subagent tools for the Planner's ReAct loop.

RunSubagentTool: delegates a measurement/analysis task to a registered agent type.
  - Runs the target agent synchronously and returns ToolResponse.
  - Appends the WorkerOutput dict to ctx.job_history for Orchestrator collection.

RunSubagentParallelTool: runs multiple independent subagent tasks concurrently.
  - Validates all agent_types up front, then dispatches with ThreadPoolExecutor.
  - Each subagent gets its own tool instances and AgentContext (isolated circuit breakers).
  - Safe to share Executor across threads (designed thread-safe).

MarkReadyForCriticTool: signals the Planner has finished all work.
  - Raises _Terminated to exit the AgentLoop cleanly (same pattern as submit_results).
"""
from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from agents.core.agent import SubAgent
from agents.core.types import AgentContext, Step, WorkerOutput
from agents.tools.base import Tool, ToolParameter
from agents.tools.registry import ToolFactory, _Terminated
from agents.tools.response import ToolErrorCode, ToolResponse


# ---------------------------------------------------------------------------
# Shared execution helper
# ---------------------------------------------------------------------------

def _execute_one(
    llm: Any,
    tool_factory: ToolFactory,
    agent_registry: dict,
    agent_cfg: Any,
    verbose: bool,
    agent_type: str,
    targets: list,
    retry_context: dict | None,
    ctx: AgentContext | None,
    instructions: str = "",
) -> WorkerOutput:
    """Run a single subagent call. Appends to ctx.job_history if ctx is set.

    Returns WorkerOutput with agent_type populated.
    Caller must have already validated that agent_type exists in agent_registry.
    """
    agent_def = agent_registry[agent_type]
    tools = tool_factory.build(agent_def.required_tools)
    effective_llm = llm.with_max_tokens(agent_def.max_tokens) if agent_def.max_tokens else llm
    agent: SubAgent = agent_def.agent_class(llm=effective_llm, agent_cfg=agent_cfg, verbose=verbose)

    if ctx is not None:
        agent.run_id = ctx.run_id
        agent.shared_store = ctx.shared_store
    agent.agent_id = f"sub_{agent_type}"

    step_id = str(uuid.uuid4())[:12]
    step = Step(
        id=step_id,
        worker=agent_type,
        targets=targets,
        task=", ".join(str(t) for t in targets),
        instructions=instructions,
        retry_context=retry_context,
    )

    out = agent.run(step, tools)
    out.agent_type = agent_type

    # Write per-agent log immediately so it's available even if the pipeline crashes.
    if ctx is not None and ctx.log_manager is not None:
        ctx.log_manager.write_worker_log(out)
    elif ctx is not None and ctx.run_id:
        # Fallback: legacy step_logs/ path when no LogManager is configured.
        log_dir = Path("step_logs") / ctx.run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{agent_type}_{step_id}.json"
        log_file.write_text(
            json.dumps(
                {"step_id": step_id, "agent_type": agent_type,
                 "reasoning_log": out.reasoning_log, "events": out.events},
                indent=2, default=str,
            ),
            encoding="utf-8",
        )

    if ctx is not None:
        ctx.job_history.append(out)  # list.append() is atomic in CPython

    return out


# ---------------------------------------------------------------------------
# RunSubagentTool
# ---------------------------------------------------------------------------

class RunSubagentTool(Tool):
    """Delegate measurement or analysis to a registered agent type (sequential).

    The agent runs its full ReAct loop and returns results via ToolResponse.
    Results are also accumulated in ctx.job_history so Orchestrator can collect
    them for Critic review after the Planner loop exits.
    """

    _ctx: AgentContext | None = None  # injected by ToolRegistry.dispatch()

    def __init__(
        self,
        llm: Any,
        executor: Any,
        agent_registry: dict,
        agent_cfg: Any,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            name="run_subagent",
            description=(
                "Delegate a measurement or analysis task to a specialized agent. "
                "The agent runs autonomously and returns its results. "
                "Use for hardware probing, operator profiling, or bottleneck analysis. "
                "For independent tasks that can run concurrently, prefer run_subagent_parallel."
            ),
        )
        self._llm = llm
        self._executor = executor
        self._agent_registry = agent_registry
        self._agent_cfg = agent_cfg
        self._verbose = verbose
        self._tool_factory = ToolFactory(executor)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="agent_type",
                type="string",
                description="The agent type key (e.g. 'hardware_probe').",
            ),
            ToolParameter(
                name="targets",
                type="array",
                description="List of target metric names or analysis goals (max 4 per call).",
            ),
        ]

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "run_subagent",
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_type": {
                            "type": "string",
                            "description": "Agent type key. Must be one of the available agent types.",
                        },
                        "targets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Metric names to measure (max 4). "
                                "Provide for measurement agents (hardware_probe, op_profiler) "
                                "to enable Critic coverage checks. "
                                "Omit for analysis/optimization agents — use instructions instead."
                            ),
                            "minItems": 1,
                        },
                        "instructions": {
                            "type": "string",
                            "description": (
                                "Task description and upstream context for the agent. "
                                "This becomes the agent's initial prompt. "
                                "For analysis agents (bottleneck_analyst, kernel_optimizer), "
                                "include relevant measurements from upstream stages. "
                                "If omitted, a default message is generated from targets."
                            ),
                        },
                        "retry_context": {
                            "type": "object",
                            "description": (
                                "Optional. Provide when re-measuring failed targets. "
                                "Keys: 'reason' (str), 'previous_bad_values' "
                                "(dict of metric → {value, unit, method})."
                            ),
                        },
                    },
                    "required": ["agent_type"],
                },
            },
        }

    def run(self, args: Dict[str, Any]) -> ToolResponse:
        agent_type = args.get("agent_type", "")
        targets = list(args.get("targets", []))
        retry_context = args.get("retry_context")
        instructions = args.get("instructions", "")

        if self._agent_registry.get(agent_type) is None:
            return ToolResponse.error(
                code=ToolErrorCode.UNKNOWN_TOOL,
                message=(
                    f"Unknown agent type '{agent_type}'. "
                    f"Available: {list(self._agent_registry.keys())}"
                ),
            )

        out = _execute_one(
            self._llm, self._tool_factory, self._agent_registry, self._agent_cfg,
            self._verbose, agent_type, targets, retry_context, self._ctx,
            instructions=instructions,
        )

        targets_measured = [r.get("metric") for r in out.results if r.get("metric")]
        missing = [t for t in targets if t not in targets_measured]

        parts = [f"Agent '{agent_type}' {'completed' if out.success else 'completed with issues'}."]
        if targets_measured:
            parts.append(f"Measured: {targets_measured}.")
        if missing:
            parts.append(f"Not measured: {missing}.")
        if out.summary:
            parts.append(f"Summary: {out.summary}")

        return ToolResponse.success(
            text=" ".join(parts),
            data={
                "step_id": out.step_id,
                "success": out.success,
                "results": out.results,
                "targets_requested": targets,
                "targets_measured": targets_measured,
                "missing_targets": missing,
                "summary": out.summary,
            },
        )


# ---------------------------------------------------------------------------
# RunSubagentParallelTool
# ---------------------------------------------------------------------------

class RunSubagentParallelTool(Tool):
    """Run multiple independent subagent tasks concurrently.

    Validates all agent_types before spawning threads. Each subagent gets its
    own tool instances and isolated AgentContext. Results from all calls are
    accumulated in ctx.job_history (list.append is atomic in CPython).

    Use when tasks have no dependency on each other's results (e.g. hardware
    probing and baseline profiling can run at the same time).
    """

    _ctx: AgentContext | None = None  # injected by ToolRegistry.dispatch()

    def __init__(
        self,
        llm: Any,
        executor: Any,
        agent_registry: dict,
        agent_cfg: Any,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            name="run_subagent_parallel",
            description=(
                "Run multiple independent subagent tasks concurrently. "
                "Use when tasks have no dependency on each other's results — "
                "e.g. hardware probing and baseline profiling. "
                "All calls must use valid agent_type keys. "
                "Minimum 2 calls; use run_subagent for a single call."
            ),
        )
        self._llm = llm
        self._executor = executor
        self._agent_registry = agent_registry
        self._agent_cfg = agent_cfg
        self._verbose = verbose
        self._tool_factory = ToolFactory(executor)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="calls",
                type="array",
                description="List of independent subagent calls to run concurrently (min 2).",
            ),
        ]

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "run_subagent_parallel",
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "calls": {
                            "type": "array",
                            "minItems": 2,
                            "description": (
                                "Independent subagent calls to execute concurrently. "
                                "Minimum 2 entries. Use run_subagent for a single call."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "agent_type": {
                                        "type": "string",
                                        "description": "Agent type key.",
                                    },
                                    "targets": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": (
                                            "Metric names for measurement agents (max 4). "
                                            "Omit for analysis/optimization agents."
                                        ),
                                        "minItems": 1,
                                    },
                                    "instructions": {
                                        "type": "string",
                                        "description": (
                                            "Task description and upstream context. "
                                            "Becomes the agent's initial prompt. "
                                            "If omitted, generated from targets."
                                        ),
                                    },
                                    "retry_context": {
                                        "type": "object",
                                        "description": "Optional retry context.",
                                    },
                                },
                                "required": ["agent_type"],
                            },
                        },
                    },
                    "required": ["calls"],
                },
            },
        }

    def run(self, args: Dict[str, Any]) -> ToolResponse:
        calls = list(args.get("calls", []))
        if not calls:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message="'calls' must be a non-empty list.",
            )

        # Validate all agent_types before spawning any threads
        unknown = [
            c.get("agent_type", "")
            for c in calls
            if self._agent_registry.get(c.get("agent_type", "")) is None
        ]
        if unknown:
            return ToolResponse.error(
                code=ToolErrorCode.UNKNOWN_TOOL,
                message=(
                    f"Unknown agent type(s): {unknown}. "
                    f"Available: {list(self._agent_registry.keys())}"
                ),
            )

        ctx = self._ctx
        call_results: List[dict] = [{}] * len(calls)  # preserve order

        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            future_to_idx = {
                pool.submit(
                    _execute_one,
                    self._llm,
                    self._tool_factory,
                    self._agent_registry,
                    self._agent_cfg,
                    self._verbose,
                    c.get("agent_type", ""),
                    list(c.get("targets", [])),
                    c.get("retry_context"),
                    ctx,
                    c.get("instructions", ""),
                ): i
                for i, c in enumerate(calls)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    out = future.result()
                    targets_measured = [r.get("metric") for r in out.results if r.get("metric")]
                    missing = [t for t in calls[idx].get("targets", []) if t not in targets_measured]
                    call_results[idx] = {
                        "step_id": out.step_id,
                        "agent_type": out.agent_type,
                        "success": out.success,
                        "targets_measured": targets_measured,
                        "missing_targets": missing,
                        "summary": out.summary,
                        "error": None,
                    }
                except Exception as exc:
                    call_results[idx] = {
                        "step_id": None,
                        "agent_type": calls[idx].get("agent_type", ""),
                        "success": False,
                        "targets_measured": [],
                        "missing_targets": list(calls[idx].get("targets", [])),
                        "summary": "",
                        "error": str(exc),
                    }

        all_success = all(r.get("success") for r in call_results)
        summary_lines = []
        for r in call_results:
            status = "ok" if r.get("success") else "FAILED"
            measured = r.get("targets_measured", [])
            missing = r.get("missing_targets", [])
            line = f"  [{status}] {r['agent_type']}: measured={measured}"
            if missing:
                line += f", missing={missing}"
            if r.get("error"):
                line += f", error={r['error']}"
            summary_lines.append(line)

        text = (
            f"Parallel run: {len(calls)} subagent(s), "
            f"{'all succeeded' if all_success else 'some failed'}.\n"
            + "\n".join(summary_lines)
        )
        return ToolResponse.success(
            text=text,
            data={"calls": call_results, "all_success": all_success},
        )


# ---------------------------------------------------------------------------
# MarkReadyForCriticTool
# ---------------------------------------------------------------------------

class MarkReadyForCriticTool(Tool):
    """Signal the Planner has finished all work and is ready for Critic review.

    Raises _Terminated (same pattern as submit_results) to exit the AgentLoop.
    """

    _ctx: AgentContext | None = None  # injected by ToolRegistry.dispatch()

    def __init__(self) -> None:
        super().__init__(
            name="mark_ready_for_critic",
            description=(
                "Signal that all measurements, analyses, and optimizations are complete "
                "and ready for Critic review. Call this only when ALL success criteria "
                "from the operator skill are satisfied."
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="summary",
                type="string",
                description=(
                    "Comprehensive summary of all work done: hardware parameters measured, "
                    "bottleneck analysis, optimization strategy, performance results, "
                    "and correctness verification. Must address every success criterion."
                ),
            ),
        ]

    def run(self, args: Dict[str, Any]) -> ToolResponse:
        summary = args.get("summary", "")
        if self._ctx is not None:
            self._ctx.memory.set("run", "summary", summary)
        raise _Terminated(summary)
