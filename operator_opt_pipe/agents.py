"""Per-role agent entry points + per-role tool registry construction.

Five roles total:

    hardware_profiler  → write_blackboard("hardware", ...) → natural exit
    optimizer_cold     → write_candidate → submit_candidate (terminate_with)
    analyst            → write_blackboard("latest_diagnosis", ...) → natural exit
    optimizer          → write_candidate → submit_candidate (terminate_with)
    summary            → write_blackboard("final_summary", ...) → natural exit

Two termination patterns coexist:

    * "natural exit" — write_blackboard does NOT terminate; the LLM is
      instructed to summarize and stop. ReActLoop's
      ``max_consecutive_no_tool_call`` handles termination, returning
      ``reason=NO_TOOL_CALL``. The orchestrator reads the produced
      blackboard key directly.
    * "explicit terminate" — submit_candidate must hand the candidate_id
      back to the orchestrator immediately so it can run multi-shape
      benchmark + promote logic; uses ``ToolResponse.terminate_with(payload)``,
      ``reason=COMPLETED``.

Performance evaluation, multi-shape benchmark, and best promotion are NOT
exposed as tools — they live in the orchestrator.
"""
from __future__ import annotations

import json
from typing import Any

import mls_agent
from mls_agent import (
    Agent,
    AgentConfig,
    AgentResult,
    Tool,
    ToolRegistry,
)
from mls_agent.tools.builtin import make_side_effect_tools, make_skill_tools
from mls_agent.tools.cuda.profile_tools import make_profile_tools

from operator_opt_pipe.resources.contract import OperatorContract
from operator_opt_pipe.state import RunLayout, Stage, load_blackboard
from operator_opt_pipe.tools import (
    ReadBlackboardTool,
    SubmitCandidateTool,
    WriteBlackboardTool,
    WriteCandidateTool,
)


# ---------------------------------------------------------------------------
# Role list + per-role allowed blackboard keys + required payload fields
# ---------------------------------------------------------------------------


VALID_ROLES: tuple[str, ...] = (
    "hardware_profiler",
    "optimizer_cold",
    "analyst",
    "optimizer",
    "summary",
)


# Per-role write_blackboard whitelist. Value is the list of required payload
# field names (used for shallow validation); None disables validation.
ROLE_BLACKBOARD_KEYS: dict[str, dict[str, list[str] | None]] = {
    "hardware_profiler": {"hardware": ["metrics"]},
    "analyst":           {"latest_diagnosis": ["bottleneck", "evidence"]},
    "summary":           {"final_summary": ["best_speedup", "narrative"]},
}


# ---------------------------------------------------------------------------
# build_registry — per-role tool wiring
# ---------------------------------------------------------------------------


def build_registry(
    role: str,
    *,
    layout: RunLayout,
    contract: OperatorContract,
    executor: Any,
    skills_dir: Any | None = None,
) -> ToolRegistry:
    """Construct a fresh ``ToolRegistry`` populated with the role's tools.

    Skill / profile / candidate tools are constructed locally — there is no
    shared "tool bag" because ``WriteBlackboardTool`` needs different
    ``allowed_keys`` per role and ``WriteCandidateTool`` needs the
    contract.

    Tests can pass a no-op executor + ``skills_dir=None`` to skip groups
    that aren't exercised.
    """
    if role not in VALID_ROLES:
        raise ValueError(f"unknown role {role!r}; valid: {VALID_ROLES}")

    reg = ToolRegistry()

    # Always-on
    if skills_dir is not None:
        for tool in make_skill_tools(skills_dir):
            reg.register(tool)
    for tool in make_side_effect_tools():
        reg.register(tool)
    reg.register(ReadBlackboardTool(layout))

    if role == "hardware_profiler":
        if executor is not None:
            for tool in make_profile_tools(executor=executor):
                reg.register(tool)
        reg.register(WriteBlackboardTool(layout, ROLE_BLACKBOARD_KEYS[role]))
    elif role == "analyst":
        if executor is not None:
            # Same factory — agents only get the subset they need by selecting
            # their own slice. Profile tools are read-only w.r.t. workspace
            # state, safe to expose.
            for tool in make_profile_tools(executor=executor):
                if tool.NAME in ("profile_with_ncu", "profile_with_nsys", "profile_with_torch"):
                    reg.register(tool)
        reg.register(WriteBlackboardTool(layout, ROLE_BLACKBOARD_KEYS[role]))
    elif role in ("optimizer_cold", "optimizer"):
        reg.register(WriteCandidateTool(layout, contract, executor))
        reg.register(SubmitCandidateTool(layout))
    elif role == "summary":
        reg.register(WriteBlackboardTool(layout, ROLE_BLACKBOARD_KEYS[role]))
    return reg


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


SYSTEM_PROMPT_HARDWARE_PROFILER = """\
You are the Hardware Profiler stage of a CUDA-operator optimization pipeline.

Your job: characterize the GPU and toolchain so later stages can reason about
achievable performance. Available tools include CUDA probes, ncu/nsys
profilers, environment inspection, and the side-effect channels
(record_measurement / flag_event).

Run the minimum probes needed to fill in: device name, compute capability,
SM count, DRAM bandwidth, L2 size, peak boost clock. Cite the metric
numbers for each finding.

When done, call write_blackboard with key="hardware" and a payload
containing at least:
  metrics: { sm_count, dram_bw_gbps, l2_kb, peak_clock_mhz, ... }
  device_name, compute_capability, caveats (list of strings).

After write_blackboard succeeds, summarize what you wrote in plain text and
do NOT call any further tools — the loop will end naturally."""


SYSTEM_PROMPT_OPTIMIZER_COLD = """\
You are the cold-start Optimizer for a CUDA operator. The PyTorch baseline
has already been measured. Read it via read_blackboard("baseline") and
read_blackboard("operator") to understand the contract.

Goal for this round: produce a FIRST CORRECT candidate. Speed is secondary.
A naive but correct fused kernel is far better than a clever one that fails
correctness — the orchestrator will keep iterating later.

Workflow:
  1. read_blackboard for "operator" and "baseline" (and "hardware" for SM count).
  2. write_candidate(source=...) — the tool compiles + runs correctness on
     a single shape and reports the result. If compile_ok=false or
     correctness_ok=false, study the log and call write_candidate again
     with a fixed source. Drafts are cheap.
  3. Once a candidate passes both gates, call submit_candidate with the
     returned candidate_id, plus hypothesis / experiment_type /
     expected_effect / risk fields.

The forward signature MUST match the contract verbatim. Use
torch::Tensor and PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{ m.def("forward", &forward); }}."""


SYSTEM_PROMPT_ANALYST = """\
You are the Analyst inside a tuning round. Read evidence, write a focused
diagnosis. You do NOT propose code.

Workflow:
  1. read_blackboard("best") and read_blackboard("history") to see prior
     attempts; read_blackboard("hardware") for headroom anchors.
  2. Choose a profile tool (profile_with_ncu / profile_with_nsys /
     profile_with_torch) and run it against the best candidate to gather
     concrete numbers. Cite the candidate file path explicitly.
  3. Call write_blackboard with key="latest_diagnosis" and a payload like:
     {
       "bottleneck": "dram_bound" | "compute_bound" | "occupancy_low" | ...,
       "evidence": ["dram_throughput=82% of peak (ncu)", ...],
       "hypothesis_for_optimizer": "fuse low-rank correction into WX kernel
                                   to eliminate one DRAM round-trip",
       "tile_hints": {"BM": 128, "BN": 128, "BK": 16}   # optional
     }

After a successful write_blackboard, summarize and stop calling tools."""


SYSTEM_PROMPT_OPTIMIZER = """\
You are the Optimizer inside a tuning round. The Analyst has already written
latest_diagnosis to the blackboard.

Workflow:
  1. read_blackboard("latest_diagnosis") and read_blackboard("best") to see
     the current target speedup baseline.
  2. write_candidate(source=...) — the tool compiles + runs correctness on
     a single shape. If anything fails, iterate (call write_candidate again
     with a fixed source).
  3. submit_candidate with hypothesis referencing the diagnosis: which
     bottleneck this candidate addresses and how.

You do NOT see runtime numbers — the orchestrator runs the multi-shape
benchmark after submit_candidate and decides promotion."""


SYSTEM_PROMPT_SUMMARY = """\
You are the Finalize stage. Read the best candidate, baseline, and history
from the blackboard. Call write_blackboard with key="final_summary" and a
payload containing:
  best_speedup: float | null
  best_candidate_id: string | null
  narrative: 3-6 sentences describing the path that got there
  caveats: list of remaining concerns

After a successful write, summarize and stop calling tools."""


# ---------------------------------------------------------------------------
# User-message builders
# ---------------------------------------------------------------------------


def _section(title: str, body: str) -> str:
    return f"=== {title} ===\n{body.rstrip()}\n"


def _format_keys(blackboard: dict, keys: tuple[str, ...]) -> str:
    parts: list[str] = []
    for k in keys:
        if k in blackboard:
            try:
                rendered = json.dumps(blackboard[k], ensure_ascii=False, indent=2, default=str)
            except (TypeError, ValueError):
                rendered = repr(blackboard[k])
            parts.append(f"{k}:\n{rendered}")
        else:
            parts.append(f"{k}: <absent>")
    return "\n\n".join(parts)


def _build_user_msg_hardware(contract: OperatorContract) -> str:
    return (
        contract.summary_for_prompt() + "\n\n"
        + "Probe the GPU and call write_blackboard with key='hardware' when finished."
    )


def _build_user_msg_optimizer_cold(contract: OperatorContract, blackboard: dict) -> str:
    return (
        contract.summary_for_prompt() + "\n\n"
        + _section(
            "Blackboard snapshot",
            _format_keys(blackboard, ("hardware", "baseline")),
        )
        + "\n"
        + "Produce the first correct candidate and submit it."
    )


def _build_user_msg_analyst(contract: OperatorContract, blackboard: dict) -> str:
    return (
        contract.summary_for_prompt() + "\n\n"
        + _section(
            "Blackboard snapshot",
            _format_keys(blackboard, ("baseline", "best", "history", "round")),
        )
        + "\n"
        + "Diagnose the current bottleneck and write_blackboard('latest_diagnosis', ...)."
    )


def _build_user_msg_optimizer(contract: OperatorContract, blackboard: dict) -> str:
    return (
        contract.summary_for_prompt() + "\n\n"
        + _section(
            "Blackboard snapshot",
            _format_keys(blackboard, ("baseline", "best", "latest_diagnosis", "history", "round")),
        )
        + "\n"
        + "Propose and submit the next candidate."
    )


def _build_user_msg_summary(contract: OperatorContract, blackboard: dict) -> str:
    return (
        contract.summary_for_prompt() + "\n\n"
        + _section(
            "Blackboard snapshot",
            _format_keys(blackboard, ("baseline", "best", "history")),
        )
        + "\n"
        + "Write the final report via write_blackboard('final_summary', ...)."
    )


# ---------------------------------------------------------------------------
# Common runner — translates AgentResult into orchestrator-consumable dict
# ---------------------------------------------------------------------------


# Termination reasons that count as "agent succeeded"
_OK_REASONS = ("completed", "no_tool_call")


def _result_to_dict(
    result: AgentResult,
    *,
    expected_stage: Stage | None,
    blackboard_key: str | None = None,
    layout: RunLayout | None = None,
) -> dict:
    """Convert ``AgentResult`` into the dict the orchestrator consumes.

    Two cases:

    * ``submit_candidate`` agents return ``COMPLETED`` with the payload
      attached. Pass it through, optionally validating ``stage``.
    * ``write_blackboard`` agents return ``NO_TOOL_CALL`` (the loop ended
      after the LLM stopped calling tools). The orchestrator reads the
      blackboard key the agent was supposed to write, so we synthesize
      a success payload referencing it.
    """
    if result.reason not in _OK_REASONS:
        return {
            "status": "failed",
            "stage": expected_stage.value if expected_stage else None,
            "caveats": [
                f"agent terminated with reason={result.reason!r}; "
                f"summary={result.summary!r}"
            ],
        }

    if result.reason == "completed":
        if isinstance(result.payload, dict):
            payload = dict(result.payload)
            payload.setdefault("status", "success")
            if expected_stage is not None:
                payload.setdefault("stage", expected_stage.value)
            return payload
        return {
            "status": "failed",
            "stage": expected_stage.value if expected_stage else None,
            "caveats": ["terminate_with payload was not a dict"],
        }

    # NO_TOOL_CALL: did the agent actually write the blackboard key?
    if blackboard_key is not None and layout is not None:
        bb = load_blackboard(layout)
        if blackboard_key in bb:
            payload = bb[blackboard_key]
            return {
                "status": "success",
                "stage": expected_stage.value if expected_stage else None,
                "blackboard_key": blackboard_key,
                "payload": payload,
                "summary": result.summary,
            }
        return {
            "status": "failed",
            "stage": expected_stage.value if expected_stage else None,
            "caveats": [
                f"agent ended without writing blackboard[{blackboard_key!r}]"
            ],
        }

    # Generic NO_TOOL_CALL with no expected key — treat as partial success.
    return {
        "status": "partial",
        "stage": expected_stage.value if expected_stage else None,
        "caveats": ["agent ended without calling a tool"],
        "summary": result.summary,
    }


def _run_agent(
    *,
    backend: mls_agent.LLMBackend,
    registry: ToolRegistry,
    system_prompt: str,
    user_message: str,
    agent_cfg: AgentConfig,
    observer: mls_agent.AgentObserver,
    expected_stage: Stage | None,
    blackboard_key: str | None,
    layout: RunLayout | None,
) -> dict:
    agent = Agent(
        backend=backend,
        registry=registry,
        system_prompt=system_prompt,
        config=agent_cfg,
        observer=observer,
    )
    result = agent.run(user_message)
    return _result_to_dict(
        result,
        expected_stage=expected_stage,
        blackboard_key=blackboard_key,
        layout=layout,
    )


# ---------------------------------------------------------------------------
# Per-role entry points
# ---------------------------------------------------------------------------


def run_hardware_profiler(
    *,
    backend: mls_agent.LLMBackend,
    registry: ToolRegistry,
    layout: RunLayout,
    contract: OperatorContract,
    agent_cfg: AgentConfig,
    observer: mls_agent.AgentObserver,
) -> dict:
    user_msg = _build_user_msg_hardware(contract)
    return _run_agent(
        backend=backend, registry=registry,
        system_prompt=SYSTEM_PROMPT_HARDWARE_PROFILER,
        user_message=user_msg, agent_cfg=agent_cfg, observer=observer,
        expected_stage=Stage.HARDWARE_PROFILE,
        blackboard_key="hardware", layout=layout,
    )


def run_optimizer_cold(
    *,
    backend: mls_agent.LLMBackend,
    registry: ToolRegistry,
    layout: RunLayout,
    contract: OperatorContract,
    agent_cfg: AgentConfig,
    observer: mls_agent.AgentObserver,
) -> dict:
    blackboard = load_blackboard(layout)
    user_msg = _build_user_msg_optimizer_cold(contract, blackboard)
    return _run_agent(
        backend=backend, registry=registry,
        system_prompt=SYSTEM_PROMPT_OPTIMIZER_COLD,
        user_message=user_msg, agent_cfg=agent_cfg, observer=observer,
        expected_stage=Stage.INITIAL_CANDIDATE,
        blackboard_key=None, layout=None,
    )


def run_analyst(
    *,
    backend: mls_agent.LLMBackend,
    registry: ToolRegistry,
    layout: RunLayout,
    contract: OperatorContract,
    agent_cfg: AgentConfig,
    observer: mls_agent.AgentObserver,
) -> dict:
    blackboard = load_blackboard(layout)
    user_msg = _build_user_msg_analyst(contract, blackboard)
    return _run_agent(
        backend=backend, registry=registry,
        system_prompt=SYSTEM_PROMPT_ANALYST,
        user_message=user_msg, agent_cfg=agent_cfg, observer=observer,
        expected_stage=Stage.TUNING_LOOP,
        blackboard_key="latest_diagnosis", layout=layout,
    )


def run_optimizer(
    *,
    backend: mls_agent.LLMBackend,
    registry: ToolRegistry,
    layout: RunLayout,
    contract: OperatorContract,
    agent_cfg: AgentConfig,
    observer: mls_agent.AgentObserver,
) -> dict:
    blackboard = load_blackboard(layout)
    user_msg = _build_user_msg_optimizer(contract, blackboard)
    return _run_agent(
        backend=backend, registry=registry,
        system_prompt=SYSTEM_PROMPT_OPTIMIZER,
        user_message=user_msg, agent_cfg=agent_cfg, observer=observer,
        expected_stage=Stage.TUNING_LOOP,
        blackboard_key=None, layout=None,
    )


def run_summary(
    *,
    backend: mls_agent.LLMBackend,
    registry: ToolRegistry,
    layout: RunLayout,
    contract: OperatorContract,
    agent_cfg: AgentConfig,
    observer: mls_agent.AgentObserver,
) -> dict:
    blackboard = load_blackboard(layout)
    user_msg = _build_user_msg_summary(contract, blackboard)
    return _run_agent(
        backend=backend, registry=registry,
        system_prompt=SYSTEM_PROMPT_SUMMARY,
        user_message=user_msg, agent_cfg=agent_cfg, observer=observer,
        expected_stage=Stage.FINALIZE,
        blackboard_key="final_summary", layout=layout,
    )


__all__ = [
    "VALID_ROLES",
    "ROLE_BLACKBOARD_KEYS",
    "build_registry",
    "SYSTEM_PROMPT_HARDWARE_PROFILER",
    "SYSTEM_PROMPT_OPTIMIZER_COLD",
    "SYSTEM_PROMPT_ANALYST",
    "SYSTEM_PROMPT_OPTIMIZER",
    "SYSTEM_PROMPT_SUMMARY",
    "run_hardware_profiler",
    "run_optimizer_cold",
    "run_analyst",
    "run_optimizer",
    "run_summary",
]
