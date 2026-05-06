"""Per-role agent entry points.

There is intentionally **no** ``LLMStageAgent`` base class. Each role is a
plain function that:

  1. Builds a user-message string from the blackboard + LoRA contract.
  2. Constructs an ``mls_agent.Agent`` with the caller-provided backend and
     registry.
  3. Runs the agent and converts the resulting ``AgentResult.payload`` into
     a dict — falling back to ``{"status": "failed", ...}`` when the agent
     terminated abnormally or the payload fails the lightweight schema check.

Prompts emphasize the ReAct framework's contract: the agent's job is to read
the evidence and submit a decision; performance evaluation, leaderboard, and
best-promotion are owned by ``RoundRunner`` and are NOT visible as tools.
"""
from __future__ import annotations

import json
from typing import Any

import mls_agent
from mls_agent import Agent, AgentConfig, AgentResult, ToolRegistry

from operator_opt_pipe.lora_resources.contract import LoRAContract
from operator_opt_pipe.state import (
    RunLayout,
    Stage,
    check_submit_payload,
    load_blackboard,
)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


SYSTEM_PROMPT_HARDWARE_PROFILER = """\
You are the Hardware Profiler stage of an autonomous CUDA-operator
optimization pipeline. Your single job is to characterize the GPU and the
toolchain so later stages can reason about achievable performance.

Available tools include CUDA probes, ncu/nsys profilers, environment probes,
and a side-effect channel for measurements/events. Run the minimum set of
probes needed to fill in: device name, compute capability, SM count, DRAM
bandwidth, L2 size, and clock behaviour. Record each fact via
record_measurement so it is auditable.

When you have enough evidence, call submit_hardware_profile with status,
metrics, and any caveats (e.g. clocks not locked). Do NOT speculate beyond
what the probes returned — caveats are cheaper than wrong numbers.
"""

SYSTEM_PROMPT_OPTIMIZER_COLD = """\
You are the cold-start Optimizer. The benchmark and PyTorch baseline have
already been measured. Read them with read_blackboard, design a first
correct CUDA candidate for the operator, and submit it.

Workflow:
  1. read_blackboard("benchmark") and read_blackboard("baseline").
  2. write_candidate(source=...) creates a draft.
  3. verify_candidate(candidate_id) checks compile + correctness only —
     it does NOT report runtime. If verification fails, edit_candidate and
     verify again. Repeat until correctness passes.
  4. submit_candidate with hypothesis / experiment_type / expected_effect /
     risk fields populated.

Performance is judged by the RoundRunner after submission; you do not see
runtime numbers from this stage's tools.
"""

SYSTEM_PROMPT_ANALYST = """\
You are the Analyst inside a tuning round. Read the blackboard (benchmark,
baseline, best, latest history entries) and any profiler outputs you choose
to gather, then call submit_diagnosis with a structured judgement of the
current bottleneck.

Your job is NOT to propose code; the Optimizer reads your diagnosis and
decides what to write. Be concrete about what evidence supports each claim:
DRAM-bound vs compute-bound, occupancy ceiling, low-rank correction
overhead, launch overhead, etc. Cite the metric numbers you used.
"""

SYSTEM_PROMPT_OPTIMIZER = """\
You are the Optimizer inside a tuning round. The Analyst has already
written latest_diagnosis to the blackboard. Read it (and the recent history
of attempts) and decide on the next candidate worth trying.

Workflow is identical to the cold-start Optimizer: write_candidate →
verify_candidate (compile + correctness only) → submit_candidate. Submit
only when correctness passes. The hypothesis you attach to the submission
should reference the diagnosis: which bottleneck this candidate addresses
and how.
"""

SYSTEM_PROMPT_SUMMARY = """\
You are the Finalize stage. Read the best candidate, the baseline, and the
history of attempts from the blackboard, and call submit_summary with a
concise account of the best speedup, the path that got there, and any
remaining caveats.
"""


# ---------------------------------------------------------------------------
# User-message builders
# ---------------------------------------------------------------------------


def _section(title: str, body: str) -> str:
    return f"=== {title} ===\n{body.rstrip()}\n"


def _format_contract(contract: LoRAContract) -> str:
    lo, hi = contract.d_range
    return (
        f"Operator: {contract.operator}\n"
        f"Reference formula: {contract.output_name} = {contract.reference_pytorch}\n"
        f"Forward args order: {', '.join(contract.forward_args)}\n"
        f"Shape param d ∈ [{lo}, {hi}]; LoRA rank r = {contract.r}; dtype = {contract.dtype}\n"
        f"Tolerance: atol={contract.tolerance_atol}, rtol={contract.tolerance_rtol}"
    )


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


def _build_user_msg_hardware(contract: LoRAContract) -> str:
    return (
        _section("Operator contract", _format_contract(contract))
        + "\n"
        + "Probe the GPU and submit_hardware_profile when finished."
    )


def _build_user_msg_optimizer_cold(contract: LoRAContract, blackboard: dict) -> str:
    return (
        _section("Operator contract", _format_contract(contract))
        + "\n"
        + _section("Blackboard snapshot", _format_keys(blackboard, ("hardware", "benchmark", "baseline")))
        + "\n"
        + "Produce the first correct candidate and submit it."
    )


def _build_user_msg_analyst(contract: LoRAContract, blackboard: dict) -> str:
    return (
        _section("Operator contract", _format_contract(contract))
        + "\n"
        + _section(
            "Blackboard snapshot",
            _format_keys(blackboard, ("benchmark", "baseline", "best", "history", "round")),
        )
        + "\n"
        + "Diagnose the current bottleneck and call submit_diagnosis."
    )


def _build_user_msg_optimizer(contract: LoRAContract, blackboard: dict) -> str:
    return (
        _section("Operator contract", _format_contract(contract))
        + "\n"
        + _section(
            "Blackboard snapshot",
            _format_keys(blackboard, ("baseline", "best", "latest_diagnosis", "history", "round")),
        )
        + "\n"
        + "Propose and submit the next candidate."
    )


def _build_user_msg_summary(contract: LoRAContract, blackboard: dict) -> str:
    return (
        _section("Operator contract", _format_contract(contract))
        + "\n"
        + _section(
            "Blackboard snapshot",
            _format_keys(blackboard, ("baseline", "best", "history")),
        )
        + "\n"
        + "Write the final report via submit_summary."
    )


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _payload_or_failure(result: AgentResult, expected_stage: Stage | None) -> dict:
    """Convert an ``AgentResult`` into a dict the orchestrator can consume.

    - ``reason != 'completed'`` → ``{"status": "failed", ...}`` with a caveat
      describing how the agent terminated.
    - Missing/invalid payload → same.
    - Otherwise return the payload unchanged (``status`` / ``stage`` already
      validated by ``check_submit_payload``).
    """
    base_caveat = (
        f"agent terminated with reason={result.reason!r}; summary={result.summary!r}"
    )
    if result.reason != "completed" or not isinstance(result.payload, dict):
        return {
            "status": "failed",
            "stage": expected_stage.value if expected_stage else None,
            "caveats": [base_caveat],
        }
    ok, errs = check_submit_payload(result.payload, expected_stage)
    if not ok:
        return {
            "status": "failed",
            "stage": expected_stage.value if expected_stage else None,
            "caveats": ["payload validation failed: " + "; ".join(errs)],
        }
    return result.payload


def _run_agent(
    *,
    backend: mls_agent.LLMBackend,
    registry: ToolRegistry,
    system_prompt: str,
    user_message: str,
    agent_cfg: AgentConfig,
    observer: mls_agent.AgentObserver,
    expected_stage: Stage | None,
) -> dict:
    agent = Agent(
        backend=backend,
        registry=registry,
        system_prompt=system_prompt,
        config=agent_cfg,
        observer=observer,
    )
    result = agent.run(user_message)
    return _payload_or_failure(result, expected_stage)


# ---------------------------------------------------------------------------
# Per-role entry points
# ---------------------------------------------------------------------------


def run_hardware_profiler(
    *,
    backend: mls_agent.LLMBackend,
    registry: ToolRegistry,
    layout: RunLayout,
    contract: LoRAContract,
    agent_cfg: AgentConfig,
    observer: mls_agent.AgentObserver,
) -> dict:
    user_msg = _build_user_msg_hardware(contract)
    return _run_agent(
        backend=backend, registry=registry,
        system_prompt=SYSTEM_PROMPT_HARDWARE_PROFILER,
        user_message=user_msg, agent_cfg=agent_cfg, observer=observer,
        expected_stage=Stage.HARDWARE_PROFILE,
    )


def run_optimizer_cold(
    *,
    backend: mls_agent.LLMBackend,
    registry: ToolRegistry,
    layout: RunLayout,
    contract: LoRAContract,
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
    )


def run_analyst(
    *,
    backend: mls_agent.LLMBackend,
    registry: ToolRegistry,
    layout: RunLayout,
    contract: LoRAContract,
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
    )


def run_optimizer(
    *,
    backend: mls_agent.LLMBackend,
    registry: ToolRegistry,
    layout: RunLayout,
    contract: LoRAContract,
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
    )


def run_summary(
    *,
    backend: mls_agent.LLMBackend,
    registry: ToolRegistry,
    layout: RunLayout,
    contract: LoRAContract,
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
    )


__all__ = [
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
