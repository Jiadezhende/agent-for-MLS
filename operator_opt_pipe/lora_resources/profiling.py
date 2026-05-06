"""Optional profiler hooks (ncu / nsys / torch). **Stub — filled in next PR.**

Profiling is not part of the ranking path. The Analyst agent calls it via
LLM-facing tools to investigate bottlenecks; results land in the blackboard's
``latest_diagnosis`` and inform the next Optimizer iteration but never affect
leaderboard order.
"""
from __future__ import annotations

from dataclasses import dataclass

from operator_opt_pipe.state import RunLayout


@dataclass(frozen=True)
class ProfileResult:
    candidate_id: str
    mode: str          # "ncu" | "nsys" | "torch"
    ok: bool
    metrics: dict
    artifact_paths: dict


def profile_candidate(
    layout: RunLayout,
    executor,
    candidate_id: str,
    *,
    mode: str = "ncu",
) -> ProfileResult:
    raise NotImplementedError(
        "operator_opt_pipe.lora_resources.profiling.profile_candidate is a "
        "stub; it will be implemented in the next PR (operator_opt_pipe-impl)."
    )
