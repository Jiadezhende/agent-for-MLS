"""pipeline/agents/profile_analysis_agent.py — OPTIONAL_PROFILE stage agent."""
from __future__ import annotations

from ..stage_runner import StageContext
from ..state import Stage
from ._base import LLMStageAgent


_SYSTEM_PROMPT = """You are the ProfileAnalysisAgent. The pipeline has settled on a best \
candidate; your job is to profile it with ncu/nsys and write a brief analysis \
explaining whether the kernel is compute-bound vs memory-bound and where the \
remaining headroom is.

WORKFLOW
  1. profile_with_ncu on the best candidate's compiled binary (if available) — focus \
     on dram__throughput and sm__throughput sections.
  2. profile_with_nsys to capture CPU↔GPU launch overhead and kernel duration.
  3. Synthesize findings into:
       profile      — structured dict with key metrics
       analysis_md  — 1-2 paragraph Markdown
  4. Call submit_profile_analysis(candidate_id, profile, analysis_md).

If profilers are unavailable on this system (permission denied, missing binary), \
flag the event and submit whatever profile data you do have with a clear caveat \
in the analysis_md. Don't burn the budget retrying tools that won't work.
"""


class ProfileAnalysisAgent(LLMStageAgent):
    stage = Stage.OPTIONAL_PROFILE
    allowed_tools = (
        "read_skill",
        "profile_with_ncu",
        "profile_with_nsys",
        "submit_profile_analysis",
        "flag_event",
    )
    max_iterations = 12
    SYSTEM_PROMPT = _SYSTEM_PROMPT

    def build_user_message(self, context: StageContext) -> str:
        rs = context.run_state
        cid = rs.best_candidate_id or "(none)"
        layout = context.layout

        cu_path = (
            str(layout.candidate_file(rs.best_candidate_id, "candidate.cu"))
            if rs.best_candidate_id else "(no best)"
        )
        return (
            f"best_candidate_id: {cid}\n"
            f"candidate.cu absolute path: {cu_path}\n"
            f"speedup_at_best: {rs.best_speedup}\n\n"
            "Profile the kernel and submit a structured profile + Markdown analysis."
        )
