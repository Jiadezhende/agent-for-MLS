"""pipeline/agents/summary_agent.py — FINALIZE stage agent (final report writer)."""
from __future__ import annotations

import json

from ..stage_runner import StageContext
from ..state import Stage
from ._base import LLMStageAgent


_SYSTEM_PROMPT = """You are the SummaryAgent. The pipeline is finished or has run out of budget. \
Synthesize the run into:

  report      — structured dict, including:
                  run_id, operator, best_candidate_id, best_speedup,
                  per_d_speedups (list of {d, speedup}),
                  candidates_evaluated (int), elapsed_s, time_budget_s,
                  failure_counts, caveats.

  summary_md  — 4-8 paragraph Markdown, written for an evaluator.
                Sections (MAY be implicit):
                  - What was attempted
                  - Final result (one-line top line + per-d table)
                  - Headline tradeoffs / caveats
                  - What further work would help if time permitted

Call submit_summary(report, summary_md) exactly once. Do NOT modify any other files.
"""


class SummaryAgent(LLMStageAgent):
    stage = Stage.FINALIZE
    allowed_tools = ("read_skill", "submit_summary", "flag_event")
    max_iterations = 5
    SYSTEM_PROMPT = _SYSTEM_PROMPT

    def build_user_message(self, context: StageContext) -> str:
        rs = context.run_state
        layout = context.layout

        # Truncate the leaderboard so very long runs don't blow up the prompt.
        lb_lines: list[str] = []
        if layout.leaderboard_path.is_file():
            lb_lines = layout.leaderboard_path.read_text(encoding="utf-8").strip().splitlines()
        lb_tail = "\n".join(lb_lines[-15:]) if lb_lines else "(empty)"

        baseline_blob = "(baseline.json not present)"
        if layout.has_baseline():
            try:
                baseline_blob = layout.baseline_path.read_text(encoding="utf-8")[:2000]
            except OSError:
                pass

        best_result = "(no best_result.json)"
        if layout.best_result_path.is_file():
            try:
                best_result = layout.best_result_path.read_text(encoding="utf-8")
            except OSError:
                pass

        return (
            "Generate the final report and Markdown summary for this run.\n\n"
            f"=== run state ===\n{json.dumps(rs.to_dict(), indent=2)}\n\n"
            f"=== best_result.json ===\n{best_result}\n\n"
            f"=== baseline.json (truncated) ===\n{baseline_blob}\n\n"
            f"=== leaderboard tail (last 15) ===\n{lb_tail}\n"
        )
