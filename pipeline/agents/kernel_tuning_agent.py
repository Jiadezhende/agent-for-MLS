"""pipeline/agents/kernel_tuning_agent.py — INITIAL_CANDIDATE / TUNING_LOOP stage agent."""
from __future__ import annotations

import json
from pathlib import Path

from ..stage_runner import StageContext
from ..state import Stage
from ._base import LLMStageAgent

_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"


_SYSTEM_PROMPT = """You are the KernelTuningAgent. Each invocation must produce ONE candidate \
CUDA kernel and submit a CandidateRecord describing how it performed.

The operator contract (formula, tensor shapes, forward signature, shape range) AND the \
tuning guide (search space, navigation rules, low-rank exploitation, tile size reference) \
are both injected into the user message below — treat them as authoritative. \
The candidate .cu file must export the declared forward signature via PYBIND11_MODULE so \
that torch.utils.cpp_extension.load can compile it.

WORKFLOW
  1. Read the injected operator contract and tuning guide. Use the Navigation Rules table \
     to decide your approach based on prior leaderboard results.
  2. Decide an approach: start from the search space dimensions (kernel architecture, \
     memory access, low-rank path, compute unit, tile shape). DO NOT copy losing candidates verbatim.
  3. Call write_candidate(source) — wait for the returned candidate_id.
  4. Call evaluate_candidate(candidate_id, d_list, mode="quick") with the d_list from \
     the candidate_quick spec.
       - compile_ok=False → analyze compile_error; write a fixed candidate (≤2 retries).
       - all_correct=False → fix logic; ≤2 retries.
       - speedup_median ≤ 1.0 → apply the matching Navigation Rule; submit as \
         accepted_for="strategy_guidance" and stop.
  5. If quick passed and speedup > 1.0, call evaluate_candidate(..., mode="confirm").
       - variance_pct > 15 → submit as accepted_for="strategy_guidance".
       - else if confirm speedup > current best → submit accepted_for="best_update".
       - else submit accepted_for="quick_ranking".
  6. Call submit_candidate_result(record) — exactly once, then stop.

REQUIRED FIELDS in record (CandidateRecord):
  candidate_id, compile_ok, correctness_ok, accepted_for
  + quick/confirm speedup_median + samples + variance_pct when measured.

ENGINEERING DISCIPLINE
  - Don't claim best_update when compile or correctness failed.
  - Don't fabricate speedup numbers — pull them from evaluate_candidate's summary.
  - One submit_candidate_result per stage. After it the loop ends.
"""


class KernelTuningAgent(LLMStageAgent):
    stage = Stage.INITIAL_CANDIDATE  # may be re-tagged to TUNING_LOOP at construction
    allowed_tools = (
        "write_candidate",
        "evaluate_candidate",
        "submit_candidate_result",
        "flag_event",
    )
    max_iterations = 30
    SYSTEM_PROMPT = _SYSTEM_PROMPT

    def __init__(self, llm, *, agent_cfg=None, verbose=False, stage: Stage | None = None):
        super().__init__(llm, agent_cfg=agent_cfg, verbose=verbose)
        if stage is not None:
            # Same agent class drives INITIAL_CANDIDATE and TUNING_LOOP; the
            # orchestrator tells us which stage we're in so submit_candidate_result
            # tags the StageResult correctly (StageToolFactory passes current_stage
            # at build time as well).
            self.stage = stage

    def build_user_message(self, context: StageContext) -> str:
        rs = context.run_state
        layout = context.layout

        baseline_summary = "(baseline.json not found)"
        if layout.has_baseline():
            try:
                baseline_summary = layout.baseline_path.read_text(encoding="utf-8")[:2000]
            except OSError:
                pass

        quick_spec = self._read_spec(layout, "candidate_quick")
        confirm_spec = self._read_spec(layout, "candidate_confirm")

        lb_tail = "(empty)"
        if layout.leaderboard_path.is_file():
            lines = layout.leaderboard_path.read_text(encoding="utf-8").strip().splitlines()
            lb_tail = "\n".join(lines[-5:]) if lines else "(empty)"

        existing_cands = []
        if layout.candidates_dir.is_dir():
            for c in sorted(layout.candidates_dir.iterdir()):
                if c.is_dir():
                    existing_cands.append(c.name)

        best_block = "(no best yet)"
        if rs.best_candidate_id:
            best_block = f"id={rs.best_candidate_id} speedup={rs.best_speedup}"

        op_block = context.op_spec.summary_for_prompt() if context.op_spec else "(operator contract unavailable)"
        tuning_guide = self._read_tuning_guide(context)
        return (
            f"{op_block}\n\n"
            f"=== tuning guide ===\n{tuning_guide}\n\n"
            f"Stage: {self.stage.value}\n"
            f"Iteration: {rs.current_iteration}\n"
            f"Current best: {best_block}\n"
            f"Existing candidates: {existing_cands}\n\n"
            f"=== candidate_quick spec ===\n{quick_spec}\n\n"
            f"=== candidate_confirm spec ===\n{confirm_spec}\n\n"
            f"=== baseline.json (truncated) ===\n{baseline_summary}\n\n"
            f"=== leaderboard tail ===\n{lb_tail}\n\n"
            "Produce ONE new candidate, evaluate it, and submit a CandidateRecord."
        )

    @staticmethod
    def _read_tuning_guide(context: StageContext) -> str:
        op_name = context.op_spec.name if context.op_spec else ""
        path = _SKILLS_DIR / "operators" / f"{op_name}_tuning.md"
        if not path.is_file():
            return "(tuning guide not found)"
        text = path.read_text(encoding="utf-8")
        # Strip YAML frontmatter so the agent sees clean Markdown.
        if text.startswith("---"):
            end = text.find("---", 3)
            if end != -1:
                text = text[end + 3:].lstrip()
        return text

    @staticmethod
    def _read_spec(layout, slot: str) -> str:
        p = layout.benchmark_spec_path(slot)
        if not p.is_file():
            return "(spec not found)"
        try:
            return json.dumps(json.loads(p.read_text(encoding="utf-8")), indent=2)
        except json.JSONDecodeError:
            return p.read_text(encoding="utf-8")
