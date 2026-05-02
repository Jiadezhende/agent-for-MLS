"""pipeline/agents/kernel_tuning_agent.py — INITIAL_CANDIDATE / TUNING_LOOP stage agent."""
from __future__ import annotations

import json

from ..stage_runner import StageContext
from ..state import Stage
from ._base import LLMStageAgent


_SYSTEM_PROMPT = """You are the KernelTuningAgent. Each invocation must produce ONE candidate \
CUDA kernel and submit a CandidateRecord describing how it performed.

The operator contract (formula, tensor shapes, forward signature, shape range) is \
injected into the user message below — treat it as authoritative. The candidate \
.cu file must export the declared forward signature via PYBIND11_MODULE so that \
torch.utils.cpp_extension.load can compile it.

WORKFLOW
  1. Read the injected operator contract; consult the operator skill via read_skill for strategy.
  2. Read prior leaderboard / strategy-guidance candidates for context (their dirs are \
     under candidates/). DO NOT copy losing candidates verbatim.
  3. Decide an approach (fused vs split, tiled GEMM + low-rank correction, vectorization, \
     tensor cores when applicable).
  4. Call write_candidate(source) — wait for the returned candidate_id.
  5. Call evaluate_candidate(candidate_id, d_list, mode="quick") with the d_list from \
     the candidate_quick spec.
       - compile_ok=False → analyze compile_error; write a fixed candidate (≤2 retries).
       - all_correct=False → fix logic; ≤2 retries.
       - speedup_median ≤ 1.0 → submit as accepted_for="strategy_guidance" and stop.
  6. If quick passed and speedup > 1.0, call evaluate_candidate(..., mode="confirm").
       - variance_pct > 15 → submit as accepted_for="strategy_guidance".
       - else if confirm speedup > current best → submit accepted_for="best_update".
       - else submit accepted_for="quick_ranking".
  7. Call submit_candidate_result(record) — exactly once, then stop.

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
        "list_skills",
        "read_skill",
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

        # Read what we have on disk that the agent should consider.
        baseline_summary = "(baseline.json not found)"
        if layout.has_baseline():
            try:
                baseline_summary = layout.baseline_path.read_text(encoding="utf-8")[:2000]
            except OSError:
                pass

        quick_spec = self._read_spec(layout, "candidate_quick")
        confirm_spec = self._read_spec(layout, "candidate_confirm")

        # Leaderboard digest: just the last few entries so prompt size stays bounded.
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
        return (
            f"{op_block}\n\n"
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
    def _read_spec(layout, slot: str) -> str:
        p = layout.benchmark_spec_path(slot)
        if not p.is_file():
            return "(spec not found)"
        try:
            return json.dumps(json.loads(p.read_text(encoding="utf-8")), indent=2)
        except json.JSONDecodeError:
            return p.read_text(encoding="utf-8")
