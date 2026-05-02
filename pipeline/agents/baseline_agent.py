"""pipeline/agents/baseline_agent.py — BASELINE_PROFILE stage agent."""
from __future__ import annotations

import json

from ..stage_runner import StageContext
from ..state import Stage
from ._base import LLMStageAgent


_SYSTEM_PROMPT = """You are the BaselineAgent. Generate the operator's input tensors and the \
reference output that KernelTuningAgent needs for correctness checks, and \
benchmark the PyTorch reference latency that defines the speedup denominator.

The operator contract (formula, tensor list, forward signature, shape range) is \
injected into the user message below. Read the corresponding operator skill for \
strategy context if needed.

WORKFLOW
  1. Read the injected operator contract (and optionally the operator skill via read_skill).
  2. Read the baseline benchmark spec at the path given below for d_list + samples.
  3. Call generate_baseline(d_list, samples) — the tool generates the operator \
inputs + reference output and benchmarks the PyTorch reference for each d \
in one subprocess.
  4. Inspect the per-d torch_ms_median values. If any are missing or 0, flag \
and retry once.
  5. Call submit_baseline(per_d, notes) to finalize.

DO NOT call evaluate_candidate or write_candidate here — those are KernelTuningAgent's job.
"""


class BaselineAgent(LLMStageAgent):
    stage = Stage.BASELINE_PROFILE
    allowed_tools = (
        "list_skills",
        "read_skill",
        "generate_baseline",
        "submit_baseline",
        "flag_event",
    )
    max_iterations = 12
    SYSTEM_PROMPT = _SYSTEM_PROMPT

    def build_user_message(self, context: StageContext) -> str:
        spec_path = context.layout.benchmark_spec_path("baseline")
        spec_blob = ""
        if spec_path.is_file():
            try:
                spec_blob = json.dumps(
                    json.loads(spec_path.read_text(encoding="utf-8")), indent=2
                )
            except json.JSONDecodeError:
                spec_blob = spec_path.read_text(encoding="utf-8")
        op_block = context.op_spec.summary_for_prompt() if context.op_spec else "(operator contract unavailable)"
        return (
            f"{op_block}\n\n"
            "Generate inputs + references and benchmark the PyTorch reference. "
            "Inputs go to baseline/inputs/, references to baseline/references/.\n\n"
            f"=== baseline spec ===\n{spec_blob or '(spec not found)'}\n"
        )
