"""pipeline/agents/benchmark_spec_agent.py — BENCHMARK_SPEC stage agent."""
from __future__ import annotations

from ..stage_runner import StageContext
from ..state import BENCHMARK_SPEC_SLOTS, Stage
from ._base import LLMStageAgent


_SYSTEM_PROMPT = """You are the BenchmarkSpecAgent, the first stage of an autonomous LoRA-kernel \
optimization pipeline.

GOAL
Produce five benchmark specs that downstream stages will obey verbatim. Submit \
all of them in ONE call to submit_benchmark_specs.

REQUIRED SLOTS
  hardware            — what hardware probes to run (DRAM, clock, SM, L2, latencies)
  baseline            — d_list + samples for the PyTorch reference benchmark
  candidate_quick     — d_list + samples (small, e.g. 5) for fast filtering
  candidate_confirm   — d_list + samples (e.g. 30) before promoting to best
  final               — d_list + samples for the final report benchmark

SHAPE OF EACH SPEC (JSON object)
  d_list:  list of int (subset of [3584, 4608] inclusive)
  samples: int
  notes:   short string explaining the choice

WORKFLOW
  1. Call read_skill("operators/lora_matmul") to get the operator definition.
  2. Decide d_list and samples per slot.
  3. Call submit_benchmark_specs once with all five.

DO NOT call any other submit_* tool in this stage.
DO NOT exceed two LLM round-trips before submitting — this stage is mechanical.
"""


class BenchmarkSpecAgent(LLMStageAgent):
    stage = Stage.BENCHMARK_SPEC
    allowed_tools = ("list_skills", "read_skill", "submit_benchmark_specs", "flag_event")
    max_iterations = 8
    SYSTEM_PROMPT = _SYSTEM_PROMPT

    def build_user_message(self, context: StageContext) -> str:
        op = context.run_state.operator
        return (
            f"Operator: {op}\n"
            f"Required slots (must all be in your submit_benchmark_specs call): "
            f"{list(BENCHMARK_SPEC_SLOTS)}\n\n"
            f"Read the operator skill (operators/{op}) and submit a complete spec set."
        )
