"""Candidate compile + correctness + perf benchmark. **Stub — filled in next PR.**

This module is the only code that turns a ``candidate.cu`` file into a
speedup number. Importantly, it is NOT exposed as an LLM tool — only the
orchestrator and ``RoundRunner`` call it. LLM agents see the result indirectly
through the leaderboard / blackboard.
"""
from __future__ import annotations

from dataclasses import dataclass

from operator_opt_pipe.state import RunLayout


@dataclass(frozen=True)
class EvalResult:
    candidate_id: str
    compile_ok: bool
    correctness_ok: bool
    candidate_ms_median: float | None
    speedup: float | None                  # baseline_ms_median / candidate_ms_median
    diagnostics: dict
    samples: int

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "compile_ok": self.compile_ok,
            "correctness_ok": self.correctness_ok,
            "candidate_ms_median": self.candidate_ms_median,
            "speedup": self.speedup,
            "samples": self.samples,
            "diagnostics": self.diagnostics,
        }


def evaluate_candidate(
    layout: RunLayout,
    executor,
    candidate_id: str,
    baseline_ms_median: float,
    *,
    samples: int = 30,
) -> EvalResult:
    """Compile candidate → correctness → benchmark → ``EvalResult``.

    Stub — the next PR ports the cpp_extension.load + cudaEvent harness from
    ``pipeline/tools/candidate_tools.py:_build_eval_script``. The harness must
    use the same compile/load invocation as the Phase-2 evaluator
    (``extra_cuda_cflags=["-O3"]``) so local "best" matches scoring "best".
    """
    raise NotImplementedError(
        "operator_opt_pipe.lora_resources.evaluation.evaluate_candidate is a "
        "stub; it will be implemented in the next PR (operator_opt_pipe-impl)."
    )
