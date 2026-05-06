"""PyTorch reference-implementation baseline. **Stub — filled in next PR.**

Will run the standard PyTorch implementation (``Y = W @ X + A @ (B.T @ X)``)
under the same benchmark harness used for candidates and record per-d median
ms in ``baseline.json``. Speedup is defined as
``pytorch_ms_median / candidate_ms_median`` and is the only ranking metric.
"""
from __future__ import annotations

from dataclasses import dataclass

from operator_opt_pipe.lora_resources.benchmark import BenchmarkSpec
from operator_opt_pipe.lora_resources.contract import LoRAContract
from operator_opt_pipe.state import RunLayout


@dataclass(frozen=True)
class BaselineResult:
    """Per-d PyTorch reference timings.

    ``per_d`` maps each d to ``{"ms_median": ..., "ms_min": ..., "ms_max": ...,
    "samples": int}``. ``ms_median_overall`` is the single number used as the
    speedup denominator.
    """

    spec: BenchmarkSpec
    per_d: dict[int, dict]
    ms_median_overall: float

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "per_d": {str(k): v for k, v in self.per_d.items()},
            "ms_median_overall": self.ms_median_overall,
            "standard_impl": "Y = W @ X + A @ (B.T @ X)",
        }


def run_pytorch_baseline(
    layout: RunLayout,
    executor,
    contract: LoRAContract,
    spec: BenchmarkSpec,
) -> BaselineResult:
    """Time the PyTorch reference and persist ``layout.baseline_path``.

    Stub — the next PR ports the cudaEvent-based timing subprocess from
    ``pipeline/tools/baseline_tools.py:_build_baseline_script``.
    """
    raise NotImplementedError(
        "operator_opt_pipe.lora_resources.baseline.run_pytorch_baseline is a "
        "stub; it will be implemented in the next PR (operator_opt_pipe-impl)."
    )
