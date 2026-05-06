"""Benchmark spec + synthetic input generation. **Stub — filled in next PR.**

The next iteration will port the subprocess script from
``pipeline/tools/baseline_tools.py`` so it materializes ``W/X/A/B`` tensors
and saves them under ``layout.baseline_dir/inputs/``. For now this module
only fixes the function signatures so the orchestrator can wire them up.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from operator_opt_pipe.lora_resources.contract import LoRAContract
from operator_opt_pipe.state import RunLayout


@dataclass(frozen=True)
class BenchmarkSpec:
    d_list: tuple[int, ...]
    samples: int
    warmup: int
    seed: int
    dtype: str
    device: str

    def to_dict(self) -> dict:
        return {
            "d_list": list(self.d_list),
            "samples": self.samples,
            "warmup": self.warmup,
            "seed": self.seed,
            "dtype": self.dtype,
            "device": self.device,
        }


def generate_benchmark_spec(
    contract: LoRAContract,
    *,
    d_list: tuple[int, ...] | None = None,
    samples: int = 30,
    warmup: int = 2,
    seed: int = 0,
) -> BenchmarkSpec:
    """Pick a fixed set of d values inside ``contract.d_range`` and return a spec.

    Default ``d_list`` covers both ends of the range plus a midpoint so a
    single benchmark run gives the LLM enough signal to detect d-sensitive
    bottlenecks (low-rank correction is dominant for smaller d).
    """
    if d_list is None:
        lo, hi = contract.d_range
        mid = (lo + hi) // 2
        d_list = (lo, mid, hi)
    return BenchmarkSpec(
        d_list=tuple(d_list),
        samples=int(samples),
        warmup=int(warmup),
        seed=int(seed),
        dtype=contract.dtype,
        device=contract.device,
    )


def materialize_inputs(layout: RunLayout, contract: LoRAContract, spec: BenchmarkSpec) -> dict[str, Path]:
    """Write ``W/X/A/B`` and ``Y_ref`` tensors for each d in ``spec.d_list``.

    Returns a mapping ``"<tensor>_d{d}" → Path``. Stub — the implementation
    will spawn a torch subprocess via ``mls_agent.tools.cuda.cuda_executor``.
    """
    raise NotImplementedError(
        "operator_opt_pipe.lora_resources.benchmark.materialize_inputs is a "
        "stub; it will be implemented in the next PR (operator_opt_pipe-impl)."
    )
