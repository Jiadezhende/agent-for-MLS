"""Correctness fixtures + PyTorch reference latency.

Two responsibilities, two functions:

* ``build_correctness_fixtures`` — generate per-shape input tensors AND
  the PyTorch reference output (oracle). Used by both candidate
  correctness checks and the latency measurement that follows.
* ``measure_pytorch_latency`` — time the PyTorch reference implementation
  on the inputs already on disk; returns per-shape median ms + the
  speedup denominator.

Both run in-process via ``OperatorOps``: no string-template subprocess
indirection. Baseline only invokes ``torch`` ops and never loads a
candidate ``.cu`` file, so there is no CUDA-context contamination risk
to justify subprocess isolation.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from operator_opt_pipe.operators._base import OperatorOps
from operator_opt_pipe.resources.benchmark import BenchmarkSpec


@dataclass(frozen=True)
class BaselineResult:
    """Per-shape PyTorch reference timings + speedup denominator."""

    spec: BenchmarkSpec
    per_shape: dict[str, dict[str, Any]]
    ms_median_overall: float
    reference_pytorch: str

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "per_shape": self.per_shape,
            "ms_median_overall": self.ms_median_overall,
            "reference_pytorch": self.reference_pytorch,
        }


def build_correctness_fixtures(
    ops: OperatorOps,
    spec: BenchmarkSpec,
    inputs_dir: Path,
    oracle_dir: Path,
) -> dict:
    """Generate per-shape inputs + oracle outputs.

    For each shape in ``spec.shape_grid``: draw inputs via
    ``ops.make_inputs`` (single seeded generator advanced across shapes,
    matching the old ``torch.manual_seed + randn`` sequencing), compute
    the oracle via ``ops.reference``, save both to disk.

    Returns ``{"shape_ids": [...]}``.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("build_correctness_fixtures: cuda_unavailable")

    inputs_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(spec.seed))

    # TF32 on Ampere truncates float32 mantissa to 10 bits, introducing
    # max abs errors ~0.1 for d≥3584 matmuls — far above atol=1e-4.
    # Disable for oracle generation so custom FP32 kernels can pass.
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False

    shape_ids: list[str] = []
    try:
        for d in spec.shape_grid:
            inputs = ops.make_inputs(int(d), device=device, generator=generator)
            sid = ops.shape_id(int(d))
            ops.save_inputs(inputs, inputs_dir, sid)
            with torch.no_grad():
                Y = ops.reference(inputs)
            ops.save_oracle(Y, oracle_dir, sid)
            shape_ids.append(sid)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32

    return {"shape_ids": shape_ids}


def measure_pytorch_latency(
    ops: OperatorOps,
    spec: BenchmarkSpec,
    inputs_dir: Path,
) -> BaselineResult:
    """Time the PyTorch reference implementation per shape.

    Reads inputs already materialized by ``build_correctness_fixtures``;
    does NOT recompute or save the oracle.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("measure_pytorch_latency: cuda_unavailable")

    device = torch.device("cuda")
    per_shape: dict[str, dict[str, Any]] = {}

    for d in spec.shape_grid:
        sid = ops.shape_id(int(d))
        inputs = ops.load_inputs(inputs_dir, sid, device=device)

        with torch.no_grad():
            for _ in range(int(spec.warmup)):
                _ = ops.reference(inputs)
            torch.cuda.synchronize()

            times: list[float] = []
            for _ in range(int(spec.samples)):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = ops.reference(inputs)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))

        per_shape[sid] = {
            "ms_median": float(statistics.median(times)),
            "ms_min": float(min(times)),
            "ms_max": float(max(times)),
            "samples": int(spec.samples),
        }

    medians = [v["ms_median"] for v in per_shape.values()]
    overall = float(statistics.median(medians)) if medians else 0.0
    return BaselineResult(
        spec=spec,
        per_shape=per_shape,
        ms_median_overall=overall,
        reference_pytorch=ops.reference_doc(),
    )


def run_pytorch_baseline(
    ops: OperatorOps,
    spec: BenchmarkSpec,
    inputs_dir: Path,
    oracle_dir: Path,
) -> BaselineResult:
    """One-shot helper: build fixtures, then measure latency."""
    build_correctness_fixtures(
        ops=ops, spec=spec, inputs_dir=inputs_dir, oracle_dir=oracle_dir,
    )
    return measure_pytorch_latency(ops=ops, spec=spec, inputs_dir=inputs_dir)


__all__ = [
    "BaselineResult",
    "build_correctness_fixtures",
    "measure_pytorch_latency",
    "run_pytorch_baseline",
]
