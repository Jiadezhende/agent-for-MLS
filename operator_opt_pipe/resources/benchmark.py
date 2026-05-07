"""Benchmark spec.

The spec is a small, in-memory dataclass shared by baseline and
candidate-evaluation. It carries no operator-specific knowledge — each
function that uses it pulls inputs from the contract.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


_MARKER_PREFIX = "==="


@dataclass(frozen=True)
class BenchmarkSpec:
    """The single configuration object for benchmark + baseline.

    ``shape_grid`` is the list of shape-parameter values to test; for LoRA
    this is e.g. ``(3584, 4096, 4608)``. Tolerances live on the contract.
    """

    shape_grid: tuple[int, ...]
    samples: int = 30
    warmup: int = 5
    seed: int = 0

    def to_dict(self) -> dict:
        return {
            "shape_grid": list(self.shape_grid),
            "samples": self.samples,
            "warmup": self.warmup,
            "seed": self.seed,
        }

    @classmethod
    def for_contract(cls, contract, **overrides) -> "BenchmarkSpec":
        grid = overrides.pop("shape_grid", None) or contract.default_shape_grid()
        return cls(
            shape_grid=tuple(grid),
            samples=int(overrides.get("samples", 30)),
            warmup=int(overrides.get("warmup", 5)),
            seed=int(overrides.get("seed", 0)),
        )


# Kept here for backwards-compat with the previous module surface (still
# used by baseline.py and evaluation.py to parse marked subprocess output).
def parse_marked_output(job, marker: str) -> dict:
    """Parse the JSON object printed after ``marker`` in a profile_with_torch job."""
    if isinstance(job, dict):
        stdout = job.get("stdout", "") or ""
        returncode = job.get("returncode")
        timed_out = job.get("timed_out", False)
        stderr = job.get("stderr", "") or ""
    else:
        summary = getattr(job, "summary", None) or {}
        stdout = summary.get("stdout", "") or ""
        returncode = summary.get("returncode")
        timed_out = summary.get("timed_out", False)
        stderr = summary.get("stderr", "") or ""

    if timed_out:
        raise RuntimeError(f"subprocess timed out before emitting {marker!r}")

    if marker not in stdout:
        tail = (stderr[-800:] if stderr else stdout[-800:]) or "<empty>"
        raise RuntimeError(
            f"subprocess output missing marker {marker!r} "
            f"(returncode={returncode}); tail:\n{tail}"
        )
    tail = stdout.split(marker, 1)[1].strip()
    for line in tail.splitlines():
        line = line.strip()
        if line:
            try:
                return json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"subprocess marker {marker!r} not followed by valid JSON: "
                    f"{line[:200]}"
                ) from exc
    raise RuntimeError(f"subprocess marker {marker!r} not followed by any output")


# Backward-compat alias for code still expecting the old generate_*
# function name (within this package).
def generate_benchmark_spec(contract, *, shape_grid=None, samples=30, warmup=5, seed=0):
    return BenchmarkSpec.for_contract(
        contract, shape_grid=shape_grid, samples=samples, warmup=warmup, seed=seed,
    )


__all__ = [
    "BenchmarkSpec",
    "generate_benchmark_spec",
    "parse_marked_output",
]
