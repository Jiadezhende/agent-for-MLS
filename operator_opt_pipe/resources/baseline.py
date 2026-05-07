"""PyTorch reference baseline — speedup denominator + correctness oracle.

Runs ``contract.reference_pytorch`` over every shape in ``spec.shape_grid``,
saves the reference output tensor (oracle for candidate correctness), and
records cudaEvent-timed median latency (per-shape). Operator-agnostic via
``contract.render_*`` helpers.
"""
from __future__ import annotations

import json
import statistics
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from operator_opt_pipe.resources.benchmark import (
    BenchmarkSpec,
    _parse_marked_output,
)
from operator_opt_pipe.resources.contract import OperatorContract


_BASELINE_MARKER = "=== BASELINE_RESULT ==="


@dataclass(frozen=True)
class BaselineResult:
    """Per-shape PyTorch reference timings.

    ``per_shape`` is keyed by shape_id (e.g. ``"d3584"``) → ``{ms_median,
    ms_min, ms_max, samples}``. ``ms_median_overall`` is the median of
    per-shape medians (used as a single-number speedup baseline).
    """

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


def run_pytorch_baseline(
    contract: OperatorContract,
    spec: BenchmarkSpec,
    inputs_dir: Path,
    references_dir: Path,
    *,
    executor,
) -> BaselineResult:
    """Time the PyTorch reference and save the oracle tensors.

    Reads inputs from ``inputs_dir`` (must already be materialized via
    ``benchmark.materialize_inputs``) and writes ``references/<output>_<shape_id>.pt``
    plus per-shape median latency. Spawns a single torch subprocess.
    """
    references_dir.mkdir(parents=True, exist_ok=True)
    script = _build_baseline_script(
        contract=contract,
        inputs_dir=inputs_dir,
        references_dir=references_dir,
        spec=spec,
    )
    job = executor.profile_with_torch(
        python_code=script,
        op_name=f"baseline/{contract.name.replace('/', '_')}",
        timeout_s=600,
    )
    raw = _parse_marked_output(job, _BASELINE_MARKER)
    if "error" in raw:
        raise RuntimeError(f"baseline subprocess error: {raw['error']}")

    per_shape: dict[str, dict[str, Any]] = {}
    for entry in raw["per_shape"]:
        sid = entry["shape_id"]
        per_shape[sid] = {
            "ms_median": float(entry["ms_median"]),
            "ms_min": float(entry["ms_min"]),
            "ms_max": float(entry["ms_max"]),
            "samples": int(entry["samples"]),
        }

    medians = [v["ms_median"] for v in per_shape.values()]
    overall = float(statistics.median(medians)) if medians else 0.0

    return BaselineResult(
        spec=spec,
        per_shape=per_shape,
        ms_median_overall=overall,
        reference_pytorch=contract.reference_pytorch,
    )


def _build_baseline_script(
    *,
    contract: OperatorContract,
    inputs_dir: Path,
    references_dir: Path,
    spec: BenchmarkSpec,
) -> str:
    """Render the per-shape benchmark script using contract helpers."""
    shape_var = contract.shape_param
    inputs_dir_s = inputs_dir.resolve().as_posix()
    refs_dir_s = references_dir.resolve().as_posix()
    grid_repr = json.dumps(list(spec.shape_grid))

    load_inputs = textwrap.indent(
        contract.render_load_inputs(dir_var="INPUT_DIR", shape_id_expr="shape_id"),
        "    ",
    )
    compute_ref = textwrap.indent(contract.render_reference_compute(), "    ")
    save_ref = textwrap.indent(
        contract.render_save_reference(dir_var="REF_DIR", shape_id_expr="shape_id"),
        "    ",
    )
    ref_expr = contract.reference_pytorch

    return f'''import json
import os
import statistics
import sys

import torch

INPUT_DIR = r"{inputs_dir_s}"
REF_DIR   = r"{refs_dir_s}"
SHAPE_GRID = {grid_repr}
SAMPLES   = {int(spec.samples)}
WARMUP    = {int(spec.warmup)}

os.makedirs(REF_DIR, exist_ok=True)

if not torch.cuda.is_available():
    print("{_BASELINE_MARKER}")
    print(json.dumps({{"error": "cuda_unavailable"}}))
    sys.exit(0)

device = torch.device("cuda")

per_shape = []
for {shape_var} in SHAPE_GRID:
    shape_id = f"{shape_var}{{{shape_var}}}"
{load_inputs}

{compute_ref}
{save_ref}

    # Warmup
    for _ in range(WARMUP):
        _ = {ref_expr}
    torch.cuda.synchronize()

    times = []
    for _ in range(SAMPLES):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _ = {ref_expr}
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    per_shape.append({{
        "shape_id": shape_id,
        "ms_median": float(statistics.median(times)),
        "ms_min":    float(min(times)),
        "ms_max":    float(max(times)),
        "samples":   int(SAMPLES),
    }})

print("{_BASELINE_MARKER}")
print(json.dumps({{"per_shape": per_shape}}))
'''


__all__ = [
    "BaselineResult",
    "run_pytorch_baseline",
]
