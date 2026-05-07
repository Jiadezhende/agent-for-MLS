"""Correctness fixtures + PyTorch reference latency.

Two responsibilities, two functions:

* ``build_correctness_fixtures`` — generate per-shape input tensors AND
  the PyTorch reference output (oracle). Used by both candidate
  correctness checks and the latency measurement that follows.
* ``measure_pytorch_latency`` — time the PyTorch reference implementation
  on the inputs already on disk; returns per-shape median ms + the
  speedup denominator.

Both are operator-agnostic via ``contract.render_*`` helpers.
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
    parse_marked_output,
)
from operator_opt_pipe.resources.contract import OperatorContract


_FIXTURES_MARKER = "=== FIXTURES_RESULT ==="
_LATENCY_MARKER = "=== LATENCY_RESULT ==="


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
    contract: OperatorContract,
    spec: BenchmarkSpec,
    inputs_dir: Path,
    oracle_dir: Path,
    *,
    executor,
) -> dict:
    """Generate per-shape inputs + oracle outputs.

    For each shape in ``spec.shape_grid``:
      - draw all ``contract.inputs`` tensors with seeded torch.randn → save to ``inputs_dir``
      - compute ``contract.reference_pytorch`` once → save to ``oracle_dir``

    Returns ``{"shape_ids": [...]}``.
    """
    inputs_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    script = _build_fixtures_script(
        contract=contract, inputs_dir=inputs_dir, oracle_dir=oracle_dir, spec=spec,
    )
    job = executor.profile_with_torch(
        python_code=script,
        op_name=f"fixtures/{contract.name.replace('/', '_')}",
        timeout_s=300,
    )
    raw = parse_marked_output(job, _FIXTURES_MARKER)
    if "error" in raw:
        raise RuntimeError(f"build_correctness_fixtures: {raw['error']}")
    return raw


def measure_pytorch_latency(
    contract: OperatorContract,
    spec: BenchmarkSpec,
    inputs_dir: Path,
    *,
    executor,
) -> BaselineResult:
    """Time the PyTorch reference implementation per shape.

    Reads inputs already materialized by ``build_correctness_fixtures``;
    does NOT recompute or save the oracle.
    """
    script = _build_latency_script(contract=contract, inputs_dir=inputs_dir, spec=spec)
    job = executor.profile_with_torch(
        python_code=script,
        op_name=f"baseline/{contract.name.replace('/', '_')}",
        timeout_s=600,
    )
    raw = parse_marked_output(job, _LATENCY_MARKER)
    if "error" in raw:
        raise RuntimeError(f"measure_pytorch_latency: {raw['error']}")

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


def run_pytorch_baseline(
    contract: OperatorContract,
    spec: BenchmarkSpec,
    inputs_dir: Path,
    oracle_dir: Path,
    *,
    executor,
) -> BaselineResult:
    """One-shot helper: build fixtures, then measure latency."""
    build_correctness_fixtures(
        contract=contract, spec=spec,
        inputs_dir=inputs_dir, oracle_dir=oracle_dir, executor=executor,
    )
    return measure_pytorch_latency(
        contract=contract, spec=spec, inputs_dir=inputs_dir, executor=executor,
    )


# ---------------------------------------------------------------------------
# Subprocess script generators
# ---------------------------------------------------------------------------


def _build_fixtures_script(
    *,
    contract: OperatorContract,
    inputs_dir: Path,
    oracle_dir: Path,
    spec: BenchmarkSpec,
) -> str:
    shape_var = contract.shape_param
    inp_dir_s = inputs_dir.resolve().as_posix()
    ora_dir_s = oracle_dir.resolve().as_posix()
    grid_repr = json.dumps(list(spec.shape_grid))

    create_inputs = textwrap.indent(
        contract.render_input_creation(device_var="device"), "    ",
    )
    save_inputs = textwrap.indent(
        contract.render_save_inputs(dir_var="INPUT_DIR", shape_id_expr="shape_id"), "    ",
    )
    compute_ref = textwrap.indent(contract.render_reference_compute(), "    ")
    save_ref = textwrap.indent(
        contract.render_save_reference(dir_var="ORACLE_DIR", shape_id_expr="shape_id"), "    ",
    )

    return f'''import json
import os
import sys

import torch

INPUT_DIR  = r"{inp_dir_s}"
ORACLE_DIR = r"{ora_dir_s}"
SHAPE_GRID = {grid_repr}
SEED = {int(spec.seed)}

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(ORACLE_DIR, exist_ok=True)

if not torch.cuda.is_available():
    print("{_FIXTURES_MARKER}")
    print(json.dumps({{"error": "cuda_unavailable"}}))
    sys.exit(0)

device = torch.device("cuda")
torch.manual_seed(SEED)

shape_ids = []
for {shape_var} in SHAPE_GRID:
    shape_id = f"{shape_var}{{{shape_var}}}"
{create_inputs}
{save_inputs}
{compute_ref}
{save_ref}
    shape_ids.append(shape_id)

print("{_FIXTURES_MARKER}")
print(json.dumps({{"shape_ids": shape_ids}}))
'''


def _build_latency_script(
    *,
    contract: OperatorContract,
    inputs_dir: Path,
    spec: BenchmarkSpec,
) -> str:
    shape_var = contract.shape_param
    inp_dir_s = inputs_dir.resolve().as_posix()
    grid_repr = json.dumps(list(spec.shape_grid))

    load_inputs = textwrap.indent(
        contract.render_load_inputs(dir_var="INPUT_DIR", shape_id_expr="shape_id"), "    ",
    )
    ref_expr = contract.reference_pytorch

    return f'''import json
import os
import statistics
import sys

import torch

INPUT_DIR = r"{inp_dir_s}"
SHAPE_GRID = {grid_repr}
SAMPLES = {int(spec.samples)}
WARMUP  = {int(spec.warmup)}

if not torch.cuda.is_available():
    print("{_LATENCY_MARKER}")
    print(json.dumps({{"error": "cuda_unavailable"}}))
    sys.exit(0)

device = torch.device("cuda")

per_shape = []
for {shape_var} in SHAPE_GRID:
    shape_id = f"{shape_var}{{{shape_var}}}"
{load_inputs}

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

print("{_LATENCY_MARKER}")
print(json.dumps({{"per_shape": per_shape}}))
'''


__all__ = [
    "BaselineResult",
    "build_correctness_fixtures",
    "measure_pytorch_latency",
    "run_pytorch_baseline",
]
