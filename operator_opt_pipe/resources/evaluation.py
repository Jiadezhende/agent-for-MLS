"""Candidate compile + correctness + multi-shape benchmark.

Two entry points:

* ``compile_and_check_quick`` — ``write_candidate`` calls this to give the
  LLM fast feedback. Runs ``cpp_extension.load`` once and verifies
  correctness on a single shape (the smallest one in the grid). Reports
  compile errors verbatim and the worst per-element error.

* ``benchmark_on_grid`` — orchestrator calls this after every accepted
  ``submit_candidate``. Runs the candidate on every shape in
  ``spec.shape_grid``, checks correctness against the saved oracle, and
  measures cudaEvent latency. The agent never sees this result; the
  orchestrator owns promotion decisions.

Both use ``torch.utils.cpp_extension.load`` with ``extra_cuda_cflags=["-O3"]``
to mirror the official Phase-2 evaluation harness — local "best" matches
scoring "best".

The subprocess script is a fixed template that imports the operator
``OPS`` instance via ``operator_opt_pipe.operators.load_ops`` at runtime;
all operator-specific behaviour (input load, forward call, oracle load)
goes through ``OPS`` methods. No string-template rendering of operator
formulae.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from operator_opt_pipe.operators._base import OperatorOps
from operator_opt_pipe.resources.benchmark import (
    BenchmarkSpec,
    parse_marked_output,
)


_QUICK_MARKER = "=== QUICK_RESULT ==="
_BENCH_MARKER = "=== BENCH_RESULT ==="

# Project root — added to sys.path inside subprocess scripts so that the
# generated subprocess (cwd=workspace/exec) can ``import operator_opt_pipe``.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuickEvalResult:
    """Quick (single-shape) feedback for write_candidate."""

    candidate_id: str
    compile_ok: bool
    correctness_ok: bool
    shape_id: str
    max_abs_err: float | None
    rel_l2_err: float | None
    compile_log: str
    diagnostics: dict

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "compile_ok": self.compile_ok,
            "correctness_ok": self.correctness_ok,
            "shape_id": self.shape_id,
            "max_abs_err": self.max_abs_err,
            "rel_l2_err": self.rel_l2_err,
            "compile_log": self.compile_log,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """Full multi-shape benchmark — orchestrator-only."""

    candidate_id: str
    compile_ok: bool
    per_shape: dict[str, dict[str, Any]]
    speedup_geomean: float | None
    speedup_worst: float | None
    speedup_best: float | None
    correctness_per_shape: dict[str, bool]
    all_correct: bool
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "compile_ok": self.compile_ok,
            "per_shape": self.per_shape,
            "speedup_geomean": self.speedup_geomean,
            "speedup_worst": self.speedup_worst,
            "speedup_best": self.speedup_best,
            "correctness_per_shape": self.correctness_per_shape,
            "all_correct": self.all_correct,
            "diagnostics": self.diagnostics,
        }

    # Convenience for orchestrator promotion logic
    @property
    def correctness_ok(self) -> bool:
        return self.all_correct

    @property
    def speedup(self) -> float | None:
        """Single number for promotion comparison: geomean across shapes."""
        return self.speedup_geomean


# ---------------------------------------------------------------------------
# Quick eval (single-shape, compile-focused)
# ---------------------------------------------------------------------------


def compile_and_check_quick(
    ops: OperatorOps,
    candidate_id: str,
    candidate_cu: Path,
    inputs_dir: Path,
    oracle_dir: Path,
    sample_shape: int,
    build_dir: Path,
    *,
    executor,
) -> QuickEvalResult:
    """Compile the candidate and check correctness on a single shape.

    Used by ``write_candidate`` so the LLM gets immediate feedback on
    syntax / type errors / wrong forward signatures without paying for
    a full multi-shape benchmark. ``build_dir`` is passed to
    ``cpp_extension.load(build_directory=...)`` so .so artifacts live
    under the run instead of ``~/.cache/torch_extensions/``.
    """
    shape_id_str = ops.shape_id(int(sample_shape))
    script = _build_quick_script(
        ops=ops,
        candidate_id=candidate_id,
        candidate_cu=candidate_cu,
        inputs_dir=inputs_dir,
        oracle_dir=oracle_dir,
        build_dir=build_dir,
        shape_value=int(sample_shape),
        shape_id=shape_id_str,
    )
    job = executor.profile_with_torch(
        python_code=script,
        op_name=f"quick_eval/{candidate_id}",
        timeout_s=600,
    )
    raw = parse_marked_output(job, _QUICK_MARKER)
    return QuickEvalResult(
        candidate_id=candidate_id,
        compile_ok=bool(raw.get("compile_ok")),
        correctness_ok=bool(raw.get("correctness_ok")),
        shape_id=shape_id_str,
        max_abs_err=_maybe_float(raw.get("max_abs_err")),
        rel_l2_err=_maybe_float(raw.get("rel_l2_err")),
        compile_log=str(raw.get("compile_log") or ""),
        diagnostics=raw.get("diagnostics") or {},
    )


# ---------------------------------------------------------------------------
# Multi-shape benchmark (orchestrator-only)
# ---------------------------------------------------------------------------


def benchmark_on_grid(
    ops: OperatorOps,
    spec: BenchmarkSpec,
    candidate_id: str,
    candidate_cu: Path,
    inputs_dir: Path,
    oracle_dir: Path,
    baseline_per_shape: dict[str, dict[str, Any]],
    build_dir: Path,
    *,
    executor,
) -> BenchmarkResult:
    """Full benchmark across ``spec.shape_grid``.

    For each shape: load inputs + oracle, call candidate forward, validate
    allclose, time with cudaEvent (samples + warmup from spec). Speedup is
    ``baseline_ms / candidate_ms`` per shape; geomean / worst / best
    summarize across shapes.
    """
    script = _build_bench_script(
        ops=ops,
        candidate_id=candidate_id,
        candidate_cu=candidate_cu,
        inputs_dir=inputs_dir,
        oracle_dir=oracle_dir,
        build_dir=build_dir,
        spec=spec,
    )
    job = executor.profile_with_torch(
        python_code=script,
        op_name=f"benchmark/{candidate_id}",
        timeout_s=900,
    )
    raw = parse_marked_output(job, _BENCH_MARKER)
    if not raw.get("compile_ok"):
        return BenchmarkResult(
            candidate_id=candidate_id,
            compile_ok=False,
            per_shape={},
            speedup_geomean=None,
            speedup_worst=None,
            speedup_best=None,
            correctness_per_shape={},
            all_correct=False,
            diagnostics={"compile_log": raw.get("compile_log", "")},
        )

    per_shape: dict[str, dict[str, Any]] = {}
    correctness: dict[str, bool] = {}
    speedups: list[float] = []
    for entry in raw.get("per_shape", []):
        sid = entry["shape_id"]
        cand_ms = _maybe_float(entry.get("ms_median"))
        baseline_ms = _maybe_float(
            (baseline_per_shape.get(sid) or {}).get("ms_median")
        )
        speedup = (
            baseline_ms / cand_ms
            if cand_ms is not None and cand_ms > 0 and baseline_ms is not None
            else None
        )
        per_shape[sid] = {
            "ms_median": cand_ms,
            "ms_min": _maybe_float(entry.get("ms_min")),
            "ms_max": _maybe_float(entry.get("ms_max")),
            "samples": int(entry.get("samples") or 0),
            "max_abs_err": _maybe_float(entry.get("max_abs_err")),
            "rel_l2_err": _maybe_float(entry.get("rel_l2_err")),
            "speedup": speedup,
        }
        correctness[sid] = bool(entry.get("correct"))
        if speedup is not None:
            speedups.append(speedup)

    if speedups:
        speedup_worst = min(speedups)
        speedup_best = max(speedups)
        # Geometric mean — robust to widely varying d-sensitivity.
        speedup_geomean = math.exp(
            statistics.fmean(math.log(s) for s in speedups)
        )
    else:
        speedup_worst = speedup_best = speedup_geomean = None

    all_correct = bool(correctness) and all(correctness.values())

    return BenchmarkResult(
        candidate_id=candidate_id,
        compile_ok=True,
        per_shape=per_shape,
        speedup_geomean=speedup_geomean,
        speedup_worst=speedup_worst,
        speedup_best=speedup_best,
        correctness_per_shape=correctness,
        all_correct=all_correct,
        diagnostics={},
    )


# ---------------------------------------------------------------------------
# Subprocess script generators
# ---------------------------------------------------------------------------


def _build_quick_script(
    *,
    ops: OperatorOps,
    candidate_id: str,
    candidate_cu: Path,
    inputs_dir: Path,
    oracle_dir: Path,
    build_dir: Path,
    shape_value: int,
    shape_id: str,
) -> str:
    cu_path_s = str(candidate_cu.resolve()).replace("\\", "/")
    inp_dir_s = inputs_dir.resolve().as_posix()
    ora_dir_s = oracle_dir.resolve().as_posix()
    bld_dir_s = build_dir.resolve().as_posix()
    project_root_s = _PROJECT_ROOT.as_posix()
    cand_name = f"cand_{candidate_id}".replace("-", "_")

    return f'''import json
import sys
import traceback

sys.path.insert(0, r"{project_root_s}")

import torch
from torch.utils.cpp_extension import load

from operator_opt_pipe.operators import load_ops

CU_PATH    = r"{cu_path_s}"
INPUT_DIR  = r"{inp_dir_s}"
ORACLE_DIR = r"{ora_dir_s}"
BUILD_DIR  = r"{bld_dir_s}"
SHORT_NAME = "{ops.short_name}"
SHAPE_VAL  = {int(shape_value)}
SHAPE_ID   = "{shape_id}"
RTOL = {float(ops.contract.rtol)}
ATOL = {float(ops.contract.atol)}

import os
os.makedirs(BUILD_DIR, exist_ok=True)

result = {{
    "compile_ok": False,
    "correctness_ok": False,
    "max_abs_err": None,
    "rel_l2_err": None,
    "compile_log": "",
    "diagnostics": {{}},
}}

if not torch.cuda.is_available():
    result["compile_log"] = "cuda_unavailable"
    print("{_QUICK_MARKER}")
    print(json.dumps(result))
    sys.exit(0)

ops = load_ops(SHORT_NAME)

try:
    mod = load(
        name="{cand_name}",
        sources=[CU_PATH],
        build_directory=BUILD_DIR,
        verbose=False,
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
    )
    result["compile_ok"] = True
except Exception as exc:
    result["compile_log"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    print("{_QUICK_MARKER}")
    print(json.dumps(result))
    sys.exit(0)

try:
    device = torch.device("cuda")
    inputs = ops.load_inputs(INPUT_DIR, SHAPE_ID, device=device)
    Y_ref = ops.load_oracle(ORACLE_DIR, SHAPE_ID, device=device)
    with torch.no_grad():
        Y = ops.forward_call(mod, inputs)
    diff = (Y - Y_ref).float()
    result["max_abs_err"] = float(diff.abs().max().item())
    result["rel_l2_err"] = float((diff.norm() / (Y_ref.float().norm() + 1e-12)).item())
    result["correctness_ok"] = bool(torch.allclose(Y, Y_ref, rtol=RTOL, atol=ATOL))
except Exception as exc:
    result["diagnostics"]["error"] = (
        "".join(traceback.format_exception_only(type(exc), exc)).strip()
    )

print("{_QUICK_MARKER}")
print(json.dumps(result))
'''


def _build_bench_script(
    *,
    ops: OperatorOps,
    candidate_id: str,
    candidate_cu: Path,
    inputs_dir: Path,
    oracle_dir: Path,
    build_dir: Path,
    spec: BenchmarkSpec,
) -> str:
    cu_path_s = str(candidate_cu.resolve()).replace("\\", "/")
    inp_dir_s = inputs_dir.resolve().as_posix()
    ora_dir_s = oracle_dir.resolve().as_posix()
    bld_dir_s = build_dir.resolve().as_posix()
    project_root_s = _PROJECT_ROOT.as_posix()
    cand_name = f"cand_{candidate_id}".replace("-", "_")
    grid_repr = json.dumps(list(spec.shape_grid))

    return f'''import json
import statistics
import sys
import traceback

sys.path.insert(0, r"{project_root_s}")

import torch
from torch.utils.cpp_extension import load

from operator_opt_pipe.operators import load_ops

CU_PATH    = r"{cu_path_s}"
INPUT_DIR  = r"{inp_dir_s}"
ORACLE_DIR = r"{ora_dir_s}"
BUILD_DIR  = r"{bld_dir_s}"
SHORT_NAME = "{ops.short_name}"
SHAPE_GRID = {grid_repr}
SAMPLES    = {int(spec.samples)}
WARMUP     = {int(spec.warmup)}
RTOL = {float(ops.contract.rtol)}
ATOL = {float(ops.contract.atol)}

import os
os.makedirs(BUILD_DIR, exist_ok=True)

result = {{"compile_ok": False, "compile_log": "", "per_shape": []}}

if not torch.cuda.is_available():
    result["compile_log"] = "cuda_unavailable"
    print("{_BENCH_MARKER}")
    print(json.dumps(result))
    sys.exit(0)

ops = load_ops(SHORT_NAME)

try:
    mod = load(
        name="{cand_name}",
        sources=[CU_PATH],
        build_directory=BUILD_DIR,
        verbose=False,
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
    )
    result["compile_ok"] = True
except Exception as exc:
    result["compile_log"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    print("{_BENCH_MARKER}")
    print(json.dumps(result))
    sys.exit(0)

device = torch.device("cuda")
per_shape = []
for d in SHAPE_GRID:
    shape_id = ops.shape_id(int(d))
    entry = {{
        "shape_id": shape_id,
        "correct": False,
        "max_abs_err": None,
        "rel_l2_err": None,
        "ms_median": None,
        "ms_min": None,
        "ms_max": None,
        "samples": 0,
    }}
    try:
        inputs = ops.load_inputs(INPUT_DIR, shape_id, device=device)
        Y_ref = ops.load_oracle(ORACLE_DIR, shape_id, device=device)
        with torch.no_grad():
            Y = ops.forward_call(mod, inputs)
        diff = (Y - Y_ref).float()
        entry["max_abs_err"] = float(diff.abs().max().item())
        entry["rel_l2_err"] = float((diff.norm() / (Y_ref.float().norm() + 1e-12)).item())
        entry["correct"] = bool(torch.allclose(Y, Y_ref, rtol=RTOL, atol=ATOL))
        if entry["correct"]:
            with torch.no_grad():
                for _ in range(WARMUP):
                    _ = ops.forward_call(mod, inputs)
                torch.cuda.synchronize()
                times = []
                for _ in range(SAMPLES):
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    _ = ops.forward_call(mod, inputs)
                    e.record()
                    torch.cuda.synchronize()
                    times.append(s.elapsed_time(e))
            entry["ms_median"] = float(statistics.median(times))
            entry["ms_min"] = float(min(times))
            entry["ms_max"] = float(max(times))
            entry["samples"] = SAMPLES
    except Exception as exc:
        entry["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    per_shape.append(entry)

result["per_shape"] = per_shape
print("{_BENCH_MARKER}")
print(json.dumps(result))
'''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "QuickEvalResult",
    "BenchmarkResult",
    "compile_and_check_quick",
    "benchmark_on_grid",
]
