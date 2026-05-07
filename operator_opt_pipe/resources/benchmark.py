"""Benchmark spec + synthetic input materialization.

Both functions are operator-agnostic: they read ``contract.inputs`` to
decide what to generate and where to save it. Adding a new operator does
not require touching this module — only updating the skill markdown.
"""
from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

from operator_opt_pipe.resources.contract import OperatorContract


# Mirrors the resources/_script_runner pattern: subprocess prints a
# marker line then a JSON blob; the parent picks it up.
_MATERIALIZE_MARKER = "=== MATERIALIZE_RESULT ==="


@dataclass(frozen=True)
class BenchmarkSpec:
    """The single configuration object for benchmark + baseline.

    ``shape_grid`` is the list of shape-parameter values to test; for LoRA
    this is e.g. ``(3584, 4096, 4608)``. ``samples / warmup`` control timing
    runs. Tolerances live on the contract, not here.
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


def generate_benchmark_spec(
    contract: OperatorContract,
    *,
    shape_grid: tuple[int, ...] | None = None,
    samples: int = 30,
    warmup: int = 5,
    seed: int = 0,
) -> BenchmarkSpec:
    """Default benchmark spec covers both ends of the range plus midpoint."""
    grid = tuple(shape_grid) if shape_grid else contract.default_shape_grid()
    return BenchmarkSpec(
        shape_grid=grid,
        samples=int(samples),
        warmup=int(warmup),
        seed=int(seed),
    )


def materialize_inputs(
    contract: OperatorContract,
    spec: BenchmarkSpec,
    inputs_dir: Path,
    *,
    executor,
) -> dict:
    """Generate ``contract.inputs`` tensors per shape value to ``inputs_dir/``.

    Spawns a single torch subprocess that creates and saves all tensors
    for every ``shape_grid`` value. Returns ``{"saved": {...}}`` with the
    list of files written. Idempotent — re-running overwrites.
    """
    inputs_dir.mkdir(parents=True, exist_ok=True)
    script = _build_materialize_script(
        contract=contract,
        inputs_dir=inputs_dir,
        spec=spec,
    )
    job = executor.profile_with_torch(
        python_code=script,
        op_name=f"materialize_inputs/{contract.name.replace('/', '_')}",
        timeout_s=300,
    )
    raw = _parse_marked_output(job, _MATERIALIZE_MARKER)
    if "error" in raw:
        raise RuntimeError(f"materialize_inputs subprocess: {raw['error']}")
    return raw


def _build_materialize_script(
    *,
    contract: OperatorContract,
    inputs_dir: Path,
    spec: BenchmarkSpec,
) -> str:
    """Render a per-shape-loop subprocess script using contract helpers."""
    shape_var = contract.shape_param
    inputs_dir_s = inputs_dir.resolve().as_posix()
    grid_repr = json.dumps(list(spec.shape_grid))

    create_inputs = textwrap.indent(
        contract.render_input_creation(device_var="device"), "    "
    )
    save_inputs = textwrap.indent(
        contract.render_save_inputs(dir_var="INPUT_DIR", shape_id_expr="shape_id"),
        "    ",
    )

    return f'''import json
import os
import sys

import torch

INPUT_DIR = r"{inputs_dir_s}"
SHAPE_GRID = {grid_repr}
SEED = {int(spec.seed)}

os.makedirs(INPUT_DIR, exist_ok=True)

if not torch.cuda.is_available():
    print("{_MATERIALIZE_MARKER}")
    print(json.dumps({{"error": "cuda_unavailable"}}))
    sys.exit(0)

device = torch.device("cuda")
torch.manual_seed(SEED)

saved = []
for {shape_var} in SHAPE_GRID:
    shape_id = f"{shape_var}{{{shape_var}}}"
{create_inputs}
{save_inputs}
    saved.append(shape_id)

print("{_MATERIALIZE_MARKER}")
print(json.dumps({{"saved": saved}}))
'''


def _parse_marked_output(job, marker: str) -> dict:
    """Parse the JSON object printed after ``marker`` in a profile_with_torch job.

    ``Executor.profile_with_torch`` returns a dict (the ``to_tool_result``
    shape) with ``stdout / stderr / returncode / timed_out`` at the top
    level. We accept either dict or an object with ``.summary``-style
    attributes for forward compatibility.
    """
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
        # Surface stderr in the error so compile / import failures are visible.
        tail = (stderr[-800:] if stderr else stdout[-800:]) or "<empty>"
        raise RuntimeError(
            f"subprocess output missing marker {marker!r} "
            f"(returncode={returncode}); tail:\n{tail}"
        )
    tail = stdout.split(marker, 1)[1].strip()
    # The first non-empty line after the marker is the JSON blob.
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


__all__ = [
    "BenchmarkSpec",
    "generate_benchmark_spec",
    "materialize_inputs",
]
