"""pipeline/tools/baseline_tools.py — BaselineAgent tools.

Two tools:
  - GenerateBaselineTool: in one shot, materialize per-operator inputs + reference
    output to disk for every requested d, *and* benchmark the PyTorch reference
    latency.
  - SubmitBaselineTool: write baseline.json + finalize the BASELINE_PROFILE
    stage with a StageResult.

The tool generates a Python script, hands it to ``executor.profile_with_torch``,
and parses a marked JSON blob from stdout. Inputs/references end up in absolute
paths under ``layout.baseline_inputs_dir`` / ``layout.baseline_references_dir``.
The script body is rendered from an ``OperatorSpec`` so this tool is fully
operator-agnostic — switching to a different operator only requires a different
skill file under skills/operators/.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List

from agents.tools.base import Tool, ToolParameter
from agents.tools.registry import _Terminated
from agents.tools.response import ToolErrorCode, ToolResponse

from ..agent_loop_signal import stash_stage_result
from ..operator_spec import OperatorSpec
from ..state import Stage, StageResult
from ..workspace_layout import RunLayout
from ._script_runner import collect_stdout, parse_marked_json, run_python_script


# Default seed for reproducibility — agents can override via parameters.
DEFAULT_SEED = 0


# ===========================================================================
# GenerateBaselineTool
# ===========================================================================

_BASELINE_MARKER = "=== BASELINE_RESULT ==="


def _build_baseline_script(
    *,
    op_spec: OperatorSpec,
    inputs_dir: Path,
    refs_dir: Path,
    d_list: List[int],
    samples: int,
    seed: int,
) -> str:
    """Generate a self-contained Python script for the baseline subprocess.

    The script:
      1. Seeds torch and creates the operator's input tensors per d on the GPU.
      2. Saves inputs to inputs/, the reference output to refs/ (as .pt files).
      3. Benchmarks the operator's ``reference_pytorch`` formula with cudaEvent
         over ``samples`` iterations after a 2-iter warmup.
      4. Prints a marker line followed by JSON describing per-d torch_ms_median.
    """
    inputs_dir_s = inputs_dir.resolve().as_posix()
    refs_dir_s = refs_dir.resolve().as_posix()
    d_list_repr = json.dumps(list(d_list))

    # Each rendered block is a flat string of newline-joined statements; indent
    # them to live inside the ``for d in D_LIST:`` body.
    create_inputs = textwrap.indent(op_spec.render_input_creation(), "    ")
    compute_ref = textwrap.indent(op_spec.render_reference_compute(), "    ")
    save_inputs = textwrap.indent(
        op_spec.render_save_inputs(dir_var="INPUT_DIR", d_var="d"), "    "
    )
    save_ref = textwrap.indent(
        op_spec.render_save_reference(dir_var="REF_DIR", d_var="d"), "    "
    )
    ref_expr = op_spec.reference_pytorch

    return f'''import json
import os
import statistics
import sys
import torch

INPUT_DIR = r"{inputs_dir_s}"
REF_DIR   = r"{refs_dir_s}"
D_LIST    = {d_list_repr}
SAMPLES   = {int(samples)}
SEED      = {int(seed)}

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(REF_DIR, exist_ok=True)

if not torch.cuda.is_available():
    print("=== BASELINE_RESULT ===")
    print(json.dumps({{"error": "cuda_unavailable"}}))
    sys.exit(0)

device = torch.device("cuda")
torch.manual_seed(SEED)

results = []
for d in D_LIST:
{create_inputs}
{compute_ref}

{save_inputs}
{save_ref}

    # Warmup
    for _ in range(2):
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

    results.append({{
        "d": int(d),
        "torch_ms_median": float(statistics.median(times)),
        "torch_ms_min":    float(min(times)),
        "torch_ms_max":    float(max(times)),
        "samples":         int(SAMPLES),
    }})

print("{_BASELINE_MARKER}")
print(json.dumps(results))
'''


class GenerateBaselineTool(Tool):
    """Generate operator inputs + reference AND benchmark PyTorch reference latency."""

    _ctx: Any = None

    def __init__(self, *, executor: Any, layout: RunLayout, op_spec: OperatorSpec):
        super().__init__(
            name="generate_baseline",
            description=(
                f"For each requested {op_spec.shape_param}, generate input tensors "
                f"({', '.join(t.name for t in op_spec.inputs)}) and the PyTorch "
                f"reference output {op_spec.output.name} (saved to disk for later "
                "candidate correctness checks), and benchmark the PyTorch "
                f"reference latency. Returns per-{op_spec.shape_param} torch_ms_median measurements."
            ),
        )
        self._executor = executor
        self._layout = layout
        self._op_spec = op_spec

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="d_list",  type="array",   description="List of integer d values to test."),
            ToolParameter(name="samples", type="integer", description="Number of timed iterations after warmup.", required=False, default=30),
            ToolParameter(name="seed",    type="integer", description="Torch seed for reproducibility.", required=False, default=DEFAULT_SEED),
        ]

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "d_list": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 16},
                            "minItems": 1,
                            "description": (
                                f"List of integer {self._op_spec.shape_param} values to test "
                                f"(within {list(self._op_spec.shape_param_range)})."
                            ),
                        },
                        "samples": {
                            "type": "integer",
                            "minimum": 5,
                            "maximum": 1000,
                            "description": "Iterations after warmup (default 30).",
                        },
                        "seed": {
                            "type": "integer",
                            "description": "Torch seed (default 0).",
                        },
                    },
                    "required": ["d_list"],
                },
            },
        }

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        d_list = list(parameters["d_list"])
        samples = int(parameters.get("samples", 30))
        seed = int(parameters.get("seed", DEFAULT_SEED))

        self._layout.baseline_inputs_dir.mkdir(parents=True, exist_ok=True)
        self._layout.baseline_references_dir.mkdir(parents=True, exist_ok=True)

        code = _build_baseline_script(
            op_spec=self._op_spec,
            inputs_dir=self._layout.baseline_inputs_dir,
            refs_dir=self._layout.baseline_references_dir,
            d_list=d_list,
            samples=samples,
            seed=seed,
        )

        out = run_python_script(self._executor, code=code, op_name="generate_baseline", timeout_s=600)
        if "_executor_error" in out:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"baseline subprocess failed: {out['_executor_error']}",
            )

        stdout = collect_stdout(out)
        parsed = parse_marked_json(stdout, _BASELINE_MARKER)
        if parsed is None:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message="baseline subprocess produced no parseable JSON; "
                        "stdout_tail follows: " + (stdout[-1500:] if stdout else "<empty>"),
            )
        if isinstance(parsed, dict) and "error" in parsed:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"baseline subprocess reported error: {parsed['error']}",
            )

        return ToolResponse.success(
            text=f"Baseline generated for d={d_list}; {samples} samples each.",
            data={"per_d": parsed, "d_list": d_list, "samples": samples, "seed": seed},
        )


# ===========================================================================
# SubmitBaselineTool
# ===========================================================================

class SubmitBaselineTool(Tool):
    """Persist baseline/baseline.json and finalize the BASELINE_PROFILE stage."""

    _ctx: Any = None

    def __init__(self, layout: RunLayout):
        super().__init__(
            name="submit_baseline",
            description=(
                "Finalize the BASELINE_PROFILE stage. Writes baseline.json with "
                "per-d torch_ms_median and signals stage completion. Call this "
                "exactly once after generate_baseline has returned its measurements."
            ),
        )
        self._layout = layout

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="per_d", type="array", description="Per-d baseline records as returned by generate_baseline."),
            ToolParameter(name="notes", type="string", description="Free-form notes about the run.", required=False, default=""),
        ]

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "per_d": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "d": {"type": "integer"},
                                    "torch_ms_median": {"type": "number"},
                                    "samples": {"type": "integer"},
                                },
                                "required": ["d", "torch_ms_median"],
                                "additionalProperties": True,
                            },
                            "minItems": 1,
                        },
                        "notes": {"type": "string"},
                    },
                    "required": ["per_d"],
                },
            },
        }

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        ctx = self._ctx
        per_d = parameters["per_d"]
        notes = parameters.get("notes", "")

        baseline_payload = {
            "per_d": list(per_d),
            "notes": notes,
        }
        path = self._layout.baseline_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(baseline_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        result = StageResult(
            stage=Stage.BASELINE_PROFILE.value,
            status="success",
            artifacts={"baseline": self._layout.relpath(path)},
            metrics={
                "per_d": list(per_d),
                "n_d": len(per_d),
            },
            confidence=1.0,
        )
        stash_stage_result(ctx, result)
        raise _Terminated(f"baseline submitted for {len(per_d)} d values")
