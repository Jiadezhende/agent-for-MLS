"""pipeline/tools/baseline_tools.py — BaselineAgent tools.

Two tools:
  - GenerateBaselineTool: in one shot, materialize W/X/A/B + Y_ref to disk
    for every requested d, *and* benchmark the PyTorch reference latency.
  - SubmitBaselineTool: write baseline.json + finalize the BASELINE_PROFILE
    stage with a StageResult.

The tool generates a Python script, hands it to ``executor.profile_with_torch``,
and parses a marked JSON blob from stdout. Inputs/references end up in absolute
paths under ``layout.baseline_inputs_dir`` / ``layout.baseline_references_dir``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agents.tools.base import Tool, ToolParameter
from agents.tools.registry import _Terminated
from agents.tools.response import ToolErrorCode, ToolResponse

from ..agent_loop_signal import stash_stage_result
from ..state import Stage, StageResult
from ..workspace_layout import RunLayout
from ._script_runner import collect_stdout, parse_marked_json, run_python_script


# Default seed for reproducibility — agents can override via parameters.
DEFAULT_SEED = 0
DEFAULT_DTYPE = "float32"


# ===========================================================================
# GenerateBaselineTool
# ===========================================================================

_BASELINE_MARKER = "=== BASELINE_RESULT ==="


def _build_baseline_script(
    *,
    inputs_dir: Path,
    refs_dir: Path,
    d_list: List[int],
    samples: int,
    seed: int,
) -> str:
    """Generate a self-contained Python script for the baseline subprocess.

    The script:
      1. Seeds torch and creates W/X/A/B/Y_ref tensors per d on the GPU.
      2. Saves W/X/A/B to inputs/, Y_ref to refs/ (as .pt files).
      3. Benchmarks ``Y = W @ X + A @ (B.T @ X)`` with cudaEvent over
         ``samples`` iterations after a 2-iter warmup.
      4. Prints a marker line followed by JSON describing per-d torch_ms_median.
    """
    inputs_dir_s = inputs_dir.resolve().as_posix()
    refs_dir_s = refs_dir.resolve().as_posix()
    d_list_repr = json.dumps(list(d_list))
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
dtype = torch.float32
torch.manual_seed(SEED)

results = []
for d in D_LIST:
    W = torch.randn(d, d, device=device, dtype=dtype)
    X = torch.randn(d, d, device=device, dtype=dtype)
    A = torch.randn(d, 16, device=device, dtype=dtype)
    B = torch.randn(d, 16, device=device, dtype=dtype)
    Y = W @ X + A @ (B.T @ X)

    torch.save(W.cpu(), os.path.join(INPUT_DIR, f"W_d{{d}}.pt"))
    torch.save(X.cpu(), os.path.join(INPUT_DIR, f"X_d{{d}}.pt"))
    torch.save(A.cpu(), os.path.join(INPUT_DIR, f"A_d{{d}}.pt"))
    torch.save(B.cpu(), os.path.join(INPUT_DIR, f"B_d{{d}}.pt"))
    torch.save(Y.cpu(), os.path.join(REF_DIR, f"Y_d{{d}}.pt"))

    # Warmup
    for _ in range(2):
        _ = W @ X + A @ (B.T @ X)
    torch.cuda.synchronize()

    times = []
    for _ in range(SAMPLES):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _ = W @ X + A @ (B.T @ X)
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
    """Generate W/X/A/B/Y_ref for each d AND benchmark PyTorch reference."""

    _ctx: Any = None

    def __init__(self, *, executor: Any, layout: RunLayout):
        super().__init__(
            name="generate_baseline",
            description=(
                "For each requested d, generate W/X/A/B input tensors and "
                "the PyTorch reference output Y_ref (saved to disk for later "
                "candidate correctness checks), and benchmark the PyTorch "
                "reference latency. Returns per-d torch_ms_median measurements."
            ),
        )
        self._executor = executor
        self._layout = layout

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
                            "description": "List of integer d values to test (must be >=16, multiples used by LoRA).",
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
