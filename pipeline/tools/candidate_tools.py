"""pipeline/tools/candidate_tools.py — KernelTuningAgent tools.

Three tools form the candidate-evaluation triplet:

  - WriteCandidateTool: write a fresh ``candidates/candidate_NNN/candidate.cu``
    file (NNN auto-allocated based on existing candidate dirs).
  - EvaluateCandidateTool: in one subprocess invocation, compile the candidate
    via ``torch.utils.cpp_extension.load`` (matching the official Phase-2
    evaluation path), then run correctness against the BaselineAgent's
    Y_ref tensors and benchmark candidate vs PyTorch reference using
    cudaEvent — quick (5 samples) or confirm (30 samples, with variance).
  - SubmitCandidateResultTool: stash a StageResult whose ``metrics.candidate``
    is a CandidateRecord-shaped dict; the orchestrator appends it to
    ``leaderboard.jsonl`` and may promote it to ``best/best.cu``.

Internal benchmarking script intentionally mirrors the harness used by the
course staff (cpp_extension.load + cudaEvent + median latency vs PyTorch
reference). This guarantees ``best`` chosen locally is best at evaluation.
"""
from __future__ import annotations

import json
import re
import statistics
import textwrap
from pathlib import Path
from typing import Any, Dict, List

from agents.tools.base import Tool, ToolParameter
from agents.tools.registry import _Terminated
from agents.tools.response import ToolErrorCode, ToolResponse

from ..agent_loop_signal import stash_stage_result
from ..operator_spec import OperatorSpec
from ..state import (
    ACCEPTED_FOR_VALUES,
    CandidateRecord,
    Stage,
    StageResult,
)
from ..workspace_layout import RunLayout, candidate_id as candidate_id_for
from ._script_runner import collect_stdout, parse_marked_json, run_python_script


_EVAL_MARKER = "=== EVAL_RESULT ==="

# Allowed accepted_for values that an LLM submit_candidate_result can specify.
# (None is allowed in the dataclass but reserved for failed candidates the
# tool itself sets — agents shouldn't send None.)
_AGENT_ACCEPTED_FOR_VALUES = [v for v in ACCEPTED_FOR_VALUES if v is not None]


# ===========================================================================
# WriteCandidateTool
# ===========================================================================

_CANDIDATE_DIR_RE = re.compile(r"^candidate_(\d+)$")


def _next_candidate_index(layout: RunLayout) -> int:
    """Scan candidates/ for existing candidate_NNN dirs; return next index."""
    base = layout.candidates_dir
    if not base.is_dir():
        return 0
    best = -1
    for child in base.iterdir():
        if not child.is_dir():
            continue
        m = _CANDIDATE_DIR_RE.match(child.name)
        if m:
            try:
                best = max(best, int(m.group(1)))
            except ValueError:
                continue
    return best + 1


class WriteCandidateTool(Tool):
    """Allocate a candidate_NNN dir and write candidate.cu inside it."""

    _ctx: Any = None

    def __init__(self, layout: RunLayout, op_spec: OperatorSpec):
        super().__init__(
            name="write_candidate",
            description=(
                "Write a new candidate CUDA implementation to candidates/<id>/candidate.cu "
                "and return the allocated candidate_id + absolute path. The file must export "
                f"torch::Tensor {op_spec.forward_signature_text()} via PYBIND11_MODULE so that "
                "torch.utils.cpp_extension.load can compile it."
            ),
        )
        self._layout = layout
        self._op_spec = op_spec

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="source", type="string", description="Full CUDA source for candidate.cu."),
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
                        "source": {
                            "type": "string",
                            "description": (
                                "Full CUDA source for candidate.cu. Must include "
                                f"<torch/extension.h>, define {self._op_spec.forward_signature_text()}, "
                                "and expose it via PYBIND11_MODULE."
                            ),
                            "minLength": 100,
                        },
                    },
                    "required": ["source"],
                },
            },
        }

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        source = parameters["source"]
        if "PYBIND11_MODULE" not in source or "forward" not in source:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message=(
                    "candidate source must define `forward` and expose it via PYBIND11_MODULE — "
                    "the official harness will reject anything else."
                ),
            )

        idx = _next_candidate_index(self._layout)
        cid = candidate_id_for(idx)
        cdir = self._layout.candidate_dir(cid)
        cdir.mkdir(parents=True, exist_ok=True)
        path = self._layout.candidate_file(cid, "candidate.cu")
        path.write_text(source, encoding="utf-8")

        return ToolResponse.success(
            text=f"Wrote {cid}/candidate.cu ({len(source)} chars).",
            data={
                "candidate_id": cid,
                "path": self._layout.relpath(path),
                "abs_path": str(path),
            },
        )


# ===========================================================================
# EvaluateCandidateTool
# ===========================================================================

def _build_eval_script(
    *,
    op_spec: OperatorSpec,
    candidate_id: str,
    candidate_cu: Path,
    inputs_dir: Path,
    refs_dir: Path,
    d_list: List[int],
    samples: int,
    correctness_atol: float = 1e-4,
    correctness_rtol: float = 1e-4,
) -> str:
    """Generate the candidate evaluation Python script.

    On ``import torch.utils.cpp_extension.load`` failure the script prints a
    JSON blob with ``compile_ok=false`` and exits 0 (so subprocess success
    doesn't depend on candidate compileability).

    For each d the script:
      - loads operator inputs from inputs_dir and the reference output from refs_dir
      - calls ``mod.<forward_call>``, checks allclose vs reference
      - if correct: warmup + ``samples`` cudaEvent timings for candidate AND
        for PyTorch reference (``op_spec.reference_pytorch``); records median + min + max
    Final JSON includes per_d list, all_correct flag, overall median speedup.
    """
    cu_path_s = str(candidate_cu.resolve()).replace("\\", "/")
    inp_dir_s = inputs_dir.resolve().as_posix()
    ref_dir_s = refs_dir.resolve().as_posix()
    cand_name = f"cand_{candidate_id}".replace("-", "_")
    d_list_repr = json.dumps(list(d_list))

    load_inputs = textwrap.indent(
        op_spec.render_load_inputs(dir_var="INPUT_DIR", d_var="d"), "        "
    )
    load_ref = textwrap.indent(
        op_spec.render_load_reference(dir_var="REF_DIR", d_var="d", var_name="Y_ref"),
        "        ",
    )
    forward_call = op_spec.render_forward_call("mod")
    ref_expr = op_spec.reference_pytorch

    return f'''import json
import os
import statistics
import sys
import traceback

import torch

CAND_PATH = r"{cu_path_s}"
CAND_NAME = "{cand_name}"
INPUT_DIR = r"{inp_dir_s}"
REF_DIR   = r"{ref_dir_s}"
D_LIST    = {d_list_repr}
SAMPLES   = {int(samples)}
ATOL      = {float(correctness_atol)}
RTOL      = {float(correctness_rtol)}

result = {{
    "compile_ok":   False,
    "compile_error": None,
    "all_correct":  False,
    "per_d":        [],
    "overall_speedup_median": None,
}}

if not torch.cuda.is_available():
    result["compile_error"] = "cuda_unavailable"
    print("{_EVAL_MARKER}")
    print(json.dumps(result))
    sys.exit(0)

# --- COMPILE ----------------------------------------------------------------
try:
    from torch.utils.cpp_extension import load
    mod = load(
        name=CAND_NAME,
        sources=[CAND_PATH],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    result["compile_ok"] = True
except Exception as e:
    result["compile_error"] = (str(e) + "\\n" + traceback.format_exc())[-3000:]
    print("{_EVAL_MARKER}")
    print(json.dumps(result))
    sys.exit(0)

device = torch.device("cuda")

# --- PER-D EVALUATION -------------------------------------------------------
for d in D_LIST:
    entry = {{
        "d": int(d),
        "correctness_ok": False,
        "max_abs_diff":   None,
        "candidate_ms_median": None,
        "ref_ms_median":       None,
        "speedup": None,
        "samples": int(SAMPLES),
        "candidate_ms_min": None,
        "candidate_ms_max": None,
        "ref_ms_min": None,
        "ref_ms_max": None,
        "error": None,
    }}
    try:
{load_inputs}
{load_ref}
    except Exception as e:
        entry["error"] = f"input_load_failed: {{e}}"[-500:]
        result["per_d"].append(entry)
        continue

    # Correctness
    try:
        Y_cand = {forward_call}
        torch.cuda.synchronize()
        diff = (Y_cand - Y_ref).abs().max().item()
        entry["max_abs_diff"] = float(diff)
        # rtol*max + atol style; mimics torch.allclose
        tol = ATOL + RTOL * float(Y_ref.abs().max().item())
        entry["correctness_ok"] = bool(diff <= tol)
    except Exception as e:
        entry["error"] = f"forward_failed: {{e}}"[-500:]
        result["per_d"].append(entry)
        continue

    if not entry["correctness_ok"]:
        result["per_d"].append(entry)
        continue

    # Benchmark candidate
    try:
        # warmup
        for _ in range(3):
            _ = {forward_call}
        torch.cuda.synchronize()

        cand_times = []
        for _ in range(SAMPLES):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = {forward_call}
            e.record()
            torch.cuda.synchronize()
            cand_times.append(s.elapsed_time(e))

        # warmup ref
        for _ in range(3):
            _ = {ref_expr}
        torch.cuda.synchronize()

        ref_times = []
        for _ in range(SAMPLES):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = {ref_expr}
            e.record()
            torch.cuda.synchronize()
            ref_times.append(s.elapsed_time(e))

        cand_med = float(statistics.median(cand_times))
        ref_med  = float(statistics.median(ref_times))
        entry["candidate_ms_median"] = cand_med
        entry["candidate_ms_min"]    = float(min(cand_times))
        entry["candidate_ms_max"]    = float(max(cand_times))
        entry["ref_ms_median"]       = ref_med
        entry["ref_ms_min"]          = float(min(ref_times))
        entry["ref_ms_max"]          = float(max(ref_times))
        if cand_med > 0:
            entry["speedup"] = ref_med / cand_med
    except Exception as e:
        entry["error"] = f"benchmark_failed: {{e}}"[-500:]

    result["per_d"].append(entry)

# Aggregates
result["all_correct"] = all(e.get("correctness_ok") for e in result["per_d"]) and len(result["per_d"]) == len(D_LIST)
speedups = [e["speedup"] for e in result["per_d"] if isinstance(e.get("speedup"), (int, float))]
if speedups:
    result["overall_speedup_median"] = float(statistics.median(speedups))

print("{_EVAL_MARKER}")
print(json.dumps(result))
'''


def _summarize_evaluation(parsed: dict, samples: int) -> dict:
    """Compute speedup variance across the per-d entries (post-hoc)."""
    speedups = [e["speedup"] for e in parsed.get("per_d", []) if isinstance(e.get("speedup"), (int, float))]
    variance_pct = None
    if len(speedups) >= 2:
        med = statistics.median(speedups)
        if med:
            spread = (max(speedups) - min(speedups)) / med
            variance_pct = round(100.0 * spread, 2)
    return {
        "compile_ok": bool(parsed.get("compile_ok")),
        "all_correct": bool(parsed.get("all_correct")),
        "speedup_median": parsed.get("overall_speedup_median"),
        "samples": samples,
        "variance_pct": variance_pct,
    }


class EvaluateCandidateTool(Tool):
    """Compile + correctness + benchmark a single candidate."""

    _ctx: Any = None

    def __init__(self, *, executor: Any, layout: RunLayout, op_spec: OperatorSpec):
        super().__init__(
            name="evaluate_candidate",
            description=(
                "Compile candidates/<id>/candidate.cu via torch.utils.cpp_extension.load "
                "(same path as the official harness), run correctness against baseline/refs, "
                "and benchmark candidate vs PyTorch reference using cudaEvent. "
                "Returns per_d records with speedup + a summary dict (compile_ok, all_correct, "
                "speedup_median, variance_pct). Use mode='quick' (5 samples) for fast filtering "
                "and mode='confirm' (30 samples) before promoting to best."
            ),
        )
        self._executor = executor
        self._layout = layout
        self._op_spec = op_spec

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="candidate_id", type="string", description="As returned by write_candidate."),
            ToolParameter(name="d_list",       type="array",  description="Integer d values to evaluate."),
            ToolParameter(name="mode",         type="string", description="'quick' (5 samples) or 'confirm' (30 samples).", required=False, default="quick"),
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
                        "candidate_id": {"type": "string", "pattern": r"^candidate_\d{3,}$"},
                        "d_list": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 16},
                            "minItems": 1,
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["quick", "confirm"],
                            "description": "quick=5 samples, confirm=30 samples (default quick).",
                        },
                    },
                    "required": ["candidate_id", "d_list"],
                },
            },
        }

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        cid = parameters["candidate_id"]
        d_list = list(parameters["d_list"])
        mode = parameters.get("mode", "quick")
        samples = 30 if mode == "confirm" else 5

        cu_path = self._layout.candidate_file(cid, "candidate.cu")
        if not cu_path.is_file():
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message=f"candidate file missing: {self._layout.relpath(cu_path)}; call write_candidate first",
            )

        # Sanity check: baseline references must exist.
        ref_name = self._op_spec.output.name
        missing_refs = []
        for d in d_list:
            if not self._layout.baseline_reference_path(ref_name, d).is_file():
                missing_refs.append(d)
        if missing_refs:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message=(
                    f"baseline references missing for d={missing_refs}; "
                    "BaselineAgent must run before evaluate_candidate"
                ),
            )

        code = _build_eval_script(
            op_spec=self._op_spec,
            candidate_id=cid,
            candidate_cu=cu_path,
            inputs_dir=self._layout.baseline_inputs_dir,
            refs_dir=self._layout.baseline_references_dir,
            d_list=d_list,
            samples=samples,
        )

        out = run_python_script(self._executor, code=code, op_name=f"eval_{cid}_{mode}", timeout_s=600)
        if "_executor_error" in out:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"evaluation subprocess failed: {out['_executor_error']}",
            )

        stdout = collect_stdout(out)
        parsed = parse_marked_json(stdout, _EVAL_MARKER)
        if parsed is None:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message="evaluation subprocess produced no parseable JSON; tail: " +
                        (stdout[-1500:] if stdout else "<empty>"),
            )

        # Persist per-mode result file inside the candidate dir.
        out_name = "quick_benchmark.json" if mode == "quick" else "confirm_benchmark.json"
        (self._layout.candidate_dir(cid) / out_name).write_text(
            json.dumps(parsed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Always also update compile.json + correctness.json as small summaries.
        (self._layout.candidate_dir(cid) / "compile.json").write_text(
            json.dumps({"ok": bool(parsed.get("compile_ok")), "error": parsed.get("compile_error")}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (self._layout.candidate_dir(cid) / "correctness.json").write_text(
            json.dumps(
                {
                    "all_correct": bool(parsed.get("all_correct")),
                    "per_d": [
                        {"d": e.get("d"), "ok": e.get("correctness_ok"), "max_abs_diff": e.get("max_abs_diff")}
                        for e in parsed.get("per_d", [])
                    ],
                },
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        summary = _summarize_evaluation(parsed, samples)
        return ToolResponse.success(
            text=(
                f"{cid} mode={mode}: compile_ok={summary['compile_ok']} "
                f"all_correct={summary['all_correct']} "
                f"speedup_median={summary['speedup_median']} "
                f"variance_pct={summary['variance_pct']}"
            ),
            data={"summary": summary, "per_d": parsed.get("per_d", []),
                  "compile_error": parsed.get("compile_error"),
                  "candidate_id": cid, "mode": mode, "samples": samples},
        )


# ===========================================================================
# SubmitCandidateResultTool
# ===========================================================================

class SubmitCandidateResultTool(Tool):
    """Stash a CandidateRecord and end the candidate-producing stage."""

    _ctx: Any = None

    def __init__(self, *, layout: RunLayout, stage: Stage):
        super().__init__(
            name="submit_candidate_result",
            description=(
                "Finalize the current candidate stage with a CandidateRecord. The orchestrator "
                "appends it to leaderboard.jsonl and may promote it to best/best.cu when "
                "accepted_for=='best_update'. Call exactly once per stage invocation."
            ),
        )
        self._layout = layout
        self._stage = stage

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="record",       type="object", description="CandidateRecord-shaped dict."),
            ToolParameter(name="status",       type="string", description="StageResult status: success | partial | failed.", required=False, default="success"),
            ToolParameter(name="confidence",   type="number", description="StageResult confidence 0..1.", required=False, default=1.0),
            ToolParameter(name="caveats",      type="array",  description="Optional caveats list.", required=False, default=[]),
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
                        "record": {
                            "type": "object",
                            "properties": {
                                "candidate_id":            {"type": "string"},
                                "compile_ok":              {"type": "boolean"},
                                "correctness_ok":          {"type": "boolean"},
                                "quick_speedup_median":    {"type": ["number", "null"]},
                                "quick_samples":           {"type": ["integer", "null"]},
                                "quick_variance_pct":      {"type": ["number", "null"]},
                                "confirm_speedup_median":  {"type": ["number", "null"]},
                                "confirm_samples":         {"type": ["integer", "null"]},
                                "confirm_variance_pct":    {"type": ["number", "null"]},
                                "accepted_for": {
                                    "type": ["string", "null"],
                                    "enum": _AGENT_ACCEPTED_FOR_VALUES + [None],
                                },
                            },
                            "required": ["candidate_id", "compile_ok", "correctness_ok", "accepted_for"],
                            "additionalProperties": True,
                        },
                        "status":     {"type": "string", "enum": ["success", "partial", "failed"]},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "caveats":    {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["record"],
                },
            },
        }

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        ctx = self._ctx
        record = dict(parameters["record"])
        status = parameters.get("status", "success")
        confidence = float(parameters.get("confidence", 1.0))
        caveats = list(parameters.get("caveats", []) or [])

        # Construct via dataclass to fill timestamp + validate field set.
        try:
            cr = CandidateRecord(
                candidate_id=record["candidate_id"],
                compile_ok=bool(record.get("compile_ok")),
                correctness_ok=bool(record.get("correctness_ok")),
                quick_speedup_median=record.get("quick_speedup_median"),
                quick_samples=record.get("quick_samples"),
                quick_variance_pct=record.get("quick_variance_pct"),
                confirm_speedup_median=record.get("confirm_speedup_median"),
                confirm_samples=record.get("confirm_samples"),
                confirm_variance_pct=record.get("confirm_variance_pct"),
                accepted_for=record.get("accepted_for"),
            )
        except (KeyError, TypeError) as e:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message=f"invalid CandidateRecord: {type(e).__name__}: {e}",
            )

        # Refuse best_update for failed candidates — orchestrator would promote
        # a broken kernel to optimized_lora.cu.
        if cr.accepted_for == "best_update" and not (cr.compile_ok and cr.correctness_ok):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message="accepted_for='best_update' requires compile_ok and correctness_ok both true",
            )

        result = StageResult(
            stage=self._stage.value,
            status=status,
            artifacts={"candidate_dir": self._layout.relpath(self._layout.candidate_dir(cr.candidate_id))},
            metrics={"candidate": cr.to_dict()},
            confidence=confidence,
            caveats=caveats,
        )
        stash_stage_result(ctx, result)
        raise _Terminated(f"submitted candidate {cr.candidate_id} accepted_for={cr.accepted_for}")
