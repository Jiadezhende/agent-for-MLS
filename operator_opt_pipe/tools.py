"""LLM-facing tools for operator_opt_pipe.

Four classes total:

* ``ReadBlackboardTool`` — read a key from the run's blackboard.
* ``WriteBlackboardTool`` — write a key (with per-instance schema +
  allowed-key whitelist). Replaces v1's three SubmitTool instances. Does
  NOT terminate the agent — write-and-finish is signaled by the agent
  returning a final text response with no further tool calls
  (``ReActLoop.max_consecutive_no_tool_call``).
* ``WriteCandidateTool`` — allocate a new ``candidate_NNN/`` slot, write
  ``candidate.cu``, and synchronously compile + correctness-check it on a
  single shape. Returns the structured ``QuickEvalResult`` so the LLM has
  immediate compile + correctness feedback.
* ``SubmitCandidateTool`` — freeze a candidate and ``terminate_with`` the
  candidate_id payload. The orchestrator picks it up, runs a multi-shape
  benchmark, and decides whether to promote.

Performance evaluation, multi-shape benchmarking, leaderboard, and
best-promotion are NOT exposed as tools — they belong to the orchestrator
so the agent has no avenue to fabricate or short-circuit them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from mls_agent import Event, Tool, ToolErrorCode, ToolResponse

from operator_opt_pipe.operators._base import OperatorOps
from operator_opt_pipe.resources.evaluation import (
    QuickEvalResult,
    compile_and_check_quick,
)
from operator_opt_pipe.state import (
    RunLayout,
    load_blackboard,
    save_blackboard,
)


_CANDIDATE_DIR_RE = re.compile(r"^candidate_(\d+)$")


# ---------------------------------------------------------------------------
# read_blackboard
# ---------------------------------------------------------------------------


class ReadBlackboardTool(Tool):
    NAME = "read_blackboard"
    DESCRIPTION = (
        "Read a single key from the run's blackboard (workspace state shared "
        "between stages). Returns the value as JSON. Use this to look up "
        "operator/hardware/benchmark/baseline/best/latest_diagnosis/history."
    )

    def __init__(self, layout: RunLayout) -> None:
        self._layout = layout

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Top-level blackboard key to read.",
                },
                "default": {
                    "description": "Value to return if the key is absent.",
                },
            },
            "required": ["key"],
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        bb = load_blackboard(self._layout)
        key = parameters["key"]
        if key in bb:
            value = bb[key]
            present = True
        else:
            value = parameters.get("default")
            present = False
        return ToolResponse.success(
            text=json.dumps(value, ensure_ascii=False, default=str),
            data={"value": value, "present": present, "key": key},
        )


# ---------------------------------------------------------------------------
# write_blackboard
# ---------------------------------------------------------------------------


class WriteBlackboardTool(Tool):
    """Write a structured payload to a designated blackboard key.

    One class, instantiated per agent role with a different ``allowed_keys``
    mapping. ``allowed_keys[k]`` is an optional list of required field names
    in the payload — None to skip schema validation entirely.

    Does NOT terminate the agent. After a successful write, the LLM is
    expected to summarize and stop calling tools — ReActLoop's
    ``max_consecutive_no_tool_call`` handles natural termination.
    """

    NAME = "write_blackboard"
    DESCRIPTION = (
        "Write a structured payload to a specific top-level key in the run's "
        "blackboard. Each role is restricted to a small set of keys it owns. "
        "After a successful write you do not need to call any further tools — "
        "summarize what you wrote and the loop will end."
    )

    def __init__(
        self,
        layout: RunLayout,
        allowed_keys: Mapping[str, list[str] | None],
    ) -> None:
        if not allowed_keys:
            raise ValueError("WriteBlackboardTool requires at least one allowed key")
        self._layout = layout
        self._allowed: dict[str, list[str] | None] = {
            k: (list(v) if v is not None else None) for k, v in allowed_keys.items()
        }

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "enum": sorted(self._allowed),
                    "description": "Blackboard key to write. Each role is limited to its owned keys.",
                },
                "payload": {
                    "type": "object",
                    "description": "Structured payload to persist under the key.",
                },
            },
            "required": ["key", "payload"],
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        key = parameters["key"]
        payload = parameters["payload"]
        if key not in self._allowed:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message=(
                    f"key {key!r} is not allowed for this role; "
                    f"allowed: {sorted(self._allowed)}"
                ),
            )
        if not isinstance(payload, dict):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message=f"payload must be an object/dict, got {type(payload).__name__}",
            )
        required = self._allowed[key]
        if required:
            missing = [f for f in required if f not in payload]
            if missing:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_ARGS,
                    message=(
                        f"payload for {key!r} missing required fields: {missing}; "
                        f"required: {required}"
                    ),
                )
        bb = load_blackboard(self._layout)
        bb[key] = payload
        save_blackboard(self._layout, bb)
        return ToolResponse.success(
            text=f"wrote blackboard[{key!r}] ({len(json.dumps(payload, default=str))} bytes)",
            data={"key": key, "fields": sorted(payload)},
            events=(Event(type="blackboard_write", severity="info", detail=key),),
        )


# ---------------------------------------------------------------------------
# write_candidate
# ---------------------------------------------------------------------------


class WriteCandidateTool(Tool):
    """Allocate ``candidates/candidate_NNN/``, write ``candidate.cu``, and
    immediately compile + correctness-check it on a single shape.

    The compile / correctness result is returned to the LLM verbatim so it
    has tight feedback. The LLM may call ``write_candidate`` again with a
    revised source if compile fails or correctness fails — the next call
    allocates a fresh slot (NO destructive edit). The `submit_candidate`
    tool freezes a specific candidate_id and hands it to the orchestrator.
    """

    NAME = "write_candidate"
    DESCRIPTION = (
        "Write a new draft candidate CUDA source to candidates/<id>/candidate.cu, "
        "then compile it (cpp_extension.load + -O3) and validate correctness "
        "on a single shape. Returns candidate_id, compile_ok, correctness_ok, "
        "max_abs_err, and the compile log. If compile or correctness fails, "
        "call write_candidate again with a corrected source — drafts are "
        "cheap and never block."
    )

    def __init__(
        self,
        layout: RunLayout,
        ops: OperatorOps,
        executor,
    ) -> None:
        self._layout = layout
        self._ops = ops
        self._contract = ops.contract
        self._executor = executor

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": (
                        "Full CUDA source for candidate.cu. Must include "
                        "<torch/extension.h>, define "
                        f"`{self._contract.forward_signature_text()}`, and expose it "
                        "via PYBIND11_MODULE so torch.utils.cpp_extension.load can "
                        "compile it."
                    ),
                    "minLength": 50,
                },
                "notes": {
                    "type": "string",
                    "description": "Optional human-readable note about this draft.",
                },
            },
            "required": ["source"],
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        source = parameters["source"]
        if "PYBIND11_MODULE" not in source or "forward" not in source:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message=(
                    "candidate source must define `forward` and expose it via "
                    "PYBIND11_MODULE — the official harness will reject anything else."
                ),
            )

        idx = _next_candidate_index(self._layout)
        cid = self._layout.candidate_id(idx)
        cdir = self._layout.candidate_dir(cid)
        cdir.mkdir(parents=True, exist_ok=True)
        cu_path = self._layout.candidate_file(cid, "candidate.cu")
        cu_path.write_text(source, encoding="utf-8")

        # Pick the smallest shape in the default grid for the quick check —
        # smaller is faster to compile and validate while still exercising
        # the full forward path.
        sample_shape = self._contract.default_shape_grid()[0]

        try:
            quick = compile_and_check_quick(
                ops=self._ops,
                candidate_id=cid,
                candidate_cu=cu_path,
                inputs_dir=self._layout.inputs_dir,
                oracle_dir=self._layout.oracle_dir,
                sample_shape=sample_shape,
                build_dir=self._layout.build_dir,
                executor=self._executor,
            )
        except Exception as exc:  # noqa: BLE001
            # Surface infrastructure failures so the agent doesn't burn iterations
            # spinning on an unrecoverable error.
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"quick eval crashed: {type(exc).__name__}: {exc}",
            )

        # Persist artifacts for downstream stages / debugging.
        quick_dict = quick.to_dict()
        cdir.joinpath("compile.json").write_text(
            json.dumps({
                "candidate_id": cid,
                "compile_ok": quick.compile_ok,
                "compile_log": quick.compile_log,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        cdir.joinpath("correctness_quick.json").write_text(
            json.dumps(quick_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        text_lines = [
            f"candidate_id: {cid}",
            f"compile_ok: {quick.compile_ok}",
            f"correctness_ok: {quick.correctness_ok}",
            f"shape: {quick.shape_id}",
        ]
        if quick.max_abs_err is not None:
            text_lines.append(f"max_abs_err: {quick.max_abs_err:.3e}")
        if not quick.compile_ok and quick.compile_log:
            text_lines.append("compile_log:")
            text_lines.append(quick.compile_log[-1500:])

        return ToolResponse.success(
            text="\n".join(text_lines),
            data=quick_dict,
            events=(
                Event(
                    type="candidate_drafted",
                    severity="info" if (quick.compile_ok and quick.correctness_ok) else "warn",
                    detail=cid,
                ),
            ),
        )


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


# ---------------------------------------------------------------------------
# submit_candidate
# ---------------------------------------------------------------------------


class SubmitCandidateTool(Tool):
    """Freeze a candidate and hand it to the orchestrator for benchmark + promote.

    The orchestrator picks the candidate up from ``AgentResult.payload``,
    runs the multi-shape benchmark, and decides promotion. The agent never
    sees benchmark numbers from this tool's reply (it terminates first).
    """

    NAME = "submit_candidate"
    DESCRIPTION = (
        "Freeze a candidate by id and end this turn. The orchestrator will "
        "benchmark it across the full d_grid and may promote it to best. "
        "Attach hypothesis / experiment_type / expected_effect / risk so the "
        "decision history stays explainable even when the candidate loses."
    )

    def __init__(self, layout: RunLayout) -> None:
        self._layout = layout

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "candidate_id": {
                    "type": "string",
                    "description": "candidate_NNN from a prior write_candidate call.",
                },
                "hypothesis": {
                    "type": "string",
                    "description": "What the agent expects to gain from this candidate.",
                },
                "experiment_type": {
                    "type": "string",
                    "description": "e.g. fused-correction, tile-size-sweep, smem-tiling.",
                },
                "expected_effect": {
                    "type": "string",
                    "description": "Predicted speedup direction / magnitude.",
                },
                "risk": {
                    "type": "string",
                    "description": "Known risks: numerical instability, register pressure, etc.",
                },
            },
            "required": ["candidate_id", "hypothesis", "experiment_type"],
            "additionalProperties": True,
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        cid = parameters["candidate_id"]
        cu_path = self._layout.candidate_file(cid, "candidate.cu")
        if not cu_path.is_file():
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message=f"candidate {cid!r} has no candidate.cu on disk; "
                        "call write_candidate first.",
            )
        # Pass through whatever the LLM provided. status / stage are owned
        # by the orchestrator (it adds them in agents._result_to_dict based
        # on benchmark outcome and the current pipeline stage).
        return ToolResponse.terminate_with(
            summary=f"submit_candidate {cid}",
            payload=dict(parameters),
        )


__all__ = [
    "ReadBlackboardTool",
    "WriteBlackboardTool",
    "WriteCandidateTool",
    "SubmitCandidateTool",
]
