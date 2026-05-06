"""LLM-facing tools for operator_opt_pipe.

Only four kinds of tools are defined here:

* ``ReadBlackboardTool`` — read a key from ``blackboard.json``.
* ``SubmitTool`` — generic submission: validates the payload, optionally
  persists it under a designated blackboard key, and terminates the agent.
  One class with a per-instance ``NAME`` covers ``submit_hardware_profile``,
  ``submit_diagnosis``, and ``submit_summary``.
* ``Write/Edit/Verify/SubmitCandidateTool`` — candidate-lifecycle stubs.
  Schemas are pinned now; bodies raise ``NotImplementedError`` and will be
  filled in the next PR.

What is NOT here, deliberately: ``evaluate_candidate``, ``benchmark_candidate``,
``run_python``, ``edit_file``. Performance evaluation and free-form code
execution must never reach the LLM — they belong to ``RoundRunner`` and
``lora_resources/``.
"""
from __future__ import annotations

import json
from typing import Any

import mls_agent
from mls_agent import Tool, ToolErrorCode, ToolResponse

from operator_opt_pipe.state import (
    RunLayout,
    Stage,
    check_submit_payload,
    load_blackboard,
    save_blackboard,
)


# ---------------------------------------------------------------------------
# read_blackboard
# ---------------------------------------------------------------------------


class ReadBlackboardTool(Tool):
    NAME = "read_blackboard"
    DESCRIPTION = (
        "Read a single key from the run's blackboard (workspace state shared "
        "between stages). Returns the value as JSON. Use this to look up "
        "hardware/benchmark/baseline/best/latest_diagnosis/history."
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
# submit (generic)
# ---------------------------------------------------------------------------


class SubmitTool(Tool):
    """Generic stage-submission tool.

    Each agent gets a ``SubmitTool`` instance whose ``NAME`` is the
    stage-appropriate verb (``submit_hardware_profile``, ``submit_diagnosis``,
    ``submit_summary``, …). The payload schema is shared; constructor params
    decide where the payload lands in the blackboard and which stage tag to
    expect.

    The class-level ``NAME = "submit"`` is a placeholder that satisfies
    ``Tool.__init_subclass__``. Per-instance ``NAME`` set in ``__init__``
    overrides it everywhere ``Tool`` looks the attribute up (registry
    registration, OpenAI schema export).
    """

    NAME = "submit"
    DESCRIPTION = (
        "Submit the final structured result for the current stage. The agent "
        "must attach status='success'|'partial'|'failed', plus stage-specific "
        "metrics, artifacts, and caveats. Calling this tool terminates the "
        "agent run."
    )

    def __init__(
        self,
        *,
        name: str,
        layout: RunLayout,
        blackboard_key: str | None,
        expected_stage: Stage,
        description: str | None = None,
    ) -> None:
        if not name:
            raise ValueError("SubmitTool requires a non-empty name")
        # Shadow the class attribute with a per-instance one.
        self.NAME = name
        if description:
            self.DESCRIPTION = description
        self._layout = layout
        self._key = blackboard_key
        self._stage = expected_stage

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["success", "partial", "failed"],
                    "description": "Outcome of the stage.",
                },
                "stage": {
                    "type": "string",
                    "description": "Stage tag (must match the stage this tool serves).",
                },
                "metrics": {
                    "type": "object",
                    "description": "Numeric or structured findings to persist.",
                },
                "artifacts": {
                    "type": "object",
                    "description": "Mapping of logical name → workspace-relative path.",
                },
                "caveats": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Warnings, partial-results notes, or explicit failure reasons.",
                },
                "next_recommendation": {
                    "type": ["string", "null"],
                    "description": "Optional hint for the next stage / round.",
                },
            },
            "required": ["status"],
            "additionalProperties": True,
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        ok, errors = check_submit_payload(parameters, self._stage)
        if not ok:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message="invalid submit payload: " + "; ".join(errors),
            )
        # Stamp the stage tag if the agent didn't supply one — keeps the
        # blackboard entry self-describing.
        payload = dict(parameters)
        payload.setdefault("stage", self._stage.value)
        if self._key:
            bb = load_blackboard(self._layout)
            bb[self._key] = payload
            save_blackboard(self._layout, bb)
        return ToolResponse.terminate_with(
            summary=f"{self.NAME} ok",
            payload=payload,
        )


# ---------------------------------------------------------------------------
# Candidate lifecycle — stubs
# ---------------------------------------------------------------------------
#
# Schemas are pinned so registry tests pass and the LLM-facing contract is
# settled. Bodies raise NotImplementedError; the next PR ports the real
# implementations from pipeline/tools/candidate_tools.py and integrates them
# with lora_resources.evaluation.


class _CandidateStub(Tool):
    """Shared stub base — every method is abstract or raises NotImplementedError."""

    # Marked as private (leading underscore) so the framework doesn't enforce
    # NAME/DESCRIPTION on this intermediate class.

    def __init__(self, layout: RunLayout) -> None:
        self._layout = layout

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        raise NotImplementedError(
            f"{self.NAME} is a stub; the candidate-lifecycle tools will be "
            "implemented in the next PR (operator_opt_pipe-impl)."
        )


class WriteCandidateTool(_CandidateStub):
    NAME = "write_candidate"
    DESCRIPTION = (
        "Allocate a new draft candidate slot and write its CUDA source. "
        "Returns the candidate_id; use edit_candidate for follow-up changes "
        "until the candidate is submitted."
    )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Full CUDA source for candidate.cu.",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional human-readable note about this draft.",
                },
            },
            "required": ["source"],
        }


class EditCandidateTool(_CandidateStub):
    NAME = "edit_candidate"
    DESCRIPTION = (
        "Replace the source of an existing draft candidate. Submitted "
        "candidates are frozen and cannot be edited."
    )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string"},
                "source": {
                    "type": "string",
                    "description": "Full replacement CUDA source.",
                },
            },
            "required": ["candidate_id", "source"],
        }


class VerifyCandidateTool(_CandidateStub):
    NAME = "verify_candidate"
    DESCRIPTION = (
        "Compile and correctness-check a candidate. Returns compile_ok and "
        "correctness_ok plus diagnostic text. Does NOT return runtime, "
        "speedup, or ranking — those are produced only by the RoundRunner "
        "after submit_candidate."
    )

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"candidate_id": {"type": "string"}},
            "required": ["candidate_id"],
        }


class SubmitCandidateTool(Tool):
    """Freeze a candidate and terminate the agent run with its metadata.

    In v1 the freezing semantics are minimal — write/edit/verify are stubs
    and there is no on-disk candidate state to lock. This tool simply
    validates the payload, stamps the round-step tag, and terminates so
    ``RoundRunner`` can pick the candidate up from ``AgentResult.payload``
    and hand it to ``lora_resources.evaluation.evaluate_candidate``. The
    full freeze-state guard is part of the candidate-lifecycle PR.
    """

    NAME = "submit_candidate"
    DESCRIPTION = (
        "Freeze a candidate and hand it to the RoundRunner for evaluation. "
        "The agent MUST attach a hypothesis, experiment_type, expected_effect, "
        "and risk — these are recorded with the candidate even if it loses."
    )

    def __init__(self, layout: RunLayout) -> None:
        self._layout = layout

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string"},
                "hypothesis": {
                    "type": "string",
                    "description": "What the agent expects to gain or learn from this candidate.",
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
        payload = {
            "status": "success",
            "stage": Stage.TUNING_LOOP.value,
            **parameters,
        }
        return ToolResponse.terminate_with(
            summary=f"submit_candidate {parameters['candidate_id']}",
            payload=payload,
        )


__all__ = [
    "ReadBlackboardTool",
    "SubmitTool",
    "WriteCandidateTool",
    "EditCandidateTool",
    "VerifyCandidateTool",
    "SubmitCandidateTool",
]
