"""Unit tests for operator_opt_pipe.tools."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mls_agent import ToolErrorCode, ToolRegistry, ToolStatus

from operator_opt_pipe.state import (
    RunLayout,
    Stage,
    load_blackboard,
    save_blackboard,
)
from operator_opt_pipe.tools import (
    EditCandidateTool,
    ReadBlackboardTool,
    SubmitCandidateTool,
    SubmitTool,
    VerifyCandidateTool,
    WriteCandidateTool,
)


@pytest.fixture
def layout(tmp_path: Path) -> RunLayout:
    lay = RunLayout(workspace_root=tmp_path, run_id="r")
    lay.mkdir()
    return lay


# ---------------------------------------------------------------------------
# ReadBlackboardTool
# ---------------------------------------------------------------------------


def test_read_blackboard_returns_existing_key(layout: RunLayout):
    save_blackboard(layout, {"schema_version": 1, "history": [], "best": {"speedup": 1.4}})
    tool = ReadBlackboardTool(layout)
    resp = tool.run({"key": "best"})
    assert resp.status is ToolStatus.SUCCESS
    assert resp.data["present"] is True
    assert resp.data["value"] == {"speedup": 1.4}


def test_read_blackboard_returns_default_when_missing(layout: RunLayout):
    tool = ReadBlackboardTool(layout)
    resp = tool.run({"key": "absent", "default": "fallback"})
    assert resp.data["value"] == "fallback"
    assert resp.data["present"] is False


def test_read_blackboard_schema_via_registry(layout: RunLayout):
    """End-to-end: registry-level dispatch + JSON-Schema validation."""
    reg = ToolRegistry()
    reg.register(ReadBlackboardTool(layout))
    resp = reg.dispatch("read_blackboard", {})  # missing required 'key'
    assert resp.status is ToolStatus.ERROR
    assert resp.error_info["code"] == ToolErrorCode.INVALID_ARGS


# ---------------------------------------------------------------------------
# SubmitTool (generic)
# ---------------------------------------------------------------------------


def test_submit_tool_writes_blackboard_and_terminates(layout: RunLayout):
    tool = SubmitTool(
        name="submit_hardware_profile",
        layout=layout,
        blackboard_key="hardware",
        expected_stage=Stage.HARDWARE_PROFILE,
    )
    payload_in = {"status": "success", "metrics": {"sm": 30}}
    resp = tool.run(payload_in)

    assert resp.status is ToolStatus.SUCCESS
    assert resp.terminate is True
    assert resp.terminate_payload["status"] == "success"
    # Stage is auto-stamped from expected_stage when missing.
    assert resp.terminate_payload["stage"] == Stage.HARDWARE_PROFILE.value
    # Blackboard now carries the payload under the configured key.
    bb = load_blackboard(layout)
    assert bb["hardware"]["metrics"]["sm"] == 30


def test_submit_tool_rejects_invalid_status(layout: RunLayout):
    tool = SubmitTool(
        name="submit_diagnosis",
        layout=layout,
        blackboard_key="latest_diagnosis",
        expected_stage=Stage.TUNING_LOOP,
    )
    resp = tool.run({"status": "bogus"})
    assert resp.status is ToolStatus.ERROR
    assert "invalid submit payload" in resp.text


def test_submit_tool_rejects_stage_mismatch(layout: RunLayout):
    tool = SubmitTool(
        name="submit_summary",
        layout=layout,
        blackboard_key="final_summary",
        expected_stage=Stage.FINALIZE,
    )
    resp = tool.run({"status": "success", "stage": "HARDWARE_PROFILE"})
    assert resp.status is ToolStatus.ERROR


def test_submit_tool_per_instance_name():
    """SubmitTool's NAME is set per-instance so one class covers many roles."""
    layout = RunLayout(workspace_root=Path("."), run_id="r")
    a = SubmitTool(name="submit_diagnosis", layout=layout, blackboard_key=None,
                   expected_stage=Stage.TUNING_LOOP)
    b = SubmitTool(name="submit_summary", layout=layout, blackboard_key=None,
                   expected_stage=Stage.FINALIZE)
    assert a.NAME == "submit_diagnosis"
    assert b.NAME == "submit_summary"


# ---------------------------------------------------------------------------
# Candidate stubs
# ---------------------------------------------------------------------------


def test_candidate_lifecycle_stubs_raise(layout: RunLayout):
    for cls in (WriteCandidateTool, EditCandidateTool, VerifyCandidateTool):
        tool = cls(layout)
        with pytest.raises(NotImplementedError):
            tool.run({"candidate_id": "candidate_000", "source": "// ..."})


def test_submit_candidate_terminates_with_payload(layout: RunLayout):
    tool = SubmitCandidateTool(layout)
    resp = tool.run({
        "candidate_id": "candidate_005",
        "hypothesis": "fused W*X with low-rank correction",
        "experiment_type": "fused-correction",
    })
    assert resp.terminate is True
    payload = resp.terminate_payload
    assert payload["candidate_id"] == "candidate_005"
    assert payload["status"] == "success"
    assert payload["stage"] == Stage.TUNING_LOOP.value
