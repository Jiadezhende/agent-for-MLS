"""
tests/test_tools.py — Unit tests for tools/recording.py and tools/skills.py
"""
from __future__ import annotations

import pytest

from agents.tools.registry import _Terminated
from agents.tools.builtin.recording import (
    FlagEventTool,
    RecordMeasurementTool,
    SubmitResultsTool,
)
from agents.tools.builtin.skills import ListSkillsTool, ReadSkillTool
from agents.tools.response import ToolErrorCode, ToolStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record_tool(ctx):
    tool = RecordMeasurementTool()
    tool._ctx = ctx
    return tool


def _make_flag_tool(ctx):
    tool = FlagEventTool()
    tool._ctx = ctx
    return tool


def _make_submit_tool(ctx):
    tool = SubmitResultsTool()
    tool._ctx = ctx
    return tool


# ===========================================================================
# recording.py — RecordMeasurementTool
# ===========================================================================

class TestRecordMeasurement:
    def test_records_result_in_ctx(self, agent_ctx):
        tool = _make_record_tool(agent_ctx)
        resp = tool.run({
            "metric":     "dram_latency_cycles",
            "value":      876,
            "unit":       "cycles",
            "confidence": 0.9,
            "method":     "pointer-chase 256 MB",
            "evidence":   ["latency_cycles=876"],
        })
        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["ok"] is True
        assert resp.data["count"] == 1
        assert len(agent_ctx.results) == 1
        r = agent_ctx.results[0]
        assert r.metric == "dram_latency_cycles"
        assert r.value == 876
        assert r.unit == "cycles"
        assert r.confidence == 0.9

    def test_multiple_records_accumulate(self, agent_ctx):
        tool = _make_record_tool(agent_ctx)
        for i in range(3):
            tool.run({
                "metric": f"metric_{i}", "value": i,
                "unit": None, "confidence": 0.5,
                "method": "test", "evidence": [f"evidence_{i}"],
            })
        assert len(agent_ctx.results) == 3

    def test_empty_evidence_returns_error(self, agent_ctx):
        tool = _make_record_tool(agent_ctx)
        resp = tool.run({
            "metric": "x", "value": 1, "unit": None,
            "confidence": 1.0, "method": "test", "evidence": [],
        })
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == ToolErrorCode.EMPTY_EVIDENCE
        assert len(agent_ctx.results) == 0

    def test_count_field_reflects_total(self, agent_ctx):
        tool = _make_record_tool(agent_ctx)
        tool.run({"metric": "a", "value": 1, "unit": None,
                  "confidence": 1.0, "method": "m", "evidence": ["e"]})
        resp = tool.run({"metric": "b", "value": 2, "unit": None,
                         "confidence": 1.0, "method": "m", "evidence": ["e"]})
        assert resp.data["count"] == 2

    def test_string_value_accepted(self, agent_ctx):
        tool = _make_record_tool(agent_ctx)
        resp = tool.run({
            "metric": "device_name", "value": "RTX 5060", "unit": None,
            "confidence": 1.0, "method": "cudaGetDeviceProperties",
            "evidence": ["device_name=RTX 5060"],
        })
        assert resp.status == ToolStatus.SUCCESS

    def test_dict_value_accepted(self, agent_ctx):
        tool = _make_record_tool(agent_ctx)
        resp = tool.run({
            "metric": "breakdown", "value": {"l1": 30, "l2": 200, "dram": 876},
            "unit": "cycles", "confidence": 0.8,
            "method": "pointer-chase multi-tier", "evidence": ["latency_cycles=876"],
        })
        assert resp.status == ToolStatus.SUCCESS


# ===========================================================================
# recording.py — FlagEventTool
# ===========================================================================

class TestFlagEvent:
    def test_appends_to_events(self, agent_ctx):
        agent_ctx.iteration = 3
        tool = _make_flag_tool(agent_ctx)
        resp = tool.run({
            "type": "clock_throttled", "severity": "warn",
            "detail": "Measured 800 MHz vs 1500 MHz reported",
        })
        assert resp.status == ToolStatus.SUCCESS
        assert resp.data["ok"] is True
        assert len(agent_ctx.events) == 1
        e = agent_ctx.events[0]
        assert e["type"] == "clock_throttled"
        assert e["severity"] == "warn"
        assert e["iteration"] == 3

    def test_multiple_events_accumulate(self, agent_ctx):
        tool = _make_flag_tool(agent_ctx)
        tool.run({"type": "e1", "severity": "info", "detail": "d1"})
        tool.run({"type": "e2", "severity": "error", "detail": "d2"})
        assert len(agent_ctx.events) == 2

    def test_event_detail_preserved(self, agent_ctx):
        tool = _make_flag_tool(agent_ctx)
        detail = "Some very long detail string about what happened"
        tool.run({"type": "test", "severity": "info", "detail": detail})
        assert agent_ctx.events[0]["detail"] == detail


# ===========================================================================
# recording.py — SubmitResultsTool
# ===========================================================================

class TestSubmitResults:
    def test_raises_terminated(self, agent_ctx):
        tool = _make_submit_tool(agent_ctx)
        with pytest.raises(_Terminated) as exc_info:
            tool.run({"summary": "All metrics collected."})
        assert exc_info.value.summary == "All metrics collected."

    def test_stores_summary_in_memory(self, agent_ctx):
        tool = _make_submit_tool(agent_ctx)
        with pytest.raises(_Terminated):
            tool.run({"summary": "Done."})
        assert agent_ctx.memory.get("run", "summary") == "Done."

    def test_works_after_recording(self, agent_ctx):
        rec_tool = _make_record_tool(agent_ctx)
        rec_tool.run({
            "metric": "x", "value": 1, "unit": None,
            "confidence": 1.0, "method": "m", "evidence": ["e"],
        })
        sub_tool = _make_submit_tool(agent_ctx)
        with pytest.raises(_Terminated):
            sub_tool.run({"summary": "complete"})
        assert len(agent_ctx.results) == 1


# ===========================================================================
# skills.py — ListSkillsTool
# ===========================================================================

class TestListSkills:
    def test_returns_dict_with_skills_key(self):
        resp = ListSkillsTool().run({})
        assert resp.status == ToolStatus.SUCCESS
        assert "skills" in resp.data
        assert isinstance(resp.data["skills"], list)

    def test_includes_known_skills(self):
        resp = ListSkillsTool().run({})
        names = {s["name"] for s in resp.data["skills"]}
        assert "gpu_profiling_overview" in names
        assert "memory_hierarchy" in names
        assert "clock_environment" in names

    def test_excludes_template(self):
        resp = ListSkillsTool().run({})
        names = {s["name"] for s in resp.data["skills"]}
        assert "_template" not in names

    def test_excludes_readme(self):
        resp = ListSkillsTool().run({})
        names = {s["name"] for s in resp.data["skills"]}
        assert "README" not in names

    def test_each_skill_has_description(self):
        resp = ListSkillsTool().run({})
        for s in resp.data["skills"]:
            assert "description" in s
            assert len(s["description"]) > 0


# ===========================================================================
# skills.py — ReadSkillTool
# ===========================================================================

class TestReadSkill:
    def test_reads_existing_skill(self):
        resp = ReadSkillTool().run({"name": "gpu_profiling_overview"})
        assert resp.status == ToolStatus.SUCCESS
        assert "content" in resp.data
        assert len(resp.data["content"]) > 100

    def test_reads_memory_hierarchy(self):
        resp = ReadSkillTool().run({"name": "memory_hierarchy"})
        assert resp.status == ToolStatus.SUCCESS
        assert "pointer" in resp.data["content"].lower()

    def test_reads_clock_environment(self):
        resp = ReadSkillTool().run({"name": "clock_environment"})
        assert resp.status == ToolStatus.SUCCESS
        assert "clock" in resp.data["content"].lower()

    def test_missing_skill_returns_error(self):
        resp = ReadSkillTool().run({"name": "nonexistent_skill_xyz"})
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == ToolErrorCode.SKILL_NOT_FOUND

    def test_invalid_name_returns_error(self):
        resp = ReadSkillTool().run({"name": "../../etc/passwd"})
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == ToolErrorCode.INVALID_NAME

    def test_invalid_name_with_spaces(self):
        resp = ReadSkillTool().run({"name": "has spaces"})
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == ToolErrorCode.INVALID_NAME

    def test_truncated_flag_false_for_small_file(self):
        resp = ReadSkillTool().run({"name": "gpu_profiling_overview"})
        assert resp.data["truncated"] is False
