"""
tests/test_tools.py — Unit tests for tools/recording.py and tools/skills.py
"""
from __future__ import annotations

import pytest

from agents.tools.registry import _Terminated
from agents.tools.builtin.recording import flag_event, record_measurement, submit_results
from agents.tools.builtin.skills import list_skills, read_skill
from agents.tools.schemas import TOOL_SCHEMAS


# ===========================================================================
# recording.py — record_measurement
# ===========================================================================

class TestRecordMeasurement:
    def test_records_result_in_ctx(self, agent_ctx):
        ret = record_measurement(
            agent_ctx,
            metric="dram_latency_cycles",
            value=876,
            unit="cycles",
            confidence=0.9,
            method="pointer-chase 256 MB",
            evidence=["latency_cycles=876"],
        )
        assert ret["ok"] is True
        assert ret["count"] == 1
        assert len(agent_ctx.results) == 1
        r = agent_ctx.results[0]
        assert r.metric == "dram_latency_cycles"
        assert r.value == 876
        assert r.unit == "cycles"
        assert r.confidence == 0.9

    def test_multiple_records_accumulate(self, agent_ctx):
        for i in range(3):
            record_measurement(
                agent_ctx,
                metric=f"metric_{i}",
                value=i,
                unit=None,
                confidence=0.5,
                method="test",
                evidence=[f"evidence_{i}"],
            )
        assert len(agent_ctx.results) == 3

    def test_empty_evidence_returns_error(self, agent_ctx):
        ret = record_measurement(
            agent_ctx,
            metric="x",
            value=1,
            unit=None,
            confidence=1.0,
            method="test",
            evidence=[],          # empty — must be rejected
        )
        assert "error" in ret
        assert ret["error"] == "empty_evidence"
        assert len(agent_ctx.results) == 0   # nothing recorded

    def test_count_field_reflects_total(self, agent_ctx):
        record_measurement(
            agent_ctx, metric="a", value=1, unit=None,
            confidence=1.0, method="m", evidence=["e"],
        )
        ret = record_measurement(
            agent_ctx, metric="b", value=2, unit=None,
            confidence=1.0, method="m", evidence=["e"],
        )
        assert ret["count"] == 2

    def test_string_value_accepted(self, agent_ctx):
        ret = record_measurement(
            agent_ctx,
            metric="device_name",
            value="RTX 5060",
            unit=None,
            confidence=1.0,
            method="cudaGetDeviceProperties",
            evidence=["device_name=RTX 5060"],
        )
        assert ret["ok"] is True

    def test_dict_value_accepted(self, agent_ctx):
        ret = record_measurement(
            agent_ctx,
            metric="breakdown",
            value={"l1": 30, "l2": 200, "dram": 876},
            unit="cycles",
            confidence=0.8,
            method="pointer-chase multi-tier",
            evidence=["latency_cycles=876"],
        )
        assert ret["ok"] is True


# ===========================================================================
# recording.py — flag_event
# ===========================================================================

class TestFlagEvent:
    def test_appends_to_events(self, agent_ctx):
        agent_ctx.iteration = 3
        ret = flag_event(agent_ctx, type="clock_throttled", severity="warn",
                         detail="Measured 800 MHz vs 1500 MHz reported")
        assert ret["ok"] is True
        assert len(agent_ctx.events) == 1
        e = agent_ctx.events[0]
        assert e["type"] == "clock_throttled"
        assert e["severity"] == "warn"
        assert e["iteration"] == 3

    def test_multiple_events_accumulate(self, agent_ctx):
        flag_event(agent_ctx, type="e1", severity="info", detail="d1")
        flag_event(agent_ctx, type="e2", severity="error", detail="d2")
        assert len(agent_ctx.events) == 2

    def test_event_detail_preserved(self, agent_ctx):
        detail = "Some very long detail string about what happened"
        flag_event(agent_ctx, type="test", severity="info", detail=detail)
        assert agent_ctx.events[0]["detail"] == detail


# ===========================================================================
# recording.py — submit_results
# ===========================================================================

class TestSubmitResults:
    def test_raises_terminated(self, agent_ctx):
        with pytest.raises(_Terminated) as exc_info:
            submit_results(agent_ctx, summary="All metrics collected.")
        assert exc_info.value.summary == "All metrics collected."

    def test_stores_summary_in_memory(self, agent_ctx):
        with pytest.raises(_Terminated):
            submit_results(agent_ctx, summary="Done.")
        assert agent_ctx.memory.get("run", "summary") == "Done."

    def test_works_after_recording(self, agent_ctx):
        record_measurement(
            agent_ctx, metric="x", value=1, unit=None,
            confidence=1.0, method="m", evidence=["e"],
        )
        with pytest.raises(_Terminated):
            submit_results(agent_ctx, summary="complete")
        assert len(agent_ctx.results) == 1


# ===========================================================================
# skills.py — list_skills
# ===========================================================================

class TestListSkills:
    def test_returns_dict_with_skills_key(self):
        result = list_skills()
        assert "skills" in result
        assert isinstance(result["skills"], list)

    def test_includes_known_skills(self):
        result = list_skills()
        names = {s["name"] for s in result["skills"]}
        assert names == {
            "gpu_profiling_overview",
            "memory_hierarchy",
            "throughput_resources",
            "clock_environment",
        }

    def test_excludes_template(self):
        result = list_skills()
        names = {s["name"] for s in result["skills"]}
        assert "_template" not in names

    def test_excludes_readme(self):
        result = list_skills()
        names = {s["name"] for s in result["skills"]}
        assert "README" not in names

    def test_each_skill_has_summary(self):
        result = list_skills()
        for s in result["skills"]:
            assert "summary" in s
            assert len(s["summary"]) > 0


# ===========================================================================
# skills.py — read_skill
# ===========================================================================

class TestReadSkill:
    def test_reads_existing_skill(self):
        result = read_skill("gpu_profiling_overview")
        assert "error" not in result
        assert "content" in result
        assert len(result["content"]) > 100

    def test_reads_memory_hierarchy(self):
        result = read_skill("memory_hierarchy")
        assert "error" not in result
        assert "pointer" in result["content"].lower()

    def test_reads_clock_environment(self):
        result = read_skill("clock_environment")
        assert "error" not in result
        assert "clock" in result["content"].lower()

    def test_reads_throughput_resources(self):
        result = read_skill("throughput_resources")
        assert "error" not in result
        assert "bandwidth" in result["content"].lower()

    def test_missing_skill_returns_error(self):
        result = read_skill("nonexistent_skill_xyz")
        assert result["error"] == "skill_not_found"
        assert "available" in result

    def test_invalid_name_returns_error(self):
        result = read_skill("../../etc/passwd")
        assert result["error"] == "invalid_name"

    def test_invalid_name_with_spaces(self):
        result = read_skill("has spaces")
        assert result["error"] == "invalid_name"

    def test_truncated_flag_false_for_small_file(self):
        result = read_skill("gpu_profiling_overview")
        assert result["truncated"] is False


class TestToolSchemas:
    def test_profile_with_ncu_requires_binary_path_only(self):
        schema_map = {s["function"]["name"]: s for s in TOOL_SCHEMAS}
        params = schema_map["profile_with_ncu"]["function"]["parameters"]
        props = params["properties"]
        assert "binary_path" in props
        assert "source_type" not in props
        assert "source_or_path" not in props
        assert "compile_flags" not in props
        assert params["required"] == ["binary_path", "kernel_name", "metrics"]
