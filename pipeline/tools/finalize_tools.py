"""pipeline/tools/finalize_tools.py — submit_* tools for the remaining stages.

Three closing tools:
  - SubmitHardwareProfileTool — writes hardware_profile.json
  - SubmitProfileAnalysisTool — writes candidate profile.json + analysis.md
  - SubmitSummaryTool         — writes final_report.json + summary.md

Each follows the same pattern: persist artifacts, stash a StageResult, raise
_Terminated to unwind the AgentLoop.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from agents.tools.base import Tool, ToolParameter
from agents.tools.registry import _Terminated
from agents.tools.response import ToolErrorCode, ToolResponse

from ..agent_loop_signal import stash_stage_result
from ..state import Stage, StageResult
from ..workspace_layout import RunLayout


# ===========================================================================
# SubmitHardwareProfileTool
# ===========================================================================

# Required hardware metrics per the operator skill (see lora_matmul.md §
# Required Hardware Measurements). Missing any of these downgrades the result
# to status="partial".
_HW_REQUIRED_METRICS = (
    "dram_bandwidth_gbps",
    "boost_clock_mhz",
    "sm_count",
    "l2_cache_size_mb",
    "dram_latency_cycles",
    "l2_latency_cycles",
)


class SubmitHardwareProfileTool(Tool):
    """Persist hardware/hardware_profile.json and finalize HARDWARE_PROFILE."""

    _ctx: Any = None

    def __init__(self, layout: RunLayout):
        super().__init__(
            name="submit_hardware_profile",
            description=(
                "Finalize the HARDWARE_PROFILE stage. Writes hardware_profile.json "
                "with the measured GPU parameters (DRAM bandwidth, boost clock, SM "
                "count, L2 size, DRAM/L2 latency). Missing metrics downgrade the "
                "result to status=partial — no hard failure since hardware profile "
                "is advisory for kernel optimization."
            ),
        )
        self._layout = layout

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="metrics",    type="object", description="Map of metric name → number/string."),
            ToolParameter(name="confidence", type="number", description="0..1; lower if measurements were noisy.", required=False, default=0.8),
            ToolParameter(name="caveats",    type="array",  description="Notes about throttling, locked clocks, etc.", required=False, default=[]),
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
                        "metrics":    {"type": "object", "additionalProperties": True},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "caveats":    {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["metrics"],
                },
            },
        }

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        ctx = self._ctx
        metrics = dict(parameters["metrics"])
        confidence = float(parameters.get("confidence", 0.8))
        caveats = list(parameters.get("caveats") or [])

        missing = [m for m in _HW_REQUIRED_METRICS if m not in metrics]
        status = "partial" if missing else "success"
        if missing:
            caveats.append(f"missing_metrics={missing}")

        path = self._layout.hardware_profile_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"metrics": metrics, "confidence": confidence, "caveats": caveats},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        result = StageResult(
            stage=Stage.HARDWARE_PROFILE.value,
            status=status,
            artifacts={"hardware_profile": self._layout.relpath(path)},
            metrics={"hardware": metrics, "missing_metrics": missing},
            confidence=confidence,
            caveats=caveats,
        )
        stash_stage_result(ctx, result)
        raise _Terminated(f"hardware profile submitted ({len(metrics)} metrics, {len(missing)} missing)")


# ===========================================================================
# SubmitProfileAnalysisTool
# ===========================================================================

class SubmitProfileAnalysisTool(Tool):
    """Persist a profile + Markdown analysis for the best candidate."""

    _ctx: Any = None

    def __init__(self, layout: RunLayout):
        super().__init__(
            name="submit_profile_analysis",
            description=(
                "Finalize the OPTIONAL_PROFILE stage. Writes candidates/<id>/profile.json "
                "(structured ncu/nsys metrics) and candidates/<id>/analysis.md (a 1-2 "
                "paragraph human-readable analysis of compute vs memory boundedness). "
                "Call exactly once per OPTIONAL_PROFILE invocation."
            ),
        )
        self._layout = layout

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="candidate_id", type="string", description="The best_candidate_id being profiled."),
            ToolParameter(name="profile",      type="object", description="Structured profiling metrics."),
            ToolParameter(name="analysis_md",  type="string", description="Markdown analysis text."),
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
                        "profile":      {"type": "object", "additionalProperties": True},
                        "analysis_md":  {"type": "string", "minLength": 1},
                    },
                    "required": ["candidate_id", "profile", "analysis_md"],
                },
            },
        }

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        ctx = self._ctx
        cid = parameters["candidate_id"]
        profile = parameters["profile"]
        analysis = parameters["analysis_md"]

        cdir = self._layout.candidate_dir(cid)
        if not cdir.is_dir():
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message=f"candidate_id {cid} has no directory at {self._layout.relpath(cdir)}",
            )

        prof_path = self._layout.candidate_file(cid, "profile.json")
        analysis_path = self._layout.candidate_file(cid, "analysis.md")
        prof_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
        analysis_path.write_text(analysis, encoding="utf-8")

        result = StageResult(
            stage=Stage.OPTIONAL_PROFILE.value,
            status="success",
            artifacts={
                "profile": self._layout.relpath(prof_path),
                "analysis": self._layout.relpath(analysis_path),
            },
            metrics={"candidate_id": cid},
            confidence=0.9,
        )
        stash_stage_result(ctx, result)
        raise _Terminated(f"profile + analysis submitted for {cid}")


# ===========================================================================
# SubmitSummaryTool
# ===========================================================================

class SubmitSummaryTool(Tool):
    """Persist final_report.json + summary.md at FINALIZE."""

    _ctx: Any = None

    def __init__(self, layout: RunLayout):
        super().__init__(
            name="submit_summary",
            description=(
                "Finalize the run. Writes final/final_report.json (structured) and "
                "final/summary.md (human-readable narrative). Call exactly once."
            ),
        )
        self._layout = layout

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="report",  type="object", description="Structured final report."),
            ToolParameter(name="summary_md", type="string", description="Markdown narrative."),
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
                        "report":     {"type": "object", "additionalProperties": True},
                        "summary_md": {"type": "string", "minLength": 1},
                    },
                    "required": ["report", "summary_md"],
                },
            },
        }

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        ctx = self._ctx
        report = parameters["report"]
        summary_md = parameters["summary_md"]

        self._layout.final_dir.mkdir(parents=True, exist_ok=True)
        self._layout.final_report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._layout.summary_path.write_text(summary_md, encoding="utf-8")

        result = StageResult(
            stage=Stage.FINALIZE.value,
            status="success",
            artifacts={
                "final_report": self._layout.relpath(self._layout.final_report_path),
                "summary":      self._layout.relpath(self._layout.summary_path),
            },
            metrics={"report_keys": list(report.keys()) if isinstance(report, dict) else []},
            confidence=1.0,
        )
        stash_stage_result(ctx, result)
        raise _Terminated("final summary submitted")
