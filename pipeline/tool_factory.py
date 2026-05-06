"""pipeline/tool_factory.py — Stage-scoped ToolRegistry builder.

Replaces the old, single ToolFactory. Differences:
  - Layout-aware: layout is passed to tools that need it (e.g. submit_*).
  - Per-stage: each stage_runner.run_stage() call asks for a fresh registry
    restricted to the agent's allowed_tools whitelist.
  - Reuses existing builtin tools (record_measurement, flag_event,
    profile_with_torch, run_cuda_probe, ...) by delegating to the same
    underlying classes — no duplication.
"""
from __future__ import annotations

from typing import Any, Sequence

from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry

from .operator_spec import OperatorSpec
from .state import Stage
from .workspace_layout import RunLayout


class StageToolFactory:
    """Build a per-stage ToolRegistry restricted to ``allowed_tools``.

    Constructor takes the durable bits (executor, layout, op_spec). Each
    ``build()`` call returns a fresh ToolRegistry with brand-new tool
    instances so per-stage state (like an internal counter) doesn't leak
    across stages.

    The orchestrator passes ``current_stage`` into ``build()`` so that
    submit_candidate_result knows whether to tag its StageResult as
    INITIAL_CANDIDATE or TUNING_LOOP.

    ``op_spec`` is the operator schema parsed from skills/operators/<name>.md;
    it is injected into the generate_baseline / write_candidate /
    evaluate_candidate tools so their runtime templates render the correct
    tensors and reference formula.
    """

    def __init__(self, *, executor: Any, layout: RunLayout, op_spec: OperatorSpec | None = None):
        self._executor = executor
        self._layout = layout
        self._op_spec = op_spec

    def build(self, allowed_tools: Sequence[str], *, current_stage: Stage = Stage.INITIAL_CANDIDATE) -> ToolRegistry:
        # Lazy imports keep this module importable in environments without
        # the full executor + torch chain installed (useful for unit tests
        # that build only spec_tools / recording).
        all_tools = self._build_all_tools(current_stage)

        unknown = [t for t in allowed_tools if t not in all_tools]
        if unknown:
            raise ValueError(
                f"unknown tool(s) for StageToolFactory: {unknown}; "
                f"available={sorted(all_tools)}"
            )

        reg = ToolRegistry()
        for name in allowed_tools:
            reg.register(all_tools[name])
        return reg

    # ---- the master tool catalogue --------------------------------------

    def _build_all_tools(self, current_stage: Stage) -> dict[str, Tool]:
        # Reused builtin tools (no behavioural changes).
        from agent.tools.builtin.recording import (
            FlagEventTool,
            RecordMeasurementTool,
        )
        from agent.tools.builtin.skills import ListSkillsTool, ReadSkillTool

        # New stage-specific tools.
        from .tools.spec_tools import SubmitBenchmarkSpecsTool
        from .tools.candidate_tools import (
            EvaluateCandidateTool,
            SubmitCandidateResultTool,
            WriteCandidateTool,
        )
        from .tools.baseline_tools import (
            GenerateBaselineTool,
            SubmitBaselineTool,
        )
        from .tools.finalize_tools import (
            SubmitHardwareProfileTool,
            SubmitProfileAnalysisTool,
            SubmitSummaryTool,
        )

        catalogue: dict[str, Tool] = {
            # Reused — read-only knowledge access.
            "list_skills":          ListSkillsTool(),
            "read_skill":           ReadSkillTool(),
            # Reused — generic recording for any stage.
            "record_measurement":   RecordMeasurementTool(),
            "flag_event":           FlagEventTool(),
            # New — BenchmarkSpec stage.
            "submit_benchmark_specs": SubmitBenchmarkSpecsTool(self._layout),
            # New — Candidate stages.
            "submit_candidate_result": SubmitCandidateResultTool(layout=self._layout, stage=current_stage),
            # New — Baseline stage finalize.
            "submit_baseline":         SubmitBaselineTool(self._layout),
            # New — Other stage finalizers.
            "submit_hardware_profile": SubmitHardwareProfileTool(self._layout),
            "submit_profile_analysis": SubmitProfileAnalysisTool(self._layout),
            "submit_summary":          SubmitSummaryTool(self._layout),
        }

        # Operator-aware tools. write_candidate is always registered when
        # op_spec is provided (no executor needed); generate_baseline and
        # evaluate_candidate also require an executor.
        if self._op_spec is not None:
            catalogue["write_candidate"] = WriteCandidateTool(self._layout, self._op_spec)

        # Optional reused tools that need the executor; only register when
        # an executor is available so unit tests without GPU still work.
        if self._executor is not None:
            from agent.tools.executor_tools import (
                FindBinaryTool,
                ProbeEnvironmentTool,
                ProfileWithNcuTool,
                ProfileWithNsysTool,
                ProfileWithTorchTool,
                RunCudaProbeTool,
                WriteWorkspaceFileTool,
            )
            catalogue.update({
                "write_workspace_file": WriteWorkspaceFileTool(self._executor),
                "run_cuda_probe":       RunCudaProbeTool(self._executor),
                "profile_with_ncu":     ProfileWithNcuTool(self._executor),
                "profile_with_nsys":    ProfileWithNsysTool(self._executor),
                "profile_with_torch":   ProfileWithTorchTool(self._executor),
                "probe_environment":    ProbeEnvironmentTool(self._executor),
                "find_binary":          FindBinaryTool(self._executor),
            })
            if self._op_spec is not None:
                catalogue["generate_baseline"] = GenerateBaselineTool(
                    executor=self._executor, layout=self._layout, op_spec=self._op_spec
                )
                catalogue["evaluate_candidate"] = EvaluateCandidateTool(
                    executor=self._executor, layout=self._layout, op_spec=self._op_spec
                )

        return catalogue
