"""pipeline/tools/spec_tools.py — BenchmarkSpec stage tool.

The single tool exposed in this module finalizes the BENCHMARK_SPEC stage by
writing all five benchmark spec slots to disk and stashing a StageResult.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from agents.tools.base import Tool, ToolParameter
from agents.tools.registry import _Terminated
from agents.tools.response import ToolErrorCode, ToolResponse

from ..agent_loop_signal import stash_stage_result
from ..state import BENCHMARK_SPEC_SLOTS, Stage, StageResult
from ..workspace_layout import RunLayout


# Spec versioning policy: spec content is hashed-tagged "v1" by default. We
# don't compute a content hash here — the spec_versions value is just a tag
# the orchestrator stores in RunState so resume can detect stale spec files.
DEFAULT_SPEC_VERSION = "v1"


class SubmitBenchmarkSpecsTool(Tool):
    """Persist all five benchmark specs and end the BENCHMARK_SPEC stage.

    Expects one parameter, ``specs``, an object whose keys are the five slot
    names (hardware / baseline / candidate_quick / candidate_confirm / final)
    and whose values are arbitrary JSON objects describing the benchmark
    configuration for that slot.
    """

    _ctx: Any = None  # injected by ToolRegistry

    def __init__(self, layout: RunLayout):
        super().__init__(
            name="submit_benchmark_specs",
            description=(
                "Write all five benchmark spec slots (hardware, baseline, "
                "candidate_quick, candidate_confirm, final) to disk and "
                "finalize the BENCHMARK_SPEC stage. Call this exactly once; "
                "after this call the agent loop terminates."
            ),
        )
        self._layout = layout

    # ---- schema ---------------------------------------------------------

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="specs",
                type="object",
                description=(
                    "Object with keys: hardware, baseline, candidate_quick, "
                    "candidate_confirm, final. Each value is a JSON object "
                    "describing that benchmark."
                ),
            ),
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
                        "specs": {
                            "type": "object",
                            "description": (
                                "Map of slot name → spec object. All five slots "
                                f"required: {list(BENCHMARK_SPEC_SLOTS)}."
                            ),
                            "additionalProperties": True,
                        },
                    },
                    "required": ["specs"],
                },
            },
        }

    # ---- execution ------------------------------------------------------

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        ctx = self._ctx
        specs = parameters.get("specs")
        if not isinstance(specs, dict):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message="'specs' must be an object mapping slot → spec content",
            )

        missing = [s for s in BENCHMARK_SPEC_SLOTS if s not in specs]
        if missing:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message=f"missing required slot(s): {missing}; expected {list(BENCHMARK_SPEC_SLOTS)}",
            )

        # Persist each slot. We don't reject extra keys — they're written as
        # informational artifacts but don't enter spec_versions.
        artifacts: Dict[str, str] = {}
        spec_versions: Dict[str, str] = {}

        self._layout.benchmark_specs_dir.mkdir(parents=True, exist_ok=True)
        for slot, content in specs.items():
            path = self._layout.benchmark_spec_path(slot)
            try:
                path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
            except (OSError, TypeError) as e:
                return ToolResponse.error(
                    code=ToolErrorCode.EXECUTION_ERROR,
                    message=f"failed to write {slot} spec: {type(e).__name__}: {e}",
                )
            artifacts[f"spec_{slot}"] = self._layout.relpath(path)
            if slot in BENCHMARK_SPEC_SLOTS:
                spec_versions[slot] = DEFAULT_SPEC_VERSION

        # Stash a StageResult for the StageAgent to return, then terminate
        # the AgentLoop.
        result = StageResult(
            stage=Stage.BENCHMARK_SPEC.value,
            status="success",
            artifacts=artifacts,
            metrics={"spec_versions": spec_versions, "slot_count": len(spec_versions)},
            confidence=1.0,
        )
        stash_stage_result(ctx, result)
        summary = f"submitted {len(spec_versions)} benchmark specs"
        raise _Terminated(summary)
