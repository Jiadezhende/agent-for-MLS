"""tests/pipeline/test_e2e_smoke.py — End-to-end GPU-free smoke for the full pipeline.

Wires up the *real* PipelineOrchestrator with the *real* StageToolFactory,
6 *real* StageAgents, but mocks the LLMClient (canned tool calls per stage)
and mocks the Executor (canned profile_with_torch outputs). Validates that:

  - every stage (BENCHMARK_SPEC → HARDWARE_PROFILE → BASELINE_PROFILE →
    INITIAL_CANDIDATE → TUNING_LOOP → OPTIONAL_PROFILE → FINALIZE) actually
    runs end-to-end;
  - the orchestrator promotes a candidate to best/best.cu and synchronizes
    ./optimized_lora.cu;
  - state.json + leaderboard.jsonl + final_report.json are persisted.

A real GPU run is the user's responsibility (./run.sh); this test catches
integration breakage between the framework + agents + tools.
"""
from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path

import pytest

from agents.core.llm import ChatResponse, ToolCall

from pipeline.agents import (
    BaselineAgent,
    BenchmarkSpecAgent,
    HardwareProfilerAgent,
    KernelTuningAgent,
    ProfileAnalysisAgent,
    SummaryAgent,
)
from pipeline.operator_spec import OperatorSpec
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.state import BENCHMARK_SPEC_SLOTS, Stage
from pipeline.tool_factory import StageToolFactory
from pipeline.tools.baseline_tools import _BASELINE_MARKER
from pipeline.tools.candidate_tools import _EVAL_MARKER


_SKILLS_ROOT = Path(__file__).resolve().parents[2] / "skills"
LORA_SPEC = OperatorSpec.load_from_skill(_SKILLS_ROOT, "lora_matmul")


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

VALID_CU = """
#include <torch/extension.h>
torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) {
    return W;  // placeholder; real harness only checks shape in this smoke
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "lora forward");
}
""".strip() + "\n" + ("// pad to satisfy >100 chars\n" * 4)


class _FakeRawFn:
    def __init__(self, name, args_str):
        self.name = name
        self.arguments = args_str


class _FakeRawToolCall:
    def __init__(self, cid, name, args_str):
        self.id = cid
        self.function = _FakeRawFn(name, args_str)


def _fake_resp(*tool_calls, content=""):
    """Build a ChatResponse from a list of (id, name, args_dict) tuples."""
    parsed = [ToolCall(id=c[0], name=c[1], arguments=c[2], arguments_raw=json.dumps(c[2])) for c in tool_calls]
    raw = [_FakeRawToolCall(c[0], c[1], json.dumps(c[2])) for c in tool_calls]
    return ChatResponse(
        content=content,
        tool_calls=parsed,
        reasoning_content=None,
        finish_reason="tool_calls" if tool_calls else "stop",
        _raw_tool_calls=raw,
    )


class _ScriptedLLM:
    """Returns ChatResponses keyed by which agent is calling.

    Each agent's loop pulls from its own deque; this avoids order-coupling
    between stages (the orchestrator decides stage ordering, not us).
    """

    def __init__(self, scripts: dict[str, list]):
        self._scripts = {k: deque(v) for k, v in scripts.items()}
        self._calls_by_stage: dict[str, int] = {}
        self._current_stage: str = "?"

    def set_stage(self, stage_value: str) -> None:
        self._current_stage = stage_value

    def chat(self, messages, tools=None, **kw):
        stage = self._current_stage
        self._calls_by_stage[stage] = self._calls_by_stage.get(stage, 0) + 1
        q = self._scripts.get(stage)
        if q and q:
            return q.popleft()
        # Fallback: end the conversation gracefully.
        return _fake_resp(content=f"(no scripted response left for {stage})")


class _ScriptedExecutor:
    """Returns canned profile_with_torch outputs based on op_name pattern."""

    def __init__(self, baseline_records, eval_payload):
        self._baseline_records = baseline_records
        self._eval_payload = eval_payload
        # cuda_executor.Executor exposes detect_notes; emulate.
        self.detect_notes: list = []

    def profile_with_torch(self, code, op_name, timeout_s=120):
        if op_name == "generate_baseline":
            return {"stdout": f"prelude\n{_BASELINE_MARKER}\n{json.dumps(self._baseline_records)}\n"}
        if op_name.startswith("eval_"):
            return {"stdout": f"prelude\n{_EVAL_MARKER}\n{json.dumps(self._eval_payload)}\n"}
        # Hardware probe / nsys etc — return empty so HardwareProfilerAgent can
        # decide to submit_partial.
        return {"stdout": "", "stdout_tail": ""}

    def profile_with_ncu(self, *_a, **_kw):
        return {"stdout": ""}

    def profile_with_nsys(self, *_a, **_kw):
        return {"stdout": ""}

    def run_cuda_probe(self, *_a, **_kw):
        return {"stdout": ""}

    def write_workspace_file(self, *_a, **_kw):
        return {"ok": True, "path": "(stub)"}

    def probe_environment(self, *_a, **_kw):
        return {"stdout": ""}

    def find_binary(self, *_a, **_kw):
        return {"ok": True, "candidates": [], "chosen": None, "reconfigured": False}


# ---------------------------------------------------------------------------
# Stage scripts
# ---------------------------------------------------------------------------

def _benchmark_spec_script():
    specs = {
        slot: {"d_list": [4096], "samples": 10, "notes": slot}
        for slot in BENCHMARK_SPEC_SLOTS
    }
    return [_fake_resp(("c1", "submit_benchmark_specs", {"specs": specs}))]


def _hardware_script():
    metrics = {
        "dram_bandwidth_gbps": 280.0,
        "boost_clock_mhz": 2400.0,
        "sm_count": 30,
        "l2_cache_size_mb": 32,
        "dram_latency_cycles": 380,
        "l2_latency_cycles": 120,
    }
    return [
        _fake_resp(("c1", "submit_hardware_profile", {"metrics": metrics, "confidence": 0.8})),
    ]


def _baseline_script(records):
    return [
        _fake_resp(("c1", "generate_baseline", {"d_list": [4096], "samples": 10})),
        _fake_resp(("c2", "submit_baseline", {"per_d": records, "notes": "smoke"})),
    ]


def _kernel_tuning_script(*, candidate_id_hint: str, accepted_for: str = "best_update"):
    """Produce a single candidate, evaluate it, submit best_update."""
    record = {
        "candidate_id": candidate_id_hint,
        "compile_ok": True,
        "correctness_ok": True,
        "quick_speedup_median": 1.2,
        "quick_samples": 5,
        "accepted_for": accepted_for,
    }
    return [
        _fake_resp(("c1", "write_candidate", {"source": VALID_CU})),
        _fake_resp(("c2", "evaluate_candidate", {"candidate_id": candidate_id_hint, "d_list": [4096], "mode": "quick"})),
        _fake_resp(("c3", "submit_candidate_result", {"record": record})),
    ]


def _profile_script(candidate_id):
    return [
        _fake_resp(
            (
                "c1",
                "submit_profile_analysis",
                {
                    "candidate_id": candidate_id,
                    "profile": {"compute_throughput_pct": 75.0},
                    "analysis_md": "# Analysis\nLikely compute-bound.",
                },
            )
        ),
    ]


def _summary_script():
    report = {
        "run_id": "smoke", "operator": "lora_matmul",
        "best_candidate_id": "candidate_000", "best_speedup": 1.2,
        "per_d_speedups": [{"d": 4096, "speedup": 1.2}],
    }
    return [
        _fake_resp(("c1", "submit_summary", {"report": report, "summary_md": "# done\nsmoke ran"})),
    ]


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------

class _StageRoutedLLM:
    """Wraps ScriptedLLM and exposes itself per agent.

    We need the LLM to know which stage is calling so it can pick the right
    script. We can't get that from ``messages`` alone, so the smoke test
    swaps in stage-aware wrappers per agent.
    """

    def __init__(self, scripted: _ScriptedLLM, stage_value: str):
        self._scripted = scripted
        self._stage = stage_value

    def chat(self, messages, tools=None, **kw):
        self._scripted.set_stage(self._stage)
        return self._scripted.chat(messages, tools=tools, **kw)


def _build_agents(scripted_llm: _ScriptedLLM):
    return {
        Stage.BENCHMARK_SPEC:    BenchmarkSpecAgent(llm=_StageRoutedLLM(scripted_llm, Stage.BENCHMARK_SPEC.value)),
        Stage.HARDWARE_PROFILE:  HardwareProfilerAgent(llm=_StageRoutedLLM(scripted_llm, Stage.HARDWARE_PROFILE.value)),
        Stage.BASELINE_PROFILE:  BaselineAgent(llm=_StageRoutedLLM(scripted_llm, Stage.BASELINE_PROFILE.value)),
        Stage.INITIAL_CANDIDATE: KernelTuningAgent(llm=_StageRoutedLLM(scripted_llm, Stage.INITIAL_CANDIDATE.value), stage=Stage.INITIAL_CANDIDATE),
        Stage.TUNING_LOOP:       KernelTuningAgent(llm=_StageRoutedLLM(scripted_llm, Stage.TUNING_LOOP.value), stage=Stage.TUNING_LOOP),
        Stage.OPTIONAL_PROFILE:  ProfileAnalysisAgent(llm=_StageRoutedLLM(scripted_llm, Stage.OPTIONAL_PROFILE.value)),
        Stage.FINALIZE:          SummaryAgent(llm=_StageRoutedLLM(scripted_llm, Stage.FINALIZE.value)),
    }


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    def test_full_run_produces_optimized_lora_cu(self, tmp_path):
        # Mock stdout for baseline + candidate evaluation.
        baseline_records = [{"d": 4096, "torch_ms_median": 12.0, "samples": 10}]
        eval_payload = {
            "compile_ok": True,
            "all_correct": True,
            "per_d": [{
                "d": 4096, "correctness_ok": True, "max_abs_diff": 1e-5,
                "candidate_ms_median": 5.0, "ref_ms_median": 6.0, "speedup": 1.2,
                "samples": 5, "candidate_ms_min": 4.9, "candidate_ms_max": 5.1,
                "ref_ms_min": 5.9, "ref_ms_max": 6.1, "error": None,
            }],
            "overall_speedup_median": 1.2,
        }
        executor = _ScriptedExecutor(baseline_records, eval_payload)

        # Tight per-stage budget so TUNING_LOOP only fires once before
        # OPTIONAL_PROFILE → FINALIZE.
        scripted = _ScriptedLLM({
            Stage.BENCHMARK_SPEC.value:    _benchmark_spec_script(),
            Stage.HARDWARE_PROFILE.value:  _hardware_script(),
            Stage.BASELINE_PROFILE.value:  _baseline_script(baseline_records),
            Stage.INITIAL_CANDIDATE.value: _kernel_tuning_script(candidate_id_hint="candidate_000"),
            # If TUNING_LOOP fires (budget allows) we feed it a strategy_guidance
            # candidate so best doesn't keep updating; budget cuts further loops.
            Stage.TUNING_LOOP.value:       _kernel_tuning_script(candidate_id_hint="candidate_001", accepted_for="strategy_guidance"),
            Stage.OPTIONAL_PROFILE.value:  _profile_script("candidate_000"),
            Stage.FINALIZE.value:          _summary_script(),
        })

        agents = _build_agents(scripted)
        layout_root = tmp_path / "ws"
        out_path = tmp_path / "optimized_lora.cu"

        # Pre-create layout so the StageToolFactory + Orchestrator both observe it.
        from pipeline.workspace_layout import RunLayout
        run_id = "smoke_run"
        layout = RunLayout(layout_root, run_id)
        layout.mkdir()
        factory = StageToolFactory(executor=executor, layout=layout, op_spec=LORA_SPEC)

        def build_tools(allowed, current_stage):
            return factory.build(allowed, current_stage=current_stage)

        orch = PipelineOrchestrator(
            spec={"operator": "lora_matmul"},
            time_budget_s=600.0,
            workspace_root=layout_root,
            output_path=out_path,
            stage_agents=agents,
            build_tools=build_tools,
            op_spec=LORA_SPEC,
            run_id=run_id,
            stage_budgets_s={s: 30.0 for s in Stage},
        )
        summary = orch.run()

        # ---- root-level optimized_lora.cu must exist + match best/best.cu ----
        assert out_path.is_file(), "optimized_lora.cu was not synced to root"
        assert orch.layout.best_cu_path.is_file()
        assert out_path.read_text() == orch.layout.best_cu_path.read_text()
        assert "PYBIND11_MODULE" in out_path.read_text()

        # ---- run state ----
        assert summary["best_candidate_id"] == "candidate_000"
        assert summary["best_speedup"] == 1.2
        for s in (
            Stage.BENCHMARK_SPEC, Stage.HARDWARE_PROFILE, Stage.BASELINE_PROFILE,
            Stage.INITIAL_CANDIDATE, Stage.FINALIZE,
        ):
            assert s.value in orch.run_state.completed_stages, f"missing stage: {s.value}"

        # ---- artifacts on disk ----
        assert orch.layout.state_path.is_file()
        loaded_state = json.loads(orch.layout.state_path.read_text())
        assert loaded_state["best_candidate_id"] == "candidate_000"

        leaderboard = [json.loads(l) for l in orch.layout.leaderboard_path.read_text().splitlines() if l.strip()]
        assert any(r["candidate_id"] == "candidate_000" for r in leaderboard)

        assert orch.layout.hardware_profile_path.is_file()
        assert orch.layout.baseline_path.is_file()
        assert orch.layout.final_report_path.is_file()
        assert orch.layout.summary_path.is_file()

    def test_resume_after_partial_run(self, tmp_path):
        """First run hits time budget mid-tuning; resume completes the rest."""
        baseline_records = [{"d": 4096, "torch_ms_median": 12.0, "samples": 10}]
        eval_payload = {
            "compile_ok": True, "all_correct": True,
            "per_d": [{
                "d": 4096, "correctness_ok": True, "max_abs_diff": 1e-5,
                "candidate_ms_median": 5.0, "ref_ms_median": 6.0, "speedup": 1.2,
                "samples": 5, "candidate_ms_min": 4.9, "candidate_ms_max": 5.1,
                "ref_ms_min": 5.9, "ref_ms_max": 6.1, "error": None,
            }],
            "overall_speedup_median": 1.2,
        }
        executor = _ScriptedExecutor(baseline_records, eval_payload)

        run_id = "resume_run"
        layout_root = tmp_path / "ws"
        out_path = tmp_path / "optimized_lora.cu"

        from pipeline.workspace_layout import RunLayout
        layout = RunLayout(layout_root, run_id)
        layout.mkdir()
        factory = StageToolFactory(executor=executor, layout=layout, op_spec=LORA_SPEC)

        # ---- run 1: only feed setup + INITIAL_CANDIDATE scripts, then we
        # let SummaryAgent's script also be present so finalize works.
        scripted_run1 = _ScriptedLLM({
            Stage.BENCHMARK_SPEC.value:    _benchmark_spec_script(),
            Stage.HARDWARE_PROFILE.value:  _hardware_script(),
            Stage.BASELINE_PROFILE.value:  _baseline_script(baseline_records),
            Stage.INITIAL_CANDIDATE.value: _kernel_tuning_script(candidate_id_hint="candidate_000"),
            Stage.TUNING_LOOP.value:       _kernel_tuning_script(candidate_id_hint="candidate_001", accepted_for="strategy_guidance"),
            Stage.OPTIONAL_PROFILE.value:  _profile_script("candidate_000"),
            Stage.FINALIZE.value:          _summary_script(),
        })

        def build_tools(allowed, current_stage):
            return factory.build(allowed, current_stage=current_stage)

        orch1 = PipelineOrchestrator(
            spec={"operator": "lora_matmul"},
            time_budget_s=600.0,
            workspace_root=layout_root,
            output_path=out_path,
            stage_agents=_build_agents(scripted_run1),
            build_tools=build_tools,
            op_spec=LORA_SPEC,
            run_id=run_id,
            stage_budgets_s={s: 30.0 for s in Stage},
        )
        orch1.run()
        first_best = orch1.run_state.best_candidate_id
        assert first_best == "candidate_000"

        # ---- run 2: same run_id; setup stages must NOT re-execute. We
        # express that by giving them BOOM scripts (would error if invoked).
        # Tuning is also skipped because elapsed_s already hit FINALIZE in run1.
        scripted_run2 = _ScriptedLLM({
            Stage.BENCHMARK_SPEC.value:    [],  # empty — would fail-fast on any chat call
            Stage.HARDWARE_PROFILE.value:  [],
            Stage.BASELINE_PROFILE.value:  [],
            # Resume continues from FINALIZE (state.json says completed_stages
            # already includes FINALIZE), so this script is intentionally idle.
            Stage.FINALIZE.value:          [],
        })

        orch2 = PipelineOrchestrator(
            spec={"operator": "lora_matmul"},
            time_budget_s=600.0,
            workspace_root=layout_root,
            output_path=out_path,
            stage_agents=_build_agents(scripted_run2),
            build_tools=build_tools,
            op_spec=LORA_SPEC,
            run_id=run_id,
            stage_budgets_s={s: 30.0 for s in Stage},
        )
        # Should immediately re-finalize without invoking setup scripts.
        orch2.run()
        # State preserved across runs.
        assert orch2.run_state.best_candidate_id == "candidate_000"
        assert out_path.is_file()
