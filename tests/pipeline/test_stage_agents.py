"""tests/pipeline/test_stage_agents.py — LLMStageAgent base + concrete agent integration.

These tests mock the LLMClient: each test feeds a pre-baked sequence of tool
calls and verifies that the stage agent drives the AgentLoop correctly,
ending in a valid StageResult.
"""
from __future__ import annotations

import json

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
from pipeline.stage_runner import StageContext
from pipeline.state import (
    BENCHMARK_SPEC_SLOTS,
    RunState,
    Stage,
)
from pipeline.tool_factory import StageToolFactory
from pipeline.workspace_layout import RunLayout


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class _FakeChatResponse(ChatResponse):
    """ChatResponse with a fake _raw_tool_calls that to_openai_message can serialize."""

    @classmethod
    def from_calls(cls, calls: list[tuple[str, str, dict]], content: str = "") -> "_FakeChatResponse":
        """calls = [(id, name, arguments_dict), ...]"""
        tool_calls = [ToolCall(id=cid, name=name, arguments=args, arguments_raw=json.dumps(args)) for (cid, name, args) in calls]
        # Fake _raw_tool_calls that to_openai_message can serialize.
        raw = [_FakeRawToolCall(cid, name, json.dumps(args)) for (cid, name, args) in calls]
        return cls(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=None,
            finish_reason="tool_calls" if calls else "stop",
            _raw_tool_calls=raw,
        )


class _FakeRawToolCall:
    def __init__(self, id, name, args_str):
        self.id = id
        self.function = _FakeRawFn(name, args_str)


class _FakeRawFn:
    def __init__(self, name, args):
        self.name = name
        self.arguments = args


class _FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list = []

    def chat(self, messages, tools=None, **kw):
        self.calls.append({"messages_n": len(messages), "tools_n": len(tools) if tools else 0})
        if self._responses:
            return self._responses.pop(0)
        return _FakeChatResponse.from_calls([], content="(out of canned responses)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layout(tmp_path) -> RunLayout:
    layout = RunLayout(tmp_path, "run_test")
    layout.mkdir()
    return layout


def _ctx_factory(layout, tools, *, run_state=None):
    if run_state is None:
        run_state = RunState(run_id=layout.run_id, operator="lora_matmul")
    return StageContext(
        run_state=run_state,
        layout=layout,
        tools=tools,
        stage_budget_s=60.0,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# BenchmarkSpecAgent
# ---------------------------------------------------------------------------

class TestBenchmarkSpecAgent:
    def test_one_shot_submit(self, tmp_path):
        layout = _layout(tmp_path)
        factory = StageToolFactory(executor=None, layout=layout)
        tools = factory.build(BenchmarkSpecAgent.allowed_tools, current_stage=Stage.BENCHMARK_SPEC)

        specs = {slot: {"d_list": [3584, 4096, 4608], "samples": 30, "notes": slot} for slot in BENCHMARK_SPEC_SLOTS}
        llm = _FakeLLM([
            _FakeChatResponse.from_calls([("c1", "submit_benchmark_specs", {"specs": specs})]),
        ])

        agent = BenchmarkSpecAgent(llm=llm)
        ctx = _ctx_factory(layout, tools)
        result = agent.run(ctx)

        assert result.status == "success"
        assert result.stage == Stage.BENCHMARK_SPEC.value
        assert set(result.metrics["spec_versions"].keys()) == set(BENCHMARK_SPEC_SLOTS)

    def test_no_submit_yields_failed_result(self, tmp_path):
        layout = _layout(tmp_path)
        factory = StageToolFactory(executor=None, layout=layout)
        tools = factory.build(BenchmarkSpecAgent.allowed_tools, current_stage=Stage.BENCHMARK_SPEC)

        # LLM keeps emitting empty content (no tool calls). After two such
        # responses AgentLoop raises RuntimeError → agent returns failed.
        llm = _FakeLLM([
            _FakeChatResponse.from_calls([], content="thinking..."),
            _FakeChatResponse.from_calls([], content="still thinking..."),
        ])
        agent = BenchmarkSpecAgent(llm=llm)
        ctx = _ctx_factory(layout, tools)
        result = agent.run(ctx)

        assert result.status == "failed"
        assert any("agent_loop_aborted" in c for c in result.caveats)


# ---------------------------------------------------------------------------
# BaselineAgent (mocked tool — no real GPU)
# ---------------------------------------------------------------------------

class TestBaselineAgent:
    def test_generate_then_submit(self, tmp_path):
        layout = _layout(tmp_path)

        # Mock executor returns canned baseline subprocess output.
        records = [{"d": 4096, "torch_ms_median": 12.0, "samples": 30}]
        from pipeline.tools.baseline_tools import _BASELINE_MARKER

        class _MockExec:
            def profile_with_torch(self, code, op_name, timeout_s=120):
                return {"stdout": f"{_BASELINE_MARKER}\n{json.dumps(records)}\n"}

        factory = StageToolFactory(executor=_MockExec(), layout=layout)
        tools = factory.build(BaselineAgent.allowed_tools, current_stage=Stage.BASELINE_PROFILE)

        llm = _FakeLLM([
            _FakeChatResponse.from_calls([("c1", "generate_baseline", {"d_list": [4096], "samples": 30})]),
            _FakeChatResponse.from_calls([("c2", "submit_baseline", {"per_d": records, "notes": "ok"})]),
        ])
        agent = BaselineAgent(llm=llm)
        result = agent.run(_ctx_factory(layout, tools))
        assert result.status == "success"
        assert result.stage == Stage.BASELINE_PROFILE.value
        assert layout.baseline_path.is_file()


# ---------------------------------------------------------------------------
# KernelTuningAgent
# ---------------------------------------------------------------------------

VALID_CU = """
#include <torch/extension.h>
torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B){return W;}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward, "lora"); }
""".strip() + "\n" + ("// pad\n" * 30)


class TestKernelTuningAgent:
    def test_full_cycle_writes_evaluates_submits(self, tmp_path):
        from pipeline.tools.candidate_tools import _EVAL_MARKER

        layout = _layout(tmp_path)
        # Pretend BaselineAgent ran.
        for d in (3584, 4096):
            layout.baseline_input_path("W", d).write_bytes(b"x")
            layout.baseline_input_path("X", d).write_bytes(b"x")
            layout.baseline_input_path("A", d).write_bytes(b"x")
            layout.baseline_input_path("B", d).write_bytes(b"x")
            layout.baseline_reference_path(d).write_bytes(b"x")

        # Successful evaluation result the mock executor will hand back.
        eval_payload = {
            "compile_ok": True,
            "all_correct": True,
            "per_d": [
                {"d": 3584, "correctness_ok": True, "max_abs_diff": 1e-5,
                 "candidate_ms_median": 4.0, "ref_ms_median": 5.0, "speedup": 1.25,
                 "samples": 5, "candidate_ms_min": 3.9, "candidate_ms_max": 4.1,
                 "ref_ms_min": 4.9, "ref_ms_max": 5.1, "error": None},
                {"d": 4096, "correctness_ok": True, "max_abs_diff": 1e-5,
                 "candidate_ms_median": 5.0, "ref_ms_median": 6.0, "speedup": 1.20,
                 "samples": 5, "candidate_ms_min": 4.9, "candidate_ms_max": 5.1,
                 "ref_ms_min": 5.9, "ref_ms_max": 6.1, "error": None},
            ],
            "overall_speedup_median": 1.225,
        }

        class _MockExec:
            def profile_with_torch(self, code, op_name, timeout_s=120):
                return {"stdout": f"{_EVAL_MARKER}\n{json.dumps(eval_payload)}\n"}

        factory = StageToolFactory(executor=_MockExec(), layout=layout)
        tools = factory.build(KernelTuningAgent.allowed_tools, current_stage=Stage.INITIAL_CANDIDATE)

        record = {
            "candidate_id": "candidate_000",
            "compile_ok": True,
            "correctness_ok": True,
            "quick_speedup_median": 1.225,
            "quick_samples": 5,
            "accepted_for": "best_update",
        }

        llm = _FakeLLM([
            _FakeChatResponse.from_calls([("c1", "write_candidate", {"source": VALID_CU})]),
            _FakeChatResponse.from_calls([("c2", "evaluate_candidate", {"candidate_id": "candidate_000", "d_list": [3584, 4096], "mode": "quick"})]),
            _FakeChatResponse.from_calls([("c3", "submit_candidate_result", {"record": record})]),
        ])
        agent = KernelTuningAgent(llm=llm, stage=Stage.INITIAL_CANDIDATE)
        result = agent.run(_ctx_factory(layout, tools))

        assert result.status == "success"
        assert result.stage == Stage.INITIAL_CANDIDATE.value
        assert result.metrics["candidate"]["candidate_id"] == "candidate_000"
        assert result.metrics["candidate"]["accepted_for"] == "best_update"
        # Side-effect: candidate.cu on disk
        assert layout.candidate_file("candidate_000", "candidate.cu").is_file()


# ---------------------------------------------------------------------------
# SummaryAgent
# ---------------------------------------------------------------------------

class TestSummaryAgent:
    def test_writes_final_report_and_summary(self, tmp_path):
        layout = _layout(tmp_path)
        factory = StageToolFactory(executor=None, layout=layout)
        tools = factory.build(SummaryAgent.allowed_tools, current_stage=Stage.FINALIZE)

        report = {"run_id": "run_test", "operator": "lora_matmul", "best_candidate_id": "candidate_002", "best_speedup": 1.34, "per_d_speedups": [{"d": 4096, "speedup": 1.34}]}
        summary_md = "# Run summary\n\nThe best candidate achieved 1.34x speedup."

        llm = _FakeLLM([
            _FakeChatResponse.from_calls([("c1", "submit_summary", {"report": report, "summary_md": summary_md})]),
        ])
        agent = SummaryAgent(llm=llm)
        result = agent.run(_ctx_factory(layout, tools))

        assert result.status == "success"
        assert layout.final_report_path.is_file()
        assert layout.summary_path.is_file()
        assert "1.34" in layout.summary_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# HardwareProfilerAgent + ProfileAnalysisAgent — exercise allowed_tools registration
# ---------------------------------------------------------------------------

class TestOtherAgentsToolWiring:
    """Even without a real GPU we confirm that the StageToolFactory accepts
    the allowed_tools each of these agents declares."""

    def test_hardware_profiler_tools_resolvable(self, tmp_path):
        # HardwareProfilerAgent needs run_cuda_probe etc which require an executor.
        class _DummyExec:
            workspace = None
        factory = StageToolFactory(executor=_DummyExec(), layout=_layout(tmp_path))
        reg = factory.build(HardwareProfilerAgent.allowed_tools, current_stage=Stage.HARDWARE_PROFILE)
        for t in HardwareProfilerAgent.allowed_tools:
            assert t in reg._tools

    def test_profile_analysis_tools_resolvable(self, tmp_path):
        class _DummyExec:
            workspace = None
        factory = StageToolFactory(executor=_DummyExec(), layout=_layout(tmp_path))
        reg = factory.build(ProfileAnalysisAgent.allowed_tools, current_stage=Stage.OPTIONAL_PROFILE)
        for t in ProfileAnalysisAgent.allowed_tools:
            assert t in reg._tools

    def test_summary_agent_only_text_tools(self, tmp_path):
        # SummaryAgent intentionally has no executor-dependent tools.
        factory = StageToolFactory(executor=None, layout=_layout(tmp_path))
        reg = factory.build(SummaryAgent.allowed_tools, current_stage=Stage.FINALIZE)
        for t in SummaryAgent.allowed_tools:
            assert t in reg._tools
