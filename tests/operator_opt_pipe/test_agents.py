"""End-to-end agent glue: scripted ``LLMBackend`` exercises ``agents.run_*``.

Verifies the ``mls_agent.Agent → tool → AgentResult → operator_opt_pipe dict``
round-trip without involving the network or any real LLM.

Two termination paths are tested:
  * COMPLETED via ``submit_candidate.terminate_with(payload)``
  * NO_TOOL_CALL after a successful ``write_blackboard`` call followed by
    a plain text response
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from mls_agent import (
    AgentConfig,
    ChatResponse,
    LLMBackend,
    Message,
    NullObserver,
    ToolCall,
)

from operator_opt_pipe import agents
from operator_opt_pipe.resources import OperatorContract, TensorSpec
from operator_opt_pipe.state import RunLayout, Stage, load_blackboard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def contract() -> OperatorContract:
    return OperatorContract(
        name="operators/lora_matmul",
        inputs=(
            TensorSpec(name="W", shape=("d", "d"), dtype="float32"),
            TensorSpec(name="X", shape=("d", "d"), dtype="float32"),
            TensorSpec(name="A", shape=("d", 16), dtype="float32"),
            TensorSpec(name="B", shape=("d", 16), dtype="float32"),
        ),
        output=TensorSpec(name="Y", shape=("d", "d"), dtype="float32"),
        reference_pytorch="W @ X + A @ (B.transpose(0, 1).contiguous() @ X)",
        forward_args=("W", "X", "A", "B"),
        shape_param="d",
        shape_param_range=(3584, 4608),
    )


@pytest.fixture
def layout(tmp_path: Path) -> RunLayout:
    lay = RunLayout(workspace_root=tmp_path, run_id="r")
    lay.mkdir()
    return lay


class _ScriptedBackend(LLMBackend):
    """Returns pre-baked ChatResponses; raises if exhausted unexpectedly."""

    def __init__(self, responses: list[ChatResponse]):
        self._responses = list(responses)

    def chat(self, messages: Sequence[Message], tools: Sequence[dict]):
        if not self._responses:
            raise RuntimeError("scripted backend exhausted")
        return self._responses.pop(0)


def _tool_call(call_id: str, name: str, arguments: dict, content: str | None = None) -> ChatResponse:
    return ChatResponse(
        message=Message.assistant(
            content=content,
            tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
        ),
        finish_reason="tool_calls",
    )


def _final_text(content: str) -> ChatResponse:
    return ChatResponse(
        message=Message.assistant(content=content, tool_calls=()),
        finish_reason="stop",
    )


class _NoopExecutor:
    """Stand-in for Executor — methods raise loudly if called by mistake."""

    def __getattr__(self, name):
        def _err(*a, **kw):
            raise AssertionError(f"_NoopExecutor.{name} should not be called in tests")
        return _err


# ---------------------------------------------------------------------------
# Hardware profiler — natural exit (write_blackboard + plain text)
# ---------------------------------------------------------------------------


def test_run_hardware_profiler_natural_exit(layout: RunLayout, contract: OperatorContract):
    registry = agents.build_registry(
        "hardware_profiler",
        layout=layout, contract=contract,
        executor=_NoopExecutor(), skills_dir=None,
    )
    backend = _ScriptedBackend([
        _tool_call("c1", "write_blackboard",
                   {"key": "hardware",
                    "payload": {"metrics": {"sm": 30, "dram_bw_gbps": 384.0}}},
                   content="probing complete."),
        # After write succeeds the LLM stops calling tools — natural exit.
        _final_text("hardware profile recorded; SM=30, DRAM=384 GB/s."),
        # Extra response in case max_consecutive_no_tool_call needs a second turn.
        _final_text("done."),
    ])
    out = agents.run_hardware_profiler(
        backend=backend, registry=registry,
        layout=layout, contract=contract,
        agent_cfg=AgentConfig(max_iterations=8),
        observer=NullObserver(),
    )
    assert out["status"] == "success"
    assert out["blackboard_key"] == "hardware"
    assert out["payload"]["metrics"]["sm"] == 30
    bb = load_blackboard(layout)
    assert bb["hardware"]["metrics"]["sm"] == 30


# ---------------------------------------------------------------------------
# Summary — same natural-exit pattern
# ---------------------------------------------------------------------------


def test_run_summary_natural_exit(layout: RunLayout, contract: OperatorContract):
    registry = agents.build_registry(
        "summary",
        layout=layout, contract=contract,
        executor=_NoopExecutor(), skills_dir=None,
    )
    backend = _ScriptedBackend([
        _tool_call("c1", "write_blackboard",
                   {"key": "final_summary",
                    "payload": {
                        "best_speedup": 2.5,
                        "narrative": "Fused W*X with low-rank correction; saturated DRAM.",
                    }},
                   content="summary written."),
        _final_text("done."),
        _final_text("nothing else."),
    ])
    out = agents.run_summary(
        backend=backend, registry=registry,
        layout=layout, contract=contract,
        agent_cfg=AgentConfig(max_iterations=8),
        observer=NullObserver(),
    )
    assert out["status"] == "success"
    assert out["payload"]["best_speedup"] == 2.5
    bb = load_blackboard(layout)
    assert bb["final_summary"]["best_speedup"] == 2.5


# ---------------------------------------------------------------------------
# Optimizer cold — explicit terminate via submit_candidate
# ---------------------------------------------------------------------------


def test_run_optimizer_cold_completed_path(
    layout: RunLayout, contract: OperatorContract, monkeypatch,
):
    """Mock compile_and_check_quick so write_candidate doesn't spawn subprocesses,
    then verify the agent terminates COMPLETED via submit_candidate."""
    from operator_opt_pipe.resources.evaluation import QuickEvalResult
    import operator_opt_pipe.tools as tools_mod

    def fake_quick(**kw):
        return QuickEvalResult(
            candidate_id=kw["candidate_id"], compile_ok=True, correctness_ok=True,
            shape_id="d3584", max_abs_err=1e-6, rel_l2_err=1e-7,
            compile_log="", diagnostics={},
        )

    monkeypatch.setattr(tools_mod, "compile_and_check_quick", fake_quick)

    source = (
        "#include <torch/extension.h>\n"
        "torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B) { return W; }\n"
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"forward\", &forward); }\n"
    )

    registry = agents.build_registry(
        "optimizer_cold",
        layout=layout, contract=contract,
        executor=_NoopExecutor(), skills_dir=None,
    )
    backend = _ScriptedBackend([
        _tool_call("c1", "write_candidate", {"source": source}),
        _tool_call("c2", "submit_candidate", {
            "candidate_id": "candidate_000",
            "hypothesis": "naive correctness baseline",
            "experiment_type": "baseline",
        }),
    ])
    out = agents.run_optimizer_cold(
        backend=backend, registry=registry,
        layout=layout, contract=contract,
        agent_cfg=AgentConfig(max_iterations=8),
        observer=NullObserver(),
    )
    assert out["status"] == "success"
    assert out["candidate_id"] == "candidate_000"
    # SubmitCandidateTool stamps stage=TUNING_LOOP regardless of who submitted;
    # orchestrator distinguishes initial vs tuning by RunState, not by tag.
    assert out["stage"] == Stage.TUNING_LOOP.value


# ---------------------------------------------------------------------------
# Failure surfaces as status="failed" with caveats
# ---------------------------------------------------------------------------


def test_invalid_blackboard_write_falls_back_to_failure(
    layout: RunLayout, contract: OperatorContract,
):
    """If the agent fails to write the expected key, the runner returns failed."""
    registry = agents.build_registry(
        "hardware_profiler",
        layout=layout, contract=contract,
        executor=_NoopExecutor(), skills_dir=None,
    )
    # Write to a forbidden key — tool returns ERROR; LLM never writes "hardware".
    backend = _ScriptedBackend([
        _tool_call("c1", "write_blackboard",
                   {"key": "best", "payload": {"speedup": 1.0}}),
        _final_text("oops, can't do that."),
        _final_text("giving up."),
    ])
    out = agents.run_hardware_profiler(
        backend=backend, registry=registry,
        layout=layout, contract=contract,
        agent_cfg=AgentConfig(max_iterations=4),
        observer=NullObserver(),
    )
    assert out["status"] == "failed"
    assert out["caveats"]


# ---------------------------------------------------------------------------
# build_registry shape sanity checks
# ---------------------------------------------------------------------------


def test_build_registry_unknown_role(layout: RunLayout, contract: OperatorContract):
    with pytest.raises(ValueError, match="unknown role"):
        agents.build_registry(
            "nope", layout=layout, contract=contract,
            executor=_NoopExecutor(), skills_dir=None,
        )


def test_build_registry_per_role_tool_set(layout: RunLayout, contract: OperatorContract):
    optimizer = agents.build_registry(
        "optimizer", layout=layout, contract=contract,
        executor=_NoopExecutor(), skills_dir=None,
    )
    names = optimizer.names()
    # Optimizer must NOT see profile / write_blackboard tools.
    assert "write_blackboard" not in names
    assert "profile_with_ncu" not in names
    # Optimizer DOES get write_candidate + submit_candidate.
    assert "write_candidate" in names
    assert "submit_candidate" in names
    assert "read_blackboard" in names

    analyst = agents.build_registry(
        "analyst", layout=layout, contract=contract,
        executor=_NoopExecutor(), skills_dir=None,
    )
    a_names = analyst.names()
    # Analyst gets profile tools + write_blackboard but NOT candidate tools.
    assert "profile_with_ncu" in a_names
    assert "write_blackboard" in a_names
    assert "write_candidate" not in a_names
    assert "submit_candidate" not in a_names
