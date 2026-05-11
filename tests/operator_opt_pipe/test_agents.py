"""End-to-end agent glue: scripted ``LLMBackend`` exercises ``agents.run_*``.

Verifies the ``mls_agent.Agent → tool → AgentResult → operator_opt_pipe dict``
round-trip without involving the network or any real LLM.

All success paths are now COMPLETED:
  * ``submit_candidate.terminate_with(payload)`` for optimizer roles
  * ``terminate`` (no payload) for hardware_profiler / analyst / summary
A no-tool-call streak now surfaces as ``status="failed"`` with caveats.
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
from operator_opt_pipe.operators import load_ops
from operator_opt_pipe.operators._base import OperatorOps
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


@pytest.fixture
def ops() -> OperatorOps:
    return load_ops("lora_matmul")


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
# Hardware profiler — explicit terminate after write_blackboard
# ---------------------------------------------------------------------------


def test_run_hardware_profiler_terminate(layout: RunLayout, contract: OperatorContract, ops: OperatorOps):
    registry = agents.build_registry(
        "hardware_profiler",
        layout=layout, contract=contract, ops=ops,
        executor=_NoopExecutor(), skills_dir=None,
    )
    backend = _ScriptedBackend([
        _tool_call("c1", "write_blackboard",
                   {"key": "hardware",
                    "payload": {"metrics": {"sm": 30, "dram_bw_gbps": 384.0}}},
                   content="probing complete."),
        _tool_call("c2", "terminate", {"summary": "SM=30 DRAM=384"}),
    ])
    out = agents.run_hardware_profiler(
        backend=backend, registry=registry,
        layout=layout, contract=contract,
        agent_cfg=AgentConfig(max_iterations=8),
        observer=NullObserver(),
    )
    assert out["status"] == "success"
    assert out["stage"] == Stage.HARDWARE_PROFILE.value
    assert out["summary"] == "SM=30 DRAM=384"
    # Blackboard content is verified separately — orchestrator reads it from disk.
    bb = load_blackboard(layout)
    assert bb["hardware"]["metrics"]["sm"] == 30


# ---------------------------------------------------------------------------
# Summary — same terminate pattern
# ---------------------------------------------------------------------------


def test_run_summary_terminate(layout: RunLayout, contract: OperatorContract, ops: OperatorOps):
    registry = agents.build_registry(
        "summary",
        layout=layout, contract=contract, ops=ops,
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
        _tool_call("c2", "terminate", {}),
    ])
    out = agents.run_summary(
        backend=backend, registry=registry,
        layout=layout, contract=contract,
        agent_cfg=AgentConfig(max_iterations=8),
        observer=NullObserver(),
    )
    assert out["status"] == "success"
    assert out["stage"] == Stage.FINALIZE.value
    bb = load_blackboard(layout)
    assert bb["final_summary"]["best_speedup"] == 2.5


# ---------------------------------------------------------------------------
# Optimizer cold — explicit terminate via submit_candidate
# ---------------------------------------------------------------------------


def test_run_optimizer_cold_completed_path(
    layout: RunLayout, contract: OperatorContract, ops: OperatorOps, monkeypatch,
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
        layout=layout, contract=contract, ops=ops,
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
    # The agents.run_optimizer_cold runner stamps stage=INITIAL_CANDIDATE
    # at the orchestrator boundary (its expected_stage). SubmitCandidateTool
    # itself no longer hard-codes a stage tag.
    assert out["stage"] == Stage.INITIAL_CANDIDATE.value


# ---------------------------------------------------------------------------
# Failure surfaces as status="failed" with caveats
# ---------------------------------------------------------------------------


def test_no_terminate_call_surfaces_as_failure(
    layout: RunLayout, contract: OperatorContract, ops: OperatorOps,
):
    """If the agent stops calling tools without invoking ``terminate``,
    the loop ends with reason=no_tool_call and the runner returns failed.
    """
    registry = agents.build_registry(
        "hardware_profiler",
        layout=layout, contract=contract, ops=ops,
        executor=_NoopExecutor(), skills_dir=None,
    )
    backend = _ScriptedBackend([
        _tool_call("c1", "write_blackboard",
                   {"key": "hardware",
                    "payload": {"metrics": {"sm": 30}}}),
        # LLM forgets to call terminate — keeps producing plain text.
        _final_text("done; SM=30."),
        _final_text("really done."),
    ])
    out = agents.run_hardware_profiler(
        backend=backend, registry=registry,
        layout=layout, contract=contract,
        agent_cfg=AgentConfig(max_iterations=4),
        observer=NullObserver(),
    )
    assert out["status"] == "failed"
    assert out["caveats"]
    assert "no_tool_call" in out["caveats"][0]


# ---------------------------------------------------------------------------
# build_registry shape sanity checks
# ---------------------------------------------------------------------------


def test_build_registry_unknown_role(layout: RunLayout, contract: OperatorContract, ops: OperatorOps):
    with pytest.raises(ValueError, match="unknown role"):
        agents.build_registry(
            "nope", layout=layout, contract=contract, ops=ops,
            executor=_NoopExecutor(), skills_dir=None,
        )


def test_build_registry_per_role_tool_set(layout: RunLayout, contract: OperatorContract, ops: OperatorOps):
    optimizer = agents.build_registry(
        "optimizer", layout=layout, contract=contract, ops=ops,
        executor=_NoopExecutor(), skills_dir=None,
    )
    names = optimizer.names()
    # Optimizer must NOT see profile / write_blackboard / terminate tools —
    # it terminates via submit_candidate.
    assert "write_blackboard" not in names
    assert "profile_with_ncu" not in names
    assert "terminate" not in names
    # Optimizer DOES get write_candidate + submit_candidate.
    assert "write_candidate" in names
    assert "submit_candidate" in names
    assert "read_blackboard" in names

    analyst = agents.build_registry(
        "analyst", layout=layout, contract=contract, ops=ops,
        executor=_NoopExecutor(), skills_dir=None,
    )
    a_names = analyst.names()
    # Analyst gets profile tools + write_blackboard + terminate but NOT
    # candidate tools.
    assert "profile_with_ncu" in a_names
    assert "write_blackboard" in a_names
    assert "terminate" in a_names
    assert "write_candidate" not in a_names
    assert "submit_candidate" not in a_names

    # hardware_profiler and summary also get terminate.
    for natural_exit_role in ("hardware_profiler", "summary"):
        reg = agents.build_registry(
            natural_exit_role, layout=layout, contract=contract, ops=ops,
            executor=_NoopExecutor(), skills_dir=None,
        )
        assert "terminate" in reg.names()
