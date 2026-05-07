"""End-to-end agent glue: scripted ``LLMBackend`` exercises ``agents.run_*``.

Verifies the ``mls_agent.Agent → ToolResponse.terminate_with(payload) →
AgentResult.payload → operator_opt_pipe dict`` round-trip without involving
the network or any real LLM.
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
from operator_opt_pipe.lora_resources.contract import LoRAContract
from operator_opt_pipe.orchestrator import build_registry, make_default_tools
from operator_opt_pipe.state import RunLayout, Stage, load_blackboard


@pytest.fixture
def contract() -> LoRAContract:
    return LoRAContract(
        operator="lora_matmul",
        d_range=(3584, 4608),
        r=16,
        dtype="float32",
        device="cuda",
        forward_args=("W", "X", "A", "B"),
        reference_pytorch="W @ X + A @ (B.T @ X)",
        output_name="Y",
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


class _NoopExecutor:
    def __getattr__(self, name):
        def _err(*a, **kw):
            raise AssertionError(f"_NoopExecutor.{name} should not be called in tests")
        return _err


def _full_tool_bag(layout: RunLayout) -> dict:
    """Real default tool set with a no-op executor + nonexistent skills dir."""
    return make_default_tools(
        layout=layout,
        executor=_NoopExecutor(),
        skills_dir=layout.workspace_root / "skills",
    )


def test_run_hardware_profiler_returns_payload(layout: RunLayout, contract: LoRAContract):
    tools = _full_tool_bag(layout)
    registry = build_registry("hardware_profiler", tools)
    backend = _ScriptedBackend([
        _tool_call("c1", "submit_hardware_profile",
                   {"status": "success", "metrics": {"sm": 30, "dram_bw_gbps": 384.0}},
                   content="probing complete."),
    ])
    out = agents.run_hardware_profiler(
        backend=backend, registry=registry,
        layout=layout, contract=contract,
        agent_cfg=AgentConfig(max_iterations=4),
        observer=NullObserver(),
    )
    assert out["status"] == "success"
    assert out["metrics"]["sm"] == 30
    assert out["stage"] == Stage.HARDWARE_PROFILE.value

    # Blackboard reflects the persisted hardware payload (SubmitTool wrote it).
    bb = load_blackboard(layout)
    assert bb["hardware"]["metrics"]["sm"] == 30


def test_run_summary_returns_payload(layout: RunLayout, contract: LoRAContract):
    tools = _full_tool_bag(layout)
    registry = build_registry("summary", tools)
    backend = _ScriptedBackend([
        _tool_call("c1", "submit_summary",
                   {"status": "success", "metrics": {"final_speedup": 2.5}}),
    ])
    out = agents.run_summary(
        backend=backend, registry=registry,
        layout=layout, contract=contract,
        agent_cfg=AgentConfig(max_iterations=4),
        observer=NullObserver(),
    )
    assert out["status"] == "success"
    assert out["metrics"]["final_speedup"] == 2.5
    assert out["stage"] == Stage.FINALIZE.value


def test_invalid_payload_falls_back_to_failure(layout: RunLayout, contract: LoRAContract):
    """If the agent submits a payload with the wrong stage tag, _payload_or_failure
    converts it to status=failed with diagnostic caveats."""
    tools = _full_tool_bag(layout)
    registry = build_registry("hardware_profiler", tools)
    backend = _ScriptedBackend([
        _tool_call("c1", "submit_hardware_profile",
                   {"status": "success", "stage": "FINALIZE", "metrics": {}}),
    ])
    out = agents.run_hardware_profiler(
        backend=backend, registry=registry,
        layout=layout, contract=contract,
        agent_cfg=AgentConfig(max_iterations=4),
        observer=NullObserver(),
    )
    # SubmitTool itself rejects the bad stage (returns ToolResponse.error,
    # the agent never terminates), so the agent eventually exhausts iterations
    # OR keeps calling it. With max_iterations=4 and a single scripted reply,
    # the loop terminates with reason='no_tool_call' or similar.
    assert out["status"] == "failed"
    assert out["caveats"]
