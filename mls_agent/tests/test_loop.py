"""End-to-end tests for ReActLoop / Agent with a scripted fake backend."""
from __future__ import annotations

from typing import Sequence

import pytest

from mls_agent.llm.backend import LLMBackend
from mls_agent.llm.types import ChatResponse, FinishReason, Message, ToolCall
from mls_agent.runtime.agent import Agent
from mls_agent.runtime.config import AgentConfig
from mls_agent.runtime.loop import ReActLoop
from mls_agent.runtime.observer import NullObserver
from mls_agent.runtime.state import ReActPhase
from mls_agent.tools.base import Tool, ToolParameter
from mls_agent.tools.circuit_breaker import CircuitBreaker
from mls_agent.tools.registry import ToolRegistry
from mls_agent.tools.response import (
    Event,
    Measurement,
    ToolResponse,
)


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


class _ScriptedBackend(LLMBackend):
    """Backend that yields a pre-baked sequence of ChatResponses."""

    def __init__(self, responses: Sequence[ChatResponse]):
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def chat(self, messages, tools):
        self.calls.append(list(messages))
        if not self._responses:
            raise RuntimeError("scripted backend exhausted")
        return self._responses.pop(0)


def _assistant_text(content: str, finish_reason: FinishReason = "stop") -> ChatResponse:
    return ChatResponse(
        message=Message.assistant(content=content),
        finish_reason=finish_reason,
    )


def _assistant_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: dict,
    text: str | None = None,
) -> ChatResponse:
    tc = ToolCall(id=call_id, name=name, arguments=arguments)
    return ChatResponse(
        message=Message.assistant(content=text, tool_calls=(tc,)),
        finish_reason="tool_calls",
    )


# ---------------------------------------------------------------------------
# Sample tools
# ---------------------------------------------------------------------------


class _Echo(Tool):
    NAME = "echo"
    DESCRIPTION = "Echo back."

    def parameters_schema(self):
        return Tool.schema_from_parameters([
            ToolParameter(name="text", type="string", description="text"),
        ])

    def run(self, parameters):
        return ToolResponse.success(parameters["text"])


class _Submit(Tool):
    NAME = "submit"
    DESCRIPTION = "Finalize and submit results."

    def parameters_schema(self):
        return Tool.schema_from_parameters([
            ToolParameter(name="summary", type="string", description="summary"),
        ])

    def run(self, parameters):
        return ToolResponse.terminate_with(
            summary=parameters["summary"],
            payload={"summary": parameters["summary"]},
        )


class _BoomTool(Tool):
    NAME = "boom"
    DESCRIPTION = "Always raises."

    def parameters_schema(self):
        return {"type": "object", "properties": {}}

    def run(self, parameters):
        raise RuntimeError("boom!")


class _RecorderTool(Tool):
    """Returns events + measurements via the side-effect channel."""

    NAME = "record"
    DESCRIPTION = "Record an event and a measurement."

    def parameters_schema(self):
        return {"type": "object", "properties": {}}

    def run(self, parameters):
        return ToolResponse.success(
            "recorded",
            events=(Event(type="recorded", severity="info", detail="ok"),),
            measurements=(
                Measurement(
                    metric="m",
                    value=1.0,
                    unit="x",
                    confidence=0.9,
                    method="constant",
                    evidence=("from test",),
                ),
            ),
        )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _registry_with(*tools: Tool, breaker: CircuitBreaker | None = None) -> ToolRegistry:
    reg = ToolRegistry(breaker=breaker)
    for t in tools:
        reg.register(t)
    return reg


# ---------------------------------------------------------------------------
# Termination paths
# ---------------------------------------------------------------------------


class TestTermination:
    def test_simple_terminate_path(self):
        backend = _ScriptedBackend([
            _assistant_tool_call(
                call_id="c1", name="submit", arguments={"summary": "all done"}
            ),
        ])
        registry = _registry_with(_Submit())
        loop = ReActLoop(backend, registry, "sys", AgentConfig(), NullObserver())
        result = loop.run("solve it")
        assert result.reason == "completed"
        assert result.summary == "all done"
        assert result.payload == {"summary": "all done"}
        assert result.iterations == 1

    def test_two_step_then_terminate(self):
        backend = _ScriptedBackend([
            _assistant_tool_call(
                call_id="c1", name="echo", arguments={"text": "hi"}
            ),
            _assistant_tool_call(
                call_id="c2", name="submit", arguments={"summary": "done"}
            ),
        ])
        registry = _registry_with(_Echo(), _Submit())
        loop = ReActLoop(backend, registry, "sys", AgentConfig(), NullObserver())
        result = loop.run("go")
        assert result.reason == "completed"
        assert result.iterations == 2
        # Conversation was: system, user, assistant_call_1, tool_1,
        # assistant_call_2, tool_2 = 6 messages
        assert len(result.context.messages) == 6

    def test_max_iterations_reason(self):
        # Always echo, never submit.
        backend = _ScriptedBackend([
            _assistant_tool_call(
                call_id=f"c{i}", name="echo", arguments={"text": "x"}
            )
            for i in range(10)
        ])
        registry = _registry_with(_Echo())
        cfg = AgentConfig(max_iterations=3)
        loop = ReActLoop(backend, registry, "sys", cfg, NullObserver())
        result = loop.run("go")
        assert result.reason == "max_iterations"
        assert result.iterations == 3

    def test_no_tool_call_triggers_nudge_then_aborts(self):
        # First two responses have no tool call.
        backend = _ScriptedBackend([
            _assistant_text("I don't know."),
            _assistant_text("Still don't know."),
        ])
        registry = _registry_with(_Submit())
        cfg = AgentConfig(max_consecutive_no_tool_call=2)
        loop = ReActLoop(backend, registry, "sys", cfg, NullObserver())
        result = loop.run("go")
        assert result.reason == "no_tool_call"
        assert result.summary == "Still don't know."
        assert result.iterations == 2
        # Verify the second LLM call saw the nudge as the latest user message.
        second_call_msgs = backend.calls[1]
        assert any(
            m.role == "user" and "must call a tool" in (m.content or "")
            for m in second_call_msgs
        )

    def test_nudge_then_recovery(self):
        # First: no tool call. Second: assistant calls submit.
        backend = _ScriptedBackend([
            _assistant_text("Sorry, thinking."),
            _assistant_tool_call(
                call_id="c", name="submit", arguments={"summary": "ok"}
            ),
        ])
        registry = _registry_with(_Submit())
        cfg = AgentConfig(max_consecutive_no_tool_call=2)
        loop = ReActLoop(backend, registry, "sys", cfg, NullObserver())
        result = loop.run("go")
        assert result.reason == "completed"
        assert result.summary == "ok"

    def test_llm_error_returns_result_not_raises(self):
        class _Broken(LLMBackend):
            def chat(self, messages, tools):
                raise RuntimeError("backend down")

        registry = _registry_with(_Submit())
        loop = ReActLoop(_Broken(), registry, "sys", AgentConfig(), NullObserver())
        result = loop.run("go")
        assert result.reason == "llm_error"
        assert any(e.type == "llm_error" for e in result.context.events)
        assert result.iterations == 0


# ---------------------------------------------------------------------------
# Side effects
# ---------------------------------------------------------------------------


class TestSideEffects:
    def test_events_and_measurements_landed_on_context(self):
        backend = _ScriptedBackend([
            _assistant_tool_call(call_id="c1", name="record", arguments={}),
            _assistant_tool_call(
                call_id="c2", name="submit", arguments={"summary": "done"}
            ),
        ])
        registry = _registry_with(_RecorderTool(), _Submit())
        loop = ReActLoop(backend, registry, "sys", AgentConfig(), NullObserver())
        result = loop.run("go")
        assert len(result.context.events) == 1
        assert result.context.events[0].type == "recorded"
        assert len(result.context.measurements) == 1
        assert result.context.measurements[0].metric == "m"

    def test_tool_results_appear_as_tool_messages(self):
        backend = _ScriptedBackend([
            _assistant_tool_call(
                call_id="c1", name="echo", arguments={"text": "hello"}
            ),
            _assistant_tool_call(
                call_id="c2", name="submit", arguments={"summary": "done"}
            ),
        ])
        registry = _registry_with(_Echo(), _Submit())
        loop = ReActLoop(backend, registry, "sys", AgentConfig(), NullObserver())
        result = loop.run("go")
        tool_messages = [m for m in result.context.messages if m.role == "tool"]
        assert len(tool_messages) == 2
        # The echo tool's result message should embed "hello".
        assert "hello" in tool_messages[0].content

    def test_tool_exception_does_not_break_loop(self):
        backend = _ScriptedBackend([
            _assistant_tool_call(call_id="c1", name="boom", arguments={}),
            _assistant_tool_call(
                call_id="c2", name="submit", arguments={"summary": "ok"}
            ),
        ])
        registry = _registry_with(_BoomTool(), _Submit())
        loop = ReActLoop(backend, registry, "sys", AgentConfig(), NullObserver())
        result = loop.run("go")
        assert result.reason == "completed"


# ---------------------------------------------------------------------------
# Circuit breaker integration
# ---------------------------------------------------------------------------


class TestCircuitBreakerIntegration:
    def test_repeated_failures_open_circuit(self):
        # Always call boom. Threshold=3 means iteration 4 short-circuits.
        backend = _ScriptedBackend([
            _assistant_tool_call(call_id=f"c{i}", name="boom", arguments={})
            for i in range(10)
        ])
        breaker = CircuitBreaker(threshold=3, half_open_timeout_s=60.0)
        registry = _registry_with(_BoomTool(), breaker=breaker)
        cfg = AgentConfig(max_iterations=10)
        loop = ReActLoop(backend, registry, "sys", cfg, NullObserver())
        result = loop.run("go")
        # Eventually the loop hits max_iterations because the LLM keeps
        # asking for boom, and dispatch returns CIRCUIT_OPEN. The loop
        # itself doesn't terminate on circuit_open — it's the LLM's
        # responsibility to switch tactics.
        assert result.reason == "max_iterations"
        assert breaker.open_circuits()


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------


class TestAgentEntry:
    def test_agent_run_delegates_to_loop(self):
        backend = _ScriptedBackend([
            _assistant_tool_call(
                call_id="c", name="submit", arguments={"summary": "done"}
            ),
        ])
        registry = _registry_with(_Submit())
        agent = Agent(
            backend=backend, registry=registry, system_prompt="sys",
        )
        r = agent.run("go")
        assert r.reason == "completed"
        assert r.summary == "done"


# ---------------------------------------------------------------------------
# Observer integration
# ---------------------------------------------------------------------------


class _RecordingObserver:
    """Records all hook calls in order."""

    def __init__(self):
        self.events: list[tuple[str, tuple]] = []

    def on_run_start(self, ctx):
        self.events.append(("run_start", (ctx,)))

    def on_iteration_start(self, iteration, ctx):
        self.events.append(("iter_start", (iteration,)))

    def on_phase_transition(self, prev, next):
        self.events.append(("phase", (prev, next)))

    def on_llm_call(self, messages):
        self.events.append(("llm_call", (len(messages),)))

    def on_llm_response(self, response):
        self.events.append(("llm_response", (response.finish_reason,)))

    def on_tool_call(self, call, response):
        self.events.append(("tool_call", (call.name, response.status.value)))

    def on_terminate(self, result):
        self.events.append(("terminate", (result.reason,)))

    def on_error(self, exc, phase):
        self.events.append(("error", (type(exc).__name__, phase.value)))


class TestObserver:
    def test_hooks_fire_in_expected_order(self):
        backend = _ScriptedBackend([
            _assistant_tool_call(
                call_id="c", name="submit", arguments={"summary": "done"}
            ),
        ])
        registry = _registry_with(_Submit())
        obs = _RecordingObserver()
        loop = ReActLoop(backend, registry, "sys", AgentConfig(), obs)
        loop.run("go")

        kinds = [e[0] for e in obs.events]
        # We expect at minimum: run_start, iter_start, llm_call,
        # llm_response, tool_call, terminate.
        assert "run_start" in kinds
        assert "iter_start" in kinds
        assert "llm_call" in kinds
        assert "llm_response" in kinds
        assert "tool_call" in kinds
        assert "terminate" in kinds
        # llm_call must precede llm_response which must precede tool_call.
        assert kinds.index("llm_call") < kinds.index("llm_response")
        assert kinds.index("llm_response") < kinds.index("tool_call")

    def test_error_hook_on_llm_failure(self):
        class _Broken(LLMBackend):
            def chat(self, messages, tools):
                raise RuntimeError("nope")

        registry = _registry_with(_Submit())
        obs = _RecordingObserver()
        loop = ReActLoop(_Broken(), registry, "sys", AgentConfig(), obs)
        loop.run("go")
        kinds = [e[0] for e in obs.events]
        assert "error" in kinds
