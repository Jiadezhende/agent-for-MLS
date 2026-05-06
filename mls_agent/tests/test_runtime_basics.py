"""Unit tests for runtime config / context / result / state."""
from __future__ import annotations

import pytest

from mls_agent.llm.types import Message
from mls_agent.runtime.config import AgentConfig
from mls_agent.runtime.context import AgentContext
from mls_agent.runtime.result import AgentResult
from mls_agent.runtime.state import ReActPhase
from mls_agent.tools.response import Event, Measurement


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_defaults(self):
        c = AgentConfig()
        assert c.max_iterations == 40
        assert c.max_consecutive_no_tool_call == 2
        assert c.circuit_breaker_threshold == 3

    def test_max_iterations_must_be_positive(self):
        with pytest.raises(ValueError):
            AgentConfig(max_iterations=0)
        with pytest.raises(ValueError):
            AgentConfig(max_iterations=-1)

    def test_max_consecutive_no_tool_call_must_be_positive(self):
        with pytest.raises(ValueError):
            AgentConfig(max_consecutive_no_tool_call=0)

    def test_threshold_must_be_positive(self):
        with pytest.raises(ValueError):
            AgentConfig(circuit_breaker_threshold=0)

    def test_negative_half_open_rejected(self):
        with pytest.raises(ValueError):
            AgentConfig(circuit_breaker_half_open_s=-1.0)

    def test_truncate_must_be_positive(self):
        with pytest.raises(ValueError):
            AgentConfig(truncate_arg_log_at=0)

    def test_frozen(self):
        c = AgentConfig()
        with pytest.raises(Exception):
            c.max_iterations = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AgentContext
# ---------------------------------------------------------------------------


class TestAgentContext:
    def test_default_empty(self):
        ctx = AgentContext()
        assert ctx.messages == []
        assert ctx.iteration == 0
        assert ctx.events == []
        assert ctx.measurements == []

    def test_appends_independently(self):
        ctx = AgentContext()
        ctx.messages.append(Message.user("hi"))
        ctx.events.append(Event(type="x", severity="info", detail="d"))
        ctx.measurements.append(
            Measurement(metric="m", value=1, unit=None, confidence=0.5,
                        method="x", evidence=("e",))
        )
        assert len(ctx.messages) == 1
        assert len(ctx.events) == 1
        assert len(ctx.measurements) == 1

    def test_snapshot(self):
        ctx = AgentContext()
        ctx.messages.append(Message.user("hi"))
        ctx.iteration = 2
        snap = ctx.snapshot()
        assert snap == {
            "iteration": 2, "n_messages": 1, "n_events": 0, "n_measurements": 0,
        }


# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------


class TestAgentResult:
    def test_basic(self):
        ctx = AgentContext()
        r = AgentResult(reason="completed", summary="ok", payload={"x": 1},
                        iterations=3, context=ctx)
        assert r.reason == "completed"
        assert r.payload == {"x": 1}

    def test_invalid_reason_rejected(self):
        ctx = AgentContext()
        with pytest.raises(ValueError):
            AgentResult(reason="banana", summary=None, payload=None,  # type: ignore[arg-type]
                        iterations=0, context=ctx)

    def test_negative_iterations_rejected(self):
        ctx = AgentContext()
        with pytest.raises(ValueError):
            AgentResult(reason="completed", summary=None, payload=None,
                        iterations=-1, context=ctx)

    def test_frozen(self):
        ctx = AgentContext()
        r = AgentResult(reason="completed", summary=None, payload=None,
                        iterations=0, context=ctx)
        with pytest.raises(Exception):
            r.summary = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ReActPhase
# ---------------------------------------------------------------------------


class TestReActPhase:
    def test_all_phases_have_distinct_values(self):
        values = {p.value for p in ReActPhase}
        assert len(values) == len(list(ReActPhase))

    def test_phase_names(self):
        names = {p.name for p in ReActPhase}
        assert names == {
            "THOUGHT", "VALIDATE", "ACT", "OBSERVE", "APPLY", "DECIDE",
        }
