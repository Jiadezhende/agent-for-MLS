"""Unit tests for mls_agent.llm.types."""
from __future__ import annotations

import pytest

from mls_agent.llm.types import ChatResponse, Message, ToolCall


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_basic_construction(self):
        tc = ToolCall(id="call_1", name="search", arguments={"q": "foo"})
        assert tc.id == "call_1"
        assert tc.name == "search"
        assert tc.arguments == {"q": "foo"}
        assert tc.parse_error is None

    def test_empty_id_rejected(self):
        with pytest.raises(ValueError, match="id"):
            ToolCall(id="", name="search", arguments={})

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            ToolCall(id="call_1", name="", arguments={})

    def test_arguments_default_empty_dict(self):
        tc = ToolCall(id="call_1", name="search")
        assert tc.arguments == {}

    def test_non_dict_arguments_rejected(self):
        with pytest.raises(TypeError, match="dict"):
            ToolCall(id="call_1", name="search", arguments="oops")  # type: ignore[arg-type]

    def test_parse_failure_carries_raw(self):
        tc = ToolCall(
            id="call_1",
            name="search",
            arguments={},
            arguments_raw='{"q":',
            parse_error="unexpected end of input",
        )
        assert tc.arguments == {}
        assert tc.parse_error == "unexpected end of input"
        assert tc.arguments_raw == '{"q":'

    def test_frozen(self):
        tc = ToolCall(id="call_1", name="search")
        with pytest.raises(Exception):  # FrozenInstanceError
            tc.id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class TestMessage:
    def test_system_factory(self):
        m = Message.system("hello")
        assert m.role == "system"
        assert m.content == "hello"
        assert m.tool_calls == ()

    def test_user_factory(self):
        m = Message.user("hi")
        assert m.role == "user"
        assert m.content == "hi"

    def test_assistant_with_tool_calls(self):
        tc = ToolCall(id="c", name="t")
        m = Message.assistant(content=None, tool_calls=(tc,))
        assert m.role == "assistant"
        assert m.tool_calls == (tc,)

    def test_assistant_with_reasoning(self):
        m = Message.assistant(content="answer", reasoning_content="thinking…")
        assert m.reasoning_content == "thinking…"

    def test_tool_result_factory(self):
        m = Message.tool_result(tool_call_id="c1", content="42")
        assert m.role == "tool"
        assert m.tool_call_id == "c1"
        assert m.content == "42"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValueError, match="role"):
            Message(role="banana", content="x")  # type: ignore[arg-type]

    def test_tool_calls_only_on_assistant(self):
        tc = ToolCall(id="c", name="t")
        with pytest.raises(ValueError, match="tool_calls"):
            Message(role="user", content="hi", tool_calls=(tc,))

    def test_tool_call_id_only_on_tool_role(self):
        with pytest.raises(ValueError, match="tool_call_id"):
            Message(role="user", content="hi", tool_call_id="c1")

    def test_tool_role_requires_call_id(self):
        with pytest.raises(ValueError, match="tool_call_id"):
            Message(role="tool", content="42")

    def test_tool_role_requires_content(self):
        with pytest.raises(ValueError, match="content"):
            Message(role="tool", content=None, tool_call_id="c1")

    def test_tool_calls_must_be_tuple(self):
        tc = ToolCall(id="c", name="t")
        with pytest.raises(TypeError, match="tuple"):
            Message(
                role="assistant",
                content=None,
                tool_calls=[tc],  # type: ignore[arg-type]
            )

    def test_frozen(self):
        m = Message.user("hi")
        with pytest.raises(Exception):  # FrozenInstanceError
            m.content = "bye"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ChatResponse
# ---------------------------------------------------------------------------


class TestChatResponse:
    def test_basic(self):
        msg = Message.assistant(content="hi")
        r = ChatResponse(message=msg, finish_reason="stop")
        assert r.message is msg
        assert r.finish_reason == "stop"
        assert r.usage is None

    def test_carries_usage(self):
        msg = Message.assistant(content="hi")
        r = ChatResponse(
            message=msg,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 3},
        )
        assert r.usage == {"prompt_tokens": 10, "completion_tokens": 3}

    def test_non_assistant_message_rejected(self):
        msg = Message.user("hi")
        with pytest.raises(ValueError, match="assistant"):
            ChatResponse(message=msg, finish_reason="stop")

    def test_invalid_finish_reason_rejected(self):
        msg = Message.assistant(content="hi")
        with pytest.raises(ValueError, match="finish_reason"):
            ChatResponse(message=msg, finish_reason="banana")  # type: ignore[arg-type]

    def test_with_tool_calls_finish_reason(self):
        tc = ToolCall(id="c", name="t")
        msg = Message.assistant(content=None, tool_calls=(tc,))
        r = ChatResponse(message=msg, finish_reason="tool_calls")
        assert r.message.tool_calls == (tc,)
