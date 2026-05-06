"""Unit tests for mls_agent.llm.openai_backend.

Uses a fake OpenAI client (duck-typed) so the tests run without network
or API keys. Covers: response parsing, tool-call argument decode,
GLM artifact stripping, reasoning_content extraction, sticky parameter
learning on 400 errors, exponential backoff on 429.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import openai
import pytest

from mls_agent.llm.config import LLMConfig
from mls_agent.llm.openai_backend import (
    OpenAIBackend,
    _is_reasoning_model,
    _max_tokens_param,
    _strip_glm_artifacts,
)


# ---------------------------------------------------------------------------
# Fake OpenAI client
# ---------------------------------------------------------------------------


def _mk_choice(*, content=None, tool_calls=None, finish_reason="stop",
               reasoning_content=None):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
    )
    return SimpleNamespace(message=msg, finish_reason=finish_reason)


def _mk_completion(*, choices, usage=None):
    return SimpleNamespace(choices=choices, usage=usage)


def _mk_tool_call(*, id, name, arguments):
    return SimpleNamespace(
        id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _FakeClient:
    """Pluggable in place of openai.OpenAI."""

    def __init__(self, plan):
        # plan: list of either responses or exceptions to raise in order.
        self._plan = list(plan)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._plan.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _make_backend(client, model="gpt-4o", max_retries=3):
    cfg = LLMConfig(
        api_key="sk", model=model, max_retries=max_retries,
        request_timeout_s=5.0,
    )
    backend = OpenAIBackend(cfg, client=client)
    backend._sleep = lambda s: None  # disable real sleeping
    return backend


def _mk_status_error(message: str, status_code: int) -> openai.APIStatusError:
    """Construct an APIStatusError with just enough plumbing to satisfy
    the SDK's __init__ (it reads response.request and response.status_code)."""
    fake_request = SimpleNamespace(method="POST", url="https://x")
    fake_response = SimpleNamespace(
        request=fake_request,
        status_code=status_code,
        headers={},
    )
    err = openai.APIStatusError(message, response=fake_response, body={})  # type: ignore[arg-type]
    return err


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestPureHelpers:
    def test_is_reasoning_model(self):
        assert _is_reasoning_model("o1-mini")
        assert _is_reasoning_model("o3")
        assert _is_reasoning_model("gpt-5-turbo")
        assert _is_reasoning_model("deepseek-reasoner")
        assert _is_reasoning_model("deepseek-r1-lite")
        assert not _is_reasoning_model("gpt-4o")
        assert not _is_reasoning_model("claude-3")

    def test_max_tokens_param(self):
        assert _max_tokens_param("o1-mini") == "max_completion_tokens"
        assert _max_tokens_param("gpt-5") == "max_completion_tokens"
        assert _max_tokens_param("gpt-4o") == "max_tokens"
        assert _max_tokens_param("deepseek-chat") == "max_tokens"

    def test_strip_glm_artifacts_no_artifact(self):
        assert _strip_glm_artifacts("hello world") == "hello world"
        assert _strip_glm_artifacts(None) is None
        # Empty input passes through unchanged (early return path).
        assert _strip_glm_artifacts("") == ""

    def test_strip_glm_artifacts_only_artifact_returns_none(self):
        # Content that's nothing but artifact strips to empty → None.
        only = '{"index":0,"finish_reason":"stop","delta":{}}'
        assert _strip_glm_artifacts(only) is None

    def test_strip_glm_artifacts_removes_streaming_delta(self):
        polluted = (
            'real text {"index":0,"finish_reason":"tool_calls",'
            '"delta":{"content":null}} more text'
        )
        cleaned = _strip_glm_artifacts(polluted)
        assert cleaned == "real text  more text"

    def test_strip_glm_artifacts_handles_nested_braces(self):
        polluted = 'before {"index":0,"delta":{"x":{"y":1}}} after'
        cleaned = _strip_glm_artifacts(polluted)
        assert cleaned == "before  after"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_text_only_response(self):
        client = _FakeClient([
            _mk_completion(choices=[_mk_choice(content="hello", finish_reason="stop")])
        ])
        backend = _make_backend(client)
        resp = backend.chat([], [])
        assert resp.message.content == "hello"
        assert resp.message.tool_calls == ()
        assert resp.finish_reason == "stop"

    def test_tool_call_response_parses_arguments(self):
        tc = _mk_tool_call(id="c1", name="search", arguments='{"q":"foo"}')
        client = _FakeClient([
            _mk_completion(choices=[
                _mk_choice(content=None, tool_calls=[tc], finish_reason="tool_calls")
            ])
        ])
        backend = _make_backend(client)
        resp = backend.chat([], [])
        assert len(resp.message.tool_calls) == 1
        call = resp.message.tool_calls[0]
        assert call.id == "c1"
        assert call.name == "search"
        assert call.arguments == {"q": "foo"}
        assert call.parse_error is None

    def test_malformed_arguments_recorded_as_parse_error(self):
        tc = _mk_tool_call(id="c1", name="search", arguments='{"q":')  # bad JSON
        client = _FakeClient([
            _mk_completion(choices=[
                _mk_choice(content=None, tool_calls=[tc], finish_reason="tool_calls")
            ])
        ])
        backend = _make_backend(client)
        resp = backend.chat([], [])
        call = resp.message.tool_calls[0]
        assert call.arguments == {}
        assert call.parse_error is not None
        assert call.arguments_raw == '{"q":'

    def test_non_object_arguments_treated_as_parse_error(self):
        # Some weird providers return arrays for arguments.
        tc = _mk_tool_call(id="c1", name="search", arguments='[1, 2]')
        client = _FakeClient([
            _mk_completion(choices=[
                _mk_choice(content=None, tool_calls=[tc], finish_reason="tool_calls")
            ])
        ])
        backend = _make_backend(client)
        resp = backend.chat([], [])
        call = resp.message.tool_calls[0]
        assert call.arguments == {}
        assert "JSON object" in call.parse_error

    def test_glm_artifact_stripped(self):
        polluted = 'thought {"index":0,"finish_reason":"stop","delta":{}} done'
        client = _FakeClient([
            _mk_completion(choices=[
                _mk_choice(content=polluted, finish_reason="stop")
            ])
        ])
        backend = _make_backend(client)
        resp = backend.chat([], [])
        assert resp.message.content == "thought  done"

    def test_reasoning_content_attribute(self):
        client = _FakeClient([
            _mk_completion(choices=[
                _mk_choice(content="answer", reasoning_content="thinking...",
                           finish_reason="stop")
            ])
        ])
        backend = _make_backend(client)
        resp = backend.chat([], [])
        assert resp.message.reasoning_content == "thinking..."

    def test_finish_reason_normalized(self):
        client = _FakeClient([
            _mk_completion(choices=[_mk_choice(content="x", finish_reason="weird")])
        ])
        backend = _make_backend(client)
        resp = backend.chat([], [])
        assert resp.finish_reason == "other"

    def test_usage_extracted(self):
        usage = SimpleNamespace(
            prompt_tokens=10, completion_tokens=3, total_tokens=13,
        )
        client = _FakeClient([
            _mk_completion(
                choices=[_mk_choice(content="x", finish_reason="stop")],
                usage=usage,
            )
        ])
        backend = _make_backend(client)
        resp = backend.chat([], [])
        assert resp.usage == {
            "prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13,
        }


# ---------------------------------------------------------------------------
# Provider dict round-trip
# ---------------------------------------------------------------------------


class TestProviderDictConversion:
    def test_system_to_dict(self):
        from mls_agent.llm.types import Message

        d = OpenAIBackend._to_provider_dict(Message.system("hi"))
        assert d == {"role": "system", "content": "hi"}

    def test_user_to_dict(self):
        from mls_agent.llm.types import Message

        d = OpenAIBackend._to_provider_dict(Message.user("hi"))
        assert d == {"role": "user", "content": "hi"}

    def test_tool_to_dict(self):
        from mls_agent.llm.types import Message

        d = OpenAIBackend._to_provider_dict(
            Message.tool_result(tool_call_id="c1", content="42")
        )
        assert d == {"role": "tool", "tool_call_id": "c1", "content": "42"}

    def test_assistant_with_tool_calls_uses_raw_arguments_when_available(self):
        from mls_agent.llm.types import Message, ToolCall

        tc = ToolCall(
            id="c1",
            name="search",
            arguments={"q": "foo"},
            arguments_raw='{"q":"foo"}',
        )
        d = OpenAIBackend._to_provider_dict(Message.assistant(content=None, tool_calls=(tc,)))
        assert d["tool_calls"][0]["function"]["arguments"] == '{"q":"foo"}'

    def test_assistant_with_tool_calls_falls_back_to_serialize(self):
        from mls_agent.llm.types import Message, ToolCall

        tc = ToolCall(id="c1", name="search", arguments={"q": "foo"})
        d = OpenAIBackend._to_provider_dict(Message.assistant(content=None, tool_calls=(tc,)))
        # No raw, so serialize.
        parsed = json.loads(d["tool_calls"][0]["function"]["arguments"])
        assert parsed == {"q": "foo"}


# ---------------------------------------------------------------------------
# Param compatibility / sticky learning
# ---------------------------------------------------------------------------


class TestParamCompat:
    def test_reasoning_model_omits_temperature(self):
        client = _FakeClient([
            _mk_completion(choices=[_mk_choice(content="x", finish_reason="stop")])
        ])
        backend = _make_backend(client, model="o1-mini")
        backend.chat([], [])
        assert "temperature" not in client.calls[0]
        assert "max_completion_tokens" in client.calls[0]

    def test_normal_model_passes_temperature(self):
        client = _FakeClient([
            _mk_completion(choices=[_mk_choice(content="x", finish_reason="stop")])
        ])
        backend = _make_backend(client, model="gpt-4o")
        backend.chat([], [])
        assert client.calls[0]["temperature"] == 0.2
        assert "max_tokens" in client.calls[0]

    def test_400_temperature_triggers_sticky_skip(self):
        # First call: 400 about temperature. Second: success.
        err = _mk_status_error("Unsupported value: 'temperature'.", 400)
        client = _FakeClient([
            err,
            _mk_completion(choices=[_mk_choice(content="x", finish_reason="stop")]),
        ])
        backend = _make_backend(client)
        backend.chat([], [])
        # Sticky flag flipped, second call has no temperature.
        assert "temperature" not in client.calls[1]
        # Subsequent fresh chat call should also omit temperature.
        client._plan = [
            _mk_completion(choices=[_mk_choice(content="y", finish_reason="stop")])
        ]
        backend.chat([], [])
        assert "temperature" not in client.calls[2]

    def test_400_max_completion_tokens_triggers_sticky_fallback(self):
        err = _mk_status_error(
            "Unsupported parameter: 'max_completion_tokens'.", 400
        )
        client = _FakeClient([
            err,
            _mk_completion(choices=[_mk_choice(content="x", finish_reason="stop")]),
        ])
        backend = _make_backend(client, model="o1-mini")
        backend.chat([], [])
        assert "max_completion_tokens" not in client.calls[1]
        assert "max_tokens" in client.calls[1]

    def test_unrelated_400_raised(self):
        err = _mk_status_error("model_not_found", 400)
        client = _FakeClient([err])
        backend = _make_backend(client)
        with pytest.raises(openai.APIStatusError):
            backend.chat([], [])


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------


class TestRetries:
    def test_retries_on_timeout_then_succeeds(self):
        timeout = openai.APITimeoutError(request=SimpleNamespace())  # type: ignore[arg-type]
        client = _FakeClient([
            timeout,
            timeout,
            _mk_completion(choices=[_mk_choice(content="hi", finish_reason="stop")]),
        ])
        backend = _make_backend(client, max_retries=3)
        resp = backend.chat([], [])
        assert resp.message.content == "hi"
        assert len(client.calls) == 3

    def test_429_status_error_retried(self):
        rate = _mk_status_error("rate limited", 429)
        client = _FakeClient([
            rate,
            _mk_completion(choices=[_mk_choice(content="ok", finish_reason="stop")]),
        ])
        backend = _make_backend(client, max_retries=3)
        resp = backend.chat([], [])
        assert resp.message.content == "ok"

    def test_retries_exhausted_raises(self):
        timeout = openai.APITimeoutError(request=SimpleNamespace())  # type: ignore[arg-type]
        client = _FakeClient([timeout, timeout, timeout])
        backend = _make_backend(client, max_retries=3)
        with pytest.raises(openai.APITimeoutError):
            backend.chat([], [])
