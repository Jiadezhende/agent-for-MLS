from __future__ import annotations

from types import SimpleNamespace

from agents.core.llm import ChatResponse


def _raw_tool_call() -> SimpleNamespace:
    return SimpleNamespace(
        id="call_0",
        function=SimpleNamespace(
            name="list_skills",
            arguments="{}",
        ),
    )


def test_chat_response_round_trips_reasoning_content_with_tool_calls():
    raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content="",
                    reasoning_content="provider thinking payload",
                    tool_calls=[_raw_tool_call()],
                ),
            )
        ]
    )

    resp = ChatResponse.from_openai(raw)
    msg = resp.to_openai_message()

    assert resp.reasoning_content == "provider thinking payload"
    assert msg["role"] == "assistant"
    assert msg["content"] == ""
    assert msg["reasoning_content"] == "provider thinking payload"
    assert msg["tool_calls"] == [
        {
            "id": "call_0",
            "type": "function",
            "function": {
                "name": "list_skills",
                "arguments": "{}",
            },
        }
    ]


def test_chat_response_reads_reasoning_content_from_model_extra():
    raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=None,
                    model_extra={"reasoning_content": "extra thinking payload"},
                ),
            )
        ]
    )

    resp = ChatResponse.from_openai(raw)

    assert resp.reasoning_content == "extra thinking payload"
    assert resp.to_openai_message()["reasoning_content"] == "extra thinking payload"
