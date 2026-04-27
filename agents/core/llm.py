"""
agents/core/llm.py - The ONLY file that imports openai.

Wraps the OpenAI SDK into a simple chat() interface. Non-streaming,
basic exponential-backoff retry. Streaming + tenacity come in Phase 2.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import openai

from agents.core.config import LLMConfig


# ---------------------------------------------------------------------------
# Data returned from the LLM (provider-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None            # parsed from JSON; None if parse failed
    arguments_raw: str = ""           # raw string preserved on parse failure

    @classmethod
    def from_openai(cls, tc: object) -> "ToolCall":
        raw = tc.function.arguments or ""
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            parsed = None
        return cls(
            id=tc.id,
            name=tc.function.name,
            arguments=parsed,
            arguments_raw=raw,
        )


@dataclass
class ChatResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: str | None = None
    finish_reason: str = ""
    _raw_tool_calls: list = field(default_factory=list, repr=False)

    @classmethod
    def from_openai(cls, raw: object) -> "ChatResponse":
        choice = raw.choices[0]
        msg = choice.message
        tcs = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tcs.append(ToolCall.from_openai(tc))

        reasoning_content = getattr(msg, "reasoning_content", None)
        if reasoning_content is None:
            model_extra = getattr(msg, "model_extra", None)
            if isinstance(model_extra, dict):
                reasoning_content = model_extra.get("reasoning_content")

        return cls(
            content=msg.content,
            tool_calls=tcs,
            reasoning_content=reasoning_content,
            finish_reason=choice.finish_reason or "",
            _raw_tool_calls=msg.tool_calls or [],
        )

    def to_openai_message(self) -> dict:
        """Reconstruct the assistant message dict expected on the next turn.

        IMPORTANT: The tool_calls list must be included verbatim (as dicts)
        so the provider can match tool call IDs to the tool-result messages
        we append afterwards.
        """
        msg: dict = {"role": "assistant", "content": self.content}
        if self.reasoning_content is not None:
            msg["reasoning_content"] = self.reasoning_content
        if self._raw_tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self._raw_tool_calls
            ]
        return msg


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """Thin wrapper around openai.OpenAI.

    Exposes a single chat() method. The rest of the codebase never sees
    the openai module.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        self._client = openai.OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,          # None uses OpenAI default
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> ChatResponse:
        """Call the LLM with retry on transient errors.

        Raises the last exception if all retries are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(self._cfg.max_retries):
            try:
                raw = self._client.chat.completions.create(
                    model=self._cfg.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=self._cfg.max_tokens,
                    temperature=self._cfg.temperature,
                    timeout=self._cfg.request_timeout_s,
                )
                return ChatResponse.from_openai(raw)
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as exc:
                last_exc = exc
                wait = 2 ** attempt          # 1 s, 2 s, 4 s, ...
                print(
                    f"[llm] transient error (attempt {attempt + 1}/"
                    f"{self._cfg.max_retries}): {exc}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            except openai.APIStatusError as exc:
                # 4xx errors (except 429) are not transient; surface immediately.
                if exc.status_code == 429:
                    last_exc = exc
                    wait = 2 ** attempt
                    time.sleep(wait)
                else:
                    raise

        assert last_exc is not None
        raise last_exc
