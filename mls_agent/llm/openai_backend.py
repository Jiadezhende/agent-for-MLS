"""OpenAI-compatible chat backend.

This is the ONLY file in the project allowed to import the ``openai`` SDK.
Everything provider-specific lives behind this module: parameter shape
juggling, GLM artifact stripping, DeepSeek reasoning extraction,
exponential backoff, sticky parameter learning. Callers above the LLM
layer see only ``Message`` and ``ChatResponse``.

Adapted from the legacy ``agent/core/llm.py`` (LLMClient).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Sequence

import openai

from mls_agent.llm.backend import LLMBackend
from mls_agent.llm.config import LLMConfig
from mls_agent.llm.types import ChatResponse, FinishReason, Message, ToolCall


# ---------------------------------------------------------------------------
# Per-model parameter compatibility (sticky learning)
# ---------------------------------------------------------------------------


def _is_reasoning_model(model: str) -> bool:
    """Models that reject ``temperature`` (must omit or fix at 1)."""
    m = model.lower()
    return (
        bool(re.match(r"o\d", m))
        or m.startswith("gpt-5")
        or m.startswith("deepseek-reasoner")
        or m.startswith("deepseek-r1")
    )


def _max_tokens_param(model: str) -> str:
    """OpenAI o-series / gpt-5+ use ``max_completion_tokens``; others use ``max_tokens``."""
    m = model.lower()
    if bool(re.match(r"o\d", m)) or m.startswith("gpt-5"):
        return "max_completion_tokens"
    return "max_tokens"


class _ModelCompat:
    """Sticky parameter learning per model.

    On a 400 about ``temperature`` or ``max_completion_tokens`` we flip a
    flag so subsequent requests omit the offending parameter. The flags
    survive across calls within one backend instance.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.skip_temperature: bool = False
        self.force_max_tokens: bool = False

    def kwargs(self, cfg: LLMConfig) -> dict[str, Any]:
        kw: dict[str, Any] = {"timeout": cfg.request_timeout_s}
        if cfg.max_tokens:
            param = "max_tokens" if self.force_max_tokens else _max_tokens_param(self.model)
            kw[param] = cfg.max_tokens
        if not self.skip_temperature and not _is_reasoning_model(self.model):
            kw["temperature"] = cfg.temperature
        return kw

    def absorb_400(self, exc: Exception) -> bool:
        """Inspect a 400 body; flip flags if it complains about a known param.

        Returns True iff a flag changed (caller should retry immediately).
        """
        body = str(exc).lower()
        changed = False
        if not self.skip_temperature and "temperature" in body:
            self.skip_temperature = True
            changed = True
        if not self.force_max_tokens and "max_completion_tokens" in body:
            self.force_max_tokens = True
            changed = True
        return changed


# ---------------------------------------------------------------------------
# GLM streaming-artifact cleaner
# ---------------------------------------------------------------------------


_GLM_ARTIFACT_START = re.compile(r'\{"index":\s*\d+')


def _strip_glm_artifacts(text: str | None) -> str | None:
    """Remove GLM-API streaming delta JSON objects accidentally embedded in content.

    GLM's non-streaming mode appends raw SSE delta objects like
    ``{"index":0,"finish_reason":"tool_calls","delta":{...}}`` to the
    text content. Strip those out by brace-matching, leaving the human
    text intact.
    """
    if not text:
        return text
    parts: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        m = _GLM_ARTIFACT_START.search(text, i)
        if m is None:
            parts.append(text[i:])
            break
        parts.append(text[i:m.start()])
        # Walk forward counting braces to find the matching closing brace.
        j = m.start()
        depth = 0
        in_str = False
        esc = False
        while j < n:
            c = text[j]
            if esc:
                esc = False
            elif c == "\\" and in_str:
                esc = True
            elif c == '"':
                in_str = not in_str
            elif not in_str:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
            j += 1
        i = j
    cleaned = "".join(parts).strip()
    return cleaned or None


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------


_VALID_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter"}


def _normalize_finish_reason(raw: str | None) -> FinishReason:
    if raw in _VALID_FINISH_REASONS:
        return raw  # type: ignore[return-value]
    return "other"


class OpenAIBackend(LLMBackend):
    """``LLMBackend`` implementation for the OpenAI Python SDK family."""

    def __init__(self, config: LLMConfig, *, client: Any | None = None) -> None:
        self._cfg = config
        self._client = client or openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self._compat = _ModelCompat(config.model)

    # ------------------------------------------------------------------
    # LLMBackend
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> ChatResponse:
        provider_msgs = [self._to_provider_dict(m) for m in messages]
        last_exc: Exception | None = None
        attempt = 0
        # Note: parameter-fallback retries do NOT consume the retry budget;
        # they pass through with ``continue`` after flipping a flag.
        while attempt < self._cfg.max_retries:
            try:
                raw = self._client.chat.completions.create(
                    model=self._cfg.model,
                    messages=provider_msgs,
                    tools=list(tools),
                    tool_choice="auto",
                    **self._compat.kwargs(self._cfg),
                )
                return self._parse_response(raw)
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as exc:
                last_exc = exc
                base = 15 if isinstance(exc, openai.RateLimitError) else 1
                self._sleep(base * (2**attempt))
                attempt += 1
            except openai.APIStatusError as exc:
                if exc.status_code == 429:
                    last_exc = exc
                    self._sleep(15 * (2**attempt))
                    attempt += 1
                elif exc.status_code == 400 and self._compat.absorb_400(exc):
                    # Flag flipped; retry immediately without consuming a slot.
                    continue
                else:
                    raise

        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # Conversion: Message <-> provider dict
    # ------------------------------------------------------------------

    @staticmethod
    def _to_provider_dict(msg: Message) -> dict[str, Any]:
        if msg.role == "system":
            return {"role": "system", "content": msg.content or ""}
        if msg.role == "user":
            return {"role": "user", "content": msg.content or ""}
        if msg.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content or "",
            }
        # assistant
        out: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments_raw
                        or json.dumps(tc.arguments, default=str),
                    },
                }
                for tc in msg.tool_calls
            ]
        # Thinking-mode models (DeepSeek-reasoner, Qwen with thinking, …)
        # require the `reasoning_content` from a prior assistant turn to be
        # passed back on every subsequent request, otherwise they 400 with
        # "The `reasoning_content` in the thinking mode must be passed back
        # to the API." Forward it verbatim when present.
        if msg.reasoning_content:
            out["reasoning_content"] = msg.reasoning_content
        return out

    @staticmethod
    def _parse_tool_call(tc: Any) -> ToolCall:
        raw_args = tc.function.arguments or ""
        try:
            parsed = json.loads(raw_args) if raw_args.strip() else {}
            parse_error: str | None = None
        except json.JSONDecodeError as e:
            parsed = {}
            parse_error = str(e)
        if not isinstance(parsed, dict):
            parse_error = (
                f"arguments must be a JSON object, got {type(parsed).__name__}"
            )
            parsed = {}
        return ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=parsed,
            arguments_raw=raw_args,
            parse_error=parse_error,
        )

    @classmethod
    def _parse_response(cls, raw: Any) -> ChatResponse:
        choice = raw.choices[0]
        provider_msg = choice.message

        tool_calls: tuple[ToolCall, ...] = ()
        if provider_msg.tool_calls:
            tool_calls = tuple(cls._parse_tool_call(tc) for tc in provider_msg.tool_calls)

        # DeepSeek-style reasoning content.
        reasoning_content = getattr(provider_msg, "reasoning_content", None)
        if reasoning_content is None:
            extra = getattr(provider_msg, "model_extra", None)
            if isinstance(extra, dict):
                reasoning_content = extra.get("reasoning_content")

        cleaned_content = _strip_glm_artifacts(provider_msg.content)

        message = Message.assistant(
            content=cleaned_content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )
        finish_reason = _normalize_finish_reason(choice.finish_reason)

        usage = None
        u = getattr(raw, "usage", None)
        if u is not None:
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0),
                "completion_tokens": getattr(u, "completion_tokens", 0),
                "total_tokens": getattr(u, "total_tokens", 0),
            }

        return ChatResponse(
            message=message,
            finish_reason=finish_reason,
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Test seam
    # ------------------------------------------------------------------

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)
