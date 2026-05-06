"""
agents/core/llm.py — The ONLY file that imports openai.

Wraps the OpenAI SDK into a simple chat() interface.  Non-streaming,
basic exponential-backoff retry.  Streaming + tenacity come in Phase 2.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import openai

from agent.core.config import LLMConfig


def _is_reasoning_model(model: str) -> bool:
    """Models that don't accept temperature (must omit or fix at 1).

    - OpenAI: o-series (o1, o3, o4-mini …), gpt-5+
    - DeepSeek: deepseek-reasoner, deepseek-r1* — API error if temperature != 1
    """
    m = model.lower()
    return (
        bool(re.match(r"o\d", m))
        or m.startswith("gpt-5")
        or m.startswith("deepseek-reasoner")
        or m.startswith("deepseek-r1")
    )


def _max_tokens_param(model: str) -> str:
    """Only OpenAI o-series / gpt-5+ use max_completion_tokens; all others use max_tokens."""
    m = model.lower()
    if bool(re.match(r"o\d", m)) or m.startswith("gpt-5"):
        return "max_completion_tokens"
    return "max_tokens"


def _build_create_kwargs(
    cfg: LLMConfig,
    skip_temperature: bool = False,
    force_max_tokens: bool = False,
) -> dict[str, Any]:
    """Return only the API params this model family accepts.

    skip_temperature / force_max_tokens are runtime-learned overrides applied
    after a 400 param error is detected by LLMClient.chat().
    """
    kwargs: dict[str, Any] = {"timeout": cfg.request_timeout_s}
    if cfg.max_tokens:
        param = "max_tokens" if force_max_tokens else _max_tokens_param(cfg.model)
        kwargs[param] = cfg.max_tokens
    if not skip_temperature and not _is_reasoning_model(cfg.model):
        kwargs["temperature"] = cfg.temperature
    return kwargs


# ---------------------------------------------------------------------------
# GLM streaming-artifact cleaner
# ---------------------------------------------------------------------------

_GLM_ARTIFACT_START = re.compile(r'\{"index":\s*\d+')


def _strip_glm_artifacts(text: str | None) -> str | None:
    """Remove GLM-API streaming delta JSON objects accidentally embedded in content.

    GLM's non-streaming mode appends raw SSE delta objects like
    {"index":0,"finish_reason":"tool_calls","delta":{...}} to the text content.
    These pollute conversation history and confuse subsequent LLM calls.
    Uses brace-counting to correctly handle nested JSON.
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
        # Walk forward counting braces to find the matching closing brace
        j = m.start()
        depth = 0
        in_str = False
        esc = False
        while j < n:
            c = text[j]
            if esc:
                esc = False
            elif c == '\\' and in_str:
                esc = True
            elif c == '"':
                in_str = not in_str
            elif not in_str:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
            j += 1
        i = j
    cleaned = ''.join(parts).strip()
    return cleaned or None


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
            content=_strip_glm_artifacts(msg.content),
            tool_calls=tcs,
            reasoning_content=reasoning_content,
            finish_reason=choice.finish_reason or "",
            _raw_tool_calls=msg.tool_calls or [],
        )

    def to_openai_message(self) -> dict:
        """Reconstruct the assistant message dict the API expects on the next turn.

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

    Exposes a single chat() method.  The rest of the codebase never sees
    the openai module.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        self._client = openai.OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,          # None → uses OpenAI default
        )
        # Runtime-learned param compatibility overrides (sticky across calls).
        # Set automatically on first 400 param error; avoids repeating bad params.
        self._skip_temperature: bool = False   # learned: provider rejects temperature
        self._force_max_tokens: bool = False   # learned: provider rejects max_completion_tokens

    def with_max_tokens(self, n: int) -> "LLMClient":
        """Return a new LLMClient with the same connection but a different max_tokens."""
        import dataclasses
        new_cfg = dataclasses.replace(self._cfg, max_tokens=n)
        child = LLMClient(new_cfg)
        # Carry over runtime-learned param overrides so the child doesn't re-learn them.
        child._skip_temperature = self._skip_temperature
        child._force_max_tokens = self._force_max_tokens
        return child

    def _try_param_fallback(self, exc: openai.APIStatusError) -> bool:
        """Inspect a 400 error body and update sticky param overrides.

        Returns True if an override was applied (caller should retry immediately),
        False if the error is unrelated to parameter compatibility.
        """
        body = str(exc).lower()
        changed = False
        if not self._skip_temperature and "temperature" in body:
            self._skip_temperature = True
            print(
                f"[llm] provider rejected temperature for model '{self._cfg.model}'; "
                "disabling for all subsequent calls.",
                flush=True,
            )
            changed = True
        if not self._force_max_tokens and "max_completion_tokens" in body:
            self._force_max_tokens = True
            print(
                f"[llm] provider rejected max_completion_tokens for model '{self._cfg.model}'; "
                "falling back to max_tokens.",
                flush=True,
            )
            changed = True
        return changed

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> ChatResponse:
        """Call the LLM with retry on transient errors.

        On 400 parameter-incompatibility errors (temperature, max_completion_tokens)
        the client updates sticky overrides and retries once before giving up.
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
                    **_build_create_kwargs(
                        self._cfg,
                        skip_temperature=self._skip_temperature,
                        force_max_tokens=self._force_max_tokens,
                    ),
                )
                return ChatResponse.from_openai(raw)
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as exc:
                last_exc = exc
                is_rate_limit = isinstance(exc, openai.RateLimitError)
                base = 15 if is_rate_limit else 1
                wait = base * (2 ** attempt)   # rate-limit: 15s, 30s, 60s, …
                print(
                    f"[llm] transient error (attempt {attempt + 1}/"
                    f"{self._cfg.max_retries}): {exc}. "
                    f"Retrying in {wait}s …",
                    flush=True,
                )
                time.sleep(wait)
            except openai.APIStatusError as exc:
                if exc.status_code == 429:
                    last_exc = exc
                    wait = 15 * (2 ** attempt)  # 15s, 30s, 60s, …
                    print(
                        f"[llm] rate-limited (attempt {attempt + 1}/"
                        f"{self._cfg.max_retries}): {exc}. "
                        f"Retrying in {wait}s …",
                        flush=True,
                    )
                    time.sleep(wait)
                elif exc.status_code == 400 and self._try_param_fallback(exc):
                    # Param override applied — retry immediately, don't count as attempt.
                    continue
                else:
                    raise

        assert last_exc is not None
        raise last_exc
