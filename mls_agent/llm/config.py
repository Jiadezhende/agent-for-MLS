"""LLM connection / behavior config.

Frozen dataclass; build once at app start, pass to backend constructor.
``from_env`` is provided for parity with the legacy code path but the
config can be constructed directly in tests / scripts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str
    base_url: str | None = None
    max_tokens: int = 8192
    temperature: float = 0.2
    request_timeout_s: float = 120.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("LLMConfig.api_key must be non-empty")
        if not self.model:
            raise ValueError("LLMConfig.model must be non-empty")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        if self.request_timeout_s <= 0:
            raise ValueError(
                f"request_timeout_s must be > 0, got {self.request_timeout_s}"
            )
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = os.getenv("API_KEY", "")
        model = os.getenv("BASE_MODEL", "")
        if not api_key:
            raise OSError("API_KEY is not set in the environment")
        if not model:
            raise OSError("BASE_MODEL is not set in the environment")
        raw_base_url = (os.getenv("BASE_URL") or "").strip()
        return cls(
            api_key=api_key,
            model=model,
            base_url=raw_base_url or None,
            max_tokens=int(os.getenv("AGENT_LLM_MAX_TOKENS") or "8192"),
            temperature=float(os.getenv("AGENT_LLM_TEMPERATURE") or "0.2"),
            request_timeout_s=float(os.getenv("AGENT_LLM_TIMEOUT_S") or "120"),
            max_retries=int(os.getenv("AGENT_LLM_MAX_RETRIES") or "3"),
        )
