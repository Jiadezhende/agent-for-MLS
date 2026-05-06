"""Unit tests for mls_agent.llm.config."""
from __future__ import annotations

import pytest

from mls_agent.llm.config import LLMConfig


class TestLLMConfig:
    def test_basic_construction(self):
        cfg = LLMConfig(api_key="sk-x", model="gpt-4")
        assert cfg.api_key == "sk-x"
        assert cfg.model == "gpt-4"
        assert cfg.base_url is None
        assert cfg.max_tokens == 8192
        assert cfg.temperature == 0.2

    def test_empty_api_key_rejected(self):
        with pytest.raises(ValueError, match="api_key"):
            LLMConfig(api_key="", model="gpt-4")

    def test_empty_model_rejected(self):
        with pytest.raises(ValueError, match="model"):
            LLMConfig(api_key="sk", model="")

    def test_zero_max_tokens_rejected(self):
        with pytest.raises(ValueError, match="max_tokens"):
            LLMConfig(api_key="sk", model="gpt-4", max_tokens=0)

    def test_negative_max_tokens_rejected(self):
        with pytest.raises(ValueError, match="max_tokens"):
            LLMConfig(api_key="sk", model="gpt-4", max_tokens=-1)

    def test_temperature_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="temperature"):
            LLMConfig(api_key="sk", model="gpt-4", temperature=2.5)
        with pytest.raises(ValueError, match="temperature"):
            LLMConfig(api_key="sk", model="gpt-4", temperature=-0.1)

    def test_temperature_at_boundaries_accepted(self):
        LLMConfig(api_key="sk", model="gpt-4", temperature=0.0)
        LLMConfig(api_key="sk", model="gpt-4", temperature=2.0)

    def test_zero_timeout_rejected(self):
        with pytest.raises(ValueError, match="timeout"):
            LLMConfig(api_key="sk", model="gpt-4", request_timeout_s=0)

    def test_negative_retries_rejected(self):
        with pytest.raises(ValueError, match="retries"):
            LLMConfig(api_key="sk", model="gpt-4", max_retries=-1)

    def test_zero_retries_accepted(self):
        cfg = LLMConfig(api_key="sk", model="gpt-4", max_retries=0)
        assert cfg.max_retries == 0

    def test_frozen(self):
        cfg = LLMConfig(api_key="sk", model="gpt-4")
        with pytest.raises(Exception):
            cfg.api_key = "changed"  # type: ignore[misc]


class TestFromEnv:
    def test_loads_required_vars(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "sk-test")
        monkeypatch.setenv("BASE_MODEL", "gpt-4")
        monkeypatch.delenv("BASE_URL", raising=False)
        monkeypatch.delenv("AGENT_LLM_MAX_TOKENS", raising=False)
        cfg = LLMConfig.from_env()
        assert cfg.api_key == "sk-test"
        assert cfg.model == "gpt-4"
        assert cfg.base_url is None

    def test_loads_optional_vars(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "sk")
        monkeypatch.setenv("BASE_MODEL", "m")
        monkeypatch.setenv("BASE_URL", "https://example.com")
        monkeypatch.setenv("AGENT_LLM_MAX_TOKENS", "4096")
        monkeypatch.setenv("AGENT_LLM_TEMPERATURE", "0.7")
        cfg = LLMConfig.from_env()
        assert cfg.base_url == "https://example.com"
        assert cfg.max_tokens == 4096
        assert cfg.temperature == 0.7

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.setenv("BASE_MODEL", "m")
        with pytest.raises(OSError, match="API_KEY"):
            LLMConfig.from_env()

    def test_missing_model(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "sk")
        monkeypatch.delenv("BASE_MODEL", raising=False)
        with pytest.raises(OSError, match="BASE_MODEL"):
            LLMConfig.from_env()

    def test_blank_base_url_treated_as_none(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "sk")
        monkeypatch.setenv("BASE_MODEL", "m")
        monkeypatch.setenv("BASE_URL", "   ")
        cfg = LLMConfig.from_env()
        assert cfg.base_url is None
