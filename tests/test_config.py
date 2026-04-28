"""
tests/test_config.py — Unit tests for config.py
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import patch


# ===========================================================================
# LLMConfig
# ===========================================================================

class TestLLMConfig:
    def test_missing_api_key_raises(self):
        from agents.core.config import LLMConfig
        with patch.dict(os.environ, {"API_KEY": "", "BASE_MODEL": "gpt-4o"}, clear=False):
            with pytest.raises(EnvironmentError, match="API_KEY"):
                LLMConfig.from_env()

    def test_missing_model_raises(self):
        from agents.core.config import LLMConfig
        with patch.dict(os.environ, {"API_KEY": "sk-test", "BASE_MODEL": ""}, clear=False):
            with pytest.raises(EnvironmentError, match="BASE_MODEL"):
                LLMConfig.from_env()

    def test_base_url_none_when_blank(self):
        from agents.core.config import LLMConfig
        env = {
            "API_KEY": "sk-test",
            "BASE_MODEL": "gpt-4o",
            "BASE_URL": "",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = LLMConfig.from_env()
        assert cfg.base_url is None

    def test_base_url_set_when_provided(self):
        from agents.core.config import LLMConfig
        env = {
            "API_KEY": "sk-test",
            "BASE_MODEL": "gpt-4o",
            "BASE_URL": "https://api.example.com/v1",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = LLMConfig.from_env()
        assert cfg.base_url == "https://api.example.com/v1"

    def test_defaults_applied(self, monkeypatch):
        from agents.core.config import LLMConfig
        monkeypatch.setenv("API_KEY", "sk-x")
        monkeypatch.setenv("BASE_MODEL", "m")
        for k in ["AGENT_LLM_MAX_TOKENS", "AGENT_LLM_TEMPERATURE",
                   "AGENT_LLM_TIMEOUT_S", "AGENT_LLM_MAX_RETRIES"]:
            monkeypatch.delenv(k, raising=False)
        cfg = LLMConfig.from_env()
        assert cfg.max_tokens == 4096
        assert cfg.temperature == 0.2
        assert cfg.request_timeout_s == 120.0
        assert cfg.max_retries == 3

    def test_int_fields_parsed(self):
        from agents.core.config import LLMConfig
        env = {
            "API_KEY": "sk-x",
            "BASE_MODEL": "m",
            "AGENT_LLM_MAX_TOKENS": "8192",
            "AGENT_LLM_MAX_RETRIES": "5",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = LLMConfig.from_env()
        assert cfg.max_tokens == 8192
        assert cfg.max_retries == 5


# ===========================================================================
# AgentConfig
# ===========================================================================

class TestAgentConfig:
    def test_defaults(self, monkeypatch):
        from agents.core.config import AgentConfig
        monkeypatch.delenv("AGENT_MAX_ITERATIONS", raising=False)
        monkeypatch.delenv("AGENT_KEEP_WORKSPACE", raising=False)
        cfg = AgentConfig.from_env()
        assert cfg.max_iterations == 40
        assert cfg.keep_workspace is False

    def test_keep_workspace_true(self):
        from agents.core.config import AgentConfig
        with patch.dict(os.environ, {"AGENT_KEEP_WORKSPACE": "true"}, clear=False):
            cfg = AgentConfig.from_env()
        assert cfg.keep_workspace is True

    def test_keep_workspace_case_insensitive(self):
        from agents.core.config import AgentConfig
        with patch.dict(os.environ, {"AGENT_KEEP_WORKSPACE": "TRUE"}, clear=False):
            cfg = AgentConfig.from_env()
        assert cfg.keep_workspace is True

    def test_max_iterations_override(self):
        from agents.core.config import AgentConfig
        with patch.dict(os.environ, {"AGENT_MAX_ITERATIONS": "5"}, clear=False):
            cfg = AgentConfig.from_env()
        assert cfg.max_iterations == 5


# ===========================================================================
# ExecutorConfig
# ===========================================================================

class TestExecutorConfig:
    def test_defaults(self, monkeypatch):
        from agents.core.config import ExecutorConfig
        for k in ["AGENT_WORKSPACE_ROOT", "AGENT_NVCC_BIN", "AGENT_NVCC_CCBIN",
                   "AGENT_NVCC_FLAGS", "AGENT_NCU_BIN", "AGENT_NSYS_BIN",
                   "AGENT_PYTHON_BIN", "AGENT_CACHE_ENABLED"]:
            monkeypatch.delenv(k, raising=False)
        cfg = ExecutorConfig.from_env()
        assert cfg.workspace_root == "./workspace"
        assert cfg.nvcc_bin == "nvcc"
        assert cfg.nvcc_ccbin == ""
        assert cfg.nvcc_default_flags == []
        assert cfg.cache_enabled is True

    def test_nvcc_flags_parsed_into_list(self):
        from agents.core.config import ExecutorConfig
        with patch.dict(os.environ, {"AGENT_NVCC_FLAGS": "-O3 -arch=sm_120"}, clear=False):
            cfg = ExecutorConfig.from_env()
        assert cfg.nvcc_default_flags == ["-O3", "-arch=sm_120"]

    def test_nvcc_flags_empty_string(self):
        from agents.core.config import ExecutorConfig
        with patch.dict(os.environ, {"AGENT_NVCC_FLAGS": ""}, clear=False):
            cfg = ExecutorConfig.from_env()
        assert cfg.nvcc_default_flags == []

    def test_cache_disabled(self):
        from agents.core.config import ExecutorConfig
        with patch.dict(os.environ, {"AGENT_CACHE_ENABLED": "false"}, clear=False):
            cfg = ExecutorConfig.from_env()
        assert cfg.cache_enabled is False

    def test_allowed_binaries_default(self):
        from agents.core.config import ExecutorConfig
        cfg = ExecutorConfig()
        assert "nvcc" in cfg.allowed_binaries
        assert "ncu" in cfg.allowed_binaries
        assert "nsys" in cfg.allowed_binaries
        assert "python" in cfg.allowed_binaries

    def test_full_path_nvcc_ccbin_preserved(self):
        from agents.core.config import ExecutorConfig
        ccbin = "C:/Program Files/MSVC/bin/x64"
        with patch.dict(os.environ, {"AGENT_NVCC_CCBIN": ccbin}, clear=False):
            cfg = ExecutorConfig.from_env()
        assert cfg.nvcc_ccbin == ccbin
