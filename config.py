"""
config.py — Three configuration dataclasses loaded from environment variables.
Call load_dotenv() before from_env() in main.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    api_key: str
    model: str
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.2
    request_timeout_s: float = 120.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = os.getenv("OPENAI_API_KEY", "")
        model = os.getenv("AGENT_LLM_MODEL", "")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and fill in your API key."
            )
        if not model:
            raise EnvironmentError(
                "AGENT_LLM_MODEL is not set. "
                "Example: AGENT_LLM_MODEL=gpt-4o-mini"
            )
        raw_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        return cls(
            api_key=api_key,
            model=model,
            base_url=raw_base_url if raw_base_url else None,
            max_tokens=int(os.getenv("AGENT_LLM_MAX_TOKENS") or "4096"),
            temperature=float(os.getenv("AGENT_LLM_TEMPERATURE") or "0.2"),
            request_timeout_s=float(os.getenv("AGENT_LLM_TIMEOUT_S") or "120"),
            max_retries=int(os.getenv("AGENT_LLM_MAX_RETRIES") or "3"),
        )


# ---------------------------------------------------------------------------
# Agent Loop Configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    max_iterations: int = 40
    keep_workspace: bool = False
    circuit_breaker_threshold: int = 3  # open circuit after N consecutive failures

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS") or "40"),
            keep_workspace=(os.getenv("AGENT_KEEP_WORKSPACE") or "false").lower() == "true",
            circuit_breaker_threshold=int(os.getenv("AGENT_CB_THRESHOLD") or "3"),
        )


# ---------------------------------------------------------------------------
# Executor Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExecutorConfig:
    workspace_root: str = "./workspace"
    nvcc_bin: str = "nvcc"
    nvcc_ccbin: str = ""          # host C++ compiler dir/exe for nvcc -ccbin (Windows)
    nvcc_default_flags: list[str] = field(default_factory=list)  # prepended to every compile
    ncu_bin: str = "ncu"
    nsys_bin: str = "nsys"
    python_bin: str = "python"
    default_compile_timeout_s: int = 120
    default_run_timeout_s: int = 60
    default_profile_timeout_s: int = 600
    cache_enabled: bool = True
    allowed_binaries: list[str] = field(
        default_factory=lambda: ["nvcc", "ncu", "nsys", "python"]
    )
    stdout_truncate_bytes: int = 64_000

    @classmethod
    def from_env(cls) -> "ExecutorConfig":
        return cls(
            workspace_root=os.getenv("AGENT_WORKSPACE_ROOT") or "./workspace",
            nvcc_bin=os.getenv("AGENT_NVCC_BIN") or "nvcc",
            nvcc_ccbin=os.getenv("AGENT_NVCC_CCBIN") or "",
            nvcc_default_flags=[
                f for f in (os.getenv("AGENT_NVCC_FLAGS") or "").split() if f
            ],
            ncu_bin=os.getenv("AGENT_NCU_BIN") or "ncu",
            nsys_bin=os.getenv("AGENT_NSYS_BIN") or "nsys",
            python_bin=os.getenv("AGENT_PYTHON_BIN") or "python",
            default_compile_timeout_s=int(
                os.getenv("AGENT_COMPILE_TIMEOUT_S") or "120"
            ),
            default_run_timeout_s=int(os.getenv("AGENT_RUN_TIMEOUT_S") or "60"),
            default_profile_timeout_s=int(
                os.getenv("AGENT_PROFILE_TIMEOUT_S") or "600"
            ),
            cache_enabled=(os.getenv("AGENT_CACHE_ENABLED") or "true").lower() == "true",
            stdout_truncate_bytes=int(
                os.getenv("AGENT_STDOUT_TRUNCATE_BYTES") or "64000"
            ),
        )
