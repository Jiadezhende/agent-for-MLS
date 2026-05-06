"""Executor configuration loaded from environment variables.

Kept separate from ``mls_agent.llm.config`` so projects that don't need CUDA
tooling never import this module.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


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
