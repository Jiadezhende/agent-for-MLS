"""Fail-fast build environment check for the pipeline.

Calls _autodetect_env with verify_compile=True to run the full detection
(nvcc, g++, ncu, nsys, load_inline smoke test) in one pass. Raises
RuntimeError with a clear diagnostic on the first missing critical tool.

Returns (env_info dict, updated ExecutorConfig). Pass the updated config to
Executor() so _autodetect_env skips re-detection of already-filled fields.
"""
from __future__ import annotations

import torch

from mls_agent.tools.cuda.config import ExecutorConfig
from mls_agent.tools.cuda.cuda_executor import _autodetect_env


def check_build_env(cfg: ExecutorConfig) -> tuple[dict, ExecutorConfig]:
    """Detect and validate the full CUDA build chain. Fail fast on errors.

    Args:
        cfg: ExecutorConfig (typically from ExecutorConfig.from_env() with
             workspace_root already set).

    Returns:
        (env_info, updated_cfg) where env_info is suitable for
        blackboard["environment"] and updated_cfg has tool paths pre-filled.

    Raises:
        RuntimeError: if nvcc, g++, or the load_inline smoke test fails.
    """
    updated_cfg, notes = _autodetect_env(cfg, verify_compile=True)

    for note in notes:
        if "nvcc" in note and "not found" in note:
            raise RuntimeError(
                f"{note}\n"
                "Fix: install CUDA toolkit or set AGENT_NVCC_BIN env var."
            )
        if "g++" in note and "not found" in note:
            raise RuntimeError(
                f"{note}\n"
                "Fix: apt-get install build-essential"
            )
        if "load_inline: FAILED" in note:
            raise RuntimeError(
                f"{note}\n"
                "Fix: verify nvcc and g++ versions are compatible with "
                "the installed CUDA runtime."
            )

    env_info = {
        "detect_notes": notes,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    return env_info, updated_cfg
