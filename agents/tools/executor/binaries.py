from __future__ import annotations

import shutil
from pathlib import Path

from agents.core.config import ExecutorConfig
from agents.core.exceptions import ExecutorError


_BINARY_HINTS: dict[str, str] = {
    "ncu": "Set AGENT_NCU_BIN env var, or install Nsight Compute.",
    "nsys": "Set AGENT_NSYS_BIN env var, or install Nsight Systems.",
    "nvcc": "Set AGENT_NVCC_BIN env var, or install the CUDA Toolkit.",
}


def _check_binary(cfg: ExecutorConfig, name: str) -> str:
    """Return the resolved path for a binary name if it's on the whitelist."""
    basename = Path(name).stem
    if basename not in cfg.allowed_binaries:
        raise ExecutorError(
            "binary_not_whitelisted",
            error_class="infrastructure",
            name=name,
            allowed=cfg.allowed_binaries,
        )
    # Accept absolute paths that exist directly (e.g. from auto-detect or find_binary)
    if Path(name).is_absolute() and Path(name).exists():
        return name
    resolved = shutil.which(name)
    if resolved is None:
        raise ExecutorError(
            "binary_not_found",
            error_class="infrastructure",
            hint=_BINARY_HINTS.get(
                basename,
                f"Install {basename} or set the AGENT_{basename.upper()}_BIN env var.",
            ),
            name=name,
        )
    return resolved

