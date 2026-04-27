from __future__ import annotations

import sys
from pathlib import Path

from agents.core.config import ExecutorConfig
from agents.core.exceptions import ExecutorError
from agents.tools.executor.binaries import _check_binary
from agents.tools.executor.classifiers import (
    _classify_subprocess_failure,
    _extract_nvcc_errors,
)
from agents.tools.executor.subprocess_runner import _run_subprocess
from agents.tools.executor.workspace import _Workspace


def _compile_cuda(
    source: str,
    name: str,
    flags: list[str],
    workspace: _Workspace,
    cfg: ExecutorConfig,
) -> Path:
    """Write CUDA source to workspace/src and compile with nvcc."""
    nvcc = _check_binary(cfg, cfg.nvcc_bin)

    src_path = workspace.write(f"src/{name}.cu", source)
    suffix = ".exe" if sys.platform == "win32" else ""
    out_path = workspace.root / "bin" / f"{name}{suffix}"

    ccbin_flags = ["-ccbin", cfg.nvcc_ccbin] if cfg.nvcc_ccbin else []
    cmd = [
        nvcc,
        *ccbin_flags,
        *cfg.nvcc_default_flags,
        *flags,
        "-o",
        str(out_path),
        str(src_path),
    ]
    result = _run_subprocess(
        cmd,
        timeout_s=cfg.default_compile_timeout_s,
        truncate_bytes=cfg.stdout_truncate_bytes,
    )

    if result.returncode != 0 or result.timed_out:
        combined = (result.stdout + "\n" + result.stderr).strip()
        classified = _classify_subprocess_failure(
            combined,
            phase="compile",
            timed_out=result.timed_out,
            returncode=result.returncode,
        )
        raise ExecutorError(
            classified["error"],
            error_class=classified["error_class"],
            hint=classified.get("hint"),
            returncode=result.returncode,
            phase=classified["phase"],
            stderr=_extract_nvcc_errors(combined),
            arch_flags=[
                flag
                for flag in cmd
                if flag.startswith("-arch") or flag.startswith("--generate-code")
            ],
        )

    return out_path
