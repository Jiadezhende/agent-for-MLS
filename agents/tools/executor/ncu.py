from __future__ import annotations

import sys
from pathlib import Path

from agents.core.config import ExecutorConfig
from agents.tools.executor.binaries import _check_binary
from agents.tools.executor.subprocess_runner import _run_subprocess


def _precheck_ncu_permission() -> str | None:
    """Best-effort fast fail for common Nsight Compute counter permission issues."""
    if sys.platform == "win32":
        try:
            import ctypes
            if not ctypes.windll.shell32.IsUserAnAdmin():
                return (
                    "ncu requires Administrator privileges on Windows for many "
                    "GPU performance counters. Run as Administrator or use "
                    "run_cuda_probe self-timed kernels."
                )
        except Exception:
            return None
    else:
        try:
            paranoid = int(Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip())
            if paranoid > 2:
                return f"perf_event_paranoid={paranoid}; set it to <=2 for ncu counters."
        except Exception:
            return None
    return None


def _detect_ncu_version(cfg: ExecutorConfig) -> str | None:
    try:
        ncu = _check_binary(cfg, cfg.ncu_bin)
        result = _run_subprocess([ncu, "--version"], timeout_s=10, encoding="utf-8")
        if result.returncode == 0:
            for line in (result.stdout + "\n" + result.stderr).splitlines():
                stripped = line.strip()
                if stripped:
                    return stripped
    except Exception:
        return None
    return None
