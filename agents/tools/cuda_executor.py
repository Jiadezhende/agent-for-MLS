"""
agents/tools/cuda_executor.py — The central execution layer for CUDA tools.

The Executor is the ONLY code that runs subprocesses, compiles CUDA, or
invokes profiling tools. The LLM never calls nvcc / ncu / nsys directly;
it calls Executor public methods (registered as tools in registry.py).

This file is now the facade/glue layer. Implementation details live in
agents/tools/executor/:
  - workspace.py: workspace and path sandboxing
  - subprocess_runner.py: process tree kill, encoding, probe stdout reduction
  - binaries.py: binary whitelist and resolution
  - nvcc.py: CUDA compilation
  - reducers.py: ncu/nsys output reducers
  - classifiers.py: structured error taxonomy
  - ncu.py: Nsight Compute permission/version helpers
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agents.core.config import ExecutorConfig
from agents.core.exceptions import ExecutorError
from agents.tools.executor.binaries import _check_binary
from agents.tools.executor.classifiers import _classify_subprocess_failure
from agents.tools.executor.ncu import _detect_ncu_version, _precheck_ncu_permission
from agents.tools.executor.nvcc import _compile_cuda, _compile_cuda_for_ncu
from agents.tools.executor.subprocess_runner import (
    SubResult,
    _reduce_probe_output,
    _run_subprocess,
)
from agents.tools.executor.workspace import _safe_join, _Workspace


# ===========================================================================
# 1. Data structures
# ===========================================================================

@dataclass(frozen=True)
class JobSpec:
    backend: str       # "cuda_probe" | "ncu" | "nsys" | "torch"
    name: str          # caller-provided identifier
    payload: dict      # backend-specific parameters (frozen via tuple conversion)

    def cache_key(self, gpu_arch_tag: str = "") -> str:
        """Stable hash for this job spec."""
        canonical = json.dumps(
            {"backend": self.backend, "name": self.name, "payload": self.payload,
             "gpu": gpu_arch_tag},
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class JobResult:
    job_id: str
    backend: str
    name: str
    status: str           # "done" | "error" | "timed_out"
    summary: dict         # LLM-facing reduced view
    artifact_refs: dict   # {logical_name: workspace-relative path string}
    cache_hit: bool
    elapsed_s: float
    started_at: str       # ISO-8601

    def to_tool_result(self) -> dict:
        """Dict the LLM sees as a tool-call response."""
        return {
            "status": self.status,
            "job_id": self.job_id,
            "cache_hit": self.cache_hit,
            "elapsed_s": round(self.elapsed_s, 2),
            **self.summary,
        }

    def to_log_dict(self) -> dict:
        """Compact version stored in ctx.job_history."""
        return {
            "job_id": self.job_id,
            "backend": self.backend,
            "name": self.name,
            "status": self.status,
            "cache_hit": self.cache_hit,
            "elapsed_s": round(self.elapsed_s, 2),
            "started_at": self.started_at,
            "artifact_refs": self.artifact_refs,
        }


# ===========================================================================
# 2. Cache
# ===========================================================================

class _JobCache:
    def __init__(self) -> None:
        self._store: dict[str, JobResult] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> JobResult | None:
        with self._lock:
            return self._store.get(key)

    def put(self, key: str, result: JobResult) -> None:
        with self._lock:
            self._store[key] = result


# ===========================================================================
# 3. Compilation helpers
# ===========================================================================

# ===========================================================================
# 8. Backends
# ===========================================================================

def _execute_cuda_probe(
    spec: JobSpec,
    workspace: _Workspace,
    cfg: ExecutorConfig,
) -> dict:
    """Compile and run a CUDA kernel; return reduced stdout evidence."""
    p = spec.payload
    source: str = p["source"]
    name: str = p["probe_name"].replace(" ", "_")
    flags: list[str] = p.get("compile_flags", [])
    args: list[str] = p.get("args", [])
    timeout_s: int = p.get("timeout_s", cfg.default_run_timeout_s)

    bin_path = _compile_cuda(source, name, flags, workspace, cfg)

    run_cmd = [str(bin_path)] + args
    sub = _run_subprocess(
        run_cmd,
        timeout_s=timeout_s,
        cwd=workspace.root,
        truncate_bytes=None,
    )
    if sub.returncode != 0 or sub.timed_out:
        combined = (sub.stdout + "\n" + sub.stderr).strip()
        classified = _classify_subprocess_failure(
            combined,
            phase="run",
            timed_out=sub.timed_out,
            returncode=sub.returncode,
        )
        evidence = _reduce_probe_output(sub.stdout)
        raise ExecutorError(
            classified["error"],
            error_class=classified["error_class"],
            hint=classified.get("hint"),
            phase=classified["phase"],
            returncode=sub.returncode,
            timed_out=sub.timed_out,
            stderr=sub.stderr[-2000:] if sub.stderr else "",
            **evidence,
        )

    reduced = _reduce_probe_output(sub.stdout)
    reduced.update({
        "binary_path": workspace.rel(bin_path),
        "stderr": sub.stderr[-2000:] if sub.stderr else "",
        "returncode": sub.returncode,
        "timed_out": sub.timed_out,
    })
    return reduced


def _execute_ncu(
    spec: JobSpec,
    workspace: _Workspace,
    cfg: ExecutorConfig,
) -> dict:
    """Run Nsight Compute and return reduced metrics dict plus .ncu-rep report path."""
    p = spec.payload
    ncu = _check_binary(cfg, cfg.ncu_bin)

    binary_path: str = p["binary_path"]
    kernel_name: str = p.get("kernel_name", "")
    metrics: list[str] = p.get("metrics", [])
    args: list[str] = p.get("args", [])
    timeout_s: int = p.get("timeout_s", cfg.default_profile_timeout_s)

    if not metrics:
        raise ExecutorError(
            "invalid_args",
            error_class="user_code",
            phase="profile",
            hint="Provide at least one ncu metric name in the metrics list.",
        )

    permission_hint = _precheck_ncu_permission()
    if permission_hint:
        raise ExecutorError(
            "ncu_permission_denied",
            error_class="infrastructure",
            phase="profile",
            hint=permission_hint,
        )

    bin_path = _safe_join(workspace.root, binary_path)
    if not bin_path.exists():
        raise ExecutorError(
            "binary_not_found",
            error_class="user_code",
            phase="profile",
            path=binary_path,
            hint=(
                "Binary not found in workspace. "
                "For source_type='cuda_source', use profile_with_ncu directly — "
                "it compiles the source automatically."
            ),
        )

    report_base = workspace.allocate("ncu", "")  # ncu appends .ncu-rep automatically
    cmd = [
        ncu,
        "--replay-mode", "kernel",
        "--target-processes", "all",
        "-o", str(report_base),
    ]
    if kernel_name:
        cmd += ["--kernel-name", kernel_name]
    cmd += ["--metrics", ",".join(metrics)]
    cmd += [str(bin_path)] + args

    sub = _run_subprocess(
        cmd,
        timeout_s=timeout_s,
        cwd=workspace.root,
        truncate_bytes=None,
        encoding="utf-8",
    )

    combined = (sub.stdout + "\n" + sub.stderr).strip()

    # Check for ERR_NVGPUCTRPERM before the returncode check: some ncu versions
    # exit 0 even when GPU counter access is denied.
    if "ERR_NVGPUCTRPERM" in combined or "erf_no_privileged_mode" in combined.lower():
        raise ExecutorError(
            "ncu_permission_denied",
            error_class="infrastructure",
            phase="profile",
            hint=(
                "ERR_NVGPUCTRPERM: no permission for GPU hardware counters. "
                "Do NOT retry ncu — switch to run_cuda_probe self-timed kernels instead."
            ),
            returncode=sub.returncode,
            stderr=sub.stderr[-2000:] if sub.stderr else "",
        )

    if sub.returncode != 0 or sub.timed_out:
        if not combined:
            raise ExecutorError(
                "ncu_failed",
                error_class="infrastructure",
                phase="profile",
                hint=(
                    "ncu exited non-zero with no output. Common causes: missing admin "
                    "privileges, unsupported GPU, or broken toolkit. "
                    "Do NOT retry — switch to run_cuda_probe instead."
                ),
                returncode=sub.returncode,
                stderr=sub.stderr[-2000:] if sub.stderr else "",
            )
        classified = _classify_subprocess_failure(
            combined,
            phase="profile",
            timed_out=sub.timed_out,
            returncode=sub.returncode,
        )
        raise ExecutorError(
            classified["error"],
            error_class=classified["error_class"],
            phase=classified["phase"],
            hint=classified.get("hint"),
            returncode=sub.returncode,
            stderr=sub.stderr[-2000:] if sub.stderr else "",
            stdout_tail=sub.stdout[-2000:] if sub.stdout else "",
        )

    if "Metric Value" not in sub.stdout:
        raise ExecutorError(
            "ncu_no_kernel_found",
            error_class="data_quality",
            phase="profile",
            hint="Check kernel_name spelling or whether the binary launches the kernel.",
            kernel_name=kernel_name,
            stdout_tail=sub.stdout[-2000:] if sub.stdout else "",
            stderr=sub.stderr[-2000:] if sub.stderr else "",
            returncode=sub.returncode,
        )

    rep_file = report_base.with_suffix(".ncu-rep")
    result: dict = {
        "output": sub.stdout,
        "returncode": sub.returncode,
        "report_path": workspace.rel(rep_file) if rep_file.exists() else None,
    }
    if sub.stderr:
        result["stderr"] = sub.stderr[-2000:]
    # ncu_version injected by profile_with_ncu from cached value
    return result


def _execute_nsys(
    spec: JobSpec,
    workspace: _Workspace,
    cfg: ExecutorConfig,
) -> dict:
    """Run Nsight Systems: profile → save .nsys-rep → extract kernel stats via nsys stats."""
    p = spec.payload
    nsys = _check_binary(cfg, cfg.nsys_bin)

    source_type: str = p["source_type"]
    source_or_path: str = p["source_or_path"]
    flags: list[str] = p.get("compile_flags", [])
    args: list[str] = p.get("args", [])
    duration_s: int = p.get("duration_s", 0)   # 0 = capture full app lifetime
    timeout_s: int = p.get("timeout_s", cfg.default_profile_timeout_s)
    probe_name: str = p.get("probe_name", spec.name).replace(" ", "_")

    if source_type == "cuda_source":
        bin_path = _compile_cuda(source_or_path, probe_name, flags, workspace, cfg)
        target_cmd = [str(bin_path)] + args
    elif source_type == "python_script":
        script_path = workspace.write(f"src/{probe_name}.py", source_or_path)
        python = _check_binary(cfg, cfg.python_bin)
        target_cmd = [python, str(script_path)] + args
    else:  # binary
        bin_path = _safe_join(workspace.root, source_or_path)
        target_cmd = [str(bin_path)] + args

    report_base = workspace.allocate("nsys", "")

    # Step 1: profile and save .nsys-rep
    cmd = [
        nsys, "profile",
        "-o", str(report_base),
        "--force-overwrite", "true",
        "--trace", "cuda,nvtx",
    ]
    if duration_s > 0:
        cmd += ["-d", str(duration_s)]
    cmd += target_cmd

    sub = _run_subprocess(
        cmd,
        timeout_s=timeout_s,
        cwd=workspace.root,
        truncate_bytes=cfg.stdout_truncate_bytes,
    )

    if sub.returncode != 0 or sub.timed_out:
        combined = (sub.stdout + "\n" + sub.stderr).strip()
        classified = _classify_subprocess_failure(
            combined,
            phase="profile",
            timed_out=sub.timed_out,
            returncode=sub.returncode,
        )
        raise ExecutorError(
            classified["error"],
            error_class=classified["error_class"],
            phase=classified["phase"],
            hint=classified.get("hint"),
            returncode=sub.returncode,
            stderr=sub.stderr[-2000:] if sub.stderr else "",
        )

    # Step 2: extract kernel summary table from the saved report
    rep_file = report_base.with_suffix(".nsys-rep")
    stats_output = ""
    stats_error = ""
    if rep_file.exists():
        stats_sub = _run_subprocess(
            [nsys, "stats", str(rep_file),
             "--format", "table",
             "--report", "cuda_gpu_kern_sum"],
            timeout_s=30,
            encoding="utf-8",
            truncate_bytes=None,
        )
        if stats_sub.returncode == 0 and stats_sub.stdout.strip():
            stats_output = stats_sub.stdout
        elif stats_sub.stderr:
            stats_error = stats_sub.stderr[-2000:]

    result: dict = {
        "output": stats_output,
        "returncode": sub.returncode,
        "timed_out": sub.timed_out,
        "report_path": workspace.rel(rep_file) if rep_file.exists() else None,
    }
    if stats_error:
        result["stats_error"] = stats_error
    return result


def _execute_torch(
    spec: JobSpec,
    workspace: _Workspace,
    cfg: ExecutorConfig,
) -> dict:
    """Run a Python script (typically using torch) and return its stdout."""
    p = spec.payload
    python_code: str = p["python_code"]
    op_name: str = p.get("op_name", spec.name).replace(" ", "_")
    timeout_s: int = p.get("timeout_s", cfg.default_run_timeout_s)

    script_path = workspace.write(f"src/{op_name}.py", python_code)
    python = _check_binary(cfg, cfg.python_bin)

    sub = _run_subprocess(
        [python, str(script_path)],
        timeout_s=timeout_s,
        cwd=workspace.root,
        truncate_bytes=None,
    )

    if sub.returncode != 0 or sub.timed_out:
        if sub.timed_out:
            raise ExecutorError(
                "timeout",
                error_class="timeout",
                phase="run",
                hint="Script timed out. Reduce iterations or number of shapes tested.",
                returncode=sub.returncode,
                timed_out=True,
                stderr=sub.stderr[-2000:] if sub.stderr else "",
                stdout_tail=sub.stdout[-2000:] if sub.stdout else "",
            )
        error_class = "infrastructure" if sub.returncode == 127 else "user_code"
        hint = (
            "Python binary not found. Set AGENT_PYTHON_BIN env var."
            if error_class == "infrastructure"
            else "Script raised an exception. Check stderr for the traceback and fix the script."
        )
        raise ExecutorError(
            "torch_script_failed",
            error_class=error_class,
            phase="run",
            hint=hint,
            returncode=sub.returncode,
            timed_out=False,
            stderr=sub.stderr[-2000:] if sub.stderr else "",
            stdout_tail=sub.stdout[-2000:] if sub.stdout else "",
        )

    reduced = _reduce_probe_output(sub.stdout)
    reduced.update({
        "stderr": sub.stderr[-500:] if sub.stderr else "",
        "returncode": sub.returncode,
        "timed_out": sub.timed_out,
    })
    return reduced


# ===========================================================================
# 9. Environment auto-detection
# ===========================================================================

_NCU_SEARCH_PATHS_WIN: list[str] = [
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1\ncu.exe",
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3\ncu.exe",
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1\ncu.exe",
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.3\ncu.exe",
]

_NSYS_SEARCH_PATHS_WIN: list[str] = [
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.1\target-windows-x64\nsys.exe",
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.1\target-windows-x64\nsys.exe",
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.3.1\target-windows-x64\nsys.exe",
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2023.4.1\target-windows-x64\nsys.exe",
]

_NVCC_SEARCH_GLOBS_LIN: list[str] = [
    "/usr/local/cuda/bin/nvcc",
    "/usr/local/cuda-*/bin/nvcc",
]
_NCU_SEARCH_GLOBS_LIN: list[str] = [
    "/usr/local/cuda/bin/ncu",
    "/usr/local/cuda-*/bin/ncu",
    "/opt/nvidia/nsight-compute-*/ncu",
]
_NSYS_SEARCH_GLOBS_LIN: list[str] = [
    "/usr/local/cuda/bin/nsys",
    "/usr/local/cuda-*/bin/nsys",
    "/opt/nvidia/nsight-systems-*/target-linux-x64/nsys",
    "/opt/nvidia/nsight-systems-*/bin/nsys",
]


def _detect_arch_flags() -> str | None:
    """Query nvidia-smi for GPU compute capability and return '-arch=sm_NNN'."""
    r = _run_subprocess(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        timeout_s=10,
    )
    if r.returncode == 0:
        cc = r.stdout.strip().replace(".", "")   # "12.0" → "120"
        if cc.isdigit():
            return f"-arch=sm_{cc}"
    # Fallback for older nvidia-smi that lacks --query-gpu=compute_cap
    r2 = _run_subprocess(["nvidia-smi", "-q"], timeout_s=10)
    if r2.returncode == 0:
        m = re.search(
            r"CUDA Capability Major/Minor Version Number\s*:\s*(\d+)\.(\d+)",
            r2.stdout,
        )
        if m:
            return f"-arch=sm_{m.group(1)}{m.group(2)}"
    return None


def _detect_msvc_ccbin() -> str | None:
    """Find MSVC host compiler directory via vswhere (Windows only)."""
    if sys.platform != "win32":
        return None
    vswhere = shutil.which("vswhere") or \
        r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if not Path(vswhere).exists():
        return None
    r = _run_subprocess(
        [vswhere, "-latest", "-products", "*",
         "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        timeout_s=15,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    vs_path = Path(r.stdout.strip())
    vc_tools = vs_path / "VC" / "Tools" / "MSVC"
    if not vc_tools.exists():
        return None
    versions = sorted(vc_tools.iterdir(), reverse=True)
    for v in versions:
        cl_dir = v / "bin" / "Hostx64" / "x64"
        if (cl_dir / "cl.exe").exists():
            return str(cl_dir)
    return None


def _detect_tool_path(on_path_name: str, search_list: list[str]) -> str | None:
    """Return the first existing path in search_list if the tool is not on PATH."""
    if shutil.which(on_path_name) is not None:
        return None
    for candidate in search_list:
        if Path(candidate).exists():
            return candidate
    return None


def _detect_tool_path_linux(on_path_name: str, glob_patterns: list[str]) -> str | None:
    """Search Linux CUDA install paths via glob; pick newest version by lexicographic sort."""
    import glob as _glob
    if shutil.which(on_path_name) is not None:
        return None
    candidates: list[str] = []
    for pattern in glob_patterns:
        candidates.extend(_glob.glob(pattern))
    for candidate in sorted(candidates, reverse=True):
        if Path(candidate).is_file():
            return candidate
    return None


def _autodetect_env(cfg: "ExecutorConfig") -> tuple["ExecutorConfig", list[str]]:
    """Fill in missing ExecutorConfig values through best-effort auto-detection."""
    import dataclasses

    changes: dict[str, Any] = {}
    notes: list[str] = []

    if not any(f.startswith("-arch") for f in cfg.nvcc_default_flags):
        arch = _detect_arch_flags()
        if arch:
            changes["nvcc_default_flags"] = list(cfg.nvcc_default_flags) + [arch]
            notes.append(f"[auto-detect] GPU arch: added {arch} to nvcc flags")
        else:
            notes.append(
                "[auto-detect] GPU arch: nvidia-smi unavailable; "
                "set AGENT_NVCC_FLAGS=-arch=sm_NNN if compilation fails"
            )

    if not cfg.nvcc_ccbin:
        ccbin = _detect_msvc_ccbin()
        if ccbin:
            changes["nvcc_ccbin"] = ccbin
            notes.append(f"[auto-detect] MSVC ccbin: {ccbin}")
        elif sys.platform == "win32":
            notes.append(
                "[auto-detect] MSVC ccbin: not found via vswhere; "
                "set AGENT_NVCC_CCBIN if compilation fails on Windows"
            )

    if cfg.nvcc_bin == "nvcc" and sys.platform != "win32":
        detected = _detect_tool_path_linux("nvcc", _NVCC_SEARCH_GLOBS_LIN)
        if detected:
            changes["nvcc_bin"] = detected
            notes.append(f"[auto-detect] nvcc: {detected}")
        elif shutil.which("nvcc") is None:
            notes.append(
                "[auto-detect] nvcc: not found on PATH or known Linux paths. "
                "Set AGENT_NVCC_BIN or call probe_environment tool at runtime."
            )

    if cfg.ncu_bin == "ncu":
        detected = (
            _detect_tool_path("ncu", _NCU_SEARCH_PATHS_WIN) if sys.platform == "win32"
            else _detect_tool_path_linux("ncu", _NCU_SEARCH_GLOBS_LIN)
        )
        if detected:
            changes["ncu_bin"] = detected
            notes.append(f"[auto-detect] ncu: {detected}")

    if cfg.nsys_bin == "nsys":
        detected = (
            _detect_tool_path("nsys", _NSYS_SEARCH_PATHS_WIN) if sys.platform == "win32"
            else _detect_tool_path_linux("nsys", _NSYS_SEARCH_GLOBS_LIN)
        )
        if detected:
            changes["nsys_bin"] = detected
            notes.append(f"[auto-detect] nsys: {detected}")

    if changes:
        cfg = dataclasses.replace(cfg, **changes)

    return cfg, notes


# ===========================================================================
# 10. Executor (public API)
# ===========================================================================

class Executor:
    """Central execution layer.

    All LLM tool calls that involve running code are routed through this
    class. It owns: workspace, sandbox, cache, subprocess, compilation,
    and output reduction. The LLM never imports subprocess directly.
    """

    def __init__(
        self,
        cfg: ExecutorConfig,
        on_job_complete: Callable[[JobResult], None] | None = None,
    ) -> None:
        cfg, detect_notes = _autodetect_env(cfg)
        self._cfg = cfg
        self._cfg_lock = threading.Lock()
        self.detect_notes: list[str] = detect_notes
        self._job_listeners: list[Callable[[JobResult], None]] = []
        self._listeners_lock = threading.Lock()
        self._gpu_lock = threading.Lock()   # serializes GPU execution across all workers
        self._log_lock = threading.Lock()   # protects jobs.jsonl append on Windows
        self._ncu_version: str | None = None  # lazily populated, then cached
        if on_job_complete is not None:
            self._job_listeners.append(on_job_complete)
        self.workspace = _Workspace(cfg.workspace_root)
        self._cache = _JobCache()
        self._gpu_arch_tag = self._detect_gpu_arch()

    def _get_ncu_version(self) -> str | None:
        """Return ncu version string, cached after first successful detection."""
        if self._ncu_version is None:
            with self._cfg_lock:
                cfg = self._cfg
            self._ncu_version = _detect_ncu_version(cfg)
        return self._ncu_version

    def add_job_listener(self, fn: Callable[[JobResult], None]) -> None:
        with self._listeners_lock:
            self._job_listeners.append(fn)

    def remove_job_listener(self, fn: Callable[[JobResult], None]) -> None:
        with self._listeners_lock:
            self._job_listeners.remove(fn)

    def probe_environment(self, force_rescan: bool = False) -> dict:
        """Scan filesystem for nvcc/ncu/nsys and reconfigure Executor if found.

        Does NOT execute any binary — only Path.is_file() checks.
        Safe to call at any point during agent execution.
        """
        import dataclasses
        import glob as _glob

        with self._cfg_lock:
            cfg = self._cfg

        def _first_glob(patterns: list[str]) -> str | None:
            candidates: list[str] = []
            for p in patterns:
                candidates.extend(_glob.glob(p))
            for c in sorted(candidates, reverse=True):
                if Path(c).is_file():
                    return c
            return None

        scan = [
            ("nvcc", cfg.nvcc_bin, _NVCC_SEARCH_GLOBS_LIN if sys.platform != "win32" else []),
            ("ncu",  cfg.ncu_bin,  _NCU_SEARCH_GLOBS_LIN  if sys.platform != "win32" else _NCU_SEARCH_PATHS_WIN),
            ("nsys", cfg.nsys_bin, _NSYS_SEARCH_GLOBS_LIN if sys.platform != "win32" else _NSYS_SEARCH_PATHS_WIN),
        ]

        found: dict[str, str] = {}
        already: dict[str, str] = {}
        not_found: list[str] = []
        changes: dict[str, str] = {}

        for name, current, patterns in scan:
            if current != name and Path(current).is_file():
                already[name] = current
                continue
            hit = shutil.which(name)
            if hit and not force_rescan:
                already[name] = hit
                continue
            if sys.platform == "win32":
                hit = next((p for p in patterns if Path(p).is_file()), None)
            else:
                hit = _first_glob(patterns)
            if hit:
                found[name] = hit
                changes[f"{name}_bin"] = hit
            else:
                not_found.append(name)

        if changes:
            with self._cfg_lock:
                self._cfg = dataclasses.replace(self._cfg, **changes)

        return {
            "ok": True,
            "already_configured": already,
            "newly_found": found,
            "not_found": not_found,
            "reconfigured": bool(changes),
            "hint": (
                f"Not found: {not_found}. "
                "Set AGENT_NVCC_BIN / AGENT_NCU_BIN / AGENT_NSYS_BIN env vars."
                if not_found else "All CUDA tools resolved."
            ),
        }

    def _detect_gpu_arch(self) -> str:
        try:
            result = _run_subprocess(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                timeout_s=10,
            )
            if result.returncode == 0:
                return result.stdout.strip().replace(" ", "_")[:40]
        except Exception:
            pass
        return "unknown_gpu"

    def _run_job(
        self,
        spec: JobSpec,
        execute_fn: Callable,
    ) -> dict:
        """Common pattern for all four backends."""
        cache_key = spec.cache_key(self._gpu_arch_tag)

        if self._cfg.cache_enabled:
            cached = self._cache.get(cache_key)
            if cached is not None:
                cached_result = JobResult(
                    job_id=cached.job_id,
                    backend=cached.backend,
                    name=cached.name,
                    status=cached.status,
                    summary=cached.summary,
                    artifact_refs=cached.artifact_refs,
                    cache_hit=True,
                    elapsed_s=cached.elapsed_s,
                    started_at=cached.started_at,
                )
                with self._listeners_lock:
                    listeners = list(self._job_listeners)
                for fn in listeners:
                    fn(cached_result)
                return cached_result.to_tool_result()

        job_id = uuid.uuid4().hex[:12]
        started_at = datetime.now(timezone.utc).isoformat()
        t0 = time.monotonic()

        try:
            with self._gpu_lock:
                raw = execute_fn(spec, self.workspace, self._cfg)
            status = "done"
            if isinstance(raw, SubResult):
                summary: dict = {
                    "stdout": raw.stdout,
                    "stderr": raw.stderr,
                    "returncode": raw.returncode,
                    "timed_out": raw.timed_out,
                }
                status = "timed_out" if raw.timed_out else "done"
            else:
                summary = raw
        except ExecutorError as exc:
            status = "error"
            summary = {"error": exc.kind, "error_class": exc.error_class, **exc.details}
            if exc.hint:
                summary["hint"] = exc.hint
        except Exception as exc:
            status = "error"
            summary = {"error": exc.__class__.__name__, "error_class": "infrastructure",
                       "detail": str(exc)}

        elapsed_s = time.monotonic() - t0

        job_result = JobResult(
            job_id=job_id,
            backend=spec.backend,
            name=spec.name,
            status=status,
            summary=summary,
            artifact_refs={},
            cache_hit=False,
            elapsed_s=elapsed_s,
            started_at=started_at,
        )

        if self._cfg.cache_enabled and status == "done":
            self._cache.put(cache_key, job_result)

        try:
            log_path = self.workspace.root / "logs" / "jobs.jsonl"
            with self._log_lock:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(job_result.to_log_dict(), default=str) + "\n")
        except Exception:
            pass

        with self._listeners_lock:
            listeners = list(self._job_listeners)
        for fn in listeners:
            fn(job_result)

        return job_result.to_tool_result()

    # ------------------------------------------------------------------
    # Public API (registered as LLM tools)
    # ------------------------------------------------------------------

    def run_cuda_probe(
        self,
        source: str,
        probe_name: str,
        compile_flags: list[str] | None = None,
        args: list[str] | None = None,
        timeout_s: int = 60,
    ) -> dict:
        """Compile and run a CUDA kernel. Primary tool for hardware probing."""
        spec = JobSpec(
            backend="cuda_probe",
            name=probe_name,
            payload={
                "source": source,
                "probe_name": probe_name,
                "compile_flags": compile_flags or [],
                "args": args or [],
                "timeout_s": timeout_s,
            },
        )
        return self._run_job(spec, _execute_cuda_probe)

    def profile_with_ncu(
        self,
        source_type: str,
        source_or_path: str,
        kernel_name: str,
        metrics: list[str] | None = None,
        compile_flags: list[str] | None = None,
        args: list[str] | None = None,
        timeout_s: int = 600,
    ) -> dict:
        """Run Nsight Compute on a kernel to collect hardware counters via --metrics.

        source_type='cuda_source': compiles with -lineinfo then profiles (no pre-run).
        source_type='binary': profiles an existing workspace binary directly.
        """
        import time as _time
        import uuid as _uuid

        if source_type == "cuda_source":
            # Compile only (with -lineinfo); do NOT run the kernel before profiling.
            t0 = _time.monotonic()
            try:
                with self._cfg_lock:
                    cfg = self._cfg
                probe_name = f"ncu_target_{kernel_name}".replace(" ", "_")
                binary_path_obj = _compile_cuda_for_ncu(
                    source_or_path, probe_name, compile_flags or [], self.workspace, cfg,
                )
                binary_path = self.workspace.rel(binary_path_obj)
            except ExecutorError as exc:
                summary: dict = {"error": exc.kind, "error_class": exc.error_class, **exc.details}
                if exc.hint:
                    summary["hint"] = exc.hint
                return {
                    "status": "error",
                    "job_id": "ncu_compile_" + _uuid.uuid4().hex[:8],
                    "cache_hit": False,
                    "elapsed_s": round(_time.monotonic() - t0, 2),
                    **summary,
                }
        else:
            binary_path = source_or_path

        spec = JobSpec(
            backend="ncu",
            name=f"ncu_{kernel_name}",
            payload={
                "binary_path": binary_path,
                "kernel_name": kernel_name,
                "metrics": metrics or [],
                "args": args or [],
                "timeout_s": timeout_s,
            },
        )
        result = self._run_job(spec, _execute_ncu)
        if result.get("status") != "error":
            result["ncu_version"] = self._get_ncu_version()
        return result

    def profile_with_nsys(
        self,
        source_type: str,
        source_or_path: str,
        compile_flags: list[str] | None = None,
        args: list[str] | None = None,
        duration_s: int = 10,
        timeout_s: int = 300,
    ) -> dict:
        """Run Nsight Systems to capture CPU-GPU timeline."""
        spec = JobSpec(
            backend="nsys",
            name="nsys_profile",
            payload={
                "source_type": source_type,
                "source_or_path": source_or_path,
                "compile_flags": compile_flags or [],
                "args": args or [],
                "duration_s": duration_s,
                "timeout_s": timeout_s,
                "probe_name": "nsys_target",
            },
        )
        return self._run_job(spec, _execute_nsys)

    def profile_with_torch(
        self,
        python_code: str,
        op_name: str,
        timeout_s: int = 120,
    ) -> dict:
        """Run a Python script (with torch) and capture stdout for baseline measurement.

        The script runs with cwd=workspace root, so files written to relative paths
        (e.g. 'data/ref.npy') are placed inside the workspace sandbox.
        """
        spec = JobSpec(
            backend="torch",
            name=op_name,
            payload={
                "python_code": python_code,
                "op_name": op_name,
                "timeout_s": timeout_s,
            },
        )
        return self._run_job(spec, _execute_torch)
