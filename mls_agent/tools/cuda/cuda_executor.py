"""mls_agent/tools/cuda/cuda_executor.py — Central CUDA execution layer.

The Executor is the ONLY code that runs subprocesses, compiles CUDA, or
invokes profiling tools. The LLM never calls nvcc / ncu / nsys directly;
it calls Executor public methods (wrapped as Tool instances in
``mls_agent.tools.cuda.profile_tools``).

Implementation details live in ``mls_agent.tools.cuda.executor``:
  - workspace.py: workspace and path sandboxing
  - subprocess_runner.py: process tree kill, encoding, probe stdout reduction
  - binaries.py: binary whitelist and resolution
  - nvcc.py: CUDA compilation
  - reducers.py: ncu/nsys output reducers (currently empty)
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

from mls_agent.tools.cuda.config import ExecutorConfig
from mls_agent.tools.cuda.exceptions import ExecutorError
from mls_agent.tools.cuda.executor.binaries import _check_binary
from mls_agent.tools.cuda.executor.classifiers import _classify_subprocess_failure
from mls_agent.tools.cuda.executor.ncu import _detect_ncu_version, _precheck_ncu_permission
from mls_agent.tools.cuda.executor.nvcc import _compile_cuda, _compile_cuda_for_ncu
from mls_agent.tools.cuda.executor.subprocess_runner import (
    SubResult,
    _reduce_probe_output,
    _run_subprocess,
)
from mls_agent.tools.cuda.executor.workspace import _safe_join, _Workspace


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
# 3. Backends
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

    if "ERR_NVGPUCTRPERM" in combined or "erf_no_privileged_mode" in combined.lower():
        raise ExecutorError(
            "ncu_permission_denied",
            error_class="infrastructure",
            phase="profile",
            hint=(
                "ERR_NVGPUCTRPERM: no permission for GPU hardware counters. "
                "Do NOT retry ncu. Alternatives: "
                "(1) run_cuda_probe with self-timed kernels (clock64 / cudaEvent); "
                "(2) profile_with_torch to run a Python/torch timing script."
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

    rep_file = report_base.with_suffix(".ncu-rep")
    if not rep_file.exists():
        raise ExecutorError(
            "ncu_no_report",
            error_class="infrastructure",
            phase="profile",
            hint=(
                "ncu returned 0 but no .ncu-rep file was written. "
                "Possible workspace permission issue or ncu silent failure."
            ),
            returncode=sub.returncode,
            stdout_tail=sub.stdout[-2000:] if sub.stdout else "",
            stderr=sub.stderr[-2000:] if sub.stderr else "",
        )

    rep_path_rel = workspace.rel(rep_file)

    import_sub = _run_subprocess(
        [
            ncu,
            "--import", str(rep_file),
            "--csv",
            "--print-summary", "per-kernel",
        ],
        timeout_s=60,
        cwd=workspace.root,
        truncate_bytes=None,
        encoding="utf-8",
    )

    if import_sub.returncode != 0:
        raise ExecutorError(
            "ncu_import_failed",
            error_class="infrastructure",
            phase="profile",
            hint=(
                "ncu --import could not decode the .ncu-rep file. "
                "Report file exists but metric extraction failed; "
                "check ncu version compatibility."
            ),
            returncode=import_sub.returncode,
            report_path=rep_path_rel,
            stdout_tail=import_sub.stdout[-2000:] if import_sub.stdout else "",
            stderr=import_sub.stderr[-2000:] if import_sub.stderr else "",
        )

    non_empty_lines = [ln for ln in import_sub.stdout.splitlines() if ln.strip()]
    looks_like_csv = bool(non_empty_lines) and "," in non_empty_lines[0]
    has_data_row = len(non_empty_lines) >= 2

    if not (looks_like_csv and has_data_row):
        raise ExecutorError(
            "ncu_no_kernel_found",
            error_class="data_quality",
            phase="profile",
            hint=(
                "ncu profiled successfully but the report contains no kernel data. "
                "Check kernel_name spelling, that the binary actually launches the kernel, "
                "or that the requested metrics exist on this GPU architecture."
            ),
            kernel_name=kernel_name,
            metrics=metrics,
            report_path=rep_path_rel,
            stdout_tail=import_sub.stdout[-2000:] if import_sub.stdout else "",
            stderr=import_sub.stderr[-2000:] if import_sub.stderr else "",
            returncode=import_sub.returncode,
        )

    result: dict = {
        "output": import_sub.stdout,
        "returncode": sub.returncode,
        "report_path": rep_path_rel,
    }
    profile_stderr = sub.stderr[-2000:] if sub.stderr else ""
    if profile_stderr:
        result["stderr"] = profile_stderr
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
    duration_s: int = p.get("duration_s", 0)
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
# 4. Environment auto-detection
# ===========================================================================

_NCU_SEARCH_GLOBS_WIN: list[str] = [
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute *\ncu.exe",
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute *\ncu.bat",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\ncu.exe",
]
_NSYS_SEARCH_GLOBS_WIN: list[str] = [
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems *\target-windows-x64\nsys.exe",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nsys.exe",
]
_NVCC_SEARCH_GLOBS_WIN: list[str] = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe",
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


def _platform_globs(name: str) -> list[str]:
    """Return the built-in glob patterns for `name` on the current platform."""
    if sys.platform == "win32":
        return {
            "ncu": _NCU_SEARCH_GLOBS_WIN,
            "nsys": _NSYS_SEARCH_GLOBS_WIN,
            "nvcc": _NVCC_SEARCH_GLOBS_WIN,
        }.get(name, [])
    return {
        "ncu": _NCU_SEARCH_GLOBS_LIN,
        "nsys": _NSYS_SEARCH_GLOBS_LIN,
        "nvcc": _NVCC_SEARCH_GLOBS_LIN,
    }.get(name, [])


def _detect_arch_flags() -> str | None:
    """Query nvidia-smi for GPU compute capability and return '-arch=sm_NNN'."""
    r = _run_subprocess(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        timeout_s=10,
    )
    if r.returncode == 0:
        cc = r.stdout.strip().replace(".", "")
        if cc.isdigit():
            return f"-arch=sm_{cc}"
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


def _glob_candidates(glob_patterns: list[str]) -> list[str]:
    """Expand glob patterns and return existing files, newest first by lex sort."""
    import glob as _glob
    seen: set[str] = set()
    ordered: list[str] = []
    for pattern in glob_patterns:
        for hit in _glob.glob(pattern):
            if hit not in seen and Path(hit).is_file():
                seen.add(hit)
                ordered.append(hit)
    ordered.sort(reverse=True)
    return ordered


def _detect_tool_path_glob(on_path_name: str, glob_patterns: list[str]) -> str | None:
    """Search install paths via glob; pick newest version by lex sort."""
    if shutil.which(on_path_name) is not None:
        return None
    candidates = _glob_candidates(glob_patterns)
    return candidates[0] if candidates else None


def _autodetect_env(
    cfg: "ExecutorConfig",
    *,
    verify_compile: bool = False,
) -> tuple["ExecutorConfig", list[str]]:
    """Fill in missing ExecutorConfig values through best-effort auto-detection.

    When ``verify_compile=True`` (used by the pipeline preflight), also checks
    for g++ and runs a trivial load_inline smoke test to validate the full
    CUDA extension build chain. The default ``False`` keeps existing behavior
    so unit tests and normal Executor construction stay fast.
    """
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

    if cfg.nvcc_bin == "nvcc":
        detected = _detect_tool_path_glob("nvcc", _platform_globs("nvcc"))
        if detected:
            changes["nvcc_bin"] = detected
            notes.append(f"[auto-detect] nvcc: {detected}")
        elif shutil.which("nvcc") is None:
            notes.append(
                "[auto-detect] nvcc: not found on PATH or known install paths. "
                "Set AGENT_NVCC_BIN or call probe_environment tool at runtime."
            )

    if cfg.ncu_bin == "ncu":
        detected = _detect_tool_path_glob("ncu", _platform_globs("ncu"))
        if detected:
            changes["ncu_bin"] = detected
            notes.append(f"[auto-detect] ncu: {detected}")

    if cfg.nsys_bin == "nsys":
        detected = _detect_tool_path_glob("nsys", _platform_globs("nsys"))
        if detected:
            changes["nsys_bin"] = detected
            notes.append(f"[auto-detect] nsys: {detected}")

    # g++ detection — always included so callers can read the note
    gxx = shutil.which("g++")
    if gxx:
        notes.append(f"[auto-detect] g++: {gxx}")
    else:
        notes.append("[auto-detect] g++: not found on PATH")

    # Full compile chain smoke test — only when verify_compile=True
    if verify_compile:
        try:
            import tempfile
            import torch
            from torch.utils.cpp_extension import load_inline
            with tempfile.TemporaryDirectory() as d:
                load_inline(
                    name="autodetect_nop",
                    cpp_sources="",
                    cuda_sources=["__global__ void _nop_() {}"],
                    build_directory=d,
                    verbose=False,
                    extra_cuda_cflags=["-O0"],
                    is_python_module=False,
                )
            notes.append("[auto-detect] load_inline: ok")
        except Exception as exc:
            notes.append(f"[auto-detect] load_inline: FAILED — {exc}")

    if changes:
        cfg = dataclasses.replace(cfg, **changes)

    return cfg, notes


# ===========================================================================
# 5. Executor (public API)
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
        self._gpu_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._ncu_version: str | None = None
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
        """Scan filesystem for nvcc/ncu/nsys and reconfigure Executor if found."""
        import dataclasses

        with self._cfg_lock:
            cfg = self._cfg

        scan = [
            ("nvcc", cfg.nvcc_bin, _platform_globs("nvcc")),
            ("ncu",  cfg.ncu_bin,  _platform_globs("ncu")),
            ("nsys", cfg.nsys_bin, _platform_globs("nsys")),
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
            candidates = _glob_candidates(patterns)
            hit = candidates[0] if candidates else None
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

    def find_binary(
        self,
        name: str,
        extra_globs: list[str] | None = None,
        reconfigure: bool = True,
    ) -> dict:
        """Search for one binary using built-in glob patterns plus agent-supplied ones."""
        import dataclasses

        if name not in {"ncu", "nsys", "nvcc"}:
            return {
                "ok": False,
                "error": "invalid_name",
                "error_class": "user_code",
                "message": f"name must be one of 'ncu', 'nsys', 'nvcc'; got {name!r}",
            }

        patterns = list(_platform_globs(name))
        if extra_globs:
            patterns.extend(extra_globs)

        candidates = _glob_candidates(patterns)
        on_path = shutil.which(name)
        if on_path and on_path not in candidates:
            candidates.insert(0, on_path)

        chosen = candidates[0] if candidates else None
        reconfigured = False
        if chosen and reconfigure:
            with self._cfg_lock:
                self._cfg = dataclasses.replace(self._cfg, **{f"{name}_bin": chosen})
            reconfigured = True

        return {
            "ok": True,
            "name": name,
            "candidates": candidates,
            "chosen": chosen,
            "reconfigured": reconfigured,
            "patterns_searched": patterns,
            "hint": (
                f"No '{name}' binary matched. Pass extra_globs with the install path."
                if not candidates else None
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

    def write_workspace_file(self, content: str, filename: str) -> dict:
        """Write content to a workspace-relative path; return the path and byte size."""
        path = self.workspace.write(filename, content)
        return {
            "ok": True,
            "path": self.workspace.rel(path),
            "size_bytes": len(content.encode()),
        }

    def run_cuda_probe(
        self,
        source: str | None = None,
        probe_name: str = "",
        compile_flags: list[str] | None = None,
        args: list[str] | None = None,
        timeout_s: int = 60,
        source_path: str | None = None,
    ) -> dict:
        """Compile and run a CUDA kernel. Primary tool for hardware probing."""
        if source_path:
            file_path = _safe_join(self.workspace.root, source_path)
            source = file_path.read_text(encoding="utf-8")
        if not source:
            return {
                "status": "error",
                "error": "missing_source",
                "error_class": "user_code",
                "hint": "Provide either 'source' (inline CUDA C++) or 'source_path' (workspace path from write_workspace_file).",
            }
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
        """Run Nsight Compute on a kernel to collect hardware counters via --metrics."""
        import time as _time
        import uuid as _uuid

        if source_type == "cuda_source":
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
        """Run a Python script (with torch) and capture stdout."""
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
