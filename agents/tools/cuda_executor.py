"""
agents/tools/cuda_executor.py — The central execution layer for CUDA tools.

The Executor is the ONLY code that runs subprocesses, compiles CUDA, or
invokes profiling tools. The LLM never calls nvcc / ncu / nsys directly;
it calls Executor public methods (registered as tools in registry.py).

Sections:
  1. Data structures (JobSpec, JobResult, SubResult)
  2. Workspace (per-run temp directory)
  3. Cache (in-memory, keyed by payload hash)
  4. Sandbox (binary whitelist, path traversal guard)
  5. Subprocess helpers
  6. Compilation
  7. Post-processing (output reducers for ncu / nsys / torch)
  8. Backends (_execute_cuda_probe, _execute_ncu, _execute_nsys, _execute_torch)
  9. Environment auto-detection
 10. Executor (public API)
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agents.core.config import ExecutorConfig
from agents.core.exceptions import ExecutorError


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
class SubResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


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
# 2. Workspace
# ===========================================================================

class _Workspace:
    """Per-run temporary directory with deterministic sub-paths."""

    SUBDIRS = ("src", "bin", "ncu", "nsys", "torch", "logs")

    def __init__(self, root: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        uid = uuid.uuid4().hex[:8]
        self.root = Path(root) / f"run_{ts}_{uid}"
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in self.SUBDIRS:
            (self.root / sub).mkdir(exist_ok=True)

    def allocate(self, kind: str, suffix: str) -> Path:
        """Return a unique path inside the workspace (file not yet created)."""
        uid = uuid.uuid4().hex[:6]
        return self.root / kind / f"{kind}_{uid}{suffix}"

    def write(self, rel: str, content: str | bytes) -> Path:
        """Write content to a workspace-relative path (creates parent dirs)."""
        target = _safe_join(self.root, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            target.write_text(content, encoding="utf-8")
        else:
            target.write_bytes(content)
        return target

    def cleanup(self, keep: bool = False) -> None:
        if not keep and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def rel(self, path: Path) -> str:
        """Return POSIX-style path relative to workspace root."""
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()


# ===========================================================================
# 3. Cache
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
# 4. Sandbox helpers
# ===========================================================================

def _safe_join(root: Path, rel: str) -> Path:
    """Resolve rel relative to root and reject path traversal / absolute paths."""
    if Path(rel).is_absolute():
        raise ExecutorError("path_escape", error_class="user_code",
                            rel=rel, reason="absolute path not allowed")
    resolved = (root / rel).resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ExecutorError("path_escape", error_class="user_code",
                            rel=rel, reason="path escapes workspace")
    return resolved


def _check_binary(cfg: ExecutorConfig, name: str) -> str:
    """Return the resolved path for a binary name if it's on the whitelist."""
    basename = Path(name).stem if Path(name).suffix else name
    if basename not in cfg.allowed_binaries:
        raise ExecutorError(
            "binary_not_whitelisted",
            error_class="infrastructure",
            name=name,
            allowed=cfg.allowed_binaries,
        )
    resolved = shutil.which(name)
    if resolved is None:
        raise ExecutorError(
            "binary_not_found",
            error_class="infrastructure",
            hint=_BINARY_HINTS.get(basename, f"Install {basename} or set the AGENT_{basename.upper()}_BIN env var."),
            name=name,
        )
    return resolved


# ===========================================================================
# 5. Subprocess helpers
# ===========================================================================

def _run_subprocess(
    cmd: list[str],
    timeout_s: int,
    cwd: Path | None = None,
    env: dict | None = None,
    truncate_bytes: int = 64_000,
) -> SubResult:
    """Run cmd and return a SubResult.  Never raises; timeouts are captured."""
    TRUNC_MARKER = b"\n[... output truncated ...]\n"
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_s,
            cwd=cwd,
            env=env,
        )
        stdout_b = proc.stdout
        stderr_b = proc.stderr
        timed_out = False
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout_b = exc.stdout or b""
        stderr_b = exc.stderr or b""
        timed_out = True
        returncode = -1

    # Truncate
    if len(stdout_b) > truncate_bytes:
        stdout_b = stdout_b[:truncate_bytes] + TRUNC_MARKER
    if len(stderr_b) > truncate_bytes:
        stderr_b = stderr_b[:truncate_bytes] + TRUNC_MARKER

    # Use system locale encoding so MSVC/GBK output on Chinese Windows is readable
    _enc = sys.stdout.encoding or "utf-8"
    return SubResult(
        stdout=stdout_b.decode(_enc, errors="replace"),
        stderr=stderr_b.decode(_enc, errors="replace"),
        returncode=returncode,
        timed_out=timed_out,
    )


# ===========================================================================
# 6. Compilation helpers
# ===========================================================================

_BINARY_HINTS: dict[str, str] = {
    "ncu":  "Set AGENT_NCU_BIN env var, or install Nsight Compute.",
    "nsys": "Set AGENT_NSYS_BIN env var, or install Nsight Systems.",
    "nvcc": "Set AGENT_NVCC_BIN env var, or install the CUDA Toolkit.",
}

_DIAG_RE = re.compile(
    r"(error:|warning:|note:|undefined reference|undefined symbol|"
    r"\d+ error(s)? detected|cannot open source file|fatal error)",
    re.IGNORECASE,
)


def _extract_nvcc_errors(combined: str, max_chars: int = 3000) -> str:
    """Extract only diagnostic lines from nvcc stderr/stdout mix.

    nvcc output looks like:
        nvcc.EXE -ccbin … (invocation — skip)
        /path/file.cu(42): error: 'clockRate' is not a member of …
        1 error detected in compilation of …

    We keep lines that contain diagnostic keywords and strip the workspace
    path prefix so line references are stable across runs.
    Fallback: if nothing matches, return the last 2000 chars of combined.
    """
    lines = combined.splitlines()
    kept = [l.strip() for l in lines if l.strip() and _DIAG_RE.search(l)]
    result = "\n".join(kept) if kept else combined[-2000:]
    return result[:max_chars]


def _classify_compile_error(combined: str) -> str:
    """Return 'user_code' or 'infrastructure' based on nvcc error content."""
    low = combined.lower()
    if "command not found" in low:
        return "infrastructure"
    if "nvcc fatal" in low and "no input files" not in low:
        return "infrastructure"
    # ccbin-related: the host compiler path is wrong (env misconfiguration)
    if "-ccbin" in combined and ("cannot find" in low or "no such file" in low):
        return "infrastructure"
    return "user_code"


def _compile_cuda(
    source: str,
    name: str,
    flags: list[str],
    workspace: _Workspace,
    cfg: ExecutorConfig,
) -> Path:
    """Write CUDA source to workspace/src and compile with nvcc.

    Returns path to the compiled binary.
    Raises ExecutorError("compile_failed") on non-zero nvcc exit.
    """
    nvcc = _check_binary(cfg, cfg.nvcc_bin)

    src_path = workspace.write(f"src/{name}.cu", source)
    # On Windows nvcc produces .exe
    suffix = ".exe" if sys.platform == "win32" else ""
    out_path = workspace.root / "bin" / f"{name}{suffix}"

    ccbin_flags = ["-ccbin", cfg.nvcc_ccbin] if cfg.nvcc_ccbin else []
    cmd = [nvcc, *ccbin_flags, *cfg.nvcc_default_flags, *flags, "-o", str(out_path), str(src_path)]
    result = _run_subprocess(
        cmd,
        timeout_s=cfg.default_compile_timeout_s,
        truncate_bytes=cfg.stdout_truncate_bytes,
    )

    if result.returncode != 0 and not result.timed_out:
        # nvcc prints errors to stdout on some platforms; capture both
        combined = (result.stdout + "\n" + result.stderr).strip()
        ec = _classify_compile_error(combined)
        raise ExecutorError(
            "compile_failed",
            error_class=ec,
            returncode=result.returncode,
            # Clean stderr: only diagnostic lines, not the nvcc invocation.
            # Omitting cmd prevents LLM from misreading flags as the error cause.
            stderr=_extract_nvcc_errors(combined),
            arch_flags=[f for f in cmd if f.startswith("-arch") or f.startswith("--generate-code")],
        )
    if result.timed_out:
        raise ExecutorError("compile_timeout", error_class="timeout",
                            source_name=name)

    return out_path


# ===========================================================================
# 7. Post-processing (output reducers)
# ===========================================================================

def _reduce_ncu(raw_text: str, metrics_requested: list[str]) -> dict:
    """Parse ncu --csv output and extract requested metrics."""
    result: dict[str, Any] = {}
    notes: list[str] = []

    lines = raw_text.strip().splitlines()
    if not lines:
        return {"metrics": {}, "notes": ["empty ncu output"]}

    try:
        reader = csv.DictReader(io.StringIO(raw_text))
        rows = list(reader)
        for row in rows:
            for m in metrics_requested:
                for col in row:
                    if m in col or col in m:
                        val_str = row[col].strip().strip('"')
                        try:
                            result[m] = float(val_str.replace(",", ""))
                        except ValueError:
                            result[m] = val_str
    except Exception as exc:
        notes.append(f"CSV parse error: {exc}")

    for m in metrics_requested:
        if m not in result:
            pattern = re.compile(
                r"(?i)" + re.escape(m) + r"[^\d\-]*([0-9]+(?:\.[0-9]+)?)"
            )
            match = pattern.search(raw_text)
            if match:
                result[m] = float(match.group(1))

    missing = [m for m in metrics_requested if m not in result]
    if missing:
        notes.append(f"Could not parse metrics: {missing}")

    return {"metrics": result, "notes": notes}


def _reduce_nsys(raw_text: str) -> dict:
    """Extract top GPU kernels from nsys stats text output."""
    lines = raw_text.strip().splitlines()
    result: dict[str, Any] = {"timeline_summary": [], "notes": []}

    in_table = False
    header: list[str] = []
    rows: list[dict] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_table:
                in_table = False
            continue
        if "Time (%)" in stripped and not in_table:
            header = [h.strip() for h in stripped.split(",")]
            in_table = True
            continue
        if in_table:
            parts = stripped.split(",")
            if len(parts) == len(header):
                rows.append(dict(zip(header, [p.strip() for p in parts])))

    if rows:
        result["timeline_summary"] = rows[:10]
    else:
        result["timeline_summary"] = lines[:50]
        result["notes"].append("Could not parse nsys stats table; raw excerpt returned")

    return result


def _reduce_torch(raw_text: str) -> dict:
    """Parse torch.profiler text output and return top operators by self CPU time."""
    lines = raw_text.strip().splitlines()
    result: dict[str, Any] = {"op_stats": [], "notes": []}

    header_idx = -1
    for i, line in enumerate(lines):
        if "Self CPU" in line or "CPU total" in line:
            header_idx = i
            break

    if header_idx >= 0:
        data_lines = lines[header_idx + 1:]
        ops = []
        for line in data_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("-"):
                continue
            ops.append(stripped)
            if len(ops) >= 20:
                break
        result["op_stats"] = ops
    else:
        result["op_stats"] = lines[:30]
        result["notes"].append("Could not parse torch profiler table; raw excerpt returned")

    return result


# ===========================================================================
# 8. Backends
# ===========================================================================

def _execute_cuda_probe(
    spec: JobSpec,
    workspace: _Workspace,
    cfg: ExecutorConfig,
) -> SubResult:
    """Compile and run a CUDA kernel; return raw stdout/stderr."""
    p = spec.payload
    source: str = p["source"]
    name: str = p["probe_name"].replace(" ", "_")
    flags: list[str] = p.get("compile_flags", [])
    args: list[str] = p.get("args", [])
    timeout_s: int = p.get("timeout_s", cfg.default_run_timeout_s)

    bin_path = _compile_cuda(source, name, flags, workspace, cfg)

    run_cmd = [str(bin_path)] + args
    return _run_subprocess(
        run_cmd,
        timeout_s=timeout_s,
        cwd=workspace.root,
        truncate_bytes=cfg.stdout_truncate_bytes,
    )


def _execute_ncu(
    spec: JobSpec,
    workspace: _Workspace,
    cfg: ExecutorConfig,
) -> dict:
    """Run Nsight Compute and return reduced metrics dict."""
    p = spec.payload
    ncu = _check_binary(cfg, cfg.ncu_bin)

    source_type: str = p["source_type"]
    source_or_path: str = p["source_or_path"]
    kernel_name: str = p.get("kernel_name", "")
    metrics: list[str] = p.get("metrics", [])
    flags: list[str] = p.get("compile_flags", [])
    args: list[str] = p.get("args", [])
    timeout_s: int = p.get("timeout_s", cfg.default_profile_timeout_s)
    probe_name: str = p.get("probe_name", spec.name)

    if source_type == "cuda_source":
        bin_path = _compile_cuda(
            source_or_path, probe_name.replace(" ", "_"), flags, workspace, cfg
        )
    else:
        bin_path = _safe_join(workspace.root, source_or_path)
        if not bin_path.exists():
            raise ExecutorError("binary_not_found", path=source_or_path)

    ncu_output = workspace.allocate("ncu", ".csv")
    cmd = [
        ncu,
        "--csv",
        "--log-file", str(ncu_output),
    ]
    if kernel_name:
        cmd += ["--kernel-name", kernel_name]
    if metrics:
        cmd += ["--metrics", ",".join(metrics)]
    cmd += [str(bin_path)] + args

    sub = _run_subprocess(
        cmd,
        timeout_s=timeout_s,
        cwd=workspace.root,
        truncate_bytes=cfg.stdout_truncate_bytes,
    )

    raw_csv = ""
    if ncu_output.exists():
        raw_csv = ncu_output.read_text(encoding="utf-8", errors="replace")
    if not raw_csv:
        raw_csv = sub.stdout

    if not raw_csv and sub.returncode != 0:
        return {
            "error": "ncu_failed",
            "error_class": "infrastructure",
            "stderr": sub.stderr[-2000:] if sub.stderr else "",
            "hint": (
                "NCU command failed. Common causes: (1) requires admin/root privileges "
                "(run with elevated permissions), (2) metric names unsupported on this GPU arch, "
                "(3) --log-file path issue on Windows. Do NOT retry with different metric names — "
                "switch to run_cuda_probe self-instrumentation instead."
            ),
            "metrics": {},
            "notes": ["NCU command exited with non-zero returncode; no output produced"],
            "returncode": sub.returncode,
        }

    reduced = _reduce_ncu(raw_csv, metrics)
    reduced["stdout"] = sub.stdout
    reduced["stderr"] = sub.stderr[-2000:] if sub.stderr else ""
    reduced["returncode"] = sub.returncode
    reduced["raw_path"] = workspace.rel(ncu_output) if ncu_output.exists() else None
    return reduced


def _execute_nsys(
    spec: JobSpec,
    workspace: _Workspace,
    cfg: ExecutorConfig,
) -> dict:
    """Run Nsight Systems and return a minimal timeline summary."""
    p = spec.payload
    nsys = _check_binary(cfg, cfg.nsys_bin)

    source_type: str = p["source_type"]
    source_or_path: str = p["source_or_path"]
    flags: list[str] = p.get("compile_flags", [])
    args: list[str] = p.get("args", [])
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
    report_path = str(report_base)

    cmd = [
        nsys, "profile",
        "--output", report_path,
        "--force-overwrite", "true",
        "--stats", "true",
        "--export", "sqlite",
    ] + target_cmd

    sub = _run_subprocess(
        cmd,
        timeout_s=timeout_s,
        cwd=workspace.root,
        truncate_bytes=cfg.stdout_truncate_bytes,
    )

    reduced = _reduce_nsys(sub.stdout + sub.stderr)
    reduced["returncode"] = sub.returncode
    reduced["timed_out"] = sub.timed_out
    reduced["raw_path"] = workspace.rel(report_base.with_suffix(".nsys-rep")) \
        if (report_base.with_suffix(".nsys-rep")).exists() else None
    return reduced


def _execute_torch(
    spec: JobSpec,
    workspace: _Workspace,
    cfg: ExecutorConfig,
) -> dict:
    """Run user Python code under torch.profiler and return op stats."""
    p = spec.payload
    python_code: str = p["python_code"]
    op_name: str = p.get("op_name", spec.name)
    num_iters: int = p.get("num_iters", 100)
    timeout_s: int = p.get("timeout_s", cfg.default_profile_timeout_s)

    python = _check_binary(cfg, cfg.python_bin)

    wrapper = textwrap.dedent(f"""\
        import torch
        from torch.profiler import profile, ProfilerActivity, record_function

        # ---- user code start ----
        {textwrap.indent(python_code, '        ')}
        # ---- user code end ----

        NUM_ITERS = {num_iters}
        OP_NAME = {repr(op_name)}

        # Warmup
        for _ in range(max(1, NUM_ITERS // 10)):
            pass

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            with record_function(OP_NAME):
                for _ in range(NUM_ITERS):
                    pass

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    """)

    script_path = workspace.write(
        f"torch/{op_name.replace(' ', '_')}_prof.py", wrapper
    )

    sub = _run_subprocess(
        [python, str(script_path)],
        timeout_s=timeout_s,
        cwd=workspace.root,
        truncate_bytes=cfg.stdout_truncate_bytes,
    )

    reduced = _reduce_torch(sub.stdout)
    reduced["returncode"] = sub.returncode
    reduced["stderr"] = sub.stderr[-1000:] if sub.stderr else ""
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
    return None


def _validate_arch_flag(arch: str, nvcc_bin: str) -> bool:
    """Return True if nvcc accepts the given -arch flag (test compile a no-op kernel)."""
    import tempfile
    minimal_src = "__global__ void _k(){} int main(){return 0;}\n"
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "arch_test.cu"
        out = Path(d) / ("arch_test.exe" if sys.platform == "win32" else "arch_test")
        src.write_text(minimal_src)
        r = _run_subprocess([nvcc_bin, arch, "-o", str(out), str(src)], timeout_s=20)
        return r.returncode == 0


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


def _autodetect_env(cfg: "ExecutorConfig") -> tuple["ExecutorConfig", list[str]]:
    """Fill in missing ExecutorConfig values through best-effort auto-detection."""
    import dataclasses

    changes: dict[str, Any] = {}
    notes: list[str] = []

    if not any(f.startswith("-arch") for f in cfg.nvcc_default_flags):
        arch = _detect_arch_flags()
        if arch:
            if _validate_arch_flag(arch, cfg.nvcc_bin):
                changes["nvcc_default_flags"] = list(cfg.nvcc_default_flags) + [arch]
                notes.append(f"[auto-detect] GPU arch: added {arch} to nvcc flags")
            else:
                notes.append(
                    f"[auto-detect] GPU arch: {arch} detected but nvcc rejects it "
                    f"(toolkit too old for this GPU); compiling without -arch flag. "
                    f"Set AGENT_NVCC_FLAGS=-arch=sm_NNN to override."
                )
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

    if cfg.ncu_bin == "ncu":
        detected = _detect_tool_path("ncu", _NCU_SEARCH_PATHS_WIN if sys.platform == "win32" else [])
        if detected:
            changes["ncu_bin"] = detected
            notes.append(f"[auto-detect] ncu: {detected}")

    if cfg.nsys_bin == "nsys":
        detected = _detect_tool_path("nsys", _NSYS_SEARCH_PATHS_WIN if sys.platform == "win32" else [])
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
        self.detect_notes: list[str] = detect_notes
        self._job_listeners: list[Callable[[JobResult], None]] = []
        self._listeners_lock = threading.Lock()
        if on_job_complete is not None:
            self._job_listeners.append(on_job_complete)
        self.workspace = _Workspace(cfg.workspace_root)
        self._cache = _JobCache()
        self._gpu_arch_tag = self._detect_gpu_arch()

    def add_job_listener(self, fn: Callable[[JobResult], None]) -> None:
        with self._listeners_lock:
            self._job_listeners.append(fn)

    def remove_job_listener(self, fn: Callable[[JobResult], None]) -> None:
        with self._listeners_lock:
            self._job_listeners.remove(fn)

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
        metrics: list[str],
        compile_flags: list[str] | None = None,
        args: list[str] | None = None,
        timeout_s: int = 600,
    ) -> dict:
        """Run Nsight Compute on a kernel to collect hardware counters."""
        spec = JobSpec(
            backend="ncu",
            name=f"ncu_{kernel_name}",
            payload={
                "source_type": source_type,
                "source_or_path": source_or_path,
                "kernel_name": kernel_name,
                "metrics": metrics,
                "compile_flags": compile_flags or [],
                "args": args or [],
                "timeout_s": timeout_s,
                "probe_name": kernel_name,
            },
        )
        return self._run_job(spec, _execute_ncu)

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
        num_iters: int = 100,
        timeout_s: int = 300,
    ) -> dict:
        """Run PyTorch Profiler on user-provided Python code."""
        spec = JobSpec(
            backend="torch",
            name=op_name,
            payload={
                "python_code": python_code,
                "op_name": op_name,
                "num_iters": num_iters,
                "timeout_s": timeout_s,
            },
        )
        return self._run_job(spec, _execute_torch)
