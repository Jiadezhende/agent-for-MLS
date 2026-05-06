"""LLM-facing wrappers around ``Executor`` methods.

The Executor is the only code that runs subprocesses, compiles CUDA, or
invokes profiling binaries. These tool classes are thin schema + result
adapters — they don't add behaviour beyond what the Executor already does.
"""
from __future__ import annotations

from typing import Any

from mls_agent.tools.base import Tool
from mls_agent.tools.response import ToolErrorCode, ToolResponse


# ---------------------------------------------------------------------------
# Helper: convert Executor result dict → ToolResponse
# ---------------------------------------------------------------------------


def _exec_result_to_response(result: dict, label: str) -> ToolResponse:
    """Map an Executor JobResult-style dict to a normalized ``ToolResponse``."""
    status = result.get("status", "error")

    if status == "done":
        elapsed = result.get("elapsed_s", "?")
        cache_tag = " [cache_hit]" if result.get("cache_hit") else ""
        stdout = result.get("stdout", "")
        text = f"{label} done in {elapsed}s{cache_tag}."
        if stdout:
            text += f"\n---stdout---\n{stdout}"
        return ToolResponse.success(
            text=text,
            data=result,
            stats={"elapsed_s": elapsed, "cache_hit": bool(result.get("cache_hit"))},
        )

    if status == "timed_out":
        elapsed = result.get("elapsed_s", "?")
        return ToolResponse.partial(
            text=f"{label} timed out after {elapsed}s.",
            data=result,
            stats={"elapsed_s": elapsed},
        )

    # status == "error" (or anything unexpected)
    code = result.get("error_class") or "execution_error"
    return ToolResponse.error(
        code=code,
        message=str(result.get("error") or result.get("stderr") or f"{label} failed"),
        data=result,
    )


# ---------------------------------------------------------------------------
# write_workspace_file
# ---------------------------------------------------------------------------


class WriteWorkspaceFileTool(Tool):
    NAME = "write_workspace_file"
    DESCRIPTION = (
        "Write content to a workspace file and return its path. Stage large "
        "CUDA sources here before calling run_cuda_probe / profile_with_ncu so "
        "subsequent calls can pass a short ``source_path`` instead of the full "
        "kernel string. Overwrites any existing file at the same path."
    )

    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Full file content."},
                "filename": {
                    "type": "string",
                    "description": (
                        "Workspace-relative path. Use 'src/' prefix for source files. "
                        "Directories are created automatically."
                    ),
                },
            },
            "required": ["content", "filename"],
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        result = self._executor.write_workspace_file(**parameters)
        path = result.get("path", "")
        size = result.get("size_bytes", 0)
        return ToolResponse.success(
            text=f"wrote {size} bytes to {path!r}",
            data=result,
        )


# ---------------------------------------------------------------------------
# run_cuda_probe
# ---------------------------------------------------------------------------


class RunCudaProbeTool(Tool):
    NAME = "run_cuda_probe"
    DESCRIPTION = (
        "Compile and run a CUDA kernel source file, capturing its stdout. Use "
        "for self-timing kernels (clock64, cudaEvent) where stdout is the "
        "primary measurement. NOT for hardware-counter profiling — use "
        "profile_with_ncu instead. For large kernels (>3 KB), first call "
        "write_workspace_file and pass the returned path as source_path."
    )

    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": (
                        "Complete CUDA C++ source. Use only for short snippets. "
                        "Mutually exclusive with source_path."
                    ),
                },
                "source_path": {
                    "type": "string",
                    "description": (
                        "Workspace-relative path to a .cu file (preferred for large "
                        "kernels). Mutually exclusive with source."
                    ),
                },
                "probe_name": {
                    "type": "string",
                    "description": "Short identifier used as cache key and log label.",
                },
                "compile_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Extra nvcc flags (e.g. ['-O3', '-arch=sm_86']).",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Command-line arguments for the compiled binary.",
                },
                "timeout_s": {
                    "type": "integer",
                    "default": 60,
                    "description": "Execution timeout in seconds.",
                },
            },
            "required": ["probe_name"],
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        result = self._executor.run_cuda_probe(**parameters)
        return _exec_result_to_response(result, label=f"probe {parameters.get('probe_name', '')!r}")


# ---------------------------------------------------------------------------
# profile_with_ncu
# ---------------------------------------------------------------------------


class ProfileWithNcuTool(Tool):
    NAME = "profile_with_ncu"
    DESCRIPTION = (
        "Run a kernel under NVIDIA Nsight Compute and collect the metrics "
        "specified by ``metrics``. Auto-injects ``-lineinfo`` for source-line "
        "correlation and saves a .ncu-rep report. Use ``source_type='cuda_source'`` "
        "to compile-and-profile, or 'binary' to profile an existing workspace binary."
    )

    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source_type": {
                    "type": "string",
                    "enum": ["cuda_source", "binary"],
                },
                "source_or_path": {
                    "type": "string",
                    "description": (
                        "Full CUDA source string when source_type='cuda_source'; "
                        "workspace-relative binary path when source_type='binary'."
                    ),
                },
                "kernel_name": {
                    "type": "string",
                    "description": "Kernel function name or regex passed to ncu --kernel-name.",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Specific ncu metric names; empty defaults to a small built-in set.",
                },
                "compile_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
                "timeout_s": {
                    "type": "integer",
                    "default": 600,
                },
            },
            "required": ["source_type", "source_or_path", "kernel_name"],
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        result = self._executor.profile_with_ncu(**parameters)
        return _exec_result_to_response(result, label="ncu profile")


# ---------------------------------------------------------------------------
# profile_with_nsys
# ---------------------------------------------------------------------------


class ProfileWithNsysTool(Tool):
    NAME = "profile_with_nsys"
    DESCRIPTION = (
        "Run a program under NVIDIA Nsight Systems to capture the CPU-GPU "
        "timeline. Useful for kernel launch overhead, stream ordering, and "
        "host-device synchronization patterns."
    )

    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source_type": {
                    "type": "string",
                    "enum": ["cuda_source", "python_script", "binary"],
                },
                "source_or_path": {"type": "string"},
                "compile_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
                "duration_s": {"type": "integer", "default": 10},
                "timeout_s": {"type": "integer", "default": 300},
            },
            "required": ["source_type", "source_or_path"],
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        result = self._executor.profile_with_nsys(**parameters)
        return _exec_result_to_response(result, label="nsys profile")


# ---------------------------------------------------------------------------
# profile_with_torch
# ---------------------------------------------------------------------------


class ProfileWithTorchTool(Tool):
    NAME = "profile_with_torch"
    DESCRIPTION = (
        "Run a Python script (typically importing torch) and capture its stdout. "
        "Use to measure PyTorch baselines, generate reference tensors for "
        "correctness validation, or run any Python-based GPU workload. The "
        "script's CWD is the workspace root, so relative writes stay sandboxed."
    )

    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "python_code": {
                    "type": "string",
                    "description": "Standalone Python source. Print measurements in a parseable format.",
                },
                "op_name": {
                    "type": "string",
                    "description": "Short identifier used as filename and cache key.",
                },
                "timeout_s": {"type": "integer", "default": 120},
            },
            "required": ["python_code", "op_name"],
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        result = self._executor.profile_with_torch(**parameters)
        return _exec_result_to_response(result, label=f"torch {parameters.get('op_name', '')!r}")


# ---------------------------------------------------------------------------
# probe_environment
# ---------------------------------------------------------------------------


class ProbeEnvironmentTool(Tool):
    NAME = "probe_environment"
    DESCRIPTION = (
        "Scan the filesystem for nvcc / ncu / nsys binaries without executing "
        "them. Use when an earlier tool reported error_class='infrastructure' "
        "and error='binary_not_found'. If binaries are found, the Executor is "
        "reconfigured automatically. Pure stat checks — no subprocesses."
    )

    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "force_rescan": {
                    "type": "boolean",
                    "default": False,
                    "description": "Re-scan even if a binary appears to be on PATH.",
                }
            },
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        result = self._executor.probe_environment(**parameters)
        if "error" in result:
            return ToolResponse.error(
                code=str(result.get("error_class") or ToolErrorCode.EXECUTION_ERROR),
                message=str(result.get("error", "probe_environment failed")),
                data=result,
            )
        already = result.get("already_configured", {})
        newly = result.get("newly_found", {})
        not_found = result.get("not_found", [])
        parts: list[str] = []
        if already:
            parts.append("already: " + ", ".join(f"{k}={v}" for k, v in already.items()))
        if newly:
            parts.append("newly: " + ", ".join(f"{k}={v}" for k, v in newly.items()))
        if not_found:
            parts.append("missing: " + ", ".join(not_found))
        summary = "; ".join(parts) if parts else "no binaries known"
        return ToolResponse.success(text=f"environment scan: {summary}", data=result)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_profile_tools(*, executor: Any) -> tuple[Tool, ...]:
    """Construct fresh instances of every CUDA profile tool sharing one Executor."""
    return (
        WriteWorkspaceFileTool(executor),
        RunCudaProbeTool(executor),
        ProfileWithNcuTool(executor),
        ProfileWithNsysTool(executor),
        ProfileWithTorchTool(executor),
        ProbeEnvironmentTool(executor),
    )


__all__ = [
    "ProbeEnvironmentTool",
    "ProfileWithNcuTool",
    "ProfileWithNsysTool",
    "ProfileWithTorchTool",
    "RunCudaProbeTool",
    "WriteWorkspaceFileTool",
    "make_profile_tools",
]
