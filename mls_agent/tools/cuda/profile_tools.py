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
        "Run a CUDA kernel under NVIDIA Nsight Compute and collect the metrics "
        "in `metrics`. Saves a .ncu-rep report. Specify exactly ONE of cuda_source "
        "(tool compiles with -O3 -lineinfo) or binary_path (you control compile "
        "flags). ncu CANNOT profile Python scripts — to profile a kernel exposed "
        "via torch.utils.cpp_extension.load (no main()), use profile_with_nsys "
        "with python_source instead."
    )

    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "cuda_source": {
                    "type": "string",
                    "description": (
                        "Inline CUDA C++ source. Tool compiles with -O3 -lineinfo "
                        "automatically. Source must include a main() that launches "
                        "the kernel. Mutually exclusive with binary_path."
                    ),
                },
                "binary_path": {
                    "type": "string",
                    "description": (
                        "Workspace-relative path to a pre-compiled binary. You "
                        "must compile with -lineinfo yourself for source-line "
                        "correlation. Mutually exclusive with cuda_source."
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
                    "description": "ncu metric names; at least one required by the executor.",
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
            "required": ["kernel_name"],
            "oneOf": [
                {"required": ["cuda_source"]},
                {"required": ["binary_path"]},
            ],
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        cuda = parameters.get("cuda_source")
        bn = parameters.get("binary_path")
        given = [
            (k, v) for k, v in (
                ("cuda_source", cuda),
                ("binary_path", bn),
            ) if v is not None
        ]
        if len(given) != 1:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message=(
                    f"Exactly one of cuda_source / binary_path required "
                    f"(got {len(given)}: {[k for k, _ in given]})."
                ),
            )
        field, value = given[0]
        source_type = "cuda_source" if field == "cuda_source" else "binary"
        result = self._executor.profile_with_ncu(
            source_type=source_type,
            source_or_path=value,
            kernel_name=parameters["kernel_name"],
            metrics=parameters.get("metrics", []),
            compile_flags=parameters.get("compile_flags", []),
            args=parameters.get("args", []),
            timeout_s=parameters.get("timeout_s", 600),
        )
        return _exec_result_to_response(result, label="ncu profile")


# ---------------------------------------------------------------------------
# profile_with_nsys
# ---------------------------------------------------------------------------


class ProfileWithNsysTool(Tool):
    NAME = "profile_with_nsys"
    DESCRIPTION = (
        "Run a target program under NVIDIA Nsight Systems and capture the CPU-GPU "
        "timeline + per-kernel summary (via `nsys stats --report cuda_gpu_kern_sum`). "
        "Specify exactly ONE of: cuda_source (inline .cu with main()), python_source "
        "(inline .py to run), binary_path (workspace-relative pre-compiled artifact). "
        "For PyBind11 kernels loaded via torch.utils.cpp_extension.load, use "
        "python_source — that's the only path that reproduces the real call stack."
    )

    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "cuda_source": {
                    "type": "string",
                    "description": (
                        "Inline CUDA C++ source to compile and profile. Source must "
                        "include a main() that launches the kernel. "
                        "Mutually exclusive with python_source / binary_path."
                    ),
                },
                "python_source": {
                    "type": "string",
                    "description": (
                        "Inline Python source (typically importing torch + "
                        "cpp_extension.load) to run under nsys. Use this to profile "
                        "kernels exposed via PyBind11 that have no main() of their "
                        "own. Mutually exclusive with cuda_source / binary_path."
                    ),
                },
                "binary_path": {
                    "type": "string",
                    "description": (
                        "Workspace-relative path to a pre-compiled executable. "
                        "Mutually exclusive with cuda_source / python_source."
                    ),
                },
                "compile_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Extra nvcc flags (only used with cuda_source).",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Command-line args appended to the target program.",
                },
                "duration_s": {"type": "integer", "default": 10},
                "timeout_s": {
                    "type": "integer",
                    "default": 300,
                    "description": (
                        "Wall-clock cap. Raise to 600+ when python_source triggers "
                        "a first-time cpp_extension.load build (30-60s)."
                    ),
                },
            },
            "oneOf": [
                {"required": ["cuda_source"]},
                {"required": ["python_source"]},
                {"required": ["binary_path"]},
            ],
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        cuda = parameters.get("cuda_source")
        py = parameters.get("python_source")
        bn = parameters.get("binary_path")
        given = [
            (k, v) for k, v in (
                ("cuda_source", cuda),
                ("python_source", py),
                ("binary_path", bn),
            ) if v is not None
        ]
        if len(given) != 1:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message=(
                    f"Exactly one of cuda_source / python_source / binary_path "
                    f"required (got {len(given)}: {[k for k, _ in given]})."
                ),
            )
        field, value = given[0]
        source_type = {
            "cuda_source": "cuda_source",
            "python_source": "python_script",
            "binary_path": "binary",
        }[field]
        result = self._executor.profile_with_nsys(
            source_type=source_type,
            source_or_path=value,
            compile_flags=parameters.get("compile_flags", []),
            args=parameters.get("args", []),
            duration_s=parameters.get("duration_s", 10),
            timeout_s=parameters.get("timeout_s", 300),
        )
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
        "script's CWD is the workspace root, so relative writes stay sandboxed. "
        "Also use as a fallback when ncu/nsys are unavailable (e.g. permission "
        "denied): write a timing loop using torch.cuda.Event or torch.profiler "
        "and capture the printed output. "
        "Note: timeout_s defaults to 600s. cpp_extension.load on first call "
        "rebuilds the extension and can take 30-60s; budget for that before "
        "wrapping the timing loop."
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
                "timeout_s": {
                    "type": "integer",
                    "default": 600,
                    "description": (
                        "Wall-clock cap. Default 600s covers first-time "
                        "cpp_extension.load (30-60s build) + warmup + several "
                        "timed iterations. Bump to 900-1200 for multi-shape sweeps."
                    ),
                },
            },
            "required": ["python_code", "op_name"],
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        result = self._executor.profile_with_torch(
            python_code=parameters["python_code"],
            op_name=parameters["op_name"],
            timeout_s=parameters.get("timeout_s", 600),
        )
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
