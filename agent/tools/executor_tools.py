"""
agents/tools/executor_tools.py — Thin Tool wrappers for Executor methods.

Each class delegates to the shared Executor instance and converts the returned
dict into a standardised ToolResponse.  The Executor itself is unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List

from agents.tools.base import Tool, ToolParameter
from agents.tools.response import ToolResponse


# ---------------------------------------------------------------------------
# Shared helper: executor result dict → ToolResponse
# ---------------------------------------------------------------------------

def _exec_result_to_response(result: dict, label: str) -> ToolResponse:
    """Convert an Executor result dict to a ToolResponse.

    Executor methods return dicts with a 'status' key:
      'done'      → SUCCESS
      'timed_out' → PARTIAL
      'error'     → ERROR  (error_class carries the circuit-breaker key)
    """
    status = result.get("status", "error")

    if status == "done":
        elapsed   = result.get("elapsed_s", "?")
        cache_tag = " [cache_hit]" if result.get("cache_hit") else ""
        stdout    = result.get("stdout", "")
        # Compose a readable summary for the LLM; full data is in result dict.
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

    # error path — keep the full result dict reachable via .data so the LLM can
    # inspect hint / stdout_tail / report_path / kernel_name / returncode etc.
    error_class = result.get("error_class", "unknown")
    error_kind  = result.get("error", "unknown_error")
    hint        = result.get("hint") or ""
    detail      = result.get("stderr") or result.get("stdout_tail") or result.get("detail") or result.get("message") or ""
    snippet     = str(detail)[:300] if detail else ""

    parts = [f"{label} failed ({error_class}/{error_kind})"]
    if hint:
        parts.append(f"hint: {hint}")
    if snippet:
        parts.append(f"detail: {snippet}")
    return ToolResponse.error(
        code=error_class,
        message=" | ".join(parts),
        data=result,
        stats={"elapsed_s": result.get("elapsed_s")},
    )


# ---------------------------------------------------------------------------
# run_cuda_probe
# ---------------------------------------------------------------------------

class WriteWorkspaceFileTool(Tool):
    """Write a source file to the workspace sandbox and return its path.

    Use this to stage large CUDA source files (8-12 KB kernels) before calling
    run_cuda_probe or profile_with_ncu with source_path. This way subsequent
    tool calls only need the short path string, keeping tool call JSON small and
    well within the LLM output token budget.
    """

    def __init__(self, executor: Any) -> None:
        super().__init__(
            name="write_workspace_file",
            description=(
                "Write content to a workspace file and return its path. "
                "Use before run_cuda_probe or profile_with_ncu to stage large "
                "CUDA source files. After writing, pass the returned 'path' as "
                "source_path to run_cuda_probe (saves output tokens on re-runs). "
                "Overwrites any existing file at the same path."
            ),
        )
        self._executor = executor

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="content",
                type="string",
                description="Full file content to write (e.g. CUDA C++ source).",
            ),
            ToolParameter(
                name="filename",
                type="string",
                description=(
                    "Workspace-relative path for the file. "
                    "Use 'src/' prefix for source files (e.g. 'src/kernel_v1.cu'). "
                    "Directories are created automatically."
                ),
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        result = self._executor.write_workspace_file(**parameters)
        path = result.get("path", "")
        size = result.get("size_bytes", 0)
        return ToolResponse.success(
            text=f"Written {size} bytes to '{path}'.",
            data=result,
        )


class RunCudaProbeTool(Tool):
    def __init__(self, executor: Any) -> None:
        super().__init__(
            name="run_cuda_probe",
            description=(
                "Compile and run a CUDA kernel source file. Use this when the "
                "kernel measures hardware properties via self-timing (clock64) "
                "and you need the stdout output as your primary measurement. "
                "The Executor handles compilation, sandboxing, and stdout capture. "
                "Do NOT use this for profiling — use profile_with_ncu instead. "
                "For large kernels (>3 KB), first write the source with "
                "write_workspace_file and pass the returned path as source_path."
            ),
        )
        self._executor = executor

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="source",
                type="string",
                description=(
                    "Complete CUDA C++ source code (.cu content). "
                    "MUST be actual code starting with #include or __global__. "
                    "NEVER pass a skill name, filename, or description here. "
                    "Mutually exclusive with source_path."
                ),
                required=False,
            ),
            ToolParameter(
                name="source_path",
                type="string",
                description=(
                    "Workspace-relative path to a .cu file written by write_workspace_file. "
                    "Use this instead of source for large kernels to keep tool call JSON small. "
                    "Mutually exclusive with source."
                ),
                required=False,
            ),
            ToolParameter(
                name="probe_name",
                type="string",
                description=(
                    "Short identifier for this probe (used in cache key and logs). "
                    "E.g. 'pointer_chase_dram', 'bandwidth_global'."
                ),
            ),
            ToolParameter(
                name="compile_flags",
                type="array",
                description="Extra nvcc flags (e.g. [\"-O3\", \"-arch=sm_86\"]).",
                required=False,
                default=[],
            ),
            ToolParameter(
                name="args",
                type="array",
                description="Command-line arguments passed to the compiled binary.",
                required=False,
                default=[],
            ),
            ToolParameter(
                name="timeout_s",
                type="integer",
                description="Execution timeout in seconds.",
                required=False,
                default=60,
            ),
        ]

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "run_cuda_probe",
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": (
                                "Complete CUDA C++ source code. "
                                "Use for short snippets only. "
                                "For kernels >3 KB, prefer write_workspace_file + source_path. "
                                "Mutually exclusive with source_path."
                            ),
                        },
                        "source_path": {
                            "type": "string",
                            "description": (
                                "Workspace-relative path to a .cu file from write_workspace_file. "
                                "Preferred for large kernels — keeps this call JSON tiny. "
                                "Mutually exclusive with source."
                            ),
                        },
                        "probe_name": {
                            "type": "string",
                            "description": "Short identifier used in cache key and logs.",
                        },
                        "compile_flags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                            "description": "Extra nvcc flags.",
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
                },
            },
        }

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        result = self._executor.run_cuda_probe(**parameters)
        return _exec_result_to_response(result, label=f"Probe '{parameters.get('probe_name', '')}'")


# ---------------------------------------------------------------------------
# profile_with_ncu
# ---------------------------------------------------------------------------

class ProfileWithNcuTool(Tool):
    def __init__(self, executor: Any) -> None:
        super().__init__(
            name="profile_with_ncu",
            description=(
                "Run a kernel under NVIDIA Nsight Compute to collect hardware performance "
                "counters. Specify exact metric names via the metrics list. "
                "Compiles with -lineinfo for source-line correlation and saves a "
                ".ncu-rep report (openable in Nsight Compute GUI). "
                "source_type='cuda_source' compiles and profiles directly; "
                "source_type='binary' profiles an existing workspace binary. "
                "Output is a human-readable table with Metric Name, Metric Unit, Metric Value."
            ),
        )
        self._executor = executor

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="source_type",   type="string", description="'cuda_source' or 'binary'."),
            ToolParameter(name="source_or_path", type="string", description="CUDA C source code, or workspace-relative binary path."),
            ToolParameter(name="kernel_name",    type="string", description="Kernel function name or regex passed to ncu --kernel-name."),
            ToolParameter(name="metrics",        type="array",  description="ncu metric names to collect (--metrics).", required=False, default=[]),
            ToolParameter(name="compile_flags",  type="array",  description="Extra nvcc flags. -lineinfo is always injected automatically.", required=False, default=[]),
            ToolParameter(name="args",           type="array",  description="Command-line arguments.", required=False, default=[]),
            ToolParameter(name="timeout_s",      type="integer", description="Total timeout for the ncu run in seconds.", required=False, default=600),
        ]

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_type": {
                            "type": "string",
                            "enum": ["cuda_source", "binary"],
                            "description": (
                                "'cuda_source': provide CUDA C source — compiled with -lineinfo "
                                "internally, kernel is NOT pre-run before profiling. "
                                "'binary': workspace-relative path to an already-compiled binary."
                            ),
                        },
                        "source_or_path": {
                            "type": "string",
                            "description": (
                                "If source_type='cuda_source': full CUDA C source code. "
                                "If source_type='binary': workspace-relative binary path "
                                "(e.g. from a prior run_cuda_probe result's binary_path field)."
                            ),
                        },
                        "kernel_name": {
                            "type": "string",
                            "description": (
                                "Kernel function name or regex passed to ncu --kernel-name. "
                                "ncu will only instrument matching kernels."
                            ),
                        },
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "ncu metric names to collect (--metrics). Required. "
                                "Examples: ['sm__throughput.avg.pct_of_peak_sustained_elapsed', "
                                "'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', "
                                "'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum', "
                                "'sm__cycles_elapsed.avg.per_second']."
                            ),
                            "default": [],
                        },
                        "compile_flags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Extra nvcc flags. -lineinfo is always injected automatically.",
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
                            "description": "Total timeout for the ncu run in seconds.",
                        },
                    },
                    "required": ["source_type", "source_or_path", "kernel_name"],
                },
            },
        }

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        result = self._executor.profile_with_ncu(**parameters)
        return _exec_result_to_response(result, label="ncu profile")


# ---------------------------------------------------------------------------
# profile_with_nsys
# ---------------------------------------------------------------------------

class ProfileWithNsysTool(Tool):
    def __init__(self, executor: Any) -> None:
        super().__init__(
            name="profile_with_nsys",
            description=(
                "Run a program under NVIDIA Nsight Systems to capture the "
                "CPU-GPU timeline. Use this to understand kernel launch overhead, "
                "CUDA stream execution order, and host-device synchronization patterns."
            ),
        )
        self._executor = executor

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="source_type",   type="string",  description="'cuda_source', 'python_script', or 'binary'."),
            ToolParameter(name="source_or_path", type="string",  description="Source code string or workspace-relative path."),
            ToolParameter(name="compile_flags",  type="array",   description="Extra nvcc flags.", required=False, default=[]),
            ToolParameter(name="args",           type="array",   description="Command-line arguments.", required=False, default=[]),
            ToolParameter(name="duration_s",     type="integer", description="Approximate profiling window in seconds.", required=False, default=10),
            ToolParameter(name="timeout_s",      type="integer", description="Total timeout in seconds.", required=False, default=300),
        ]

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_type": {
                            "type": "string",
                            "enum": ["cuda_source", "python_script", "binary"],
                            "description": (
                                "'cuda_source': CUDA source code (Executor compiles and profiles). "
                                "'python_script': Python code run under nsys. "
                                "'binary': existing workspace binary."
                            ),
                        },
                        "source_or_path": {
                            "type": "string",
                            "description": "Source code string or workspace-relative path.",
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
                        "duration_s": {
                            "type": "integer",
                            "default": 10,
                            "description": "Approximate profiling window in seconds.",
                        },
                        "timeout_s": {
                            "type": "integer",
                            "default": 300,
                        },
                    },
                    "required": ["source_type", "source_or_path"],
                },
            },
        }

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        result = self._executor.profile_with_nsys(**parameters)
        return _exec_result_to_response(result, label="nsys profile")


# ---------------------------------------------------------------------------
# profile_with_torch
# ---------------------------------------------------------------------------

class ProfileWithTorchTool(Tool):
    def __init__(self, executor: Any) -> None:
        super().__init__(
            name="profile_with_torch",
            description=(
                "Run a Python script (typically importing torch) and capture its stdout. "
                "Use this to measure PyTorch baseline performance, generate reference "
                "output tensors for correctness validation, or run any Python-based "
                "GPU workload. The script executes with cwd set to the workspace root, "
                "so files written to relative paths (e.g. 'data/ref.npy') are placed "
                "inside the sandbox. stdout is returned as the tool result."
            ),
        )
        self._executor = executor

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="python_code",
                type="string",
                description=(
                    "Complete Python source code to run. Must be standalone (no external "
                    "file dependencies unless writing them first). Print measurements to "
                    "stdout in a parseable format."
                ),
            ),
            ToolParameter(
                name="op_name",
                type="string",
                description=(
                    "Short identifier for this script (used as filename and cache key). "
                    "E.g. 'lora_matmul_baseline', 'correctness_oracle'."
                ),
            ),
            ToolParameter(
                name="timeout_s",
                type="integer",
                description="Execution timeout in seconds.",
                required=False,
                default=120,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        result = self._executor.profile_with_torch(**parameters)
        return _exec_result_to_response(result, label=f"torch '{parameters.get('op_name', '')}'")


# ---------------------------------------------------------------------------
# probe_environment
# ---------------------------------------------------------------------------

class ProbeEnvironmentTool(Tool):
    def __init__(self, executor: Any) -> None:
        super().__init__(
            name="probe_environment",
            description=(
                "Scan the filesystem for nvcc, ncu, and nsys binaries without executing them. "
                "Call this when error_class='infrastructure' and error='binary_not_found'. "
                "If binaries are found, the Executor is reconfigured automatically so "
                "subsequent tool calls use the discovered paths. "
                "Does NOT run any subprocess — only filesystem stat checks."
            ),
        )
        self._executor = executor

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="force_rescan",
                type="boolean",
                description=(
                    "If true, re-scans even if a binary appears to be on PATH. "
                    "Use only if you suspect PATH-based resolution is wrong."
                ),
                required=False,
                default=False,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        result = self._executor.probe_environment(**parameters)
        # probe_environment returns a plain dict (not a JobResult).
        # Treat it as success unless it contains an error key.
        if "error" in result:
            return ToolResponse.error(
                code=result.get("error_class", "probe_error"),
                message=str(result.get("error", "probe_environment failed")),
            )
        already = result.get("already_configured", {})
        newly = result.get("newly_found", {})
        not_found = result.get("not_found", [])
        parts: List[str] = []
        if already:
            parts.append("already: " + ", ".join(f"{k}={v}" for k, v in already.items()))
        if newly:
            parts.append("newly: " + ", ".join(f"{k}={v}" for k, v in newly.items()))
        if not_found:
            parts.append("missing: " + ", ".join(not_found))
        summary = "; ".join(parts) if parts else "no binaries known"
        return ToolResponse.success(
            text=f"Environment scan complete: {summary}",
            data=result,
        )


# ---------------------------------------------------------------------------
# find_binary
# ---------------------------------------------------------------------------

class FindBinaryTool(Tool):
    def __init__(self, executor: Any) -> None:
        super().__init__(
            name="find_binary",
            description=(
                "Search for a single CUDA tool binary (ncu, nsys, or nvcc) using built-in "
                "glob patterns plus optional caller-supplied globs. Use this if "
                "probe_environment came back empty and you know the install path "
                "(e.g. a non-standard drive or a fresh Nsight version not yet in the "
                "built-in list). Returns every candidate found and reconfigures the "
                "Executor to the lex-newest one. Does NOT run any subprocess."
            ),
        )
        self._executor = executor

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="name",
                type="string",
                description="Binary name: 'ncu', 'nsys', or 'nvcc'.",
                required=True,
            ),
            ToolParameter(
                name="extra_globs",
                type="array",
                description=(
                    "Optional list of glob patterns to add to the search, e.g. "
                    "['D:/Nsight/**/ncu.bat']. Patterns are merged with built-in ones."
                ),
                required=False,
                default=[],
            ),
            ToolParameter(
                name="reconfigure",
                type="boolean",
                description="If true (default), update Executor config with the chosen path.",
                required=False,
                default=True,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        result = self._executor.find_binary(**parameters)
        if not result.get("ok", False):
            return ToolResponse.error(
                code=result.get("error_class", "find_binary_error"),
                message=str(result.get("message") or result.get("error") or "find_binary failed"),
            )
        chosen = result.get("chosen")
        candidates = result.get("candidates", [])
        if chosen:
            text = (
                f"find_binary({result.get('name')}): chose {chosen}"
                + (f" (out of {len(candidates)} candidates)" if len(candidates) > 1 else "")
                + (" [reconfigured]" if result.get("reconfigured") else "")
            )
        else:
            text = (
                f"find_binary({result.get('name')}): no candidates matched. "
                f"Searched {len(result.get('patterns_searched', []))} patterns."
            )
        return ToolResponse.success(text=text, data=result)
