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

    # error path
    error_class = result.get("error_class", "unknown")
    detail      = result.get("stderr") or result.get("detail") or result.get("message") or ""
    snippet     = str(detail)[:300] if detail else ""
    return ToolResponse.error(
        code=error_class,
        message=f"{label} failed ({error_class}): {snippet}",
        stats={"elapsed_s": result.get("elapsed_s")},
    )


# ---------------------------------------------------------------------------
# run_cuda_probe
# ---------------------------------------------------------------------------

class RunCudaProbeTool(Tool):
    def __init__(self, executor: Any) -> None:
        super().__init__(
            name="run_cuda_probe",
            description=(
                "Compile and run a CUDA kernel source file. Use this when the "
                "kernel measures hardware properties via self-timing (clock64) "
                "and you need the stdout output as your primary measurement. "
                "The Executor handles compilation, sandboxing, and stdout capture. "
                "Do NOT use this for profiling — use profile_with_ncu instead."
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
                    "NEVER pass a skill name, filename, or description here."
                ),
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
        found = result.get("found", {})
        summary = "; ".join(f"{k}={v}" for k, v in found.items()) if found else "no binaries found"
        return ToolResponse.success(
            text=f"Environment scan complete: {summary}",
            data=result,
        )
