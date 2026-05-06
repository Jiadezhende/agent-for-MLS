from __future__ import annotations

import re
from typing import Any


_DIAG_RE = re.compile(
    r"(error:|warning:|note:|undefined reference|undefined symbol|"
    r"\d+ error(s)? detected|cannot open source file|fatal error)",
    re.IGNORECASE,
)


def _extract_nvcc_errors(combined: str, max_chars: int = 3000) -> str:
    """Extract only diagnostic lines from nvcc stderr/stdout mix."""
    lines = combined.splitlines()
    kept = [line.strip() for line in lines if line.strip() and _DIAG_RE.search(line)]
    result = "\n".join(kept) if kept else combined[-2000:]
    return result[:max_chars]


def _classify_compile_error(combined: str) -> str:
    """Return 'user_code' or 'infrastructure' based on nvcc error content."""
    return _classify_subprocess_failure(combined, phase="compile")["error_class"]


def _classify_subprocess_failure(
    combined: str,
    phase: str,
    timed_out: bool = False,
    returncode: int | None = None,
) -> dict[str, Any]:
    """Priority-ordered subprocess error taxonomy for LLM-facing tool errors."""
    low = combined.lower()
    if timed_out:
        return {
            "error": f"{phase}_timeout",
            "error_class": "timeout",
            "phase": phase,
            "hint": "Reduce workload size or increase timeout_s.",
        }
    if (
        "erf_no_privileged_mode" in low
        or "err_nvgpuctrperm" in low
        or ("access denied" in low and phase == "profile")
        or ("permission denied" in low and "ncu" in low)
    ):
        return {
            "error": "ncu_permission_denied",
            "error_class": "infrastructure",
            "phase": phase,
            "hint": (
                "ERR_NVGPUCTRPERM: no permission for GPU hardware counters. "
                "Do NOT retry ncu — switch to run_cuda_probe self-timed kernels instead."
            ),
        }
    if (
        "cuda driver version is insufficient" in low
        or "no cuda-capable device" in low
        or "cudaerrornodevice" in low
        or "failed to initialize cuda" in low
        or ("cuda driver" in low and "not found" in low)
    ):
        return {
            "error": "cuda_environment_unavailable",
            "error_class": "infrastructure",
            "phase": phase,
            "hint": "CUDA driver or NVIDIA GPU is unavailable; do not retry the same code.",
        }
    if (
        "cudaerrormemoryallocation" in low
        or "out of memory" in low
        or "bad_alloc" in low
        or "memory allocation failed" in low
    ):
        return {
            "error": "cuda_oom",
            "error_class": "user_code",
            "phase": phase,
            "hint": "Reduce allocation sizes or iteration counts.",
        }
    if (
        "undefined reference" in low
        or "undefined symbol" in low
        or "unresolved external symbol" in low
        or "lnk2019" in low
    ):
        return {
            "error": "link_failed",
            "error_class": "user_code",
            "phase": phase,
            "hint": "Fix missing symbols, libraries, or CUDA runtime linkage.",
        }
    if (
        "unsupported gpu architecture" in low
        or "unsupported architecture" in low
        or ("value 'sm_" in low and "not defined" in low)
        or "invalid value for --gpu-architecture" in low
        or "not a supported gpu architecture" in low
    ):
        return {
            "error": "unsupported_arch",
            "error_class": "user_code",
            "phase": phase,
            "hint": "Use an nvcc -arch flag supported by this CUDA toolkit/GPU.",
        }
    if (
        "cl.exe" in low and ("not found" in low or "cannot find" in low)
        or ("-ccbin" in combined and ("cannot find" in low or "no such file" in low))
        or "host compiler targets unsupported os" in low
    ):
        return {
            "error": "host_compiler_missing",
            "error_class": "infrastructure",
            "phase": phase,
            "hint": "Configure AGENT_NVCC_CCBIN or install the required host compiler.",
        }
    if "command not found" in low or "is not recognized as" in low:
        return {
            "error": "binary_not_found",
            "error_class": "infrastructure",
            "phase": phase,
            "hint": "Install the required binary or configure the corresponding AGENT_*_BIN variable.",
        }
    if phase == "run" and ("no such file or directory" in low or "errno 2" in low):
        return {
            "error": "binary_missing_after_compile",
            "error_class": "infrastructure",
            "phase": phase,
            "hint": (
                "The compiled binary could not be executed — it may not have been "
                "written to disk. Check nvcc output, workspace permissions, or "
                "try a simpler kernel first."
            ),
        }
    if "nvcc fatal" in low and "no input files" not in low and phase == "compile":
        return {
            "error": "nvcc_infrastructure_failure",
            "error_class": "infrastructure",
            "phase": phase,
            "hint": "nvcc failed before compiling user code; check CUDA toolkit installation.",
        }

    default_error = "compile_failed" if phase == "compile" else f"{phase}_failed"
    return {
        "error": default_error,
        "error_class": "user_code",
        "phase": phase,
        "hint": "Inspect stdout/stderr and adjust the generated code or arguments.",
        "returncode": returncode,
    }
