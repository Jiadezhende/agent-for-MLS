"""CUDA-aware tooling for mls_agent.

Provides:
  * ``Executor`` — the central subprocess/CUDA execution layer (compile,
    run, ncu, nsys, torch profiling). The only place that runs nvcc / ncu /
    nsys binaries; tools delegate to it.
  * ``ExecutorConfig`` — env-driven configuration for the Executor.
  * ``ExecutorError`` — error type raised by the Executor on expected failures
    (compile/run/profile errors carry an ``error_class`` taxonomy tag).

Tool wrappers that turn ``Executor`` methods into ``mls_agent.Tool`` instances
live in ``mls_agent.tools.cuda.profile_tools``.
"""
from __future__ import annotations

from mls_agent.tools.cuda.config import ExecutorConfig
from mls_agent.tools.cuda.cuda_executor import Executor, JobResult, JobSpec
from mls_agent.tools.cuda.exceptions import ExecutorError

__all__ = [
    "Executor",
    "ExecutorConfig",
    "ExecutorError",
    "JobResult",
    "JobSpec",
]
