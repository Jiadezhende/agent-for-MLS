"""
agents/agents/kernel_optimizer_agent.py — KernelOptimizerAgent: writes and
validates an optimized CUDA kernel for the target operator.

Workflow (internal ReAct loop):
  1. Read bottleneck conclusion + operator spec from instructions.
  2. Draft an optimized CUDA kernel applying the recommended strategy.
  3. Validate correctness against PyTorch reference (two-step protocol).
  4. Benchmark performance for ≥ 3 shapes; compute speedup vs torch baseline.
  5. Iterate until correctness passes and speedup ≥ 1.0.
  6. submit_results with the final .cu source embedded in summary.
"""
from __future__ import annotations

from typing import Any

from agents.core.agent import SubAgent
from agents.core.llm import LLMClient
from agents.core.loop import AgentLoop
from agents.core.prompts import build_user_message
from agents.core.types import AgentContext, MemoryStore, Step, WorkerOutput
from agents.tools.circuit_breaker import CircuitBreaker
from agents.tools.registry import ToolRegistry
from agents._registry import AgentDefinition, register


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous CUDA kernel optimizer.

Your job is to write an optimized CUDA kernel for the operator specified in
the instructions, validate its correctness against a PyTorch reference, and
benchmark its performance. The instructions contain:
  - Bottleneck type (memory_bound / compute_bound) and recommended strategy
  - Torch baseline latency per shape (speedup denominator)
  - Operator specification (formula, shapes, correctness threshold, success criteria)

## Two-phase validation protocol

### Phase A: Generate reference data (PyTorch oracle)

Call profile_with_torch with a Python script that:
  1. Generates random input tensors matching the operator spec (use torch.manual_seed
     for reproducibility).
  2. Runs the PyTorch reference computation (e.g. W @ X + A @ (B.T @ X)).
  3. Saves inputs AND reference output as numpy binary files to the workspace:
       import os, numpy as np, torch
       os.makedirs("data", exist_ok=True)
       np.save("data/W.npy", W.cpu().numpy())
       np.save("data/X.npy", X.cpu().numpy())
       np.save("data/ref_out.npy", ref_out.cpu().numpy())
  4. Prints a confirmation: shape=<d> ref_saved=True

Do this for each shape you intend to validate (at least 3).

### Phase B: Validate optimized kernel

Call run_cuda_probe with a CUDA C source that:
  1. Loads the saved .npy files from Phase A (use a minimal npy reader — see below).
  2. Copies inputs to GPU.
  3. Runs the optimized kernel.
  4. Copies output back to CPU.
  5. Computes max absolute difference vs reference output.
  6. Prints for each shape:
       shape=<d> max_abs_diff=<value>
  7. Also prints kernel timing using cudaEvent:
       shape=<d> opt_ms=<value> speedup=<opt_ms / torch_baseline_ms>

Use torch_baseline_ms from instructions to compute speedup.

### Minimal .npy file reader for CUDA C

A simple approach is to write a small C function that reads .npy files:
  - Skip the first 128 bytes (numpy magic + header, padded to 128 bytes)
  - Read the remaining bytes as raw float32 data
  - This works for simple float32 arrays (C-contiguous)
  For safety, validate that the element count matches expected shape × shape.

## Iteration logic

After each validation pass:
- If max_abs_diff > threshold (check operator spec, typically 1e-2 for float32):
    → flag_event("correctness_fail", "error", detail="shape=X diff=Y")
    → Identify the numerical error (wrong accumulation order? overflow? wrong index?)
    → Fix the kernel and retry from Phase B (no need to re-generate reference data)
- If speedup < 1.0 on a shape:
    → flag_event("performance_regression", "warn", detail="shape=X speedup=Y")
    → Consider a different optimization strategy (adjust tile size, use different
      memory access pattern) or simply report with note if marginal
- Optionally call profile_with_ncu to diagnose remaining bottlenecks:
    → Useful metrics: sm__throughput.avg.pct_of_peak_sustained_elapsed,
      dram__throughput.avg.pct_of_peak_sustained_elapsed,
      l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum

## CUDA kernel writing guidelines

- Always include: #include <cuda_runtime.h>, #include <stdio.h>, #include <math.h>
- Use cudaEvent for timing (not clock64 — it measures cycles not wall time):
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel<<<grid, block>>>(args...);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
- For median timing: run 100 iterations, store results, take median.
- Always call cudaGetLastError() after kernel launch and cudaDeviceSynchronize().
- Use cudaMallocManaged or cudaMalloc + explicit transfers.
- The GPU arch flag is auto-detected; do NOT hard-code -arch in compile_flags.
- Always add bounds checking for variable-size inputs.

## Memory-bound strategies

- Fuse multiple passes into one kernel to reduce DRAM round-trips.
- Use shared memory tiling: load tiles of A and B into __shared__ memory, compute
  a block of C from the tiles (classic tiled GEMM pattern).
- Use vectorized loads: float4 loads (128-bit) where strides allow.
- Tile size guidance: BLOCK_M × BLOCK_N × 4 bytes should fit in L1/shared memory
  (typically ≤ 48 KB; use L2 cache size from bottleneck analysis for L2 reuse).

## Compute-bound strategies

- Increase ILP: unroll inner loops with #pragma unroll.
- Use register blocking to hide latency.
- WMMA (tensor cores) for float16 accumulation (if precision allows). Note: for
  float32 accumulation you need sm_80+ and TF32 (reduced precision).
- Tune BLOCK_M, BLOCK_N, BLOCK_K to keep the GPU fully occupied.

## Error classification

- "user_code": CUDA source has a compile error or runtime crash. Fix and retry.
- "infrastructure": Compiler or system issue. Call flag_event + try fallback.
- "timeout": Reduce problem size or use a simpler kernel variant.

## Critical rules

- You are FULLY AUTONOMOUS. Never ask for direction.
- Correctness FIRST. Do not claim success if max_abs_diff exceeds the threshold.
- After run_cuda_probe or profile_with_torch returns, IMMEDIATELY call
  record_measurement for each shape result found in stdout.
- Evidence: every record_measurement must quote the exact stdout line.
- The submit_results summary MUST include the full final .cu kernel source code
  (paste the entire source). Downstream tools (Critic, reporting) need it.
- Call submit_results exactly once.

## Circuit breaker

When you receive "status": "circuit_open":
1. Call flag_event(type="circuit_open", severity="error").
2. Switch to a simpler kernel variant or submit_results with partial data.
"""

_DESCRIPTION = """\
Writes, validates, and benchmarks an optimized CUDA kernel for the target operator.
Internal workflow: design kernel → validate correctness vs PyTorch oracle →
benchmark speedup → iterate. Submits final .cu source embedded in summary.
Targets belonging here: speedup_d*, max_abs_diff_d*, opt_latency_ms_d* (for each
  tested shape), and kernel_source (the optimized .cu code as a result).
This agent requires bottleneck_analyst results in its instructions.
Grouping rules: one worker per operator; handles all shapes internally.\
"""

_CRITIC_SYSTEM_PROMPT = """\
You are auditing CUDA kernel optimization results.

Your job:
1. Coverage check (MANDATORY): speedup_d* and max_abs_diff_d* must be present for
   at least 3 distinct shapes. If fewer than 3 shapes are covered, retry.
2. Correctness check:
   - max_abs_diff must be < the operator's correctness threshold (typically 1e-2
     for float32). If ANY shape has max_abs_diff ≥ threshold, retry immediately.
   - A kernel with correctness failures is not an acceptable result.
3. Performance check:
   - speedup must be ≥ 1.0 for at least 2 of 3 shapes. A marginal regression on
     one shape (speedup ≥ 0.95) with improvements on others is acceptable.
   - If speedup < 0.95 for all shapes, retry with reason "no improvement over baseline".
4. Source check: the summary must contain CUDA C source code. If it only contains
   "see summary" or references without code, retry with reason "kernel source missing".
5. Plausibility:
   - speedup > 20× without algorithmic justification → flag but do not auto-retry.
   - opt_latency_ms ≤ 0 → retry (timing error).

Accept-with-warnings: do NOT retry for warn-severity flag_events alone.
Consistency-accept: if retry_count ≥ 1 and speedup values are consistent, accept.

Call audit_results exactly once.\
"""


# ---------------------------------------------------------------------------
# KernelOptimizerAgent
# ---------------------------------------------------------------------------

class KernelOptimizerAgent(SubAgent):
    """Worker agent for optimized CUDA kernel design, validation, and benchmarking."""

    REQUIRED_TOOLS: list[str] = [
        "list_skills",
        "read_skill",
        "profile_with_torch",
        "run_cuda_probe",
        "profile_with_ncu",
        "record_measurement",
        "flag_event",
        "submit_results",
    ]

    def __init__(
        self,
        llm: LLMClient,
        agent_cfg: Any,
        verbose: bool = False,
        worker_id: int | str | None = None,
    ) -> None:
        self.llm = llm
        self.agent_cfg = agent_cfg
        self.verbose = verbose
        self.worker_id = worker_id

    def run(self, step: Step, tools: ToolRegistry) -> WorkerOutput:
        ctx = AgentContext(
            memory=MemoryStore(),
            circuit_breaker=CircuitBreaker(
                threshold=self.agent_cfg.circuit_breaker_threshold,
                half_open_timeout_s=self.agent_cfg.half_open_timeout_s,
            ),
        )

        success = False
        try:
            loop = AgentLoop(
                llm=self.llm,
                registry=tools,
                ctx=ctx,
                max_iterations=self.agent_cfg.max_iterations,
                verbose=self.verbose,
                worker_id=self.worker_id,
                system_prompt=SYSTEM_PROMPT,
                user_message=step.instructions or build_user_message(step.targets, step.retry_context),
            )
            loop.run()
            success = True
        except RuntimeError:
            measured = {r.metric for r in ctx.results}
            missing = [t for t in step.targets if t not in measured]
            success = len(missing) == 0
            if missing and self.verbose:
                print(f"[W{self.worker_id}] incomplete: missing {missing}", flush=True)
        except Exception:
            success = False

        return WorkerOutput(
            step_id=step.id,
            results=[r.to_dict() for r in ctx.results],
            success=success,
            targets_requested=step.targets,
            reasoning_log=ctx.reasoning_log,
            events=ctx.events,
            summary=ctx.memory.get("run", "summary") or "",
        )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

register(AgentDefinition(
    agent_type="kernel_optimizer",
    description=_DESCRIPTION,
    agent_class=KernelOptimizerAgent,
    required_tools=KernelOptimizerAgent.REQUIRED_TOOLS,
    critic_system_prompt=_CRITIC_SYSTEM_PROMPT,
))
