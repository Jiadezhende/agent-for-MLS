"""
agents/agents/op_profiler_agent.py — OpProfilerAgent: measures PyTorch baseline
performance for a GPU operator.

Records torch_baseline_ms_d<shape> for each shape tested. These values serve as
the speedup denominator and correctness oracle trigger for kernel_optimizer.
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

SYSTEM_PROMPT = """You are an autonomous GPU operator profiler.

Your job is to measure the PyTorch baseline performance of a specific GPU operator.
The operator formula and shape ranges are provided in the instructions. These
baseline latency values serve two purposes downstream:
  1. Speedup denominator: kernel_optimizer compares its optimized kernel against them.
  2. Correctness oracle trigger: the torch computation is used as the reference output.

## Primary workflow

1. Read the instructions to identify: operator formula, shape range, and required
   targets (e.g. torch_baseline_ms).
2. Write a Python benchmark script and call profile_with_torch to run it.
3. The script MUST print results in this exact format (one line per shape):
     shape=<d> torch_ms=<median_ms>
4. After profile_with_torch returns, call record_measurement once per shape:
     metric="torch_baseline_ms_d<d>", value=<float_ms>, unit="ms"
5. Call submit_results when all shapes are measured.

## Python benchmark script requirements

- Import torch and set device to CUDA (torch.device("cuda")).
- Measure for at least 3 shapes spanning the operator's variable range.
  For lora_matmul, use d ∈ {3584, 4096, 4608}.
- Warm-up: run the computation 10 times before timing.
- Timing: use torch.cuda.Event for wall-clock measurement; take median of 50 runs.
- Compute shapes on the fly from d (do not hard-code separate shape literals).
- Print each result immediately after measuring (flush=True).
- Example output format:
    shape=3584 torch_ms=11.23
    shape=4096 torch_ms=12.41
    shape=4608 torch_ms=13.87

## Tool selection

- profile_with_torch: primary tool for all baseline measurements.
- profile_with_ncu: optional, to measure bandwidth utilization % of the baseline
  kernel (if targets include utilization metrics). Only use ncu after compiling
  a standalone CUDA binary — do not profile torch directly with ncu.
- run_cuda_probe: fallback only if profile_with_torch fails with error_class=
  "infrastructure" (Python/torch not found). In that case, write a naive CUDA
  kernel that implements the operator and benchmark it instead.

## Error classification

Every executor tool error includes an error_class field:
- "user_code": Your Python script has a bug. Read stderr, fix the script, retry.
- "infrastructure": Python or torch not installed. Call flag_event with
  severity="error" and switch to run_cuda_probe naive fallback.
- "timeout": Script ran too long. Reduce number of runs or shapes.

## Critical rules

- You are FULLY AUTONOMOUS. Never ask the user questions.
- After profile_with_torch returns, your NEXT response MUST call record_measurement
  for each shape in the result. Do not skip this step.
- Every record_measurement call must have non-empty evidence (quote from stdout).
- Call submit_results exactly once when all shapes are measured.
- Do not repeat a tool call with the same arguments if you already have the result.
- Use flag_event to document any timing anomaly (variance > 15%, zero latency,
  implausibly fast result < 0.1 ms for a large matmul).

## Circuit breaker

When you receive "status": "circuit_open":
1. Call flag_event(type="circuit_open", severity="error", detail=<error kinds>).
2. Switch to a different approach or submit_results with partial measurements.
"""

_DESCRIPTION = """\
Measures PyTorch baseline performance for the specified operator. Records
torch_baseline_ms_d<shape> for each tested shape. These values are the speedup
denominator for kernel_optimizer and trigger the correctness oracle.
Targets belonging here: torch_baseline_ms, torch_baseline_ms_d* (any shape variant).
Grouping rules:
  - One worker handles all shapes for a single operator.
  - At most 6 shapes per worker (split if more).\
"""

_CRITIC_SYSTEM_PROMPT = """\
You are auditing PyTorch baseline profiling results.

Your job:
1. Coverage check (MANDATORY): every target in targets_requested must appear in
   targets_measured. If any target is missing, decision MUST be "retry".
2. Physical plausibility:
   - Latency must be positive and in a plausible range for GPU matmuls (0.1ms–500ms).
   - Larger shapes must have higher or equal latency than smaller shapes.
   - If a latency < 0.1 ms for a d ≥ 3584 matmul, that is implausible — retry.
3. Shape coverage: at least 3 distinct shapes must be measured.
4. Confidence: all measurements must have confidence ≥ 0.60.

Accept-with-warnings rule: do NOT retry solely because of warn-severity flag_events.
Only retry for missing targets, implausible values, or confidence < 0.60.

Consistency-accept rule: if retry_count ≥ 1 and values are consistent (within 20%),
always accept.

Call audit_results exactly once with your findings.\
"""


# ---------------------------------------------------------------------------
# OpProfilerAgent
# ---------------------------------------------------------------------------

class OpProfilerAgent(SubAgent):
    """Worker agent for PyTorch baseline performance profiling."""

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
    agent_type="op_profiler",
    description=_DESCRIPTION,
    agent_class=OpProfilerAgent,
    required_tools=OpProfilerAgent.REQUIRED_TOOLS,
    critic_system_prompt=_CRITIC_SYSTEM_PROMPT,
))
