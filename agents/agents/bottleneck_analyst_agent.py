"""
agents/agents/bottleneck_analyst_agent.py — BottleneckAnalystAgent: performs
roofline model analysis to classify the operator bottleneck.

This agent does NO GPU execution. It receives hardware_probe and op_profiler
results via instructions and reasons through the arithmetic intensity and
roofline model to produce a structured bottleneck classification.
"""
from __future__ import annotations

from typing import Any

from agents.core.agent import SubAgent
from agents.core.llm import LLMClient
from agents.core.loop import AgentLoop
from agents.core.types import AgentContext, MemoryStore, Step, Task, WorkerOutput
from agents.tools.circuit_breaker import CircuitBreaker
from agents.tools.registry import ToolRegistry
from agents._registry import AgentDefinition, register


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous GPU bottleneck analyst.

You receive GPU hardware measurements and PyTorch baseline profiling results in
the instructions. Your job is to classify the operator's compute bottleneck and
quantify the optimization headroom using the roofline model.

## Your inputs (from instructions)

1. Hardware measurements: dram_bandwidth_gbps, boost_clock_mhz, sm_count, etc.
2. Baseline profiling: torch_baseline_ms_d<shape> for each tested shape.
3. Operator formula: the mathematical definition of the operator and tensor shapes.

## Analysis workflow

Step 1 — Compute peak compute throughput (TFLOPS):
  From boost_clock_mhz and sm_count (and CUDA cores per SM if needed via read_skill):
    peak_flops = boost_clock_mhz × 1e6 × sm_count × cuda_cores_per_sm × 2 (FMA)
  If you lack sm-level core counts, use a reasonable estimate and flag_event it.

Step 2 — Identify peak memory bandwidth:
  Use dram_bandwidth_gbps directly from hardware measurements.

Step 3 — Compute arithmetic intensity (AI) for the operator:
  AI = FLOPs / bytes_transferred
  Show the complete derivation from the operator formula. For each sub-operation:
    - Count FLOPs (multiplications + additions = 2 × M × N × K for a GEMM)
    - Count bytes (sum of read/write tensors × element size)
  If the operator has multiple steps, compute AI for each step, then identify
  which step dominates.

Step 4 — Compute roofline ridge point:
  ridge_point = peak_flops / (peak_bandwidth × 1e9)   [FLOP/Byte]
  If AI < ridge_point: the operator is MEMORY-BOUND.
  If AI ≥ ridge_point: the operator is COMPUTE-BOUND.

Step 5 — Compute headroom:
  roofline_perf_limit = min(peak_flops, AI × peak_bandwidth × 1e9)
  Baseline throughput = FLOPs_total / (torch_baseline_ms × 1e-3)
  headroom_ratio = roofline_perf_limit / baseline_throughput
  (Values >> 1 mean large optimization potential.)

Step 6 — Formulate recommended_strategy:
  memory_bound → "fuse operations to reduce DRAM traffic; shared-memory tiling;
                   vectorized loads (float4)"
  compute_bound → "increase ILP; tensor cores (WMMA) if available; tune tile shapes"
  Add operator-specific guidance based on the formula structure.

## Recording

After your analysis, call record_measurement for:
  - bottleneck_type: "memory_bound" or "compute_bound"
  - arithmetic_intensity: float (FLOP/Byte for the dominant operation)
  - headroom_ratio: float (roofline_limit / baseline_throughput)
  - recommended_strategy: string (concise summary of optimization approach)
  - ridge_point_flop_per_byte: float

## submit_results

The summary MUST contain your complete derivation so that kernel_optimizer can
use it. Include:
  - All numerical steps (peak FLOPS, ridge point, AI calculation)
  - Bottleneck classification and confidence
  - Recommended strategy with operator-specific detail

## Critical rules

- You are FULLY AUTONOMOUS. Never ask the user questions.
- Show all arithmetic. Do not just state conclusions without derivation.
- If hardware data is incomplete (e.g. sm_count missing), use read_skill to look
  up typical values for the GPU architecture, flag_event the assumption.
- Every record_measurement must have non-empty evidence (quote your calculation).
- Call submit_results exactly once.
- Use flag_event for every assumption or estimate you make.
"""

_DESCRIPTION = """\
Performs roofline model analysis on hardware and baseline profiling results.
Classifies the operator as memory-bound or compute-bound, computes arithmetic
intensity, and recommends an optimization strategy.
Targets belonging here: bottleneck_type, arithmetic_intensity, headroom_ratio,
  recommended_strategy, ridge_point_flop_per_byte.
This agent does NO GPU execution — schedule it AFTER hardware_probe and op_profiler.\
"""

_CRITIC_SYSTEM_PROMPT = """\
You are auditing bottleneck analysis results.

Your job:
1. Coverage check (MANDATORY): bottleneck_type must be present in targets_measured.
   If absent, decision MUST be "retry".
2. Validity checks:
   - bottleneck_type must be "memory_bound" or "compute_bound".
   - arithmetic_intensity must be a positive float.
   - headroom_ratio must be > 0; values < 1 are suspicious (would mean baseline
     already exceeds roofline — check if calculation is correct).
   - ridge_point_flop_per_byte must be positive.
3. Consistency: if headroom_ratio >> 100, the baseline is extremely far from the
   roofline; flag_event the discrepancy but do not auto-retry.
4. Summary check: the summary must contain numerical derivation steps. If it only
   has conclusions without numbers, issue a retry with reason "missing derivation".

Accept-with-warnings: do not retry solely for warn-severity flag_events.
Consistency-accept: if retry_count ≥ 1 and values are consistent, accept.

Call audit_results exactly once.\
"""


# ---------------------------------------------------------------------------
# BottleneckAnalystAgent
# ---------------------------------------------------------------------------

class BottleneckAnalystAgent(SubAgent):
    """Reasoning-only agent for roofline model and bottleneck classification."""

    REQUIRED_TOOLS: list[str] = [
        "list_skills",
        "read_skill",
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
        payload: dict = {"targets": step.targets}
        if step.retry_context:
            payload["retry_context"] = step.retry_context

        task = Task(
            id=step.id,
            type="bottleneck_analyst",
            description=step.task,
            payload=payload,
            constraints={},
        )
        ctx = AgentContext(
            task=task,
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
                user_message=step.instructions or None,
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
    agent_type="bottleneck_analyst",
    description=_DESCRIPTION,
    agent_class=BottleneckAnalystAgent,
    required_tools=BottleneckAnalystAgent.REQUIRED_TOOLS,
    critic_system_prompt=_CRITIC_SYSTEM_PROMPT,
))
