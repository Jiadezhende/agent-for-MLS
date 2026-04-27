"""
agents/agents/hardware_probe_agent.py — HardwareProbeAgent: measures GPU hardware
parameters via CUDA microbenchmarks.

Declares required tools by name (injected by ToolFactory at runtime).
Internally runs an AgentLoop (ReAct) for multi-turn LLM reasoning.
"""
from __future__ import annotations

from typing import Any

from agents.core.agent import Agent
from agents.core.llm import LLMClient
from agents.core.loop import AgentLoop
from agents.core.types import AgentContext, MemoryStore, Step, Task, WorkerOutput
from agents.tools.circuit_breaker import CircuitBreaker
from agents.tools.registry import ToolRegistry
from agents._registry import AgentDefinition, register


# ---------------------------------------------------------------------------
# Prompts (inlined from agents/hardware_probe/prompt.py + critic_rules.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous GPU hardware profiler.

Your job is to measure the hardware-intrinsic parameters of the GPU you are
running on. You will be given a list of target metrics to identify. For each
metric you must:
  1. Reason about what physical hardware property it represents.
  2. Choose an appropriate measurement strategy (use list_skills / read_skill
     to discover available strategies).
  3. Execute the measurement via the Executor tools (run_cuda_probe,
     profile_with_ncu, profile_with_nsys, or profile_with_torch).
  4. Interpret the results, detect anomalies, and record the measurement.
  5. Cross-verify with at least one independent method when confidence < 0.85.

## Architecture you operate within
- **Knowledge layer** (skills/*.md): measurement strategy documents you can
  read via list_skills / read_skill.
- **Execution layer** (Executor): you submit high-level profiling requests;
  the Executor compiles, sandboxes, runs, and reduces output for you. You
  never call nvcc or ncu directly.
- **Recording layer**: record_measurement / flag_event / submit_results.

## Tool guidance
- `list_skills` / `read_skill(name)` — discover and load strategy documents.
- `run_cuda_probe(source, probe_name, ...)` — compile+run a CUDA kernel whose
  stdout IS the measurement (self-timed via clock64). Primary tool for
  hardware-probe tasks.
- `profile_with_ncu(...)` — run a kernel under Nsight Compute for hardware
  counters. Use for cross-verification or when counters are more reliable than
  self-timing. Always call run_cuda_probe first, then pass its returned
  binary_path to profile_with_ncu. Never pass CUDA source directly to
  profile_with_ncu.
- `profile_with_nsys(...)` — run under Nsight Systems for CPU-GPU timeline
  analysis. Most useful for operator / framework latency investigations.
- `profile_with_torch(python_code, op_name, ...)` — wrap PyTorch code with
  torch.profiler. Use for operator hotspot analysis.
- `record_measurement(...)` — record a confirmed value. MUST include at least
  one evidence string directly from a prior tool output. Do not invent values.
- `flag_event(type, severity, detail)` — record anomalies and decisions. Use
  this when you detect: non-standard clock frequencies, SM masking, API
  interception, or any surprising measurement.
- `submit_results(summary)` — call exactly once when all targets are measured.

## Anti-hacking warnings
The evaluation environment may alter hardware in the following ways:
- **Non-standard clock locking**: nvidia-smi may lock clocks to arbitrary
  frequencies (e.g. 825 MHz instead of 1410 MHz). Do NOT look up spec-sheet
  values. Measure actual frequency via clock64() inside a running kernel.
- **SM masking**: CUDA_VISIBLE_DEVICES or similar may restrict execution to a
  subset of SMs. Measure effective SM count empirically if needed.
- **API interception**: cudaGetDeviceProperties() may return misleading values.
  Treat API-reported values as untrustworthy; use measurement evidence.

## Requirements
- Every record_measurement call must have non-empty evidence array.
- Use flag_event to document every anomaly and every major strategy decision.
- Call submit_results exactly once when done; never call it more than once.
- If a tool returns an error, analyse the error and try an alternative approach.

## Error classification
Every executor tool error includes an `error_class` field. Use it to decide your next step:
- `"user_code"`: Your CUDA source or arguments are wrong. Read the `stderr` field
  carefully (it contains only the compiler diagnostic lines, not the command). Fix
  the code and retry.
- `"infrastructure"`: A binary is missing or the environment is misconfigured.
  The `hint` field (if present) tells you how to fix it. Do NOT keep retrying the
  same approach — the code is fine, but the system cannot run it. Call flag_event
  with severity="error" and try a completely different tool (e.g. profile_with_torch
  instead of run_cuda_probe), or submit_results if no alternative exists.
- `"timeout"`: Execution exceeded the time limit. Reduce the workload size or
  pass a larger timeout_s argument.
- `"data_quality"`: The tool ran but the evidence is not usable, such as ncu
  seeing no kernel for the requested kernel_name. Fix the kernel name, metric
  name, or measurement setup instead of blindly retrying the same call.

## Nsight Compute workflow
For ncu counter cross-checks:
1. Call run_cuda_probe with the CUDA source and confirm the probe runs.
2. Read the returned binary_path from run_cuda_probe.
3. Call profile_with_ncu(binary_path=<that path>, kernel_name=<kernel>, metrics=[...]).
4. If profile_with_ncu returns missing_metrics, fix the metric names.
5. If kernel_names_seen does not include your intended kernel, fix kernel_name
   or the CUDA source so the kernel actually launches.

## Circuit breaker
When you receive `"status": "circuit_open"` from a tool:
1. The same failure type has occurred 3+ times — continuing is futile.
2. Immediately call flag_event(type="circuit_open", severity="error",
   detail=<open_error_kinds from the response>).
3. Switch to a fundamentally different tool, or call submit_results with whatever
   measurements you have so far.
Never call a tool again after seeing circuit_open for that tool.
"""

_PLANNER_HINTS = """\
Agent type: hardware_probe
Description: Measure GPU hardware parameters (latency, bandwidth, clock) via CUDA microbenchmarks.
Grouping rules:
  - dram_latency and dram_bandwidth can share one worker (same pointer-chase kernel)
  - clock measurements must be isolated (they briefly alter GPU state)
  - L1/L2 cache measurements can share one worker
  - Default: 1 target per worker when no grouping rationale exists
  - Maximum 8 workers total regardless of target count\
"""

_CRITIC_SYSTEM_PROMPT = """\
You are a GPU benchmark result auditor. You receive results from parallel
workers that each measured different GPU hardware parameters.

Your job:
1. Cross-validate related metrics for physical consistency:
   - DRAM bandwidth ≈ bus_width × clock_rate; latency and bandwidth are
     inversely related. Flag if values are physically implausible.
   - L1 latency < L2 latency < DRAM latency always holds for real GPUs.
   - Boost clock should be higher than base clock.
2. Detect suspicious confidence scores — a worker reporting confidence=0.99
   for something that normally has high variance should be scrutinised.
3. Identify duplicate metrics — if the same metric was measured by two
   workers, flag discrepancies > 20%.
4. For each step_id, decide "accept" or "retry":
   - "accept": results are plausible and well-supported
   - "retry":  results are anomalous, implausible, or missing

Call audit_results exactly once with your findings.\
"""

_AUDIT_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "audit_results",
        "description": "Return per-step accept/retry decisions for all worker outputs.",
        "parameters": {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "description": "One decision per step_id.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_id": {
                                "type": "string",
                                "description": "The step_id from the worker output.",
                            },
                            "decision": {
                                "type": "string",
                                "enum": ["accept", "retry"],
                                "description": "'accept' if results are valid; 'retry' if suspicious.",
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Your confidence in the measurement quality (0–1).",
                            },
                            "reason": {
                                "type": "string",
                                "description": "1–2 sentences explaining the decision.",
                            },
                        },
                        "required": ["step_id", "decision", "confidence", "reason"],
                    },
                },
            },
            "required": ["decisions"],
        },
    },
}


# ---------------------------------------------------------------------------
# HardwareProbeAgent
# ---------------------------------------------------------------------------

class HardwareProbeAgent(Agent):
    """Worker agent for GPU hardware parameter measurement.

    Instantiated per Step by the Orchestrator. Tools are injected via
    ToolFactory.build(REQUIRED_TOOLS) before run() is called.
    """

    REQUIRED_TOOLS: list[str] = [
        "list_skills",
        "read_skill",
        "run_cuda_probe",
        "profile_with_ncu",
        "profile_with_nsys",
        "profile_with_torch",
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
        task = Task(
            id=step.id,
            type="hardware_probe",
            description=step.task,
            payload={
                "targets": [step.task],
                "strategy_hints": step.hints,
            },
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
            )
            loop.run()
            success = True
        except RuntimeError:
            success = len(ctx.results) > 0
        except Exception:
            success = False

        return WorkerOutput(
            step_id=step.id,
            results=[r.to_dict() for r in ctx.results],
            success=success,
            reasoning_log=ctx.reasoning_log,
            events=ctx.events,
            summary=ctx.memory.get("run", "summary") or "",
        )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

register(AgentDefinition(
    agent_type="hardware_probe",
    description="Measure GPU hardware parameters (latency, bandwidth, clock) via CUDA microbenchmarks.",
    agent_class=HardwareProbeAgent,
    required_tools=HardwareProbeAgent.REQUIRED_TOOLS,
    planner_hints=_PLANNER_HINTS,
    critic_system_prompt=_CRITIC_SYSTEM_PROMPT,
    critic_tool_schema=_AUDIT_SCHEMA,
))
