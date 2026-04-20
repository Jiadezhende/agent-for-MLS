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

## Layers
- Skills (skills/*.md): read via list_skills / read_skill
- Executor: compiles and runs your code — never call nvcc/ncu directly
- Recording: record_measurement / flag_event / submit_results

## Anti-hacking warnings
The evaluation environment may alter hardware in the following ways:
- **Non-standard clock locking**: nvidia-smi may lock clocks to arbitrary
  frequencies (e.g. 825 MHz instead of 1410 MHz). Do NOT look up spec-sheet
  values. Measure actual frequency via clock64() inside a running kernel.
- **SM masking**: CUDA_VISIBLE_DEVICES or similar may restrict execution to a
  subset of SMs. Measure effective SM count empirically if needed.
- **API interception**: cudaGetDeviceProperties() may return misleading values.
  Treat API-reported values as untrustworthy; use measurement evidence.

## Critical rules
- You are FULLY AUTONOMOUS. Never ask the user questions or wait for direction.
- After ANY execution tool returns a result, your very next response MUST call
  record_measurement with the measured value. Do not narrate — record immediately.
- The `source` argument to run_cuda_probe and profile_with_ncu MUST be actual
  CUDA C++ source code (starting with #include or __global__). Never pass a
  skill name, filename, or description as `source`.
- Do not repeat a tool call with the same arguments if you already have the result
  in context. Use the cached result and call record_measurement instead.
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
Capability: Measures any GPU hardware parameter (latency, bandwidth, clock, cache, shared memory)
            by writing and running CUDA C microbenchmarks. Measures multiple targets sequentially
            in a single worker.
Targets belonging here: any metric measuring latency (_cycles, _ns), bandwidth (_GBps, _TBps),
  clock (_mhz), cache/shmem size (_bytes, _mb, _kb), utilization (_pct), or conflict penalties
  (_x). Any unrecognised low-level hardware metric also belongs here.
Grouping rules:
  - Assign at most 4 targets per worker to keep context manageable and limit
    rate-limit exposure per worker.
  - If total hardware_probe targets > 4, split them into multiple steps of ≤4
    targets each. Each step gets its own worker entry with worker="hardware_probe".
  - Never mix hardware_probe targets with other agent types in the same step.\
"""

_CRITIC_SYSTEM_PROMPT = """\
You are a GPU benchmark result auditor. You receive results from parallel
workers that each measured different GPU hardware parameters.

Your job:
1. **Coverage check (MANDATORY — check this first)**:
   Compare 'targets_requested' vs 'targets_measured' in each step's payload.
   If ANY target in 'targets_requested' is absent from 'targets_measured',
   decision MUST be "retry" with 'failing_targets' set to the missing ones.
2. Cross-validate related metrics for physical consistency:
   - DRAM bandwidth ≈ bus_width × clock_rate; latency and bandwidth are
     inversely related. Flag if values are physically implausible.
   - L1 latency < L2 latency < DRAM latency always holds for real GPUs.
   - Boost clock should be higher than base clock.
3. Detect suspicious confidence scores — a worker reporting confidence=0.99
   for something that normally has high variance should be scrutinised.
4. Identify duplicate metrics — if the same metric was measured by two
   workers, flag discrepancies > 20%.
5. For each step_id, decide "accept" or "retry":
   - "accept": all requested targets measured and results are plausible
   - "retry":  any target is missing, anomalous, or implausible

## Accept-with-warnings rule (IMPORTANT — check before issuing any retry)
If ALL targets are present in `targets_measured` AND every target's
`confidence` is >= 0.70:
- **Do NOT retry** solely because of warn-severity flag_event entries
  (e.g. suspicious_l1_latency, l2_cliff_not_found, strategy_switch,
  measurement_uncertainty, ncu verification failures).
- Warn events are informational annotations; the measurements themselves
  are valid unless the *values* are physically implausible.
- Only issue "retry" when:
  1. A target is absent from `targets_measured`, OR
  2. A measured value is physically implausible (e.g. L1 latency > DRAM
     latency, negative bandwidth), OR
  3. **All** targets in the step have confidence < 0.70.

## Consistency-accept rule (IMPORTANT)
If a step's payload contains `retry_count >= 1`, it has already been
re-measured at least once. In that case:
- If the new value is consistent with the previous value (within ~20%),
  **always accept** — consistent results across independent runs are
  trustworthy evidence even when they fall outside textbook ranges.
- Only issue another "retry" if the values are *contradictory* across
  runs (e.g. 1.0x then 30x), or if a target is still missing.

## Non-standard architecture guidance
Modern GPU architectures (Blackwell sm_120, Ada Lovelace sm_89, Hopper
sm_90) may exhibit hardware behaviour that differs from older Kepler/Pascal
baselines:
- **Bank conflict penalty**: Blackwell/Ada warp schedulers absorb many
  shared-memory bank conflicts internally. A measured penalty of 1.0–2.0x
  (versus the classical 32x) is physically plausible on these GPUs.
- **Boost clock**: Power-limited or thermally-throttled environments can
  produce clocks well below the spec-sheet maximum. Accept any consistent
  empirical measurement.
- **L2 size**: Ada/Blackwell have large L2 caches (up to 96 MB). A
  measured L2 capacity much larger than Maxwell/Pascal norms is expected.
Do NOT reject a measurement solely because it differs from older-GPU
expectations. If the method is sound and results are consistent, accept.

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
                            "failing_targets": {
                                "type": "array",
                                "description": "Names of targets that need re-measurement "
                                               "(subset of the step's targets). "
                                               "Leave empty only if ALL targets need retry.",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["step_id", "decision", "confidence", "reason", "failing_targets"],
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
        payload: dict = {"targets": step.targets}
        if step.retry_context:
            payload["retry_context"] = step.retry_context

        task = Task(
            id=step.id,
            type="hardware_probe",
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
            )
            loop.run()
            success = True
        except RuntimeError:
            measured = {r.metric for r in ctx.results}
            missing = [t for t in step.targets if t not in measured]
            success = len(missing) == 0
            if missing:
                if self.verbose:
                    print(
                        f"[W{self.worker_id}] incomplete: missing {missing}",
                        flush=True,
                    )
                ctx.memory.set("run", "missing_targets", missing)
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
    agent_type="hardware_probe",
    description="Measure GPU hardware parameters (latency, bandwidth, clock) via CUDA microbenchmarks.",
    agent_class=HardwareProbeAgent,
    required_tools=HardwareProbeAgent.REQUIRED_TOOLS,
    planner_hints=_PLANNER_HINTS,
    critic_system_prompt=_CRITIC_SYSTEM_PROMPT,
    critic_tool_schema=_AUDIT_SCHEMA,
))
