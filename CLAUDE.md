# Agent for MLS — Project Guide

## What this is

An autonomous multi-agent system for **GPU kernel optimization**. Given an operator
specification (formula, tensor shapes, success criteria), the system:

1. Characterizes the hardware (DRAM bandwidth, clock, cache hierarchy) via CUDA C microbenchmarks
2. Profiles the baseline operator via Nsight / Torch Profiler
3. Identifies the bottleneck (memory-bound vs compute-bound, roofline model)
4. Implements and validates an optimized kernel
5. Reports structured results with engineering reasoning logs

Operator specifications live in `skills/operators/<name>.md` — the system is extensible to any operator without code changes.

## How to run

```bash
cp .env.example .env   # fill in API_KEY and BASE_MODEL
python main.py --spec target_spec.json --output results.json --verbose
```

`target_spec.json` format:

```json
{"operator": "lora_matmul", "targets": ["dram_bandwidth_gbps", "boost_clock_mhz"]}
```

## Architecture

### Scheduling flow

```text
python main.py --spec target_spec.json
      │
      ▼
  main.py                         load config, build LLMClient + Executor + Task
      │                           import agents → registers all agent plugins
      ▼
  Orchestrator.run()              state machine driver (no LLM, no looping of its own)
      │
      ├─ phase=planning/revising
      │     PlannerAgent (ReAct AgentLoop — optimization coordinator)
      │       Coordinator tools: read_skill, run_subagent, run_subagent_parallel,
      │                          mark_ready_for_critic, flag_event
      │       Workflow:
      │         0. read_skill("operators/<operator>")  → load spec + success criteria
      │         1. run_subagent_parallel([             → concurrent (LLM overlap)
      │               hardware_probe([...]),           → GPU _gpu_lock serializes execution
      │               op_profiler([...])               → same _gpu_lock
      │            ])
      │         2. run_subagent("bottleneck_analyst")  → depends on step 1 results
      │         3. run_subagent("kernel_optimizer")    → depends on step 2 conclusion
      │         4. mark_ready_for_critic(summary)      → raises _Terminated, exits loop
      │
      ├─ phase=ready_for_critic
      │     CriticAgent (single LLM call, forced tool audit_results)
      │       Reviews ctx.job_history against operator's Success Criteria
      │       accept  → state.accepted = True, done
      │       retry   → build critic_feedback → re-enter Planner (REVISING mode)
      │
      └─ phase=accepted / failed → collect results → exit
```

**Key distinction**:

- `Orchestrator` = pure-code state machine; drives the planning ↔ critic cycle
- `PlannerAgent` = ReAct AgentLoop; autonomous coordinator; accumulates results in `ctx.job_history`
- `AgentLoop` (inside each subagent) = the actual ReAct loop for worker agents

### Concurrency model

GPU execution is **always serialized** by `Executor._gpu_lock` regardless of how many parallel
subagents are running. `run_subagent_parallel` achieves wall-clock speedup by overlapping
**LLM reasoning** between subagents while one is waiting for the GPU lock.

```text
time →
  hardware_probe:  [LLM call] → [GPU: run_cuda_probe] → [LLM call] → [GPU: profile_with_ncu]
  op_profiler:                  [LLM call]             → wait_lock  → [GPU: profile_with_nsys]
                                 ↑ overlap here
```

Logically dependent stages (e.g. bottleneck_analyst depends on hardware_probe results)
must still use sequential `run_subagent` calls — enforced by Coordinator prompt, not by code.

### Package layers

```text
agents/
├── core/        Framework: Agent ABC, LLMClient, config, exceptions, loop, types
├── tools/       Tool system: ToolRegistry, CircuitBreaker, CUDA executor, builtin tools
└── agents/      Agent implementations: PlannerAgent, CriticAgent, HardwareProbeAgent
```

### Two-tier tool system

**Coordinator tools** (registered in PlannerAgent's ToolRegistry):

| Tool | Purpose |
| --- | --- |
| `read_skill(name)` | Load operator spec or strategy doc |
| `list_skills` | List available skill files |
| `run_subagent(agent_type, targets)` | Run one subagent synchronously |
| `run_subagent_parallel(calls)` | Run multiple independent subagents concurrently |
| `mark_ready_for_critic(summary)` | Signal completion; raises `_Terminated` |
| `flag_event(type, severity, detail)` | Log strategy decision or anomaly |

**Worker tools** (registered per-subagent via `ToolFactory.build(required_tools)`):

| Tool | Agent types | Purpose |
| --- | --- | --- |
| `run_cuda_probe(source, probe_name)` | hardware_probe | Compile + run CUDA C microbenchmark |
| `profile_with_ncu(...)` | hardware_probe | Nsight Compute hardware counters |
| `profile_with_nsys(...)` | op_profiler | CPU-GPU timeline, launch overhead |
| `profile_with_torch(python_code, op_name)` | op_profiler | Torch Profiler operator stats |
| `record_measurement(metric, value, ...)` | all workers | Write result to WorkerOutput |
| `flag_event(...)` | all workers | Log anomaly / decision |
| `submit_results(summary)` | all workers | Finalize; raises `_Terminated` |
| `read_skill(name)` | all workers | Load strategy docs on demand |
| `list_skills` | all workers | List available docs |

### Planner vs Orchestrator responsibility

| | PlannerAgent | Orchestrator |
| --- | --- | --- |
| Nature | LLM (ReAct loop coordinator) | Pure code (state machine) |
| Decides | Which subagents to run; when to use parallel; when all criteria are met | How many critic cycles; whether to accept or retry |
| Inputs | operator spec + critic_feedback (REVISING mode) | PlannerAgent ctx + CriticAgent decisions |
| Outputs | AgentContext with `ctx.job_history` | `_ExecutionState` (phase, accepted, planner_ctx) |
| Fails | Loop exhausted → partial job_history returned | Accepts partial results after max_critic_cycles |

### Agent type design principle

Agent type granularity is by **domain**, not by individual metric. Each type owns one tool set
and one domain of knowledge; the Planner coordinator decides sequencing.

| Agent Type | Tools | Domain | Status |
| --- | --- | --- | --- |
| `hardware_probe` | `run_cuda_probe`, `profile_with_ncu` | CUDA C microbenchmarks | implemented |
| `op_profiler` | `profile_with_nsys`, `profile_with_torch` | PyTorch operator timeline | future |
| `bottleneck_analyst` | reads upstream results | roofline model, arithmetic intensity | future |
| `kernel_optimizer` | `run_cuda_probe` | write + validate optimized kernel | future |

## Key files

| File | Purpose |
| --- | --- |
| `agents/tools/cuda_executor.py` | Compile, run, profile; `_gpu_lock` serializes all GPU execution; thread-safe |
| `orchestrator.py` | State machine: planning/revising → ready_for_critic → accepted/failed |
| `agents/agents/planner_agent.py` | ReAct coordinator; builds ToolRegistry with subagent tools |
| `agents/agents/critic_agent.py` | Single LLM call; reviews job_history; accept/retry per step |
| `agents/agents/hardware_probe_agent.py` | CUDA hardware probing agent + inlined prompts + plugin registration |
| `agents/tools/builtin/subagent.py` | `RunSubagentTool`, `RunSubagentParallelTool`, `MarkReadyForCriticTool` |
| `agents/core/loop.py` | ReAct loop (used by both Planner and worker subagents) |
| `agents/core/types.py` | `Task`, `Step`, `WorkerOutput`, `CriticDecision`, `AgentContext`, `MemoryStore`, `RunContext` |
| `agents/core/config.py` | `LLMConfig`, `AgentConfig`, `ExecutorConfig` |
| `agents/core/llm.py` | OpenAI SDK wrapper (only file that imports openai) |
| `agents/tools/registry.py` | `ToolRegistry` + CircuitBreaker dispatch + `ToolFactory` |
| `agents/tools/circuit_breaker.py` | Universal CircuitBreaker; per-subagent isolation |
| `agents/tools/builtin/recording.py` | `record_measurement`, `flag_event`, `submit_results` |
| `agents/tools/builtin/skills.py` | `list_skills`, `read_skill` (supports `operators/` subdir paths) |
| `agents/_registry.py` | `AgentDefinition` dataclass + `register / get / all_definitions` |
| `agents/__init__.py` | Imports all built-in agent plugins to trigger registration |
| `skills/operators/lora_matmul.md` | First operator skill (LoRA-fused MATMUL) |
| `skills/operators/_operator_template.md` | Template for adding new operators |
| `skills/*.md` | Strategy docs read by worker agents via `read_skill` |

## Orchestrator state machine

```python
for cycle in range(max_critic_cycles):          # default 3
    planner_ctx = _run_planner_loop(spec, critic_feedback)   # phase=planning/revising
    worker_outputs = _collect_outputs(planner_ctx)           # reconstruct from job_history
    decisions = critic.run(worker_outputs, system_prompt_override=task_critic_prompt)
    if all accepted:
        state.accepted = True; break
    critic_feedback = _build_feedback(failing_decisions)     # phase=revising
else:
    state.accepted = True   # accept after hard limit
```

`_ExecutionState` fields: `planner_ctx`, `accepted: bool`, `phase: str`.

## Operator skill system

Operator specs live in `skills/operators/<name>.md`. The Planner reads them via
`read_skill("operators/lora_matmul")` at runtime. The Orchestrator embeds the skill
content into the Critic's system prompt for task-level evaluation.

**Standard sections** (defined in `_operator_template.md`):

- **Formula** — math definition and tensor shapes
- **Input Specification** — dtypes, variable ranges, storage format
- **Optimization Goal** — minimize latency / maximize throughput
- **Required Hardware Measurements** — what hardware_probe must collect
- **Baseline Profiling** — how op_profiler measures the reference implementation
- **Bottleneck Analysis** — roofline model parameters
- **Success Criteria** — what Critic checks for accept/retry
- **Potential Strategies** — memory-bound vs compute-bound optimization paths

**To add a new operator**: create `skills/operators/<name>.md` following the template.
No code changes needed.

## Agent plugin system

Agent types register at startup via `agents/__init__.py`. Each plugin calls
`register(AgentDefinition(...))` from its module.

| Field | Used by |
| --- | --- |
| `agent_type` | Routing key; appears in run_subagent calls and job_history |
| `description` | Capability + routing/grouping rules injected into Coordinator system prompt |
| `agent_class` | Instantiated per subagent call by `RunSubagentTool._execute_one()` |
| `required_tools` | `ToolFactory.build()` — tools injected into the subagent's ToolRegistry |
| `critic_system_prompt` | Per-type Critic prompt (used when no task-level override) |
| `critic_tool_schema` | Forced-tool schema for the Critic LLM call |

**To add a new agent type** (e.g. `op_profiler`):

1. Create `agents/agents/op_profiler_agent.py` — agent class + inlined prompts + `register(AgentDefinition(...))`
2. Add one line to `agents/__init__.py`:

   ```python
   import agents.agents.op_profiler_agent  # noqa: F401
   ```

No changes to Orchestrator, PlannerAgent, CriticAgent, or AgentLoop.

## Real-time verbose output

`--verbose` enables prefixed output to stderr:

```text
[orchestrator] Cycle 1/3 phase=planning
[planner] ── iter 1/40 ────────────────────────────────────────
[planner]   call: read_skill  {"name":"operators/lora_matmul"}
[planner]   call: run_subagent_parallel  {"calls":[...]}
[W sub_hardware_probe] ── iter 1/40 ────────────────────────────
[W sub_hardware_probe]   call: run_cuda_probe  {"source":"<3842 chars>","probe_name":"dram_bw"}
[orchestrator] Critic decisions: step_abc=accept
[orchestrator] All accepted.
```

- Large-text args (`source`, `source_or_path`, `python_code`) replaced with `<N chars>`
- All prints use `flush=True` for real-time piped output

## Executor error handling

Every error dict has `error_class`:

- `"user_code"` — CUDA source is wrong; LLM should fix the kernel
- `"infrastructure"` — binary missing or env misconfigured; LLM should stop retrying
- `"timeout"` — reduce workload

`_gpu_lock` guarantees all GPU execution is exclusive — CUDA probes, NCU profiling,
and Nsight Systems profiling never run concurrently, ensuring measurement isolation.

## Circuit breaker

Universal — applies to every tool dispatch, including coordinator tools.

| State | Condition | Behavior |
| --- | --- | --- |
| CLOSED | default | normal operation |
| OPEN | ≥ N consecutive failures | returns `{"status": "circuit_open"}` immediately |
| HALF-OPEN | OPEN for > `half_open_timeout_s` s | one probe allowed through |

Each subagent has its own `CircuitBreaker` (isolated per `AgentContext`).
Env vars: `AGENT_CB_THRESHOLD` (default 3), `AGENT_HALF_OPEN_TIMEOUT_S` (default 60 s).

## Config env vars

| Var | Default | Purpose |
| --- | --- | --- |
| `AGENT_MAX_ITERATIONS` | `40` | Max LLM round-trips per ReAct loop (Planner or worker) |
| `AGENT_MAX_CRITIC_CYCLES` | `3` | Max Planner ↔ Critic retry cycles |
| `AGENT_WORKER_TIMEOUT_S` | `600` | (legacy; no longer applies — subagents are tool calls) |
| `AGENT_HALF_OPEN_TIMEOUT_S` | `60` | Seconds before an open circuit allows a probe |

## GPU notes (RTX 5060 / Blackwell sm_120)

- `clock64()` requires `-arch=sm_120` (auto-detected; was the root cause of a 14-iteration failure loop)
- `cudaDeviceProp.clockRate` removed in CUDA 13 — do not use in skill templates
- Use `cudaEvent` for wall-clock timing, `clock64()` for cycle counting

## Tests

```bash
pytest tests/                          # all tests (some need nvcc)
pytest tests/test_circuit_breaker.py   # no GPU required
pytest tests/test_planner_loop.py      # no GPU required
pytest tests/test_orchestrator_timeout.py  # no GPU required
```
