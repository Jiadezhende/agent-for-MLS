# Agent for MLS — Project Guide

## What this is

An autonomous agent that measures GPU hardware parameters (DRAM latency, boost clock, etc.) by writing and running CUDA C microbenchmarks. The LLM reasons about measurement strategy; the Executor compiles and runs all code.

Multiple targets are measured **in parallel** by a multi-agent pipeline: Planner → Worker Pool → Critic.

## How to run

```bash
cp .env.example .env   # fill in API_KEY and BASE_MODEL
python main.py --spec target_spec.json --output results.json --verbose
```

`target_spec.json` format: `{"targets": ["dram_latency_cycles", "actual_boost_clock_mhz"]}`

## Architecture

### Scheduling flow (single command → single pipeline → exit)

```text
python main.py --spec target_spec.json
      │
      ▼
  main.py                          load config, build LLMClient + Executor + Task
      │                            import agents → registers all agent plugins
      ▼
  Orchestrator.run()               pure-code scheduler (fixed 3-phase pipeline, NOT a loop)
      │
      ├─ Phase 1  PlannerAgent     ONE LLM call → [Step with worker agent_type]
      │           Planner decides: which agent type, how to group targets
      │
      ├─ Phase 2  ThreadPoolExecutor
      │           ├── Worker-0: looks up AgentDefinition by agent_type
      │           │     AgentLoop (ReAct loop, LLM-driven)
      │           │     ├── iter 1: LLM call → tool calls → Executor → observe
      │           │     ├── iter 2: LLM call → tool calls → Executor → observe
      │           │     └── submit_results → done
      │           └── Worker-1: AgentLoop (runs concurrently)
      │                 └── ...
      │
      └─ Phase 3  CriticAgent      reviews all WorkerOutputs
                                   accept / retry per step → retry loop until done
      │
      ▼
  main.py writes results.json + reasoning_log.json → exit
```

**Key distinction**:

- `Orchestrator` = fixed pipeline executor (like an Airflow DAG); no LLM, no looping
- `AgentLoop` = the actual ReAct loop (Reason → Act → Observe → repeat until `submit_results`)

### Package layers (three-tier, mirrors hello-agents)

```text
agents/
├── core/        Framework layer: Agent ABC, LLMClient, config, exceptions, loop, types
├── tools/       Tool system layer: ToolRegistry, CircuitBreaker, CUDA executor, builtin tools
└── agents/      Agent implementation layer: PlannerAgent, CriticAgent, HardwareProbeAgent
```

```text
main.py
  └── Orchestrator (orchestrator.py)         pure-code scheduler
        ├── PlannerAgent  (agents/agents/planner_agent.py)
        │   LLM: route + group targets → Step list
        ├── Worker Pool (ThreadPoolExecutor)
        │     ├── Worker-0: AgentContext + CircuitBreaker (isolated per worker)
        │     │     └── AgentLoop → ToolRegistry → Executor
        │     └── Worker-1: ...
        └── CriticAgent  (agents/agents/critic_agent.py)
            LLM: review WorkerOutputs → accept / retry
```

```text
Each Worker (AgentLoop):
  messages = [system_prompt, user_message(targets)]
  for iter in range(max_iterations):
      LLM call(messages, tools)
      → tool_calls → ToolRegistry.dispatch()
                          ├── CircuitBreaker check  (applies to ALL tools)
                          ├── JSON Schema validation
                          ├── fn(Executor / recording tool)
                          └── CircuitBreaker update
      → append tool results to messages
  until submit_results → exit loop
```

```text
Executor (agents/tools/cuda_executor.py)     shared across all workers (thread-safe)
  ├── run_cuda_probe    PRIMARY: compile + run .cu, stdout = measurement
  ├── profile_with_ncu  cross-verify with Nsight Compute counters
  ├── profile_with_nsys CPU-GPU timeline (operator analysis, not hardware probing)
  └── profile_with_torch PyTorch operator profiling (not hardware probing)
```

**Key rule**: hardware parameters (latency, clock) use `run_cuda_probe` with CUDA C. `profile_with_nsys` / `profile_with_torch` are for operator analysis, not hardware probing.

### Planner vs Orchestrator responsibility

| | Planner | Orchestrator |
| --- | --- | --- |
| Nature | LLM (decision maker) | Pure code (executor) |
| Decides | Which agent type handles each target; how to group | How to run: threads, timeout, retry |
| Inputs | targets list + registered agent types | [Step] from Planner |
| Outputs | [Step] with worker + hints | [WorkerOutput] |
| Fails | Falls back to 1:1 mapping | Collects partial results, continues |

### Agent type design principle

Agent type granularity is by **domain**, not by individual metric.
All GPU hardware metrics share the same tools, workflow, and domain knowledge —
they belong to one agent type (`hardware_probe`). Parallelism across metrics is
already handled at the **Step** level by the Planner.

| Agent Type | Tools | Domain | Status |
| --- | --- | --- | --- |
| `hardware_probe` | `run_cuda_probe`, `profile_with_ncu` | CUDA C microbenchmarks | implemented |
| `op_profiler` | `profile_with_nsys`, `profile_with_torch` | PyTorch operator timeline | future |
| `bottleneck_analyst` | reads upstream results | roofline model, arithmetic intensity | future |

## Key files

| File | Purpose |
| --- | --- |
| `agents/tools/cuda_executor.py` | Compilation, sandboxing, auto-detection, error classification; thread-safe |
| `orchestrator.py` | Drives Planner → Worker pool → Critic retry loop; manages `ThreadPoolExecutor` |
| `agents/agents/planner_agent.py` | One LLM call routes + groups targets into `Step` list |
| `agents/agents/critic_agent.py` | Reviews `WorkerOutput`s → accept / retry decisions |
| `agents/agents/hardware_probe_agent.py` | CUDA hardware probing agent + inlined prompts + plugin registration |
| `agents/core/loop.py` | ReAct loop per worker; `--verbose` prefixes with `[Wn]` |
| `agents/core/types.py` | `Task`, `Step`, `WorkerOutput`, `CriticDecision`, `AgentContext`, `Result`, `MemoryStore` |
| `agents/core/config.py` | Three dataclasses: `LLMConfig`, `AgentConfig`, `ExecutorConfig` |
| `agents/core/llm.py` | OpenAI SDK wrapper (the only file that imports openai) |
| `agents/core/exceptions.py` | Unified exception hierarchy: `AgentError`, `ExecutorError`, `CircuitOpenError` |
| `agents/tools/registry.py` | `ToolRegistry` class + circuit breaker enforcement (all tools) + `ToolFactory` |
| `agents/tools/circuit_breaker.py` | `CircuitBreaker` — universal, applies to every tool dispatch |
| `agents/tools/schemas.py` | OpenAI function-calling schemas for all 9 tools |
| `agents/tools/builtin/recording.py` | `record_measurement`, `flag_event`, `submit_results` |
| `agents/tools/builtin/skills.py` | `list_skills`, `read_skill` |
| `agents/_registry.py` | `AgentDefinition` dataclass + `register / get / all_definitions` |
| `agents/__init__.py` | Imports all built-in agent plugins to trigger registration |
| `skills/*.md` | Measurement strategy docs the LLM reads via `list_skills`/`read_skill` |

## Orchestrator flow

1. **PlannerAgent** — one LLM call with forced tool `assign_workers`; system prompt built dynamically from `AgentDefinition.planner_hints` of all registered types. Falls back to 1:1 on failure.
2. **Worker pool** — `ThreadPoolExecutor`; each worker looks up `AgentDefinition` by `step.worker`, instantiates `agent_def.agent_class`, builds tools via `ToolFactory`. Each worker has isolated `AgentContext` + `CircuitBreaker`.
3. **CriticAgent** — reviews all `WorkerOutput`s; issues `accept` / `retry` decisions; retry loop continues until no pending steps or max retries reached.

Output files:

- `results.json` — flat list of `Result` objects (includes `task_type`)
- `reasoning_log.json` — per-worker `reasoning_log`/`events` + Critic decisions

## Agent plugin system

Agent types register at startup via `agents/__init__.py`.
Each plugin calls `register(AgentDefinition(...))` from its module.

| Field | Used by |
| --- | --- |
| `agent_type` | `Step.worker` key; routing by Planner |
| `description` | Injected into Planner system prompt |
| `agent_class` | Instantiated per worker by Orchestrator |
| `required_tools` | `ToolFactory.build()` — which tools to inject |
| `planner_hints` | Grouping/routing rules shown to Planner LLM |
| `critic_system_prompt` | Critic LLM system prompt for reviewing this type's outputs |
| `critic_tool_schema` | Forced-tool JSON schema for the Critic call |

**To add a new agent type** (e.g. `op_profiler`):

1. Create `agents/agents/op_profiler_agent.py` — a single file containing:
   - The agent class (inherits `Agent`, implements `run(step, tools) -> WorkerOutput`)
   - All prompts inlined as module-level constants
   - `register(AgentDefinition(...))` call at the bottom

2. Add one line to `agents/__init__.py`:

   ```python
   import agents.agents.op_profiler_agent  # noqa: F401
   ```

No changes needed to Orchestrator, Planner, Critic, or AgentLoop.

## Real-time verbose output

`--verbose` enables per-worker prefixed output to stderr:

```text
[orchestrator] [plan] output: {"steps": [...]}
[orchestrator] [workers] start: {"pending": ["step-0", "step-1"]}
[W0] ── iter 1/40 ────────────────────────────────────────────
[W1] ── iter 1/40 ────────────────────────────────────────────
[W0]   call: run_cuda_probe  {"source":"<3842 chars>","probe_name":"dram_latency"}
[W1]   call: list_skills  (no args)
```

- Large-text arguments (`source`, `source_or_path`, `python_code`) are replaced with `<N chars>`.
- All prints use `flush=True` so output appears immediately even when piped (`2>&1 | tee run.log`).
- Each worker's prefix `[Wn]` makes interleaved parallel output attributable.

## Executor error handling

Every error dict has `error_class`:

- `"user_code"` — CUDA source is wrong; LLM should fix the kernel
- `"infrastructure"` — binary missing or env misconfigured; LLM should stop retrying
- `"timeout"` — reduce workload

On startup, `Executor.__init__` auto-detects GPU arch (`-arch=sm_NNN` via `nvidia-smi`), MSVC path (via `vswhere`), and ncu/nsys install paths. Results printed to stderr.

## Circuit breaker

The circuit breaker is **universal** — it applies to every tool, not just executor tools.

Three states per `(tool, error_kind)` pair:

| State | Condition | Behavior |
| --- | --- | --- |
| CLOSED | default | normal operation |
| OPEN | >= N consecutive failures | `dispatch()` returns `{"status": "circuit_open"}` |
| HALF-OPEN | OPEN for > `half_open_timeout_s` seconds | one probe allowed through; success → CLOSED, failure → OPEN (timer reset) |

Threshold: `AGENT_CB_THRESHOLD` env var (default 3).
Half-open timeout: `AGENT_HALF_OPEN_TIMEOUT_S` env var (default 60 s).
Each worker has its own `CircuitBreaker` — failures in one worker do not affect others.

## Config env vars

| Var | Default | Purpose |
| --- | --- | --- |
| `AGENT_WORKER_TIMEOUT_S` | `600` | Max wall time for the entire worker pool |
| `AGENT_HALF_OPEN_TIMEOUT_S` | `60` | Seconds before an open circuit allows a probe |

## GPU notes (RTX 5060 / Blackwell sm_120)

- `clock64()` requires `-arch=sm_120` (auto-detected; was the root cause of a 14-iteration failure loop)
- `cudaDeviceProp.clockRate` removed in CUDA 13 — do not use in skill templates
- Use `cudaEvent` for wall-clock timing, `clock64()` for cycle counting

## Tests

```bash
pytest tests/                         # all tests (some need nvcc)
pytest tests/test_circuit_breaker.py  # no GPU required
pytest tests/test_autodetect.py       # no GPU required
```
