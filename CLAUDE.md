# Agent for MLS — Project Guide

## What this is

An autonomous agent that measures GPU hardware parameters (DRAM latency, boost clock, etc.) by writing and running CUDA C microbenchmarks. The LLM reasons about measurement strategy; the Executor compiles and runs all code.

Multiple targets are measured **in parallel** by a multi-agent pipeline: Planner → Worker Pool → Critic.

## How to run

```bash
cp .env.example .env   # fill in OPENAI_API_KEY and AGENT_LLM_MODEL
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
      │                            import agent.tasks → registers all task plugins
      ▼
  Orchestrator.run()               pure-code scheduler (fixed 4-phase pipeline, NOT a loop)
      │
      ├─ Phase 1  plan_tasks()     ONE LLM call → [WorkerSpec with agent_type]
      │           Planner decides: which agent type, how to group targets
      │
      ├─ Phase 2  ThreadPoolExecutor
      │           ├── Worker-0: looks up TaskDefinition by agent_type
      │           │     AgentLoop (ReAct loop, LLM-driven)
      │           │     ├── iter 1: LLM call → tool calls → Executor → observe
      │           │     ├── iter 2: LLM call → tool calls → Executor → observe
      │           │     └── submit_results → done
      │           └── Worker-1: AgentLoop (runs concurrently)
      │                 └── ...
      │
      ├─ Phase 3  Aggregate        merge all ctx.results → flat list[Result]
      │                            each Result carries task_type
      │
      └─ Phase 4  critique_results group by task_type → one LLM call per group
      │
      ▼
  main.py writes results.json + reasoning_log.json → exit
```

**Key distinction**:

- `Orchestrator` = fixed pipeline executor (like an Airflow DAG); no LLM, no looping
- `AgentLoop` = the actual ReAct loop (Reason → Act → Observe → repeat until `submit_results`)

### Component layers

```text
main.py
  └── Orchestrator (agent/orchestrator.py)   pure-code scheduler
        ├── Planner  (agent/planner.py)       LLM: route + group targets → WorkerSpec list
        ├── Worker Pool (ThreadPoolExecutor)
        │     ├── Worker-0: AgentContext + CircuitBreaker (isolated per worker)
        │     │     └── AgentLoop → ToolRegistry → Executor
        │     └── Worker-1: ...
        └── Critic   (agent/critic.py)        LLM: per-task cross-validate → confidence adjustments
```

```text
Each Worker (AgentLoop):
  messages = [system_prompt, user_message(targets)]
  for iter in range(max_iterations):
      LLM call(messages, tools)
      → tool_calls → ToolRegistry.dispatch()
                          ├── CircuitBreaker check
                          ├── JSON Schema validation
                          ├── fn(Executor / recording tool)
                          └── CircuitBreaker update
      → append tool results to messages
  until submit_results → exit loop
```

```text
Executor (executor.py)               shared across all workers (thread-safe)
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
| Inputs | targets list + registered agent types | [WorkerSpec] from Planner |
| Outputs | [WorkerSpec] with agent_type + targets + hints | [WorkerResult] |
| Fails | Falls back to 1:1 mapping | Collects partial results, continues |

### Task type design principle

Task type granularity is by **domain**, not by individual metric.
All GPU hardware metrics share the same tools, workflow, and domain knowledge —
they belong to one task type (`hardware_probe`). Parallelism across metrics is
already handled at the **WorkerSpec** level by the Planner.

| Task Type | Tools | Domain | Status |
| --- | --- | --- | --- |
| `hardware_probe` | `run_cuda_probe`, `profile_with_ncu` | CUDA C microbenchmarks | implemented (plugin) |
| `op_profiler` | `profile_with_nsys`, `profile_with_torch` | PyTorch operator timeline | future |
| `bottleneck_analyst` | reads upstream results | roofline model, arithmetic intensity | future |

These three are genuinely distinct task types because they differ in tools, workflow,
and Critic validation rules. Splitting `dram_latency` and `boost_clock` into separate
task types would only produce near-identical prompts — the wrong level of abstraction.

## Key files

| File | Purpose |
| --- | --- |
| `executor.py` | Compilation, sandboxing, auto-detection, error classification; thread-safe |
| `agent/orchestrator.py` | Drives Planner → Worker pool → Critic; manages `ThreadPoolExecutor`; receives `task_registry` |
| `agent/planner.py` | `plan_tasks(task_registry)` — one LLM call routes + groups targets into `WorkerSpec`s |
| `agent/critic.py` | `critique_results(task_registry)` — groups by `task_type`, runs per-task critic, merges |
| `agent/loop.py` | ReAct loop per worker; accepts `system_prompt` param; `--verbose` prefixes with `[Wn]` |
| `agent/types.py` | `AgentContext`, `CircuitBreaker`, `Result` (with `task_type`), `WorkerSpec` (with `agent_type`), `WorkerResult`, `CritiqueResult` |
| `agent/tool_registry.py` | `ToolRegistry` class + circuit breaker enforcement |
| `agent/prompts.py` | Generic `build_user_message()` only; system prompts live in task plugins |
| `agent/tasks/_registry.py` | `TaskDefinition` dataclass + `register / get / all_definitions` |
| `agent/tasks/__init__.py` | Imports all built-in task plugins to trigger registration |
| `agent/tasks/hardware_probe/` | `hardware_probe` plugin: `prompt.py`, `tools.py`, `critic_rules.py`, `__init__.py` |
| `config.py` | Three dataclasses: `LLMConfig`, `AgentConfig`, `ExecutorConfig` |
| `skills/*.md` | Measurement strategy docs the LLM reads via `list_skills`/`read_skill` |
| `tools/recording.py` | `record_measurement` (sets `Result.task_type` from `ctx.task.type`), `flag_event`, `submit_results` |

## Orchestrator flow

1. **Planner** — one LLM call with forced tool `assign_workers`; system prompt built dynamically from `TaskDefinition.planner_hints` of all registered types. Each assignment includes `agent_type`. Falls back to 1:1 on failure.
2. **Worker pool** — `ThreadPoolExecutor`; each worker looks up `TaskDefinition` by `spec.agent_type`, gets task-specific `system_prompt` and `build_registry()`. Each worker has isolated `AgentContext` + `CircuitBreaker`.
3. **Aggregate** — results from all worker contexts collected into a flat list; each `Result` carries `task_type`.
4. **Critic** — results grouped by `task_type`; one LLM call per group using that task's `critic_system_prompt` + `critic_tool_schema`; `CritiqueResult`s merged. No-op on failure.

Output files:

- `results.json` — flat list of `Result` objects (post-critique confidence, includes `task_type`)
- `reasoning_log.json` — per-worker `reasoning_log`/`events`/`job_history` + merged critique block

## Task plugin system

Task types are registered at startup via `agent/tasks/__init__.py`.
Each plugin provides a `TaskDefinition` with:

| Field | Used by |
| --- | --- |
| `task_type` | `WorkerSpec.agent_type` key; `Result.task_type` tag |
| `description` | Injected into Planner system prompt |
| `system_prompt` | Passed to `AgentLoop` for workers of this type |
| `build_registry` | Called per-worker to build tool set |
| `planner_hints` | Grouping/routing rules in Planner system prompt |
| `critic_system_prompt` | Critic LLM system prompt for this task's results |
| `critic_tool_schema` | Forced tool schema for Critic call |

**To add a new task type** (e.g. `op_profiler`):

```text
agent/tasks/
  op_profiler/
    __init__.py        # calls register(TaskDefinition(...))
    prompt.py          # SYSTEM_PROMPT + PLANNER_HINTS
    tools.py           # build_registry(executor) -> ToolRegistry
    critic_rules.py    # CRITIC_SYSTEM_PROMPT + AUDIT_SCHEMA
```

Then add one line to `agent/tasks/__init__.py`:

```python
import agent.tasks.op_profiler  # noqa: F401
```

No changes needed to Orchestrator, Planner, Critic, or AgentLoop.

## Real-time verbose output

`--verbose` enables per-worker prefixed output to stderr:

```text
[orchestrator] Phase 1: Planning worker assignments …
[orchestrator] Planner produced 2 worker(s): W0(hardware_probe)=['dram_latency_cycles'], W1(hardware_probe)=['actual_boost_clock_mhz']
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

Three states per `(tool, error_kind)` pair:

| State | Condition | Behavior |
| --- | --- | --- |
| CLOSED | default | normal operation |
| OPEN | >= N consecutive failures | `dispatch()` returns `{"status": "circuit_open"}` |
| HALF-OPEN | OPEN for > `half_open_timeout_s` seconds | one probe allowed through; success -> CLOSED, failure -> OPEN (timer reset) |

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
