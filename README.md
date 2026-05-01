# agent-for-MLS

Autonomous GPU kernel optimization agent for the MLSYS course project.
Given an operator specification, the agent characterizes the hardware, profiles the
baseline, identifies the bottleneck, and produces an optimized CUDA kernel —
all autonomously, with structured reasoning logs.

---

## Features

- **Full optimization pipeline**: hardware characterization → baseline profiling →
  bottleneck analysis → kernel optimization → correctness verification
- **Operator skill system**: operator specs (`skills/operators/<name>.md`) define the
  formula, success criteria, and strategies — the system is extensible to any operator
  without code changes
- **Hardware characterization** via self-written CUDA C microbenchmarks
  (DRAM bandwidth/latency, boost clock, cache hierarchy, SM count)
- **Multi-tool profiling**: Nsight Compute, Nsight Systems, Torch Profiler
- **Concurrent subagents**: independent stages (hardware probe + baseline profiling)
  run in parallel; GPU execution is always serialized by `_gpu_lock` for measurement accuracy
- **Critic review loop**: up to 3 Planner ↔ Critic cycles; Critic re-reads operator
  success criteria and issues accept/retry decisions per step
- **Any OpenAI-compatible LLM**: OpenAI, DeepSeek, local vLLM, etc.
- **Engineering reasoning log** for LLM-as-Judge scoring (30 pts)

---

## Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│  Operator Skill Layer   skills/operators/<name>.md               │
│  Formula, success criteria, hardware requirements, strategies    │
└─────────────────────────────┬────────────────────────────────────┘
                              │ read_skill("operators/lora_matmul")
┌─────────────────────────────▼────────────────────────────────────┐
│  Coordinator Layer   PlannerAgent (ReAct AgentLoop)              │
│  Reads operator spec → orchestrates subagents → submits for review│
│    run_subagent_parallel([hardware_probe, op_profiler])          │
│    run_subagent("bottleneck_analyst")                            │
│    mark_ready_for_critic(summary)                                │
└──────────┬───────────────────────────┬───────────────────────────┘
           │ run_subagent              │ run_subagent_parallel
┌──────────▼──────────┐   ┌───────────▼────────────────────────────┐
│  Worker Subagents   │   │  Worker Subagents (concurrent)         │
│  (ReAct AgentLoop)  │   │  (ReAct AgentLoops in ThreadPoolExecutor)│
│  LLM ↔ tool calls   │   │  GPU execution serialized by _gpu_lock │
└──────────┬──────────┘   └───────────┬────────────────────────────┘
           └───────────────┬──────────┘
                           │ tool calls
┌──────────────────────────▼───────────────────────────────────────┐
│  Execution Layer   Executor (agents/tools/cuda_executor.py)      │
│  nvcc compile · subprocess run · ncu / nsys / torch              │
│  _gpu_lock · timeout · cache · sandbox                           │
└──────────────────────────────────────────────────────────────────┘
```

**Multi-agent state machine**: Orchestrator drives `PlannerAgent ↔ CriticAgent` cycles.
The Planner is a full ReAct loop (not a one-shot call); the Critic reviews accumulated
results against the operator's success criteria; retry feedback is passed back to the
Planner for targeted revision.

**Key invariant**: the LLM never calls `nvcc` or `subprocess` directly; the Executor
never constructs prompts. They interact only through the `ToolRegistry`.

### Tool layers

**Coordinator tools** (used by PlannerAgent):

| Tool | Purpose |
| --- | --- |
| `read_skill(name)` | Load operator spec or strategy document |
| `list_skills` | List available skill files |
| `run_subagent(agent_type, targets)` | Delegate to one worker agent (sequential) |
| `run_subagent_parallel(calls)` | Delegate to multiple independent agents (concurrent) |
| `mark_ready_for_critic(summary)` | Signal all criteria met; exit coordinator loop |
| `flag_event(type, severity, detail)` | Log strategy decision or anomaly |

**Worker tools** (used by subagent ReAct loops):

| Tool | Purpose |
| --- | --- |
| `run_cuda_probe(source, probe_name)` | Compile + run a CUDA C microbenchmark |
| `profile_with_ncu(...)` | Nsight Compute hardware counters |
| `profile_with_nsys(...)` | CPU-GPU timeline, launch overhead |
| `profile_with_torch(python_code, op_name)` | Torch Profiler operator stats |
| `record_measurement(metric, value, ...)` | Write a result to job output |
| `flag_event(type, severity, detail)` | Log anomaly / decision |
| `submit_results(summary)` | Finalize worker output; exit subagent loop |
| `read_skill(name)` / `list_skills` | Load strategy docs on demand |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env: fill in API_KEY and BASE_MODEL
```

### 3. Prepare input

```json
{"operator": "lora_matmul", "targets": ["dram_bandwidth_gbps", "boost_clock_mhz"]}
```

`"operator"` selects the skill file at `skills/operators/lora_matmul.md` which defines
the full optimization spec. `"targets"` optionally constrains which hardware parameters
to measure (the skill file lists all required ones).

### 4. Run

```bash
python main.py --spec target_spec.json --output results.json --verbose
```

### 5. View output

```bash
cat results.json          # measured / optimized values  (70 pts)
cat reasoning_log.json    # per-worker reasoning trace   (30 pts)
cat run_log.jsonl         # structured event log (plan.start, critic.decision, ...)
```

---

## Configuration

All settings via environment variables (`.env` file).

### LLM

| Variable | Default | Description |
| --- | --- | --- |
| `API_KEY` | — | **Required.** API key |
| `BASE_MODEL` | — | **Required.** Model name |
| `BASE_URL` | OpenAI | Leave blank for OpenAI; set for other providers |
| `AGENT_LLM_MAX_TOKENS` | `4096` | Max tokens per response |
| `AGENT_LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `AGENT_LLM_TIMEOUT_S` | `120` | Per-request timeout (seconds) |
| `AGENT_LLM_MAX_RETRIES` | `3` | Retries on transient errors |

### Agent loop

| Variable | Default | Description |
| --- | --- | --- |
| `AGENT_MAX_ITERATIONS` | `40` | Max LLM round-trips per ReAct loop (coordinator or worker) |
| `AGENT_MAX_CRITIC_CYCLES` | `3` | Max Planner ↔ Critic retry cycles |
| `AGENT_KEEP_WORKSPACE` | `false` | Retain build artifacts after run |

### Executor

| Variable | Default | Description |
| --- | --- | --- |
| `AGENT_WORKSPACE_ROOT` | `./workspace` | Per-run artifact directory root |
| `AGENT_NVCC_BIN` | `nvcc` | CUDA compiler binary |
| `AGENT_NCU_BIN` | `ncu` | Nsight Compute binary |
| `AGENT_NSYS_BIN` | `nsys` | Nsight Systems binary |
| `AGENT_COMPILE_TIMEOUT_S` | `120` | nvcc compile timeout |
| `AGENT_RUN_TIMEOUT_S` | `60` | Binary execution timeout |
| `AGENT_PROFILE_TIMEOUT_S` | `600` | ncu / nsys profiling timeout |
| `AGENT_CACHE_ENABLED` | `true` | Cache identical jobs within a run |
| `AGENT_STDOUT_TRUNCATE_BYTES` | `64000` | Max stdout fed to LLM |

### Switching LLM provider

Only `.env` changes needed — no code changes:

```bash
# OpenAI
API_KEY=sk-proj-...
BASE_MODEL=gpt-4o

# DeepSeek
API_KEY=sk-...
BASE_URL=https://api.deepseek.com
BASE_MODEL=deepseek-chat

# Local vLLM / Ollama
API_KEY=none
BASE_URL=http://localhost:8000/v1
BASE_MODEL=llama-3.1-8b
```

---

## CLI Reference

```text
python main.py --spec <path> [options]

Required:
  --spec PATH           Path to target_spec.json

Optional:
  --output PATH         Output path for results.json (default: results.json)
  --max-iterations N    Override AGENT_MAX_ITERATIONS
  --keep-workspace      Retain workspace/run_* directory (useful for debugging)
  --verbose, -v         Print progress to stderr
```

---

## Project Structure

```text
agent-for-MLS/
├── main.py                          CLI entry point
├── orchestrator.py                  State machine: planning/revising ↔ critic → accepted
├── requirements.txt
├── .env.example
│
├── agents/                          Three-tier agent package
│   ├── __init__.py                  Imports all built-in plugins to trigger registration
│   ├── _registry.py                 AgentDefinition dataclass + register/get/all_definitions
│   │
│   ├── core/                        Framework layer
│   │   ├── agent.py                 Agent ABC
│   │   ├── config.py                LLMConfig, AgentConfig, ExecutorConfig
│   │   ├── llm.py                   OpenAI SDK wrapper
│   │   ├── loop.py                  ReAct agent loop (shared by coordinator + workers)
│   │   ├── exceptions.py            AgentError, ExecutorError, CircuitOpenError
│   │   ├── prompts.py               build_user_message() for worker agents
│   │   └── types.py                 Task, Step, WorkerOutput, CriticDecision, AgentContext, RunContext
│   │
│   ├── tools/                       Tool system layer
│   │   ├── base.py                  Tool protocol/ABC
│   │   ├── registry.py              ToolRegistry + CircuitBreaker dispatch + ToolFactory
│   │   ├── circuit_breaker.py       CircuitBreaker (universal, per-subagent isolation)
│   │   ├── cuda_executor.py         Compile, run, profile; _gpu_lock; thread-safe
│   │   └── builtin/
│   │       ├── recording.py         record_measurement, flag_event, submit_results
│   │       ├── skills.py            list_skills, read_skill (supports operators/ subdir)
│   │       └── subagent.py          RunSubagentTool, RunSubagentParallelTool, MarkReadyForCriticTool
│   │
│   └── agents/                      Agent implementation layer
│       ├── planner_agent.py         PlannerAgent (ReAct coordinator) + inlined prompt
│       ├── critic_agent.py          CriticAgent + system_prompt_override support
│       └── hardware_probe_agent.py  HardwareProbeAgent + prompts + plugin registration
│
├── skills/                          Strategy docs and operator specs
│   ├── operators/
│   │   ├── lora_matmul.md           LoRA-fused MATMUL operator spec (first operator)
│   │   └── _operator_template.md   Template for adding new operators
│   ├── gpu_profiling_overview.md
│   ├── memory_hierarchy.md
│   ├── clock_environment.md
│   └── throughput_resources.md
│
└── workspace/                       Runtime artifacts (gitignored)
```

---

## I/O Format

### Input: `target_spec.json`

```json
{
  "operator": "lora_matmul",
  "targets": ["dram_bandwidth_gbps", "boost_clock_mhz", "sm_count"]
}
```

`"operator"` is required for the full optimization pipeline. `"targets"` is optional
and constrains which hardware metrics to measure (the operator skill defines all
required measurements).

### Output: `results.json`

Flat dict of metric → numeric value (highest-confidence value wins on duplicates):

```json
{
  "dram_bandwidth_gbps": 848,
  "boost_clock_mhz": 2407,
  "sm_count": 36
}
```

### Output: `reasoning_log.json`

Per-worker reasoning traces:

```json
{
  "workers": [
    {
      "step_id": "a3f1c8",
      "agent_type": "hardware_probe",
      "success": true,
      "reasoning_log": [
        {"iteration": 0, "content": "...", "tool_calls": [...]},
        "..."
      ],
      "events": [
        {"type": "strategy_decision", "severity": "info",
         "detail": "Using pointer-chasing kernel for DRAM latency"}
      ],
      "summary": "Measured dram_bandwidth_gbps=848, boost_clock_mhz=2407."
    }
  ]
}
```

### Output: `run_log.jsonl`

Structured event log (one JSON object per line):

```jsonl
{"kind": "plan.start", "source": "orchestrator", "data": {"spec": {...}}}
{"kind": "plan.complete", "source": "planner", "data": {"n_subagent_calls": 2, "cycle": 0}}
{"kind": "critic.decision", "source": "critic", "data": {"step_id": "a3f1c8", "decision": "accept"}}
{"kind": "pipeline.done", "source": "orchestrator", "data": {"phase": "accepted"}}
```

---

## Adding a New Operator

1. Create `skills/operators/<name>.md` following `_operator_template.md`.
   Define: formula, tensor shapes, required hardware measurements, success criteria,
   optimization strategies.

2. Run with `{"operator": "<name>", ...}` in `target_spec.json`.

No code changes needed.

## Adding a New Agent Type

1. Create `agents/agents/<type>_agent.py` with the agent class + inlined prompts +
   `register(AgentDefinition(...))`.

2. Add one line to `agents/__init__.py`:

   ```python
   import agents.agents.<type>_agent  # noqa: F401
   ```

No changes to Orchestrator, PlannerAgent, CriticAgent, or AgentLoop.

---

## Roadmap

- `op_profiler` agent: PyTorch operator timeline via Nsight Systems / Torch Profiler
- `bottleneck_analyst` agent: roofline model, arithmetic intensity analysis
- `kernel_optimizer` agent: write + validate optimized CUDA kernel
- Additional operator skills: `flash_attention.md`, `layer_norm.md`
- Streaming LLM client + context compaction for long runs
