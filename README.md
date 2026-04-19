# agent-for-MLS

Autonomous GPU hardware profiling agent for the MLSYS course project.
Given a list of target metrics, the agent autonomously generates measurement
code, executes profiling tools, cross-verifies results, and outputs
structured measurements with engineering reasoning logs.

---

## Features

- **Hardware parameter probing** via self-written CUDA micro-benchmarks
  (pointer chasing, streaming bandwidth, clock measurement, bank conflict)
- **Multi-tool profiling**: Nsight Compute, Nsight Systems, Torch Profiler
- **Anti-hacking**: detects non-standard clock locking, SM masking, and
  spoofed `cudaGetDeviceProperties` by measuring directly in-kernel
- **Any OpenAI-compatible LLM**: OpenAI, DeepSeek, local vLLM, etc.
- **Engineering reasoning log** for LLM-as-Judge scoring (30 pts)

---

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│  Knowledge Layer   skills/*.md                       │
│  Measurement strategy docs, loaded by LLM on demand  │
└──────────────────────┬──────────────────────────────┘
                       │ read_skill / list_skills
┌──────────────────────▼──────────────────────────────┐
│  Reasoning Layer   LLM (OpenAI-compatible)           │
│  Reads target_spec → picks strategy → writes CUDA    │
│  kernel → interprets results → detects anomalies     │
└──────────────────────┬──────────────────────────────┘
                       │ tool call (JSON)
┌──────────────────────▼──────────────────────────────┐
│  Execution Layer   Executor (agents/tools/cuda_executor.py) │
│  nvcc compile · subprocess run · ncu/nsys/torch      │
│  sandbox whitelist · timeout · cache · output trim   │
└─────────────────────────────────────────────────────┘
```

**Multi-agent pipeline**: Planner routes targets to workers → workers run in
parallel → Critic reviews outputs and triggers retries if needed.

**Key invariant**: the LLM never calls `nvcc` or `subprocess` directly;
the Executor never constructs prompts. They interact only through the
`ToolRegistry` (tool call → tool result).

### Tools exposed to the LLM

| Tool | Layer | Purpose |
| --- | --- | --- |
| `list_skills` | Knowledge | List available strategy documents |
| `read_skill(name)` | Knowledge | Load a strategy document |
| `run_cuda_probe(source, probe_name)` | Executor | Compile + run a CUDA kernel |
| `profile_with_ncu(...)` | Executor | Nsight Compute hardware counters |
| `profile_with_nsys(...)` | Executor | Nsight Systems CPU-GPU timeline |
| `profile_with_torch(python_code, op_name)` | Executor | Torch Profiler operator stats |
| `record_measurement(metric, value, ...)` | Recording | Write a result to results.json |
| `flag_event(type, severity, detail)` | Recording | Log anomaly / decision |
| `submit_results(summary)` | Recording | Finalize and exit |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY and AGENT_LLM_MODEL
```

### 3. Prepare input

```json
{"targets": ["dram_latency_cycles", "actual_boost_clock_mhz"]}
```

### 4. Run

```bash
python main.py --spec target_spec.json --output results.json
```

### 5. View output

```bash
cat results.json          # measured values  (70 pts)
cat reasoning_log.json    # reasoning trace  (30 pts)
```

---

## Configuration

All settings are controlled via environment variables (`.env` file).

### LLM

| Variable | Default | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | **Required.** API key |
| `AGENT_LLM_MODEL` | — | **Required.** Model name |
| `OPENAI_BASE_URL` | OpenAI | Leave blank for OpenAI; set for other providers |
| `AGENT_LLM_MAX_TOKENS` | `4096` | Max tokens per response |
| `AGENT_LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `AGENT_LLM_TIMEOUT_S` | `120` | Per-request timeout (seconds) |
| `AGENT_LLM_MAX_RETRIES` | `3` | Retries on transient errors |

### Agent loop

| Variable | Default | Description |
| --- | --- | --- |
| `AGENT_MAX_ITERATIONS` | `40` | Max LLM round-trips per worker |
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
| `AGENT_STDOUT_TRUNCATE_BYTES` | `64000` | Max stdout size fed to LLM |

### Switching LLM provider

Only `.env` changes are needed — no code changes:

```bash
# OpenAI (default)
OPENAI_API_KEY=sk-proj-...
AGENT_LLM_MODEL=gpt-4o

# DeepSeek
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.deepseek.com
AGENT_LLM_MODEL=deepseek-chat

# Local vLLM / Ollama
OPENAI_API_KEY=none
OPENAI_BASE_URL=http://localhost:8000/v1
AGENT_LLM_MODEL=llama-3.1-8b
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
├── main.py                        CLI entry point
├── orchestrator.py                Multi-agent pipeline scheduler (Planner → Workers → Critic)
├── requirements.txt
├── .env.example
│
├── agents/                        Three-tier agent package
│   ├── __init__.py                Imports all built-in plugins to trigger registration
│   ├── _registry.py               AgentDefinition dataclass + register/get/all_definitions
│   │
│   ├── core/                      Framework layer
│   │   ├── agent.py               Agent ABC
│   │   ├── config.py              LLMConfig, AgentConfig, ExecutorConfig
│   │   ├── llm.py                 OpenAI SDK wrapper
│   │   ├── loop.py                ReAct agent loop (LLM ↔ tool dispatch)
│   │   ├── message.py             Message dataclass
│   │   ├── exceptions.py          AgentError, ExecutorError, CircuitOpenError
│   │   ├── prompts.py             Generic build_user_message()
│   │   └── types.py               Task, Step, WorkerOutput, CriticDecision, AgentContext, Result
│   │
│   ├── tools/                     Tool system layer
│   │   ├── base.py                Tool protocol/ABC
│   │   ├── registry.py            ToolRegistry + universal CircuitBreaker dispatch + ToolFactory
│   │   ├── schemas.py             OpenAI function-calling schemas for all 9 tools
│   │   ├── circuit_breaker.py     CircuitBreaker (applies to every tool)
│   │   ├── chain.py               ToolChain (sequential tool composition)
│   │   ├── cuda_executor.py       Compile, run, profile; thread-safe; auto-detects GPU arch
│   │   └── builtin/
│   │       ├── recording.py       record_measurement, flag_event, submit_results
│   │       └── skills.py          list_skills, read_skill
│   │
│   └── agents/                    Agent implementation layer
│       ├── planner_agent.py       PlannerAgent + inlined prompt
│       ├── critic_agent.py        CriticAgent + inlined prompt
│       └── hardware_probe_agent.py  HardwareProbeAgent + inlined prompts + plugin registration
│
├── skills/                        Measurement strategy docs (read by LLM via read_skill)
│   ├── gpu_profiling_overview.md
│   ├── memory_latency.md
│   ├── clock_measurement.md
│   └── README.md
│
└── workspace/                     Runtime artifacts (gitignored)
```

---

## I/O Format

### Input: `target_spec.json`

```json
{
  "targets": ["dram_latency_cycles", "l2_cache_size_mb", "actual_boost_clock_mhz"]
}
```

Targets can also be objects with metadata:

```json
{
  "targets": [
    {"name": "dram_latency_cycles", "unit": "cycles", "description": "DRAM round-trip latency"},
    {"name": "actual_boost_clock_mhz", "unit": "MHz"}
  ]
}
```

### Output: `results.json`

```json
[
  {
    "metric": "dram_latency_cycles",
    "value": 442.5,
    "unit": "cycles",
    "confidence": 0.91,
    "method": "pointer-chasing kernel with 256 MB array (> L2)",
    "evidence": ["stdout: latency_cycles: 442.5"]
  }
]
```

### Output: `reasoning_log.json`

```json
{
  "workers": [
    {
      "step_id": "step-0",
      "worker": "hardware_probe",
      "success": true,
      "reasoning_log": [
        {"iteration": 0, "content": "...", "tool_calls": [...]},
        ...
      ],
      "events": [
        {"iteration": 2, "type": "clock_locked", "severity": "warn",
         "detail": "Measured 825 MHz vs API-reported 1410 MHz"}
      ],
      "summary": "Measured dram_latency_cycles = 442.5 cycles."
    }
  ]
}
```

---

## Roadmap

- `op_profiler` agent type: PyTorch operator timeline via Nsight Systems / Torch Profiler
- `bottleneck_analyst` agent type: roofline model, arithmetic intensity analysis
- Streaming LLM client + context compaction for long runs
- Additional skills: `memory_bandwidth.md`, `cache_capacity.md`, `bank_conflict.md`
