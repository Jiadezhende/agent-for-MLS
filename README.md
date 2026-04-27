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
│  Execution Layer   Executor (executor.py)             │
│  nvcc compile · subprocess run · ncu/nsys/torch      │
│  sandbox whitelist · timeout · cache · output trim   │
└─────────────────────────────────────────────────────┘
```

**Key invariant**: the LLM never calls `nvcc` or `subprocess` directly;
the Executor never constructs prompts. They interact only through the
`ToolRegistry` (tool call → tool result).

### Tools exposed to the LLM

| Tool                                       | Layer     | Purpose                           |
| ------------------------------------------ | --------- | --------------------------------- |
| `list_skills`                              | Knowledge | List available strategy documents |
| `read_skill(name)`                         | Knowledge | Load a strategy document          |
| `run_cuda_probe(source, probe_name)`       | Executor  | Compile + run a CUDA kernel       |
| `profile_with_ncu(...)`                    | Executor  | Nsight Compute hardware counters  |
| `profile_with_nsys(...)`                   | Executor  | Nsight Systems CPU-GPU timeline   |
| `profile_with_torch(python_code, op_name)` | Executor  | Torch Profiler operator stats     |
| `record_measurement(metric, value, ...)`   | Recording | Write a result to results.json    |
| `flag_event(type, severity, detail)`       | Recording | Log anomaly / decision            |
| `submit_results(summary)`                  | Recording | Finalize and exit                 |

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
// target_spec.json
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

| Variable                 | Default  | Description                                     |
| ------------------------ | -------- | ----------------------------------------------- |
| `OPENAI_API_KEY`         | —        | **Required.** API key                           |
| `AGENT_LLM_MODEL`        | —        | **Required.** Model name                        |
| `OPENAI_BASE_URL`        | OpenAI   | Leave blank for OpenAI; set for other providers |
| `AGENT_LLM_MAX_TOKENS`   | `4096`   | Max tokens per response (see note below)        |
| `AGENT_LLM_TEMPERATURE`  | `0.2`    | Sampling temperature (ignored for o/GPT-5+)     |
| `AGENT_LLM_TIMEOUT_S`    | `120`    | Per-request timeout (seconds)                   |
| `AGENT_LLM_MAX_RETRIES`  | `3`      | Retries on transient errors                     |

> **Model family compatibility** (`agents/core/llm.py`): the client auto-detects the model
> family and adjusts API parameters accordingly.
>
> | Model family | Detection rule | `max_tokens` param sent as | `temperature` sent |
> | --- | --- | --- | --- |
> | o-series (o1, o3, o4…) | name matches `o\d.*` | `max_completion_tokens` | no |
> | GPT-5 and later | name starts with `gpt-5` | `max_completion_tokens` | no |
> | All others (GPT-4o, DeepSeek, vLLM…) | default | `max_tokens` | yes |
>
> This means `AGENT_LLM_MAX_TOKENS` and `AGENT_LLM_TEMPERATURE` work for all providers
> without any code changes — the client picks the right parameter name automatically.

### Agent loop

| Variable                | Default | Description                         |
| ----------------------- | ------- | ----------------------------------- |
| `AGENT_MAX_ITERATIONS`  | `40`    | Max LLM round-trips per run         |
| `AGENT_KEEP_WORKSPACE`  | `false` | Retain build artifacts after run    |

### Executor

| Variable                      | Default        | Description                           |
| ----------------------------- | -------------- | ------------------------------------- |
| `AGENT_WORKSPACE_ROOT`        | `./workspace`  | Per-run artifact directory root       |
| `AGENT_NVCC_BIN`              | `nvcc`         | CUDA compiler binary                  |
| `AGENT_NCU_BIN`               | `ncu`          | Nsight Compute binary                 |
| `AGENT_NSYS_BIN`              | `nsys`         | Nsight Systems binary                 |
| `AGENT_COMPILE_TIMEOUT_S`     | `120`          | nvcc compile timeout                  |
| `AGENT_RUN_TIMEOUT_S`         | `60`           | Binary execution timeout              |
| `AGENT_PROFILE_TIMEOUT_S`     | `600`          | ncu / nsys profiling timeout          |
| `AGENT_CACHE_ENABLED`         | `true`         | Cache identical jobs within a run     |
| `AGENT_STDOUT_TRUNCATE_BYTES` | `64000`        | Max stdout size fed to LLM            |

### Switching LLM provider

Only `.env` changes are needed — no code changes:

```bash
# OpenAI (default)
OPENAI_API_KEY=sk-proj-...
AGENT_LLM_MODEL=gpt-4o

# DeepSeek
OPENAI_API_KEY=sk-...          # DeepSeek key format
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
├── main.py              CLI entry point
├── config.py            Config dataclasses (LLMConfig, AgentConfig, ExecutorConfig)
├── executor.py          Execution layer: compile, run, profile, cache, sandbox
├── requirements.txt
├── .env.example
│
├── agent/
│   ├── loop.py          Main agent loop (LLM ↔ tool dispatch)
│   ├── prompts.py       System prompt + user message builder
│   ├── tool_schemas.py  OpenAI function-calling schemas for all 9 tools
│   ├── tool_registry.py Tool dispatch table and default registry factory
│   └── types.py         Shared data structures (Task, Result, AgentContext, MemoryStore)
│
├── llm/
│   └── client.py        OpenAI SDK wrapper (the only file that imports openai)
│
├── tools/
│   ├── skills.py        list_skills / read_skill (filesystem reads)
│   └── recording.py     record_measurement / flag_event / submit_results
│
├── skills/              Measurement strategy documents (Phase 2)
│   ├── gpu_profiling_overview.md
│   ├── _template.md
│   └── README.md
│
├── workspace/           Runtime artifacts (gitignored)
└── fixtures/
    └── target_spec_minimal.json
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
  "reasoning_log": [
    {"iteration": 0, "content": "...", "tool_calls": [...]},
    ...
  ],
  "events": [
    {"iteration": 2, "type": "clock_locked", "severity": "warn",
     "detail": "Measured 825 MHz vs API-reported 1410 MHz"}
  ],
  "job_history": [...],
  "summary": "Measured 3 metrics. Detected clock lock at 825 MHz ..."
}
```

---

## Phase 2 Roadmap

- `skills/memory_latency.md` — pointer-chasing strategy
- `skills/memory_bandwidth.md` — streaming kernel strategy
- `skills/cache_capacity.md` — latency-vs-size sweep
- `skills/clock_measurement.md` — clock64() frequency measurement
- `skills/bank_conflict.md` — shared memory stride comparison
- Deep ncu/nsys output reducers (roofline, memory-bound classification)
- Streaming LLM client + context compaction
- Test suite (FakeLLM + unit tests)
