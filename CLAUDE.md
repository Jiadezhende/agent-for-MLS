# Agent for MLS — Project Guide

## What this is

An autonomous agent that measures GPU hardware parameters (DRAM latency, boost clock, etc.) by writing and running CUDA C microbenchmarks. The LLM reasons about measurement strategy; the Executor compiles and runs all code.

## How to run

```bash
cp .env.example .env   # fill in OPENAI_API_KEY and AGENT_LLM_MODEL
python main.py --spec target_spec.json --output results.json --verbose
```

`target_spec.json` format: `{"targets": ["dram_latency_cycles", "actual_boost_clock_mhz"]}`

## Architecture (three layers)

```
LLM (agent/loop.py)
  ↓ tool calls
ToolRegistry (agent/tool_registry.py)   ← circuit breaker lives here
  ↓ dispatches to
Executor (executor.py)                  ← only code that runs subprocesses
  ├── run_cuda_probe    PRIMARY: compile + run .cu, stdout = measurement
  ├── profile_with_ncu  cross-verify with Nsight Compute counters
  ├── profile_with_nsys CPU-GPU timeline (not for hardware probing)
  └── profile_with_torch PyTorch operator profiling (not for hardware probing)
```

**Key rule**: hardware parameters (latency, clock) use `run_cuda_probe` with CUDA C. `profile_with_nsys` / `profile_with_torch` are for operator analysis, not hardware probing.

## Key files

| File | Purpose |
|------|---------|
| `executor.py` | Compilation, sandboxing, auto-detection, error classification |
| `agent/loop.py` | Main LLM loop; `--verbose` real-time output via `_emit()` |
| `agent/types.py` | `AgentContext`, `CircuitBreaker`, `Result`, `MemoryStore` |
| `agent/tool_registry.py` | Tool dispatch + circuit breaker enforcement |
| `agent/prompts.py` | System prompt (edit here to change LLM behavior) |
| `config.py` | Three dataclasses: `LLMConfig`, `AgentConfig`, `ExecutorConfig` |
| `skills/*.md` | Measurement strategy docs the LLM reads via `list_skills`/`read_skill` |
| `tools/recording.py` | `record_measurement`, `flag_event`, `submit_results` |

## Real-time verbose output

`AgentLoop` accepts `verbose: bool = False`. When enabled (via `--verbose`), each iteration prints to stderr:

```text
── iter 3/40 ────────────────────────────────────────────
  <LLM reasoning text>
  call: run_cuda_probe  {"source":"<3842 chars>","probe_name":"clock_measurement"}
  result: run_cuda_probe  status=done  elapsed=4.21s
```

- Large-text arguments (`source`, `source_or_path`, `python_code`) are replaced with `<N chars>` to keep output readable.
- All prints use `flush=True` so output appears immediately even when piped (`2>&1 | tee run.log`).
- Helper functions `_summarize_args()` and `_summarize_result()` in `agent/loop.py` handle formatting.

## Executor error handling

Every error dict has `error_class`:
- `"user_code"` — CUDA source is wrong; LLM should fix the kernel
- `"infrastructure"` — binary missing or env misconfigured; LLM should stop retrying
- `"timeout"` — reduce workload

On startup, `Executor.__init__` auto-detects GPU arch (`-arch=sm_NNN` via `nvidia-smi`), MSVC path (via `vswhere`), and ncu/nsys install paths. Results printed to stderr.

## Circuit breaker

After 3 consecutive `(tool, error_kind)` failures, `tool_registry.dispatch()` returns `{"status": "circuit_open", ...}` instead of calling the tool. Threshold: `AGENT_CB_THRESHOLD` env var (default 3).

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
