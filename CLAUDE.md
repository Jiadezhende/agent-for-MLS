# Agent for MLS — Project Guide

## What this is

An autonomous pipeline that searches for an optimized CUDA implementation
of a LoRA-style operator and writes the best candidate to
`./optimized_lora.cu` for the official Phase-2 evaluation harness.

Given an operator name on the CLI, the pipeline goes through fixed stages:

1. **INIT** — load `OperatorContract` from `skills/operators/<name>.md`
   frontmatter; mkdir workspace; seed Blackboard with operator + benchmark spec
2. **HARDWARE_PROFILE** — `hardware_profiler` agent characterizes the GPU
   (DRAM bw, clock, SM, L2, latency) and writes `blackboard["hardware"]`
3. **BENCHMARK_BASELINE** — pure code: generate W/X/A/B inputs across the
   shape grid and benchmark PyTorch reference latency (the speedup
   denominator). The reference output is NOT saved — every candidate
   evaluation recomputes it online inside its own subprocess so the
   cuBLAS / TF32 state matches Phase-2 exactly
4. **INITIAL_CANDIDATE** — `optimizer_cold` agent produces the first
   compilable + correct candidate so `./optimized_lora.cu` always exists
   even if a later stage times out
5. **TUNING_LOOP** — RoundRunner cycles `analyst → optimizer →
   multi-shape benchmark → maybe-promote` until time budget is exhausted
6. **FINALIZE** — `summary` agent writes `blackboard["final_summary"]`;
   orchestrator renders `final/final_report.json` + `summary.md` and
   re-syncs the root file

The pipeline is **operator-agnostic** — `operator_opt_pipe/resources/`
reads the contract object and drives baseline / benchmark / evaluation
generically. Adding a new operator means writing a single skill markdown
file (see [Adding a new operator](#adding-a-new-operator)).

## How to run

```bash
cp .env.example .env       # API_KEY + BASE_MODEL
bash run.sh                # = python main.py --operator lora_matmul --time-budget 1800 --output ./optimized_lora.cu --verbose
```

`run.sh` is the official Phase-2 entry point. Resume an interrupted run:

```bash
python main.py --operator lora_matmul --time-budget 1800 --run-id <existing_id>
```

The orchestrator detects `workspace/runs/<run_id>/state.json` and
`next_stage` automatically skips already-completed setup stages whose
artifacts exist on disk.

## Architecture

### Stage state machine

```text
python main.py --operator lora_matmul
      │
      ▼
  PipelineOrchestrator.run()        pure-code state machine
      │
      ├── stage = HARDWARE_PROFILE  agents.run_hardware_profiler → blackboard["hardware"]
      ├── stage = BENCHMARK_BASELINE pure code: resources.baseline.build_correctness_fixtures +
      │                                          resources.baseline.measure_pytorch_latency
      │                                       → inputs/ + baseline.json
      ├── stage = INITIAL_CANDIDATE agents.run_optimizer_cold → write_candidate → submit_candidate
      │                              → orchestrator: multi-shape benchmark → promote → sync root file
      ├── stage = TUNING_LOOP       RoundRunner: analyst → optimizer → benchmark → maybe-promote
      └── stage = FINALIZE          agents.run_summary → blackboard["final_summary"]
                                    → orchestrator: render final/{report.json, summary.md}
```

`PipelineOrchestrator` decides which stage runs next, persists
`state.json` after every stage, and owns the `best/best.cu →
./optimized_lora.cu` sync. Each LLM agent is a thin call into
`operator_opt_pipe.agents.run_*` — performance evaluation, multi-shape
benchmark, and best promotion are deterministic Python in
`resources/` and are NEVER reachable from inside an agent.

### `next_stage` rules ([operator_opt_pipe/transitions.py](operator_opt_pipe/transitions.py))

```text
elapsed_s >= time_budget_s                      → FINALIZE
not has_hardware_profile()                      → HARDWARE_PROFILE
not has_baseline()                              → BENCHMARK_BASELINE
best_candidate_id is None                       → INITIAL_CANDIDATE
remaining_budget_s > MIN_TUNING_SLICE_S (60s)   → TUNING_LOOP
otherwise                                       → FINALIZE
```

Predicates are filesystem checks against `RunLayout` paths — that is
what makes resume work: a dropped run picks back up wherever the
artifacts already exist.

### Two agent termination patterns

The framework uses two `mls_agent` termination paths:

| Pattern | Used by | What the agent does | `AgentResult.reason` |
| --- | --- | --- | --- |
| **Natural exit** | hardware_profiler, analyst, summary | Calls `write_blackboard(key, payload)` then stops calling tools (plain text response) → ReActLoop's `max_consecutive_no_tool_call` triggers | `"no_tool_call"` |
| **Explicit terminate** | optimizer_cold, optimizer | Calls `write_candidate` (possibly multiple times until compile + correctness pass), then `submit_candidate` which uses `ToolResponse.terminate_with(payload)` to hand `candidate_id` to the orchestrator | `"completed"` |

Natural exit decouples write and terminate (single-purpose tools).
Explicit terminate is reserved for the candidate flow where the
orchestrator must receive the id immediately for benchmark + promote.

### Per-role tool wiring

`operator_opt_pipe.agents.build_registry(role, layout, contract,
executor, skills_dir)` constructs a fresh `ToolRegistry` for each role:

| Role | Tools |
| --- | --- |
| `hardware_profiler` | `read_skill / list_skills`, `record_measurement / flag_event`, `read_blackboard`, `run_cuda_probe / profile_with_ncu / profile_with_nsys / probe_environment`, `write_blackboard("hardware")` |
| `optimizer_cold` / `optimizer` | `read_skill / list_skills`, `record_measurement / flag_event`, `read_blackboard`, `write_candidate`, `submit_candidate` |
| `analyst` | `read_skill / list_skills`, `record_measurement / flag_event`, `read_blackboard`, `profile_with_ncu / profile_with_nsys / profile_with_torch`, `write_blackboard("latest_diagnosis")` |
| `summary` | `read_skill / list_skills`, `record_measurement / flag_event`, `read_blackboard`, `write_blackboard("final_summary")` |

`WriteBlackboardTool` is the same class instantiated with a different
`allowed_keys` schema per role — analyst can only write
`"latest_diagnosis"`, summary only `"final_summary"`, etc. There are
**no** `submit_hardware_profile / submit_diagnosis / submit_summary`
tools — they are unified into `write_blackboard`.

### Self-research tools (4 classes)

| Class | Used by | Notes |
| --- | --- | --- |
| `ReadBlackboardTool` | all agents | Read a single key |
| `WriteBlackboardTool` | hardware_profiler / analyst / summary | Per-role allowed_keys whitelist + shallow schema (required field list); does NOT terminate |
| `WriteCandidateTool` | optimizer_cold / optimizer | Allocates `candidate_NNN/`, writes `candidate.cu`, **synchronously** runs `cpp_extension.load + -O3` and validates correctness on a single shape, returns the result so the LLM can iterate |
| `SubmitCandidateTool` | optimizer_cold / optimizer | Validates `candidate.cu` exists, then `terminate_with(payload={candidate_id, hypothesis, ...})` so the orchestrator picks up the id |

Everything else is from `mls_agent` (`make_skill_tools`,
`make_side_effect_tools`, `make_profile_tools`).

## Workspace layout

```text
workspace/runs/<run_id>/
    state.json                       RunState (current_stage, round_index, best_candidate_id, elapsed_s)
    blackboard.json                  shared store: operator/hardware/baseline/best/latest_diagnosis/history/round/final_summary
    events.jsonl                     append-only audit (stage_enter, best_promoted, stage_failed, …)
    leaderboard.jsonl                append-only candidate records (id, speedup_geomean, speedup_worst, all_correct)

    hardware_profile.json            agent-written via write_blackboard("hardware") + mirrored to disk
    baseline.json                    PyTorch reference latency — speedup denominator
                                       {ms_median_overall, per_shape, reference_pytorch}

    inputs/<name>_<shape_id>.pt      contract-driven candidate inputs (e.g. W_d3584.pt, ...)
                                       Reference Y is NOT cached: every candidate eval recomputes
                                       ops.reference(inputs) online to match Phase-2's cuBLAS state.

    benchmark/                       Multi-shape candidate timings — orchestrator-only
        spec.json                    {shape_grid, samples=30, warmup=5, seed=0}
        candidate_NNN.json           per-shape ms / speedup / correctness for each candidate

    candidates/
        candidate_NNN/
            candidate.cu             ← agent via write_candidate
            compile.json             ← write_candidate (compile_ok + nvcc log)
            correctness_quick.json   ← write_candidate (single-shape quick result)
            profile.json             ← analyst (when calling profile_with_ncu)

    best/                            ★ only orchestrator writes
        best.cu                      current best candidate source
        best_result.json             {candidate_id, speedup, promoted_at}

    final/
        final_report.json            machine-readable summary
        summary.md                   human-readable narrative

./optimized_lora.cu                  framework eager-syncs from best/best.cu (Phase-2 contract)
```

`RunLayout` ([operator_opt_pipe/state.py](operator_opt_pipe/state.py))
owns every path. New artifacts get a method on `RunLayout` and are
created idempotently by `mkdir()`.

## Output contract — `./optimized_lora.cu`

The Phase-2 harness reads exactly one file:

```python
mod = torch.utils.cpp_extension.load(name=..., sources=["./optimized_lora.cu"], extra_cuda_cflags=["-O3"])
mod.forward(W, X, A, B)
```

The orchestrator copies `best/best.cu` to `./optimized_lora.cu` at three
moments:

1. After INITIAL_CANDIDATE produces a `compile_ok=True, all_correct=True`
   candidate (floor guarantee — even a slow candidate is better than
   nothing if we time out later).
2. Every time `_promote_to_best` accepts a faster candidate.
3. At the end of FINALIZE as a final safety re-sync.

The local `evaluation.benchmark_on_grid` uses the same `cpp_extension.load`
invocation as the official harness — local "best" matches scoring "best".

## Adding a new operator

The pipeline is operator-agnostic. To add `<new_op>`:

1. Copy `skills/operators/_operator_template.md` to
   `skills/operators/<new_op>.md`.
2. Fill in the frontmatter:

   ```yaml
   ---
   name: operators/<new_op>
   shape_param: d
   shape_param_range: [<lo>, <hi>]
   inputs:
     - {name: <X>, shape: [d, d], dtype: float32}
     - ...
   output: {name: Y, shape: [d, d], dtype: float32}
   reference_pytorch: "<RHS expression in input names + torch ops>"
   forward_args: [<X>, ...]
   correctness: {rtol: 1.0e-4, atol: 1.0e-4}
   ---
   ```

3. Run: `python main.py --operator <new_op> --time-budget 1800 --output ./<new_op>.cu`.

That's it. `resources/` reads `inputs / output / reference_pytorch /
forward_args` from the contract and drives synthetic input generation,
PyTorch reference computation, and candidate forward calls. **Zero
Python changes.**

Optional: write `skills/operators/<new_op>_tuning.md` to give the
optimizer agent a navigation manual (search space, decision rules, tile
hints) — agents load it via `read_skill`.

## Stage state-machine invariants

For any change to the orchestrator or its callers, these must hold:

1. `next_stage(state, layout)` is **pure**: only `RunState` fields +
   `RunLayout.has_*` filesystem predicates; no side effects.
2. `optimized_lora.cu` is synced at: (a) INITIAL_CANDIDATE pass, (b)
   every `_promote_to_best`, (c) FINALIZE end. **Never** by an agent.
3. `best/` is written **only** by the orchestrator. Agents write to
   `candidates/candidate_NNN/`.
4. `state.json` is persisted at every stage exit; resume picks up via
   artifact predicates.
5. `submit_candidate` rejects payloads referencing missing `.cu` files —
   broken kernels never reach `best/`.

## Failure policy

| Type | Handling |
| --- | --- |
| Stage agent times out / errors | Wrapped in try/except; orchestrator records `stage_failed` event + carries on |
| Candidate compile fails | `write_candidate` returns `compile_ok=False` + nvcc log; LLM sees it and can re-call `write_candidate` |
| Candidate correctness fails on quick check | Same — LLM iterates until compile + single-shape correctness pass |
| Multi-shape benchmark fails (compile/correctness mismatch) | Result has `compile_ok=False` or `all_correct=False` → orchestrator does NOT promote, leaderboard records the rejection |
| `submit_candidate` references missing file | Tool returns `INVALID_ARGS`; agent must retry |
| Time budget exceeded mid-stage | `next_stage` returns FINALIZE on the next iteration |
| `MAX_LOOP_ITERATIONS` (200) hit | Orchestrator emits `loop_guard_tripped` event and finalizes |

## Key files

| File | Purpose |
| --- | --- |
| [operator_opt_pipe/state.py](operator_opt_pipe/state.py) | `Stage` enum, `RunState`, `RunLayout`, `Blackboard` |
| [operator_opt_pipe/transitions.py](operator_opt_pipe/transitions.py) | `next_stage()` pure function |
| [operator_opt_pipe/orchestrator.py](operator_opt_pipe/orchestrator.py) | `PipelineOrchestrator`, `RoundRunner`, eager-sync, multi-shape benchmark wiring |
| [operator_opt_pipe/agents.py](operator_opt_pipe/agents.py) | 5 role functions + `build_registry(role, ...)` + system prompts |
| [operator_opt_pipe/tools.py](operator_opt_pipe/tools.py) | `ReadBlackboardTool` / `WriteBlackboardTool` / `WriteCandidateTool` / `SubmitCandidateTool` |
| [operator_opt_pipe/resources/contract.py](operator_opt_pipe/resources/contract.py) | `OperatorContract` + `load_contract()` + `eval_shape()` mini-DSL + render helpers |
| [operator_opt_pipe/resources/benchmark.py](operator_opt_pipe/resources/benchmark.py) | `BenchmarkSpec` + `materialize_inputs()` (contract-driven synthetic input generation) |
| [operator_opt_pipe/resources/baseline.py](operator_opt_pipe/resources/baseline.py) | `build_correctness_fixtures()` (save per-shape inputs) + `measure_pytorch_latency()` (PyTorch reference timing) |
| [operator_opt_pipe/resources/evaluation.py](operator_opt_pipe/resources/evaluation.py) | `compile_and_check_quick()` (single-shape, used by write_candidate) + `benchmark_on_grid()` (multi-shape, orchestrator-only) |
| [operator_opt_pipe/main.py](operator_opt_pipe/main.py) | CLI entry — reads `--operator` and constructs everything |
| [main.py](main.py) | Root-level shim → `operator_opt_pipe.main:main` |
| [run.sh](run.sh) | Phase-2 entry — `python3 main.py --operator lora_matmul --time-budget 1800 …` |
| [skills/operators/lora_matmul.md](skills/operators/lora_matmul.md) | Operator contract source (frontmatter parsed by `load_contract`) |
| [skills/operators/lora_matmul_tuning.md](skills/operators/lora_matmul_tuning.md) | Tuning navigation skill — agents load via `read_skill` |
| [mls_agent/](mls_agent/) | Standalone ReAct framework (Tool ABC, ToolRegistry, ReActLoop, Executor) |

## Tests

```bash
pytest tests/operator_opt_pipe/      # ~46 tests, no GPU required
pytest mls_agent/tests/              # 168 tests, no GPU required
pytest -m cuda                       # GPU-marked integration tests (require nvcc + CUDA + torch)
```

Suites:

- `test_state.py` — Stage enum, RunState, RunLayout, Blackboard
- `test_transitions.py` — `next_stage()` rules
- `test_resources.py` — `eval_shape`, `shape_id`, `load_contract`, OperatorContract render helpers
- `test_tools.py` — ReadBlackboard / WriteBlackboard / WriteCandidate / SubmitCandidate
- `test_agents.py` — scripted backend exercises each `run_*` (natural exit + completed paths)
- `test_orchestrator.py` — RoundRunner promotion, full pipeline with injected mocks, resume

## Real-time verbose output

`--verbose` enables prefixed output to stderr:

```text
[<run_id>] [orch] stage=HARDWARE_PROFILE elapsed=0.0s remaining=1800.0s
[<run_id>] iter 1: VALIDATE → ACT
[<run_id>]   call: run_cuda_probe {"probe_name":"clock", ...}
[<run_id>]   call: write_blackboard {"key":"hardware", "payload":<...>}
[<run_id>] [orch] stage=BENCHMARK_BASELINE elapsed=15.2s remaining=1784.8s
...
```

Large-text args (`source`, `python_code`) are truncated for readability.

## Config env vars

| Var | Default | Purpose |
| --- | --- | --- |
| `API_KEY`, `BASE_MODEL` | — | LLM credentials (`mls_agent.LLMConfig.from_env`) |
| `BASE_URL` | OpenAI's | Custom OpenAI-compatible endpoint |
| `AGENT_LLM_MAX_TOKENS` | 8192 | Per-call token cap |
| `AGENT_MAX_ITERATIONS` | 30 | Per-agent ReAct iteration cap |
| `AGENT_WORKSPACE_ROOT` | `./workspace` | Workspace root |
| `AGENT_NVCC_BIN` / `AGENT_NCU_BIN` / `AGENT_NSYS_BIN` | auto-detected | Override toolchain paths |

CLI flags override env (`--operator`, `--time-budget`, `--workspace`,
`--output`, `--run-id`, `--verbose`, `--max-agent-iterations`).

## GPU notes

- All compile commands pass `extra_cuda_cflags=["-O3"]` to mirror the
  Phase-2 evaluator. Don't change without checking Phase-2 docs.
- `evaluation.benchmark_on_grid` uses cudaEvent for wall-clock timing.
- `cudaDeviceProp.clockRate` was removed in CUDA 13 — query at runtime
  via `torch.cuda.get_device_properties` instead.
