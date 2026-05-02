# Agent for MLS — Project Guide

## What this is

An autonomous pipeline that searches for an optimized CUDA implementation of a
LoRA-style operator and writes the best candidate to `./optimized_lora.cu` for
the official Phase-2 evaluation harness.

Given an operator spec, the pipeline goes through fixed stages:

1. **BENCHMARK_SPEC** — derive 5 benchmark spec slots from the operator skill
2. **HARDWARE_PROFILE** — characterize the GPU (DRAM bw, clock, SM, L2, latency)
3. **BASELINE_PROFILE** — generate W/X/A/B inputs + Y_ref tensors + benchmark
   the PyTorch reference (the speedup denominator)
4. **INITIAL_CANDIDATE** — produce the first compileable + correct candidate so
   `./optimized_lora.cu` always exists, even if we time out later
5. **TUNING_LOOP** — iterate: write candidate → quick eval → confirm eval →
   maybe promote to best (cycles until time budget thins out)
6. **OPTIONAL_PROFILE** — ncu/nsys profile the best candidate
7. **FINALIZE** — write `final_report.json` + `summary.md`, ensure root file
   matches `best/best.cu`

The framework (PipelineOrchestrator + StageAgent ABC + state machine + workspace
layout) is operator-agnostic; the LoRA-specific PyTorch reference formula and
`forward(W, X, A, B)` signature live inside two tool modules
(`baseline_tools.py` and `candidate_tools.py`) — see [Adding a new
operator](#adding-a-new-operator).

## How to run

```bash
cp .env.example .env       # API_KEY + BASE_MODEL
bash run.sh                # = python main.py --spec target_spec.json --time-budget 1800 --output ./optimized_lora.cu --verbose
```

`target_spec.json`:

```json
{"operator": "lora_matmul"}
```

`run.sh` is the official Phase-2 entry point. The pipeline syncs
`./optimized_lora.cu` eagerly — once after INITIAL_CANDIDATE produces a working
kernel, and again every time best updates. Hard timeout → straight to FINALIZE
without abandoning the current root file.

Resume an interrupted run:

```bash
python main.py --spec target_spec.json --time-budget 1800 --run-id <existing_id>
```

The orchestrator detects `workspace/runs/<run_id>/state.json` and skips
already-completed setup stages whose artifacts exist on disk.

## Architecture

### Stage state machine

```text
python main.py --spec target_spec.json
      │
      ▼
  PipelineOrchestrator.run()        pure-code state machine, no LLM
      │
      ├── stage = BENCHMARK_SPEC    BenchmarkSpecAgent → benchmark_specs/{5 slots}.json
      ├── stage = HARDWARE_PROFILE  HardwareProfilerAgent → hardware/hardware_profile.json
      ├── stage = BASELINE_PROFILE  BaselineAgent → baseline/{baseline.json, references/, inputs/}
      ├── stage = INITIAL_CANDIDATE KernelTuningAgent (one shot) → candidates/candidate_000/* + best/best.cu + ./optimized_lora.cu
      ├── stage = TUNING_LOOP       KernelTuningAgent (looped)   → candidates/candidate_NNN/* (best updated only on confirm)
      ├── stage = OPTIONAL_PROFILE  ProfileAnalysisAgent → candidates/<best>/profile.json + analysis.md
      └── stage = FINALIZE          SummaryAgent → final/{final_report.json, summary.md}; framework re-syncs root file
```

`PipelineOrchestrator` (pure code) decides which stage runs next, persists
`state.json` after every stage, and owns the `best/best.cu → ./optimized_lora.cu`
sync. **Each StageAgent** runs an LLM ReAct loop scoped to that stage's
allowed tools and returns a `StageResult` — there is no cross-stage critic loop.

### `next_stage` rules

`pipeline/transitions.py:next_stage()` is a pure function:

```text
elapsed_s >= time_budget_s                      → FINALIZE
not specs_complete()                            → BENCHMARK_SPEC
not has_hardware_profile()                      → HARDWARE_PROFILE
not has_baseline()                              → BASELINE_PROFILE
best_candidate_id is None                       → INITIAL_CANDIDATE
remaining_budget_s > MIN_TUNING_SLICE_S (60s)   → TUNING_LOOP
best exists, no profile yet                     → OPTIONAL_PROFILE
otherwise                                       → FINALIZE
```

Predicates (`has_hardware_profile`, `has_baseline`, `has_candidate_profile`)
are filesystem checks against `RunLayout` paths — that is what makes resume
work: a dropped run picks back up wherever the artifacts already exist.

### StageAgent contract

```python
# pipeline/stage_agent.py
class StageAgent(ABC):
    stage: Stage                           # which Stage enum value this serves
    allowed_tools: tuple[str, ...] = ()    # exhaustive whitelist
    @abstractmethod
    def run(self, context: StageContext) -> StageResult: ...

# pipeline/agents/_base.py
class LLMStageAgent(StageAgent):
    SYSTEM_PROMPT: str = ""
    max_iterations: int = 20
    def build_user_message(self, context: StageContext) -> str: ...
    # run() is provided: builds AgentLoop, catches RuntimeError → failed
    # StageResult, calls pop_stage_result(ctx) for the structured result.
```

| StageAgent | Stage | Allowed tools | Iterations |
| --- | --- | --- | --- |
| `BenchmarkSpecAgent` | BENCHMARK_SPEC | `read_skill`, `list_skills`, `submit_benchmark_specs`, `flag_event` | 8 |
| `HardwareProfilerAgent` | HARDWARE_PROFILE | + `run_cuda_probe`, `profile_with_ncu`, `profile_with_nsys`, `record_measurement`, `submit_hardware_profile`, `write_workspace_file`, `probe_environment` | 30 |
| `BaselineAgent` | BASELINE_PROFILE | `read_skill`, `generate_baseline`, `submit_baseline`, `flag_event` | 12 |
| `KernelTuningAgent` | INITIAL_CANDIDATE / TUNING_LOOP | `read_skill`, `write_candidate`, `evaluate_candidate`, `submit_candidate_result`, `flag_event` | 30 |
| `ProfileAnalysisAgent` | OPTIONAL_PROFILE | `read_skill`, `profile_with_ncu`, `profile_with_nsys`, `submit_profile_analysis`, `flag_event` | 12 |
| `SummaryAgent` | FINALIZE | `read_skill`, `submit_summary`, `flag_event` | 5 |

`KernelTuningAgent` is the same class wired up for two stages — the
orchestrator passes `current_stage` to `StageToolFactory.build()` so
`SubmitCandidateResultTool` tags its `StageResult` with the correct enum.

### Tool factory and authorization

```text
StageToolFactory(executor, layout)
   └── build(allowed_tools, current_stage=...) → ToolRegistry
         └── only registers tools the agent declared; unknown name → ValueError
```

Tools split into three groups:

| Tool | Purpose | Notes |
| --- | --- | --- |
| `read_skill`, `list_skills` | read `skills/*.md` and `skills/operators/*.md` | reused from old codebase |
| `record_measurement`, `flag_event` | append to AgentContext logs | reused |
| `submit_benchmark_specs` | persist 5 spec slots + StageResult | new |
| `generate_baseline` | one-shot: emit W/X/A/B/Y_ref + benchmark torch ref | LoRA-specific embedded script |
| `submit_baseline` | persist `baseline.json` + StageResult | new |
| `write_candidate` | alloc `candidate_NNN/`, write `candidate.cu` | new |
| `evaluate_candidate` | `cpp_extension.load` + correctness + cudaEvent benchmark; quick (5) or confirm (30) samples | LoRA-specific embedded script — same harness path as Phase-2 evaluator |
| `submit_candidate_result` | stash `CandidateRecord`; orchestrator promotes to best on `accepted_for=best_update` | new |
| `submit_hardware_profile`, `submit_profile_analysis`, `submit_summary` | finalize the corresponding stage | new |
| `run_cuda_probe`, `profile_with_ncu`, `profile_with_nsys`, `profile_with_torch`, `write_workspace_file`, `probe_environment` | reused from `agents/tools/executor_tools.py` | requires `Executor` |

### AgentLoop ↔ StageAgent signal

`pipeline/agent_loop_signal.py` bridges the two without modifying the existing
`AgentLoop`:

```text
submit_*_tool.run():
   stash_stage_result(ctx, StageResult(...))
   raise _Terminated(summary)         # caught inside AgentLoop, returns ctx

LLMStageAgent.run():
   try: AgentLoop(...).run()
   except RuntimeError: return _failed_result(...)
   result = pop_stage_result(agent_ctx) or _failed_result("no_submit_called", ...)
   return result
```

`stash` writes the dict to `ctx.memory["_stage"]["result"]`; `pop` round-trips
through `StageResult.from_dict` so any schema drift surfaces as a None return
and the agent reports `failed`.

### StageResult schema

Every stage returns a validated `StageResult`:

```python
@dataclass
class StageResult:
    stage: str                       # must match agent.stage.value
    status: Literal["success", "partial", "failed"]
    artifacts: dict[str, str]        # name → workspace-relative POSIX path
    metrics: dict[str, Any]
    confidence: float                # 0..1
    caveats: list[str]
    next_recommendation: str | None
    agent_trace: str | None
```

`pipeline/state.py:validate_stage_result()` is run by `stage_runner` before
the orchestrator absorbs it; schema violations / wrong stage tag / agent
exception → `status="failed"` with diagnostic caveats and the orchestrator
moves on (it does not crash the run).

## Workspace layout

```text
workspace/runs/<run_id>/
    state.json                       run state machine + best_candidate_id
    events.jsonl                     append-only audit (stage outcomes, sync events)
    leaderboard.jsonl                append-only candidate records
    benchmark_specs/                 5 JSON spec slots
    hardware/hardware_profile.json
    baseline/
        baseline.json
        inputs/{W,X,A,B}_d{N}.pt     LoRA-specific filenames (see operator extension below)
        references/Y_d{N}.pt
    candidates/candidate_NNN/
        candidate.cu
        compile.json correctness.json
        quick_benchmark.json confirm_benchmark.json
        profile.json analysis.md     only for the chosen best
    best/{best.cu, best_result.json} only the framework writes here
    final/{final_report.json, summary.md}

./optimized_lora.cu                  framework-synced from best/best.cu
```

`RunLayout` ([pipeline/workspace_layout.py](pipeline/workspace_layout.py))
owns every path. Subdirs are created by `mkdir()` once at INIT and re-created
idempotently on resume. **Only the framework writes to `best/`** — agents
write to `candidates/` and the orchestrator promotes via `shutil.copyfile`.

## Output contract — `./optimized_lora.cu`

The Phase-2 harness reads exactly one file:

```python
# inside the official harness
mod = torch.utils.cpp_extension.load(name=..., sources=["./optimized_lora.cu"], extra_cuda_cflags=["-O3"])
mod.forward(W, X, A, B)
```

The orchestrator copies `best/best.cu` to `./optimized_lora.cu` at two
moments:

1. After INITIAL_CANDIDATE produces a `compile_ok=True, correctness_ok=True`
   candidate, even if no `best_update` was claimed (floor guarantee).
2. Every time `_promote_to_best` accepts a `best_update` candidate.

This means:

- **A timed-out run still has a submittable file.** As long as
  INITIAL_CANDIDATE succeeded once, the root file is non-empty and compileable.
- The local `evaluate_candidate` tool uses the same `cpp_extension.load`
  invocation as the harness — best-locally ⇒ best-at-eval (no harness gap).

## Resume

```bash
python main.py --spec ... --run-id 20260502_153012_ab12
```

`PipelineOrchestrator._init_or_resume`:

- Reads `workspace/runs/<run_id>/state.json` if present, deserializes
  `RunState`.
- `_elapsed_at_start = state.elapsed_s`; wall clock continues from there.
- `mkdir()` is idempotent — skeleton dirs are recreated if missing.
- `next_stage()` automatically skips setup stages whose artifacts exist on
  disk (the predicates `has_hardware_profile`, `has_baseline`, etc.).

Setup stages (BENCHMARK_SPEC / HARDWARE_PROFILE / BASELINE_PROFILE) gate on
artifact presence, so even a half-completed earlier run resumes cleanly.

## Failure policy

| Type | Handling |
| --- | --- |
| Stage agent crashes mid-loop | `stage_runner` converts to `status=failed` StageResult with traceback in caveats; orchestrator keeps going |
| Setup stage fails / submits partial | Artifact predicate stays False; orchestrator either re-enters (if budget allows next iteration) or finalizes (if loop exhausted) |
| Candidate compile fails | `evaluate_candidate` returns success with `summary.compile_ok=False`; agent decides to retry (≤2) or submit `accepted_for=null` |
| Candidate correctness fails | Same; agent should not submit `best_update` |
| `accepted_for="best_update"` but compile_ok or correctness_ok is False | `submit_candidate_result` rejects with `INVALID_ARGS` (defensive — broken kernels never reach `optimized_lora.cu`) |
| Confirm benchmark unstable | Agent should downgrade to `quick_ranking` or `strategy_guidance` |
| Time budget exceeded mid-stage | Soft: stage gets a `stage_overran` caveat. Hard: `next_stage` returns FINALIZE on the next iteration |
| `MAX_LOOP_ITERATIONS` hit (200) | Orchestrator emits `loop_guard_tripped` event and finalizes — protects tests / pathological infinite agent flows |

## Adding a new operator

The framework supports it in principle; two tool modules need parameterization
because their embedded PyTorch subprocess scripts hard-code LoRA's tensors
and reference formula.

### What is operator-agnostic (no changes needed)

- `PipelineOrchestrator`, `next_stage` rules, `StageResult` schema
- `RunState`, `RunLayout` skeleton
- `read_skill`, the agent registration mechanism, `StageToolFactory`
- All `submit_*` tools (they only persist what the agent gives them)

### What is LoRA-specific today

| Site | Hard-coded item |
| --- | --- |
| [pipeline/tools/baseline_tools.py:_build_baseline_script](pipeline/tools/baseline_tools.py) | Generates `W,X,A,B` tensors with `(d,d), (d,d), (d,16), (d,16)`; computes `Y = W @ X + A @ (B.T @ X)` |
| [pipeline/tools/candidate_tools.py:_build_eval_script](pipeline/tools/candidate_tools.py) | Loads `W,X,A,B` from `baseline_inputs_dir`, calls `mod.forward(W, X, A, B)`, computes the same reference |
| [pipeline/workspace_layout.py:baseline_input_path](pipeline/workspace_layout.py) | File-naming pinned to `W_d{d}.pt`, `X_d{d}.pt`, `A_d{d}.pt`, `B_d{d}.pt` |
| Three agent SYSTEM_PROMPTs | Mention LoRA / `forward(W,X,A,B)` / `d ∈ [3584, 4608]` |

### Recommended path

Extend `skills/operators/<name>.md` with a machine-readable schema in the
frontmatter, then drive the two script generators off it:

```yaml
---
name: operators/<name>
description: ...
inputs:
  - {name: W, shape: [d, d], dtype: float32}
  - {name: X, shape: [d, d], dtype: float32}
  - {name: A, shape: [d, 16], dtype: float32}
  - {name: B, shape: [d, 16], dtype: float32}
output: {name: Y, shape: [d, d], dtype: float32}
reference_pytorch: "W @ X + A @ (B.transpose(0,1).contiguous() @ X)"
forward_signature: "torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B)"
d_range: [3584, 4608]
---
```

Then change `_build_baseline_script` and `_build_eval_script` to read this
schema and emit a templated subprocess script. Estimated ~250 lines of work
spread over those two functions, `RunLayout` (generic per-tensor file naming),
and the three SYSTEM_PROMPTs that quote LoRA specifics.

This is **not** required for Phase-2 LoRA submission — only revisit when a
second operator lands.

## Adding a new stage

1. Add `Stage.NEW_STAGE` to the enum in `pipeline/state.py`.
2. Update `next_stage()` rules in `pipeline/transitions.py`.
3. Decide artifact predicate(s) for resume — add `RunLayout.has_*` helper.
4. Implement `pipeline/agents/<new>_agent.py` subclassing `LLMStageAgent`.
5. Add any new `submit_<thing>` tool to `pipeline/tools/finalize_tools.py`
   and register it in `StageToolFactory._build_all_tools`.
6. Wire the agent into `main.py` `stage_agents` dict.
7. Tests: add to `tests/pipeline/test_state_machine.py` (transition rules)
   and ideally a `test_<new>_agent.py` with a mock LLM.

No changes to `PipelineOrchestrator` are required if the stage fits the
"agent runs ReAct loop, returns StageResult" mold.

## Key files

| File | Purpose |
| --- | --- |
| [pipeline/state.py](pipeline/state.py) | `Stage` enum, `RunState`, `StageResult` + `validate_stage_result`, `CandidateRecord`, `BENCHMARK_SPEC_SLOTS` |
| [pipeline/workspace_layout.py](pipeline/workspace_layout.py) | `RunLayout` — every path under `runs/<run_id>/`; `make_run_id()`, `candidate_id(N)` |
| [pipeline/transitions.py](pipeline/transitions.py) | Pure `next_stage(state, has_*)` function |
| [pipeline/stage_agent.py](pipeline/stage_agent.py) | `StageAgent` ABC |
| [pipeline/stage_runner.py](pipeline/stage_runner.py) | `run_stage()` — builds receipt-restricted ToolRegistry, runs agent, validates schema |
| [pipeline/orchestrator.py](pipeline/orchestrator.py) | `PipelineOrchestrator` — main loop, state.json persistence, leaderboard append, best promotion, `optimized_lora.cu` sync |
| [pipeline/agent_loop_signal.py](pipeline/agent_loop_signal.py) | `stash_stage_result` / `pop_stage_result` bridge to existing `AgentLoop` |
| [pipeline/tool_factory.py](pipeline/tool_factory.py) | `StageToolFactory.build(allowed_tools, current_stage)` |
| [pipeline/agents/_base.py](pipeline/agents/_base.py) | `LLMStageAgent` — common AgentLoop wiring + failed-StageResult fallback |
| [pipeline/agents/](pipeline/agents/) | The 6 concrete agents |
| [pipeline/tools/spec_tools.py](pipeline/tools/spec_tools.py) | `SubmitBenchmarkSpecsTool` |
| [pipeline/tools/baseline_tools.py](pipeline/tools/baseline_tools.py) | `GenerateBaselineTool`, `SubmitBaselineTool` (LoRA-specific subprocess script) |
| [pipeline/tools/candidate_tools.py](pipeline/tools/candidate_tools.py) | `WriteCandidateTool`, `EvaluateCandidateTool`, `SubmitCandidateResultTool` (LoRA-specific eval script) |
| [pipeline/tools/finalize_tools.py](pipeline/tools/finalize_tools.py) | `SubmitHardwareProfileTool`, `SubmitProfileAnalysisTool`, `SubmitSummaryTool` |
| [pipeline/tools/_script_runner.py](pipeline/tools/_script_runner.py) | `run_python_script` + `parse_marked_json` (`=== EVAL_RESULT ===` style) |
| [main.py](main.py) | CLI entry; assembles Executor + LLMClient + factory + agents + PipelineOrchestrator |
| [run.sh](run.sh) | Phase-2 entry; `python main.py --spec target_spec.json --time-budget 1800 --output ./optimized_lora.cu --verbose` |
| [agents/tools/cuda_executor.py](agents/tools/cuda_executor.py) | `Executor` — compile / run / profile_ncu / profile_nsys / profile_with_torch; global `_gpu_lock` |
| [agents/tools/executor/](agents/tools/executor/) | Subprocess runner, nvcc, ncu, error classifier, sandbox `_Workspace` |
| [agents/core/loop.py](agents/core/loop.py) | `AgentLoop` — used by every `LLMStageAgent` |
| [agents/core/types.py](agents/core/types.py) | `AgentContext`, `MemoryStore`, `EventLog`, `SharedStore` |
| [agents/core/llm.py](agents/core/llm.py) | `LLMClient`, `ChatResponse`, `ToolCall` |
| [agents/tools/registry.py](agents/tools/registry.py) | `ToolRegistry` — circuit breaker + JSON schema validation, `_Terminated` exception |
| [skills/operators/lora_matmul.md](skills/operators/lora_matmul.md) | Operator definition; agents read via `read_skill("operators/lora_matmul")` |

## Real-time verbose output

`--verbose` enables prefixed output to stderr:

```text
[main] run_id=20260502_153012_ab12 budget=1800s workspace=... output=./optimized_lora.cu
[orch] stage=BENCHMARK_SPEC budget=120.0s elapsed=0.0s
[W sub_benchmark_spec] ── iter 1/8 ──
[W sub_benchmark_spec]   call: read_skill {"name":"operators/lora_matmul"}
[W sub_benchmark_spec]   call: submit_benchmark_specs {"specs":<...>}
[orch] stage=HARDWARE_PROFILE budget=180.0s elapsed=15.2s
...
```

Large-text args (`source`, `python_code`) are replaced with `<N chars>`.
All prints are flushed for piped output.

## Executor error handling (unchanged from Phase 1)

Every `ExecutorError` carries an `error_class`:

- `"user_code"` — CUDA source is wrong → LLM should fix the kernel
- `"infrastructure"` — binary missing or env misconfigured → stop retrying
- `"timeout"` — reduce workload

`_gpu_lock` serializes every GPU operation (compile, run, ncu, nsys), so
measurements are isolated even when several stages are queued.

## Circuit breaker

Per-StageAgent (each has its own `CircuitBreaker` instance via the
`AgentContext` it constructs). Default threshold = 3 consecutive failures
per `(tool_name, error_kind)` → tool returns `status=circuit_open` until
`half_open_timeout_s` (60 s) elapses and a probe is allowed through.

## Config env vars

| Var | Default | Purpose |
| --- | --- | --- |
| `API_KEY`, `BASE_MODEL` | — | LLM credentials (`agents/core/config.py:LLMConfig.from_env`) |
| `AGENT_LLM_MAX_TOKENS` | 8192 | Per-call token cap |
| `AGENT_MAX_ITERATIONS` | 40 | Floor on AgentLoop iterations; each StageAgent further caps via its own `max_iterations` |
| `AGENT_CB_THRESHOLD` | 3 | CircuitBreaker open-state threshold |
| `AGENT_HALF_OPEN_TIMEOUT_S` | 60 | Seconds before a probe is allowed through an open circuit |

CLI flags override env (`--time-budget`, `--workspace`, `--output`,
`--run-id`, `--verbose`).

## GPU notes (RTX 5060 / Blackwell sm_120)

- `clock64()` requires `-arch=sm_120` (auto-detected)
- `cudaDeviceProp.clockRate` removed in CUDA 13 — do not use in skill
  templates
- Use `cudaEvent` for wall-clock timing, `clock64()` for cycle counting
- `evaluate_candidate` always passes `extra_cuda_cflags=["-O3"]` to mirror
  the official harness; do not change without checking Phase-2 docs

## Tests

```bash
pytest tests/                           # 224 tests; ~22s on a laptop
pytest tests/pipeline/                  # 105 — pipeline only (no GPU required)
pytest tests/pipeline/test_e2e_smoke.py # 2 — full pipeline with mock LLM + mock executor
pytest -m cuda                          # GPU-marked tests; require nvcc + nvidia-smi
```

Pipeline tests are split by concern: `test_state_machine.py` (transitions),
`test_stage_runner.py` (schema enforcement), `test_orchestrator.py` (state
machine + best sync + resume), `test_*_tools.py` (each tool module),
`test_stage_agents.py` (each agent with mock LLM), `test_e2e_smoke.py`
(everything wired up). Old planner / critic / orchestrator tests were
removed when the pipeline replaced them — see commit aea0a8b.
