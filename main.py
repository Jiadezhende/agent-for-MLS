"""
main.py — CLI entry point for the LoRA-kernel optimization pipeline.

Constructs PipelineOrchestrator with a stage-state-machine driver, six
StageAgents, and an Executor that runs CUDA / PyTorch subprocess work. The
orchestrator owns ``./optimized_lora.cu``: every time best updates (and once
INITIAL_CANDIDATE produces something compileable + correct) the file is
mirrored from workspace/runs/<run_id>/best/best.cu to the path passed via
--output.

Typical use:
    python main.py --spec target_spec.json --time-budget 1800 --output ./optimized_lora.cu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA-kernel optimization pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--spec", required=True, help="Path to operator spec JSON (must contain 'operator').")
    p.add_argument("--time-budget", type=int, default=1800, help="Total wall-clock budget in seconds.")
    p.add_argument("--output", default="./optimized_lora.cu", help="Final candidate file path the official harness reads.")
    p.add_argument("--workspace", default="./workspace", help="Workspace root containing runs/<run_id>/.")
    p.add_argument("--run-id", default=None, help="Resume an existing run by id (matches workspace/runs/<run_id>/state.json).")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    # Imports after load_dotenv so env vars are observed.
    from agents.core.config import AgentConfig, ExecutorConfig, LLMConfig
    from agents.core.llm import LLMClient
    from agents.tools.cuda_executor import Executor

    from pipeline.agents import (
        BaselineAgent,
        BenchmarkSpecAgent,
        HardwareProfilerAgent,
        KernelTuningAgent,
        ProfileAnalysisAgent,
        SummaryAgent,
    )
    from pipeline.orchestrator import PipelineOrchestrator
    from pipeline.state import Stage
    from pipeline.tool_factory import StageToolFactory
    from pipeline.workspace_layout import RunLayout, make_run_id

    # ---- config ---------------------------------------------------------
    try:
        llm_cfg = LLMConfig.from_env()
        agent_cfg = AgentConfig.from_env()
        exec_cfg = ExecutorConfig.from_env()
    except EnvironmentError as e:
        print(f"[error] config: {e}", file=sys.stderr)
        sys.exit(2)

    # ---- spec ----------------------------------------------------------
    spec_path = Path(args.spec)
    if not spec_path.is_file():
        print(f"[error] spec not found: {spec_path}", file=sys.stderr)
        sys.exit(2)
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[error] spec is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)
    if "operator" not in spec:
        print("[error] spec is missing 'operator' field", file=sys.stderr)
        sys.exit(2)

    # ---- services ------------------------------------------------------
    executor = Executor(exec_cfg)
    for note in executor.detect_notes:
        print(note, file=sys.stderr)

    llm = LLMClient(llm_cfg)

    # Mint the run_id up front so layout, tool_factory, and orchestrator all
    # share the same identity. PipelineOrchestrator will resume if state.json
    # already exists for this id.
    run_id = args.run_id or make_run_id()
    layout = RunLayout(args.workspace, run_id)
    layout.mkdir()

    tool_factory = StageToolFactory(executor=executor, layout=layout)

    def build_tools(allowed, current_stage):
        return tool_factory.build(allowed, current_stage=current_stage)

    # ---- agents --------------------------------------------------------
    common = {"llm": llm, "agent_cfg": agent_cfg, "verbose": args.verbose}
    stage_agents = {
        Stage.BENCHMARK_SPEC:    BenchmarkSpecAgent(**common),
        Stage.HARDWARE_PROFILE:  HardwareProfilerAgent(**common),
        Stage.BASELINE_PROFILE:  BaselineAgent(**common),
        # Same class instance is fine; orchestrator passes current_stage to
        # tool_factory.build so SubmitCandidateResultTool tags correctly.
        Stage.INITIAL_CANDIDATE: KernelTuningAgent(**common, stage=Stage.INITIAL_CANDIDATE),
        Stage.TUNING_LOOP:       KernelTuningAgent(**common, stage=Stage.TUNING_LOOP),
        Stage.OPTIONAL_PROFILE:  ProfileAnalysisAgent(**common),
        Stage.FINALIZE:          SummaryAgent(**common),
    }

    # ---- orchestrator --------------------------------------------------
    if args.verbose:
        print(
            f"[main] run_id={run_id} budget={args.time_budget}s "
            f"workspace={layout.root} output={args.output}",
            file=sys.stderr,
        )

    orchestrator = PipelineOrchestrator(
        spec=spec,
        time_budget_s=args.time_budget,
        workspace_root=args.workspace,
        output_path=args.output,
        stage_agents=stage_agents,
        build_tools=build_tools,
        run_id=run_id,
        verbose=args.verbose,
    )

    # ---- run -----------------------------------------------------------
    exit_code = 0
    try:
        summary = orchestrator.run()
    except KeyboardInterrupt:
        print("[main] interrupted; state.json + workspace preserved for resume", file=sys.stderr)
        sys.exit(130)
    except Exception as e:  # noqa: BLE001 — entrypoint catches all
        print(f"[error] orchestrator crashed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    # ---- summary -------------------------------------------------------
    print(json.dumps(summary, indent=2))
    if not Path(args.output).is_file():
        print(
            f"[warn] {args.output} does not exist — official harness will fail "
            "(no compileable best candidate produced).",
            file=sys.stderr,
        )
        exit_code = max(exit_code, 3)

    os._exit(exit_code)


if __name__ == "__main__":
    main()
