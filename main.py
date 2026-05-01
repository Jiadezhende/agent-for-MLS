"""
main.py — CLI entry point for the GPU profiling agent.

Usage:
    python main.py --spec target_spec.json --output results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autonomous GPU hardware profiling agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--spec",   required=True,  help="Path to target_spec.json")
    p.add_argument("--output", default="results.json", help="Output path for results.json")
    p.add_argument("--max-iterations", type=int, default=None,
                   help="Override AGENT_MAX_ITERATIONS from env")
    p.add_argument("--keep-workspace", action="store_true",
                   help="Retain workspace/run_* directory after completion")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print iteration progress to stderr")
    return p.parse_args()


def main() -> None:
    load_dotenv()

    args = parse_args()

    # Import after load_dotenv so env vars are available
    import agents  # noqa: F401  triggers registration of all agent plugins
    from agents._registry import all_definitions
    from agents.core.config import AgentConfig, ExecutorConfig, LLMConfig
    from agents.core.llm import LLMClient
    from agents.core.types import Task
    from agents.tools.cuda_executor import Executor
    from orchestrator import Orchestrator

    # --- Config -----------------------------------------------------------
    try:
        llm_cfg   = LLMConfig.from_env()
        agent_cfg = AgentConfig.from_env()
        exec_cfg  = ExecutorConfig.from_env()
    except EnvironmentError as exc:
        print(f"[error] Configuration error: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.keep_workspace:
        agent_cfg.keep_workspace = True

    # --- Load spec --------------------------------------------------------
    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"[error] Spec file not found: {spec_path}", file=sys.stderr)
        sys.exit(2)

    try:
        target_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[error] Invalid JSON in spec file: {exc}", file=sys.stderr)
        sys.exit(2)

    if "targets" not in target_spec:
        print("[warn] Spec file has no 'targets' key.", file=sys.stderr)

    # --- Build components -------------------------------------------------
    task = Task(
        id=str(uuid.uuid4()),
        type="hardware_probe",
        description="Measure requested GPU hardware parameters",
        payload=target_spec,
        constraints={},
    )

    executor = Executor(exec_cfg)
    for note in executor.detect_notes:
        print(note, file=sys.stderr)

    llm = LLMClient(llm_cfg)

    if args.verbose:
        print(
            f"[info] Starting orchestrator: model={llm_cfg.model} "
            f"max_iterations={agent_cfg.max_iterations} "
            f"workspace={executor.workspace.root}",
            file=sys.stderr,
        )

    orchestrator = Orchestrator(
        llm=llm,
        executor=executor,
        task=task,
        agent_cfg=agent_cfg,
        agent_registry=all_definitions(),
        verbose=args.verbose,
    )

    # --- Run --------------------------------------------------------------
    exit_code = 0
    state = None

    try:
        state = orchestrator.run()
        if args.verbose:
            print("[info] Orchestrator completed successfully.", file=sys.stderr)
    except RuntimeError as exc:
        print(f"[warn] {exc}", file=sys.stderr)
        exit_code = 3
    except Exception as exc:
        print(f"[error] Unexpected error: {exc}", file=sys.stderr)
        exit_code = 1

    # --- Collect results --------------------------------------------------
    all_results: list[dict] = []
    worker_logs: list[dict] = []

    if state is not None:
        raw_results: list[dict] = []
        for step in state.steps:
            out = state.outputs.get(step.id)
            if out is None:
                continue
            raw_results.extend(out.results)
            attempts = state.history.get(step.id, [out])
            worker_logs.append({
                "step_id":      step.id,
                "task":         step.task,
                "worker":       step.worker,
                "success":      out.success,
                "n_results":    len(out.results),
                "summary":      out.summary,
                "attempts": [
                    {
                        "reasoning_log": a.reasoning_log,
                        "events":        a.events,
                        "n_results":     len(a.results),
                        "success":       a.success,
                        "summary":       a.summary,
                    }
                    for a in attempts
                ],
            })

        # Deduplicate by metric: keep the entry with the highest confidence
        seen: dict[str, dict] = {}
        for r in raw_results:
            metric = r.get("metric", "")
            if metric not in seen or r.get("confidence", 0) > seen[metric].get("confidence", 0):
                seen[metric] = r
        all_results = list(seen.values())

        failed_steps = [s.id for s in state.steps if not state.outputs.get(s.id, None) or
                        not state.outputs[s.id].success]
        if failed_steps:
            exit_code = max(exit_code, 3)

    # --- Write outputs ----------------------------------------------------
    output_path  = Path(args.output)
    log_path     = output_path.with_name("reasoning_log.json")
    run_log_path = output_path.with_name("run_log.jsonl")

    flat_results: dict[str, int | float] = {}
    for r in all_results:
        raw = r.get("value")
        if raw is None:
            continue
        try:
            fval = float(raw)
            val: int | float = int(fval) if fval == int(fval) else fval
        except (TypeError, ValueError):
            continue
        flat_results[r["metric"]] = val

    try:
        output_path.write_text(
            json.dumps(flat_results, indent=2),
            encoding="utf-8",
        )
        log_path.write_text(
            json.dumps({"workers": worker_logs}, indent=2, default=str),
            encoding="utf-8",
        )
        orchestrator.run_ctx.event_log.flush(run_log_path)
        if args.verbose or exit_code != 0:
            print(
                f"[info] Wrote {output_path}, {log_path}, and {run_log_path}",
                file=sys.stderr,
            )

    except Exception as exc:
        print(f"[error] Failed to write outputs: {exc}", file=sys.stderr)
        exit_code = max(exit_code, 1)

    # --- Workspace cleanup ------------------------------------------------
    if exit_code == 0 and not agent_cfg.keep_workspace:
        executor.workspace.cleanup()
    else:
        print(f"[info] Workspace retained at: {executor.workspace.root}", file=sys.stderr)

    os._exit(exit_code)


if __name__ == "__main__":
    main()
