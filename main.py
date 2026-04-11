"""
main.py — CLI entry point for the GPU profiling agent.

Usage:
    python main.py --spec target_spec.json --output results.json

The agent reads the target spec, runs the LLM loop (which calls the Executor
via registered tools), and writes results.json + reasoning_log.json.
"""
from __future__ import annotations

import argparse
import json
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
    # Load .env before any config reads
    load_dotenv()

    args = parse_args()

    # Import after load_dotenv so env vars are available
    from agent.loop import AgentLoop
    from agent.tool_registry import build_default_registry
    from agent.types import AgentContext, CircuitBreaker, MemoryStore, Task
    from config import AgentConfig, ExecutorConfig, LLMConfig
    from executor import Executor
    from llm.client import LLMClient

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

    ctx = AgentContext(
        task=task,
        memory=MemoryStore(),
        circuit_breaker=CircuitBreaker(threshold=agent_cfg.circuit_breaker_threshold),
    )

    executor = Executor(
        exec_cfg,
        on_job_complete=lambda r: ctx.job_history.append(r.to_log_dict()),
    )
    # Always print auto-detection notes so users know what was found/missing
    for note in executor.detect_notes:
        print(note, file=sys.stderr)

    llm      = LLMClient(llm_cfg)
    registry = build_default_registry(executor)
    loop     = AgentLoop(llm, registry, ctx, max_iterations=agent_cfg.max_iterations)

    if args.verbose:
        print(
            f"[info] Starting agent: model={llm_cfg.model} "
            f"max_iterations={agent_cfg.max_iterations} "
            f"workspace={executor.workspace.root}",
            file=sys.stderr,
        )

    # --- Run loop ---------------------------------------------------------
    exit_code = 0
    try:
        loop.run()
        if args.verbose:
            print("[info] Agent completed successfully.", file=sys.stderr)
    except RuntimeError as exc:
        # Budget exhausted or protocol error — still write partial results.
        print(f"[warn] {exc}", file=sys.stderr)
        exit_code = 3
    except Exception as exc:
        print(f"[error] Unexpected error: {exc}", file=sys.stderr)
        exit_code = 1

    # --- Write outputs ----------------------------------------------------
    output_path = Path(args.output)
    log_path    = output_path.with_name("reasoning_log.json")

    try:
        serialized = ctx.serialize()

        output_path.write_text(
            json.dumps(serialized["results"], indent=2, default=str),
            encoding="utf-8",
        )

        log_data = {
            "reasoning_log": serialized["reasoning_log"],
            "events":        serialized["events"],
            "job_history":   serialized["job_history"],
            "summary":       ctx.memory.get("run", "summary"),
        }
        log_path.write_text(
            json.dumps(log_data, indent=2, default=str),
            encoding="utf-8",
        )

        if args.verbose or exit_code != 0:
            print(f"[info] Wrote {output_path} and {log_path}", file=sys.stderr)

    except Exception as exc:
        print(f"[error] Failed to write outputs: {exc}", file=sys.stderr)
        exit_code = max(exit_code, 1)

    # --- Workspace cleanup ------------------------------------------------
    if exit_code == 0 and not agent_cfg.keep_workspace:
        executor.workspace.cleanup()
    else:
        print(
            f"[info] Workspace retained at: {executor.workspace.root}",
            file=sys.stderr,
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
