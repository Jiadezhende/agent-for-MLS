"""CLI entry: ``python -m operator_opt_pipe.main --operator lora_matmul ...``.

Wires environment-driven configuration (``LLMConfig``, ``ExecutorConfig``)
to a ``PipelineOrchestrator`` instance.

The operator is selected by name (CLI ``--operator``, default
``lora_matmul``); the matching ``operator_opt_pipe.operators.<name>``
module supplies the ``OperatorContract``. The optional skill markdown at
``skills/operators/<name>.md`` is read by the LLM (via ``read_skill``)
for tuning narrative only — it does not affect the contract.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from mls_agent import AgentConfig, OpenAIBackend
from mls_agent.llm.config import LLMConfig
from mls_agent.tools.cuda.config import ExecutorConfig
from mls_agent.tools.cuda.cuda_executor import Executor

from operator_opt_pipe.operators import load_contract, load_ops
from operator_opt_pipe.orchestrator import PipelineOrchestrator
from operator_opt_pipe.state import RunLayout, make_run_id


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="operator_opt_pipe",
        description="Autonomous CUDA-operator optimization pipeline.",
    )
    p.add_argument("--operator", default="lora_matmul",
                   help="Operator name; resolves to skills/operators/<name>.md.")
    p.add_argument("--time-budget", type=float, default=1800.0,
                   help="Wall-clock time budget in seconds (default 1800 = 30 min).")
    p.add_argument("--workspace", default=os.getenv("AGENT_WORKSPACE_ROOT", "./workspace"),
                   help="Workspace root directory.")
    p.add_argument("--output", default="./optimized_lora.cu",
                   help="Path to the file synced from best/best.cu (Phase-2 contract).")
    p.add_argument("--skills-root", default="./skills",
                   help="Directory containing skills/operators/<name>.md.")
    p.add_argument("--run-id", default=None, help="Resume an existing run by id.")
    p.add_argument("--verbose", action="store_true",
                   help="Stream stage progress and per-iteration agent output to stderr.")
    p.add_argument("--max-agent-iterations", type=int,
                   default=int(os.getenv("AGENT_MAX_ITERATIONS") or "30"),
                   help="Cap on iterations for each LLM agent run.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Optional .env loading (no-op if python-dotenv missing)
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except ImportError:
        pass

    args = _parse_args(argv)

    try:
        contract = load_contract(args.operator)
        ops = load_ops(args.operator)
    except FileNotFoundError as exc:
        print(f"error: load_contract: {exc}", file=sys.stderr)
        return 2

    try:
        llm_cfg = LLMConfig.from_env()
    except EnvironmentError as exc:
        print(f"error: LLMConfig.from_env: {exc}", file=sys.stderr)
        return 2
    backend = OpenAIBackend(llm_cfg)
    # max_consecutive_no_tool_call=1: as soon as the agent stops calling
    # tools, terminate. The default (2) wastes a turn waiting for a second
    # silent reply that adds nothing after a successful write_blackboard.
    agent_cfg = AgentConfig(
        max_iterations=args.max_agent_iterations,
        max_consecutive_no_tool_call=1,
    )

    # Pin run_id up front so Executor + Orchestrator share the same path.
    run_id = args.run_id or make_run_id()
    layout = RunLayout(workspace_root=Path(args.workspace).resolve(), run_id=run_id)
    layout.mkdir()

    exec_cfg = ExecutorConfig.from_env()
    exec_cfg.workspace_root = str(layout.exec_dir)
    executor = Executor(exec_cfg)

    orchestrator = PipelineOrchestrator(
        operator=args.operator,
        time_budget_s=args.time_budget,
        workspace_root=args.workspace,
        output_path=args.output,
        backend=backend,
        agent_cfg=agent_cfg,
        executor=executor,
        contract=contract,
        ops=ops,
        skills_dir=args.skills_root,
        run_id=run_id,
        verbose=args.verbose,
    )
    summary = orchestrator.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
