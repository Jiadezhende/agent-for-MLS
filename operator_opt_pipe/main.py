"""CLI entry: ``python -m operator_opt_pipe.main --spec ... --time-budget ...``.

Wires environment-driven configuration (``LLMConfig``, ``ExecutorConfig``)
to a ``PipelineOrchestrator`` instance. The legacy ``run.sh`` continues to
point at the old ``main.py`` until the new pipeline finishes its lora_resources
implementation.
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

from operator_opt_pipe.lora_resources.contract import load_contract
from operator_opt_pipe.orchestrator import PipelineOrchestrator


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="operator_opt_pipe",
        description="Autonomous CUDA-operator optimization pipeline.",
    )
    p.add_argument("--spec", required=True, help="Path to target_spec.json (must contain 'operator').")
    p.add_argument("--time-budget", type=float, default=1800.0,
                   help="Wall-clock time budget in seconds (default 1800).")
    p.add_argument("--workspace", default=os.getenv("AGENT_WORKSPACE_ROOT", "./workspace"),
                   help="Workspace root directory.")
    p.add_argument("--output", default="./optimized_lora.cu",
                   help="Path to the file synced from best/best.cu.")
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
    args = _parse_args(argv)

    spec_path = Path(args.spec)
    if not spec_path.is_file():
        print(f"error: spec file not found: {spec_path}", file=sys.stderr)
        return 2
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    operator = spec.get("operator")
    if not operator:
        print(f"error: spec.json missing 'operator' field; got: {spec}", file=sys.stderr)
        return 2

    contract = load_contract(args.skills_root, operator=operator)

    llm_cfg = LLMConfig.from_env()
    backend = OpenAIBackend(llm_cfg)
    agent_cfg = AgentConfig(max_iterations=args.max_agent_iterations)

    exec_cfg = ExecutorConfig.from_env()
    exec_cfg.workspace_root = args.workspace
    executor = Executor(exec_cfg)

    orchestrator = PipelineOrchestrator(
        spec=spec,
        time_budget_s=args.time_budget,
        workspace_root=args.workspace,
        output_path=args.output,
        backend=backend,
        agent_cfg=agent_cfg,
        executor=executor,
        contract=contract,
        skills_dir=args.skills_root,
        run_id=args.run_id,
        verbose=args.verbose,
    )
    summary = orchestrator.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
