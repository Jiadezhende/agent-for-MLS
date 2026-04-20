"""
agents/agents/planner_agent.py — PlannerAgent: decomposes targets into Steps via one LLM call.

Returns list[Step]. Falls back to 1:1 mapping on failure.
"""
from __future__ import annotations

import json
import sys

from agents.core.agent import Agent
from agents.core.llm import LLMClient
from agents.core.types import Step

_SPLIT_THRESHOLD = 3       # targets per same-type group that triggers splitting
_MAX_WORKERS_PER_TYPE = 2  # hard cap; never split into more than 2 per type

# ---------------------------------------------------------------------------
# Prompts (inlined from agents/planner/prompt.py)
# ---------------------------------------------------------------------------

_PLANNER_BASE_PROMPT = """\
You are a multi-agent task planner. Given a list of targets, group them by
agent type domain and assign each group to workers.

Available agent types:
{agent_types_block}

You MUST respond with a JSON object in this exact format:
{{
  "workers": [
    {{
      "id": "step_0",
      "worker": "<agent_type>",
      "targets": ["<target_name>", "<target_name>"]
    }}
  ]
}}

Rules:
- Each worker has a unique id starting from "step_0"
- "worker" must be one of the available agent types listed above
- "targets" is a JSON array of target name strings — one entry per target, no comma-joining
- For each agent type group with 1-2 targets: use ONE worker entry
- For each agent type group with 3+ targets: create TWO worker entries of the same type,
  splitting targets as evenly as possible (e.g. 5 targets → 3 + 2)
- Never create more than 2 workers for the same agent type
- Do not include any text outside the JSON object\
"""

_FALLBACK_PROMPT = """\
You are a GPU benchmark task planner. Given a list of target metrics,
assign them all to one hardware_probe worker.

You MUST respond with a JSON object in this exact format:
{
  "workers": [
    {
      "id": "step_0",
      "worker": "hardware_probe",
      "targets": ["<target_name>"]
    }
  ]
}

Rules:
- Put all targets in a single worker
- Do not include any text outside the JSON object\
"""


def _build_system_prompt(agent_registry: dict) -> str:
    if not agent_registry:
        return _FALLBACK_PROMPT
    lines = []
    for defn in agent_registry.values():
        lines.append(f"### {defn.agent_type}")
        lines.append(f"  {defn.description}")
        lines.append(defn.planner_hints)
        lines.append("")
    agent_types_block = "\n".join(lines).rstrip()
    return _PLANNER_BASE_PROMPT.format(agent_types_block=agent_types_block)


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------

class PlannerAgent(Agent):
    """Single LLM call that maps targets → list[Step].

    The LLM is asked to respond with a JSON object containing a 'steps' list.
    Falls back to 1:1 mapping (one Step per target) on any failure.
    """

    def __init__(
        self,
        llm: LLMClient,
        agent_registry: dict,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.agent_registry = agent_registry
        self.verbose = verbose

    def run(self, targets: list) -> list[Step]:
        target_names = _normalize_target_names(targets)

        if not target_names:
            return []

        default_agent_type = next(iter(self.agent_registry), "hardware_probe")

        if len(target_names) == 1:
            return [Step(id="step_0", task=target_names[0], worker=default_agent_type)]

        system_prompt = _build_system_prompt(self.agent_registry)
        user_msg = (
            "Assign these targets to workers:\n"
            + "\n".join(f"  - {t}" for t in target_names)
            + "\n\nRespond with the JSON object only."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ]

        try:
            resp = self.llm.chat(messages, tools=None)
            raw = resp.content or ""
            plan = self._parse_json(raw, target_names, default_agent_type)
        except Exception as exc:
            if self.verbose:
                print(f"[planner] LLM call failed ({exc}); falling back to 1:1.", file=sys.stderr)
            plan = _fallback_plan(target_names, default_agent_type)

        plan = _maybe_split_steps(plan)

        if self.verbose:
            print(
                f"[planner] {len(plan)} step(s): "
                + ", ".join(f"{s.id}({s.worker})={s.task}" for s in plan),
                file=sys.stderr,
                flush=True,
            )

        return plan

    def _parse_json(
        self, raw: str, target_names: list[str], default_agent_type: str
    ) -> list[Step]:
        try:
            text = raw.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            data = json.loads(text)
            raw_workers = data.get("workers", data.get("steps", []))
            if not raw_workers:
                return _fallback_plan(target_names, default_agent_type)

            valid_types = set(self.agent_registry.keys()) or {"hardware_probe"}
            steps: list[Step] = []
            covered: set[str] = set()
            for i, item in enumerate(raw_workers):
                worker = item.get("worker", default_agent_type)
                if worker not in valid_types:
                    worker = default_agent_type
                targets = item.get("targets", [])
                if isinstance(targets, str):
                    targets = [t.strip() for t in targets.split(",")]
                targets = [t for t in targets if t and t not in covered]
                covered.update(targets)
                if not targets:
                    continue
                steps.append(Step(
                    id=item.get("id", f"step_{i}"),
                    worker=worker,
                    targets=targets,
                    task=", ".join(targets),
                ))

            for t in target_names:
                if t not in covered:
                    steps.append(Step(
                        id=f"step_{len(steps)}",
                        worker=default_agent_type,
                        targets=[t],
                        task=t,
                    ))

            return steps

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            if self.verbose:
                print(f"[planner] JSON parse error ({exc}); falling back.", file=sys.stderr)
            return _fallback_plan(target_names, default_agent_type)


def _maybe_split_steps(steps: list[Step]) -> list[Step]:
    """Split any Step with >= _SPLIT_THRESHOLD targets into two Steps.

    Creates new Step objects with sequential ids — no in-place mutation.
    Runs in Phase 1 (single-threaded) before any workers are spawned.
    """
    result: list[Step] = []
    for step in steps:
        if len(step.targets) >= _SPLIT_THRESHOLD:
            mid = (len(step.targets) + 1) // 2
            groups = [step.targets[:mid], step.targets[mid:]]
        else:
            groups = [step.targets]
        for targets in groups:
            result.append(Step(
                id=f"step_{len(result)}",
                worker=step.worker,
                targets=list(targets),
                task=", ".join(targets),
            ))
    return result


def _fallback_plan(target_names: list[str], default_agent_type: str) -> list[Step]:
    return [Step(
        id="step_0",
        worker=default_agent_type,
        targets=target_names,
        task=", ".join(target_names),
    )]


def _normalize_target_names(targets: list) -> list[str]:
    names: list[str] = []
    for t in targets:
        if isinstance(t, dict):
            names.append(t.get("name", str(t)))
        else:
            names.append(str(t))
    return names
