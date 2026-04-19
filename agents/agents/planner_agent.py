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


# ---------------------------------------------------------------------------
# Prompts (inlined from agents/planner/prompt.py)
# ---------------------------------------------------------------------------

_PLANNER_BASE_PROMPT = """\
You are a multi-agent task planner. Given a list of targets, assign each to
the most appropriate agent type and group related targets into workers.
Workers run in parallel, so targets that share measurement infrastructure
may be grouped to reduce overhead. Independent targets should go to separate
workers.

Available agent types:
{agent_types_block}

You MUST respond with a JSON object in this exact format:
{{
  "steps": [
    {{
      "id": "step_0",
      "task": "<target name or description>",
      "worker": "<agent_type>",
      "hints": ["<optional strategy hint>"]
    }}
  ]
}}

Rules:
- Each step has a unique id starting from "step_0"
- "worker" must be one of the available agent types listed above
- "hints" is optional; omit or leave empty if no specific hints apply
- Do not include any text outside the JSON object\
"""

_FALLBACK_PROMPT = """\
You are a GPU benchmark task planner. Given a list of target metrics,
assign each to a worker.

You MUST respond with a JSON object in this exact format:
{
  "steps": [
    {
      "id": "step_0",
      "task": "<target name>",
      "worker": "hardware_probe",
      "hints": []
    }
  ]
}

Rules:
- Each step has a unique id starting from "step_0"
- Default to "hardware_probe" as the worker type
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
            "Plan parallel workers for these targets:\n"
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
            raw_steps = data.get("steps", [])
            if not raw_steps:
                return _fallback_plan(target_names, default_agent_type)

            steps: list[Step] = []
            valid_types = set(self.agent_registry.keys()) or {"hardware_probe"}
            for item in raw_steps:
                worker = item.get("worker", default_agent_type)
                if worker not in valid_types:
                    worker = default_agent_type
                steps.append(Step(
                    id=item.get("id", f"step_{len(steps)}"),
                    task=item.get("task", ""),
                    worker=worker,
                    hints=item.get("hints", []),
                ))

            planned_tasks = {s.task for s in steps}
            for t in target_names:
                if t not in planned_tasks:
                    steps.append(Step(
                        id=f"step_{len(steps)}",
                        task=t,
                        worker=default_agent_type,
                    ))

            return steps

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            if self.verbose:
                print(f"[planner] JSON parse error ({exc}); falling back to 1:1.", file=sys.stderr)
            return _fallback_plan(target_names, default_agent_type)


def _fallback_plan(target_names: list[str], default_agent_type: str) -> list[Step]:
    return [
        Step(id=f"step_{i}", task=t, worker=default_agent_type)
        for i, t in enumerate(target_names)
    ]


def _normalize_target_names(targets: list) -> list[str]:
    names: list[str] = []
    for t in targets:
        if isinstance(t, dict):
            names.append(t.get("name", str(t)))
        else:
            names.append(str(t))
    return names
