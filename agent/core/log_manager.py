"""agents/core/log_manager.py — Structured per-cycle audit log writer.

Solves the audit problem: instead of one giant reasoning_log.json mixing all
agents and cycles, every component gets its own file under a cycle-scoped dir.

Directory layout:
    {log_root}/{run_id}/
        manifest.json          — global index (written once on run completion)
        orchestrator.jsonl     — all orchestrator EventLog records
        cycle_1/
            planner.json       — planner reasoning_log + events for this cycle
            critic.json        — critic decisions + LLM reasoning traces
            workers/
                {agent_type}_{step_id}.json   — full WorkerOutput per agent call
        cycle_2/               — only present when Critic issues a retry
            ...

All per-cycle files are written incrementally (immediately after each component
finishes) so partial logs survive pipeline crashes.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.core.types import CriticDecision, WorkerOutput


@dataclass
class LogManager:
    """Crash-safe structured log writer for one pipeline run.

    Thread-safe: write_worker_log may be called concurrently from
    run_subagent_parallel threads; all _cycle_summaries mutations are
    protected by _lock.
    """

    run_id: str
    log_root: Path
    _current_cycle: int = field(default=0, init=False, repr=False)
    _cycle_summaries: list[dict] = field(default_factory=list, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        init=False,
        repr=False,
    )

    @property
    def run_dir(self) -> Path:
        return self.log_root / self.run_id

    @property
    def cycle_dir(self) -> Path:
        return self.run_dir / f"cycle_{self._current_cycle}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin_cycle(self, cycle: int) -> None:
        """Advance to a new cycle and pre-create its subdirectory."""
        with self._lock:
            self._current_cycle = cycle
        (self.cycle_dir / "workers").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-component writes (called by Orchestrator and subagent.py)
    # ------------------------------------------------------------------

    def write_planner_log(self, reasoning_log: list[dict], events: list[dict]) -> None:
        """Write cycle_N/planner.json immediately after the Planner loop exits."""
        data = {
            "cycle": self._current_cycle,
            "n_iterations": len(reasoning_log),
            "reasoning_log": reasoning_log,
            "events": events,
        }
        self._write_json(self.cycle_dir / "planner.json", data)
        with self._lock:
            self._cycle_summary()["planner_iterations"] = len(reasoning_log)

    def write_worker_log(self, output: "WorkerOutput") -> None:
        """Write one worker file under cycle_N/workers/. Thread-safe."""
        filename = f"{output.agent_type}_{output.step_id}.json"
        self._write_json(self.cycle_dir / "workers" / filename, output.to_dict())
        with self._lock:
            self._cycle_summary().setdefault("workers", []).append({
                "step_id": output.step_id,
                "agent_type": output.agent_type,
                "success": output.success,
                "n_results": len(output.results),
            })

    def write_critic_log(
        self,
        decisions: "list[CriticDecision]",
        llm_traces: list[dict],
    ) -> None:
        """Write cycle_N/critic.json immediately after Critic returns decisions."""
        failing = [d for d in decisions if d.decision == "retry"]
        data = {
            "cycle": self._current_cycle,
            "overall": "accept" if not failing else "retry",
            "decisions": [
                {
                    "step_id": d.step_id,
                    "decision": d.decision,
                    "confidence": d.confidence,
                    "reason": d.reason,
                    "failing_targets": d.failing_targets,
                }
                for d in decisions
            ],
            "llm_traces": llm_traces,
        }
        self._write_json(self.cycle_dir / "critic.json", data)
        with self._lock:
            s = self._cycle_summary()
            s["critic_decision"] = "accept" if not failing else "retry"
            s["failing_targets"] = [t for d in failing for t in (d.failing_targets or [])]

    def flush_orchestrator_events(self, records: list[dict]) -> None:
        """Write all EventLog records to orchestrator.jsonl (called once at run end)."""
        path = self.run_dir / "orchestrator.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")

    def write_manifest(self, operator: str, accepted: bool) -> None:
        """Write manifest.json summarising the full run. Call once at pipeline end."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": self.run_id,
            "operator": operator,
            "started_at": self._started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "accepted": accepted,
            "n_cycles": self._current_cycle,
            "cycles": self._cycle_summaries,
        }
        self._write_json(self.run_dir / "manifest.json", data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cycle_summary(self) -> dict:
        """Return the summary dict for the current cycle. Caller must hold _lock."""
        while len(self._cycle_summaries) < self._current_cycle:
            self._cycle_summaries.append({
                "cycle": len(self._cycle_summaries) + 1,
                "workers": [],
            })
        return self._cycle_summaries[self._current_cycle - 1]

    def _write_json(self, path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
