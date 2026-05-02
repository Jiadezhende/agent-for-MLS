"""pipeline/state.py — RunState, StageResult, CandidateRecord.

Pure data + JSON (de)serialization. No file I/O, no scheduling logic — those
live in workspace_layout.py and orchestrator.py. Keep this module side-effect
free so it can be imported and tested without GPU / LLM dependencies.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Stage enum — single source of truth for pipeline phases.
# ---------------------------------------------------------------------------

class Stage(str, Enum):
    INIT = "INIT"
    BENCHMARK_SPEC = "BENCHMARK_SPEC"
    HARDWARE_PROFILE = "HARDWARE_PROFILE"
    BASELINE_PROFILE = "BASELINE_PROFILE"
    INITIAL_CANDIDATE = "INITIAL_CANDIDATE"
    TUNING_LOOP = "TUNING_LOOP"
    OPTIONAL_PROFILE = "OPTIONAL_PROFILE"
    FINALIZE = "FINALIZE"

    @classmethod
    def setup_stages(cls) -> tuple["Stage", ...]:
        # Stages whose artifacts can be reused on resume.
        return (cls.BENCHMARK_SPEC, cls.HARDWARE_PROFILE, cls.BASELINE_PROFILE)


# Status returned by every StageAgent.
StageStatus = str  # "success" | "partial" | "failed"
_VALID_STATUSES = {"success", "partial", "failed"}


# ---------------------------------------------------------------------------
# Required benchmark spec slots. BenchmarkSpecAgent must produce all of these
# before HARDWARE_PROFILE can run.
# ---------------------------------------------------------------------------

BENCHMARK_SPEC_SLOTS: tuple[str, ...] = (
    "hardware",
    "baseline",
    "candidate_quick",
    "candidate_confirm",
    "final",
)


# ---------------------------------------------------------------------------
# StageResult — what every StageAgent must return.
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    stage: str                       # Stage value (string form)
    status: str                      # "success" | "partial" | "failed"
    artifacts: dict[str, str] = field(default_factory=dict)  # name → workspace-relative path
    metrics: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    caveats: list[str] = field(default_factory=list)
    next_recommendation: str | None = None
    agent_trace: str | None = None   # path to per-stage / per-candidate trace file

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StageResult":
        return cls(
            stage=d["stage"],
            status=d["status"],
            artifacts=dict(d.get("artifacts") or {}),
            metrics=dict(d.get("metrics") or {}),
            confidence=float(d.get("confidence", 1.0)),
            caveats=list(d.get("caveats") or []),
            next_recommendation=d.get("next_recommendation"),
            agent_trace=d.get("agent_trace"),
        )


def validate_stage_result(d: Any) -> tuple[bool, list[str]]:
    """Schema check for a StageResult dict.

    Returns (ok, errors). Used by stage_runner to refuse advancing the
    pipeline when an agent returns a malformed result.
    """
    errors: list[str] = []
    if not isinstance(d, dict):
        return False, ["StageResult must be a dict"]

    stage = d.get("stage")
    if not isinstance(stage, str) or stage not in {s.value for s in Stage}:
        errors.append(f"stage must be one of {[s.value for s in Stage]}, got {stage!r}")

    status = d.get("status")
    if status not in _VALID_STATUSES:
        errors.append(f"status must be one of {sorted(_VALID_STATUSES)}, got {status!r}")

    arts = d.get("artifacts", {})
    if arts is not None and not isinstance(arts, dict):
        errors.append("artifacts must be a dict[str, str] or null")
    elif isinstance(arts, dict):
        for k, v in arts.items():
            if not isinstance(k, str) or not isinstance(v, str):
                errors.append(f"artifacts entry {k!r}: {v!r} must be str→str")
                break

    metrics = d.get("metrics", {})
    if metrics is not None and not isinstance(metrics, dict):
        errors.append("metrics must be a dict or null")

    conf = d.get("confidence", 1.0)
    try:
        cf = float(conf)
        if not (0.0 <= cf <= 1.0):
            errors.append(f"confidence must be in [0, 1], got {cf}")
    except (TypeError, ValueError):
        errors.append(f"confidence must be a number, got {conf!r}")

    caveats = d.get("caveats", [])
    if caveats is not None and not (isinstance(caveats, list) and all(isinstance(c, str) for c in caveats)):
        errors.append("caveats must be a list[str] or null")

    return (len(errors) == 0), errors


# ---------------------------------------------------------------------------
# CandidateRecord — one line in leaderboard.jsonl.
# ---------------------------------------------------------------------------

# accepted_for taxonomy. Drives best-update policy + final report inclusion.
ACCEPTED_FOR_VALUES = (
    None,                    # candidate failed correctness — never used
    "strategy_guidance",     # provisional; used to inform later candidates only
    "quick_ranking",         # passed quick benchmark but not yet confirmed
    "best_update",           # confirmed result that updated best/best.cu
    "final_report",          # included in final benchmark
)


@dataclass
class CandidateRecord:
    candidate_id: str
    compile_ok: bool
    correctness_ok: bool
    quick_speedup_median: float | None = None
    quick_samples: int | None = None
    quick_variance_pct: float | None = None
    confirm_speedup_median: float | None = None
    confirm_samples: int | None = None
    confirm_variance_pct: float | None = None
    accepted_for: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateRecord":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RunState — the single source of truth persisted to runs/<run_id>/state.json.
# ---------------------------------------------------------------------------

@dataclass
class RunState:
    run_id: str
    operator: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    time_budget_s: float = 1800.0
    elapsed_s: float = 0.0

    current_stage: str = Stage.INIT.value
    completed_stages: list[str] = field(default_factory=list)

    # benchmark spec slot → version tag (e.g. "v1"). All slots populated == specs_complete.
    benchmark_spec_versions: dict[str, str] = field(default_factory=dict)

    current_iteration: int = 0
    best_candidate_id: str | None = None
    best_speedup: float | None = None

    failure_counts: dict[str, int] = field(
        default_factory=lambda: {"compile": 0, "correctness": 0, "benchmark_unstable": 0}
    )

    # ---- (de)serialization ------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunState":
        valid_fields = set(cls.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**kwargs)  # type: ignore[arg-type]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> "RunState":
        return cls.from_dict(json.loads(text))

    # ---- queries ----------------------------------------------------------

    def specs_complete(self) -> bool:
        return all(slot in self.benchmark_spec_versions for slot in BENCHMARK_SPEC_SLOTS)

    def remaining_budget_s(self) -> float:
        return max(0.0, self.time_budget_s - self.elapsed_s)

    def stage_completed(self, stage: Stage | str) -> bool:
        s = stage.value if isinstance(stage, Stage) else stage
        return s in self.completed_stages

    # ---- mutations --------------------------------------------------------

    def mark_completed(self, stage: Stage | str) -> None:
        s = stage.value if isinstance(stage, Stage) else stage
        if s not in self.completed_stages:
            self.completed_stages.append(s)

    def set_current_stage(self, stage: Stage | str) -> None:
        self.current_stage = stage.value if isinstance(stage, Stage) else stage
