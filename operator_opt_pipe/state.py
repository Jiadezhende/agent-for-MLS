"""state.py — Stage enum, RunState, RunLayout, blackboard, payload validation.

Everything related to "where things live on disk" and "what the run currently
looks like" is consolidated here. There is intentionally no ``StageResult``
dataclass: the result of an LLM stage is whatever ``ToolResponse.terminate_with(payload=...)``
attached, which surfaces as ``AgentResult.payload`` (a dict). ``check_submit_payload``
provides lightweight validation at the orchestrator/runner boundary.
"""
from __future__ import annotations

import json
import os
import secrets
import string
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Stage enum + round step markers
# ---------------------------------------------------------------------------


class Stage(str, Enum):
    INIT = "INIT"
    HARDWARE_PROFILE = "HARDWARE_PROFILE"
    BENCHMARK_BASELINE = "BENCHMARK_BASELINE"
    INITIAL_CANDIDATE = "INITIAL_CANDIDATE"
    TUNING_LOOP = "TUNING_LOOP"
    FINALIZE = "FINALIZE"


# Sub-step markers used inside RoundRunner. They are NOT top-level Stages —
# they only appear in blackboard["history"] entries and event records so that
# downstream readers can tell which slice of a tuning round produced a record.
ROUND_STEP_ANALYZE = "ANALYZE"
ROUND_STEP_OPTIMIZE = "OPTIMIZE"
ROUND_STEP_EVALUATE = "EVALUATE"


_VALID_SUBMIT_STATUS = ("success", "partial", "failed")


# ---------------------------------------------------------------------------
# RunState
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id() -> str:
    """Mint a sortable run id ``<UTC-yyyymmdd_HHMMSS>_<4-char>``."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    return f"{ts}_{suffix}"


@dataclass
class RunState:
    run_id: str
    operator: str
    started_at: str = field(default_factory=_utcnow_iso)
    time_budget_s: float = 1800.0
    elapsed_s: float = 0.0
    current_stage: str = Stage.INIT.value
    completed_stages: list[str] = field(default_factory=list)
    round_index: int = 0
    best_candidate_id: str | None = None
    best_speedup: float | None = None

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------

    def remaining_budget_s(self) -> float:
        return max(0.0, self.time_budget_s - self.elapsed_s)

    def mark_stage_complete(self, stage: Stage) -> None:
        if stage.value not in self.completed_stages:
            self.completed_stages.append(stage.value)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RunState":
        data = json.loads(text)
        return cls(**data)


# ---------------------------------------------------------------------------
# RunLayout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunLayout:
    """All filesystem paths for a single run live here.

    The orchestrator and tools never compute paths from raw strings — they go
    through this type. Predicates (``has_*``) are filesystem checks so resume
    can detect partial state without reloading json blobs.
    """

    workspace_root: Path
    run_id: str

    # ------------------------------------------------------------------
    # Top-level paths
    # ------------------------------------------------------------------

    @property
    def run_dir(self) -> Path:
        return self.workspace_root / "runs" / self.run_id

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def blackboard_path(self) -> Path:
        return self.run_dir / "blackboard.json"

    @property
    def leaderboard_path(self) -> Path:
        return self.run_dir / "leaderboard.jsonl"

    @property
    def events_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def candidates_dir(self) -> Path:
        return self.run_dir / "candidates"

    @property
    def best_dir(self) -> Path:
        return self.run_dir / "best"

    @property
    def best_cu_path(self) -> Path:
        return self.best_dir / "best.cu"

    @property
    def best_result_path(self) -> Path:
        return self.best_dir / "best_result.json"

    @property
    def baseline_dir(self) -> Path:
        return self.run_dir / "baseline"

    @property
    def baseline_path(self) -> Path:
        return self.baseline_dir / "baseline.json"

    @property
    def hardware_path(self) -> Path:
        return self.run_dir / "hardware" / "hardware_profile.json"

    @property
    def final_dir(self) -> Path:
        return self.run_dir / "final"

    @property
    def final_report_path(self) -> Path:
        return self.final_dir / "final_report.json"

    @property
    def summary_path(self) -> Path:
        return self.final_dir / "summary.md"

    # ------------------------------------------------------------------
    # Per-candidate paths
    # ------------------------------------------------------------------

    @staticmethod
    def candidate_id(index: int) -> str:
        return f"candidate_{index:03d}"

    def candidate_dir(self, candidate_id: str) -> Path:
        return self.candidates_dir / candidate_id

    def candidate_file(self, candidate_id: str, name: str) -> Path:
        return self.candidate_dir(candidate_id) / name

    # ------------------------------------------------------------------
    # Predicates (filesystem-backed, used by transitions.next_stage)
    # ------------------------------------------------------------------

    def has_hardware_profile(self) -> bool:
        return self.hardware_path.is_file()

    def has_baseline(self) -> bool:
        return self.baseline_path.is_file()

    def has_best(self) -> bool:
        return self.best_cu_path.is_file()

    # ------------------------------------------------------------------
    # Skeleton creation (idempotent — safe to call on resume)
    # ------------------------------------------------------------------

    def mkdir(self) -> None:
        for d in (
            self.run_dir,
            self.candidates_dir,
            self.best_dir,
            self.baseline_dir,
            self.hardware_path.parent,
            self.final_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Blackboard — function-style API over a JSON file
# ---------------------------------------------------------------------------


SCHEMA_VERSION = 1

TOP_KEYS = (
    "schema_version",
    "hardware",
    "benchmark",
    "baseline",
    "best",
    "latest_diagnosis",
    "history",
    "round",
    "final_summary",
)


def _empty_blackboard() -> dict:
    return {"schema_version": SCHEMA_VERSION, "history": []}


def load_blackboard(layout: RunLayout) -> dict:
    """Read ``blackboard.json``; missing file → fresh dict.

    Always normalizes ``schema_version`` (defaults to 1) and ``history``
    (defaults to ``[]``) so downstream code does not need to guard.
    """
    p = layout.blackboard_path
    if not p.is_file():
        return _empty_blackboard()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Corrupt file — fail loudly rather than silently dropping data
        raise
    if not isinstance(data, dict):
        raise ValueError(f"blackboard.json must contain a JSON object, got {type(data).__name__}")
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("history", [])
    return data


def save_blackboard(layout: RunLayout, data: dict) -> None:
    """Atomic write: tmp file in the same dir, then rename onto target."""
    p = layout.blackboard_path
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".blackboard.", suffix=".json", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_name, p)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def append_history(layout: RunLayout, entry: dict) -> None:
    """Convenience: load, append to ``history``, save."""
    bb = load_blackboard(layout)
    bb.setdefault("history", []).append(entry)
    save_blackboard(layout, bb)


# ---------------------------------------------------------------------------
# Submit-payload validation
# ---------------------------------------------------------------------------


def check_submit_payload(payload: Any, expected_stage: Stage | None = None) -> tuple[bool, list[str]]:
    """Lightweight check at the orchestrator/runner boundary.

    The contract for every submit-style tool is:

      * ``payload`` is a dict
      * ``payload['status']`` is one of {"success", "partial", "failed"}
      * if ``expected_stage`` is given, ``payload['stage']`` must equal its
        ``.value`` (when present — absence is tolerated for tools that don't
        write a stage tag, e.g. ``submit_diagnosis``)

    Returns ``(ok, errors)``. ``errors`` is empty iff ``ok`` is True.
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return False, [f"payload must be a dict, got {type(payload).__name__}"]
    status = payload.get("status")
    if status not in _VALID_SUBMIT_STATUS:
        errors.append(f"payload.status must be one of {_VALID_SUBMIT_STATUS}, got {status!r}")
    if expected_stage is not None:
        stage_tag = payload.get("stage")
        if stage_tag is not None and stage_tag != expected_stage.value:
            errors.append(
                f"payload.stage tag mismatch: expected {expected_stage.value!r}, "
                f"got {stage_tag!r}"
            )
    return (not errors), errors
