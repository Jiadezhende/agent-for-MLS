"""state.py — Stage enum, RunState, RunLayout, blackboard.

Everything related to "where things live on disk" and "what the run currently
looks like" is consolidated here.
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

    def remaining_budget_s(self) -> float:
        return max(0.0, self.time_budget_s - self.elapsed_s)

    def mark_stage_complete(self, stage: Stage) -> None:
        if stage.value not in self.completed_stages:
            self.completed_stages.append(stage.value)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RunState":
        return cls(**json.loads(text))


# ---------------------------------------------------------------------------
# RunLayout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunLayout:
    """All filesystem paths for a single run live here.

    Predicates (``has_*``) are filesystem checks so resume can detect partial
    state without reloading json blobs. Single-file artifacts live directly
    under ``run_dir``; multi-file groups (inputs/candidates/best) get their
    own subdir.
    """

    workspace_root: Path
    run_id: str

    # ------------------------------------------------------------------
    # Top-level paths (single-file artifacts live directly under run_dir)
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
    def trace_path(self) -> Path:
        return self.run_dir / "agent_trace.log"

    @property
    def hardware_path(self) -> Path:
        return self.run_dir / "hardware_profile.json"

    @property
    def baseline_path(self) -> Path:
        return self.run_dir / "baseline.json"

    @property
    def final_report_path(self) -> Path:
        return self.run_dir / "final_report.json"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.md"

    # ------------------------------------------------------------------
    # Multi-file groups
    # ------------------------------------------------------------------

    @property
    def inputs_dir(self) -> Path:
        """Correctness fixtures: candidate forward inputs, indexed by shape."""
        return self.run_dir / "inputs"

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
    def benchmark_dir(self) -> Path:
        return self.run_dir / "benchmark"

    def benchmark_result_path(self, candidate_id: str) -> Path:
        return self.benchmark_dir / f"{candidate_id}.json"

    @property
    def build_dir(self) -> Path:
        """Per-run cpp_extension / nvcc build directory.

        cpp_extension.load(build_directory=...) targets this so candidate
        .so artifacts live under the run instead of ~/.cache/torch_extensions/.
        """
        return self.run_dir / "build"

    @property
    def exec_dir(self) -> Path:
        """Per-run Executor workspace (nvcc src/bin, ncu/nsys reports)."""
        return self.run_dir / "exec"

    def input_path(self, tensor_name: str, shape_id: str) -> Path:
        return self.inputs_dir / f"{tensor_name}_{shape_id}.pt"

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

    # ------------------------------------------------------------------
    # Skeleton creation (idempotent — safe to call on resume)
    # ------------------------------------------------------------------

    def mkdir(self) -> None:
        for d in (
            self.run_dir,
            self.candidates_dir,
            self.best_dir,
            self.inputs_dir,
            self.benchmark_dir,
            self.build_dir,
            self.exec_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Blackboard — function-style API over a JSON file
# ---------------------------------------------------------------------------


SCHEMA_VERSION = 1


def _empty_blackboard() -> dict:
    return {"schema_version": SCHEMA_VERSION, "history": []}


def load_blackboard(layout: RunLayout) -> dict:
    """Read ``blackboard.json``; missing file → fresh dict.

    Always normalizes ``schema_version`` and ``history`` so downstream code
    does not need to guard.
    """
    p = layout.blackboard_path
    if not p.is_file():
        return _empty_blackboard()
    data = json.loads(p.read_text(encoding="utf-8"))
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
    bb = load_blackboard(layout)
    bb.setdefault("history", []).append(entry)
    save_blackboard(layout, bb)
