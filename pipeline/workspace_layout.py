"""pipeline/workspace_layout.py — <run_id>/ directory layout helpers.

Owns all path construction so other modules never hand-build relative paths.
Pure path math + a tiny set of artifact-existence helpers; no schema logic
(that's in state.py).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from secrets import token_hex


# ---------------------------------------------------------------------------
# Directory names — keep in sync with the plan document.
# ---------------------------------------------------------------------------

DIR_BENCHMARK_SPECS = "benchmark_specs"
DIR_HARDWARE = "hardware"
DIR_BASELINE = "baseline"
DIR_BASELINE_REFS = "references"
DIR_BASELINE_INPUTS = "inputs"
DIR_CANDIDATES = "candidates"
DIR_BEST = "best"
DIR_FINAL = "final"
DIR_PROFILES = "profiles"

FILE_STATE = "state.json"
FILE_EVENTS = "events.jsonl"
FILE_LEADERBOARD = "leaderboard.jsonl"

FILE_HARDWARE_PROFILE = "hardware_profile.json"
FILE_BASELINE = "baseline.json"
FILE_BEST_CU = "best.cu"
FILE_BEST_RESULT = "best_result.json"
FILE_FINAL_BENCHMARK = "final_benchmark.json"
FILE_FINAL_REPORT = "final_report.json"
FILE_SUMMARY = "summary.md"

FILE_CANDIDATE_CU = "candidate.cu"
FILE_COMPILE = "compile.json"
FILE_CORRECTNESS = "correctness.json"
FILE_QUICK_BENCH = "quick_benchmark.json"
FILE_CONFIRM_BENCH = "confirm_benchmark.json"
FILE_PROFILE = "profile.json"
FILE_AGENT_TRACE = "agent_trace.json"


def make_run_id() -> str:
    """`YYYYMMDD_HHMMSS_<4hex>` — sortable + unique enough for human inspection."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{token_hex(2)}"


def candidate_id(index: int) -> str:
    """`candidate_NNN` — zero-padded to 3 digits to match the plan layout."""
    return f"candidate_{index:03d}"


# ---------------------------------------------------------------------------
# RunLayout — owns one <run_id>/ directory.
# ---------------------------------------------------------------------------

class RunLayout:
    """All paths are absolute Path objects, rooted at workspace_root/<run_id>.

    Use `mkdir()` once at INIT to create the skeleton; all other path methods
    just return paths without creating them. Artifact-existence helpers
    (`has_*`) are used by the state machine to decide whether to skip a
    setup stage on resume.
    """

    def __init__(self, workspace_root: str | Path, run_id: str):
        self.workspace_root = Path(workspace_root).resolve()
        self.run_id = run_id
        self.root: Path = self.workspace_root / run_id

    # ---- top-level files -------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.root / FILE_STATE

    @property
    def events_path(self) -> Path:
        return self.root / FILE_EVENTS

    @property
    def leaderboard_path(self) -> Path:
        return self.root / FILE_LEADERBOARD

    # ---- benchmark specs -------------------------------------------------

    @property
    def benchmark_specs_dir(self) -> Path:
        return self.root / DIR_BENCHMARK_SPECS

    def benchmark_spec_path(self, slot: str) -> Path:
        return self.benchmark_specs_dir / f"{slot}.json"

    # ---- hardware --------------------------------------------------------

    @property
    def hardware_dir(self) -> Path:
        return self.root / DIR_HARDWARE

    @property
    def hardware_profile_path(self) -> Path:
        return self.hardware_dir / FILE_HARDWARE_PROFILE

    # ---- baseline --------------------------------------------------------

    @property
    def baseline_dir(self) -> Path:
        return self.root / DIR_BASELINE

    @property
    def baseline_path(self) -> Path:
        return self.baseline_dir / FILE_BASELINE

    @property
    def baseline_references_dir(self) -> Path:
        return self.baseline_dir / DIR_BASELINE_REFS

    @property
    def baseline_inputs_dir(self) -> Path:
        return self.baseline_dir / DIR_BASELINE_INPUTS

    def baseline_reference_path(self, name: str, d: int) -> Path:
        return self.baseline_references_dir / f"{name}_d{d}.pt"

    def baseline_input_path(self, name: str, d: int) -> Path:
        return self.baseline_inputs_dir / f"{name}_d{d}.pt"

    # ---- candidates ------------------------------------------------------

    @property
    def candidates_dir(self) -> Path:
        return self.root / DIR_CANDIDATES

    def candidate_dir(self, candidate_id: str) -> Path:
        return self.candidates_dir / candidate_id

    def candidate_file(self, candidate_id: str, name: str) -> Path:
        return self.candidate_dir(candidate_id) / name

    # ---- best ------------------------------------------------------------

    @property
    def best_dir(self) -> Path:
        return self.root / DIR_BEST

    @property
    def best_cu_path(self) -> Path:
        return self.best_dir / FILE_BEST_CU

    @property
    def best_result_path(self) -> Path:
        return self.best_dir / FILE_BEST_RESULT

    # ---- final -----------------------------------------------------------

    @property
    def final_dir(self) -> Path:
        return self.root / DIR_FINAL

    @property
    def final_benchmark_path(self) -> Path:
        return self.final_dir / FILE_FINAL_BENCHMARK

    @property
    def final_report_path(self) -> Path:
        return self.final_dir / FILE_FINAL_REPORT

    @property
    def summary_path(self) -> Path:
        return self.final_dir / FILE_SUMMARY

    # ---- artifact existence checks --------------------------------------

    def has_hardware_profile(self) -> bool:
        return self.hardware_profile_path.is_file()

    def has_baseline(self) -> bool:
        return self.baseline_path.is_file()

    def has_benchmark_spec(self, slot: str) -> bool:
        return self.benchmark_spec_path(slot).is_file()

    def has_candidate_profile(self, candidate_id: str) -> bool:
        return self.candidate_file(candidate_id, FILE_PROFILE).is_file()

    def has_best(self) -> bool:
        return self.best_cu_path.is_file()

    # ---- skeleton creation ----------------------------------------------

    def mkdir(self) -> None:
        """Create the directory skeleton. Idempotent."""
        for p in (
            self.root,
            self.benchmark_specs_dir,
            self.hardware_dir,
            self.hardware_dir / "raw",
            self.baseline_dir,
            self.baseline_references_dir,
            self.baseline_inputs_dir,
            self.candidates_dir,
            self.best_dir,
            self.final_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    # ---- relative-path helpers ------------------------------------------

    def relpath(self, p: Path) -> str:
        """Workspace-relative POSIX path, for storing in StageResult.artifacts."""
        try:
            return p.relative_to(self.workspace_root).as_posix()
        except ValueError:
            return str(p)
