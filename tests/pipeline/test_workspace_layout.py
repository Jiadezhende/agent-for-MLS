"""tests/pipeline/test_workspace_layout.py — RunLayout path math + skeleton creation."""
from __future__ import annotations

from pipeline.workspace_layout import (
    RunLayout,
    candidate_id,
    make_run_id,
)


class TestRunIdAndCandidateId:
    def test_run_id_format(self):
        rid = make_run_id()
        # YYYYMMDD_HHMMSS_<4hex>
        parts = rid.split("_")
        assert len(parts) == 3
        assert len(parts[0]) == 8 and parts[0].isdigit()
        assert len(parts[1]) == 6 and parts[1].isdigit()
        assert len(parts[2]) == 4 and all(c in "0123456789abcdef" for c in parts[2])

    def test_run_id_uniqueness_within_same_second(self):
        # Even if generated back-to-back in the same second, the hex suffix
        # makes a collision astronomically unlikely.
        ids = {make_run_id() for _ in range(50)}
        assert len(ids) == 50

    def test_candidate_id_zero_padding(self):
        assert candidate_id(0) == "candidate_000"
        assert candidate_id(7) == "candidate_007"
        assert candidate_id(123) == "candidate_123"


class TestRunLayoutPaths:
    def test_all_top_level_paths_under_root(self, tmp_path):
        layout = RunLayout(tmp_path, "run_xyz")
        # Every advertised path should be inside runs/run_xyz
        for p in (
            layout.state_path,
            layout.events_path,
            layout.leaderboard_path,
            layout.benchmark_specs_dir,
            layout.hardware_dir,
            layout.hardware_profile_path,
            layout.baseline_dir,
            layout.baseline_path,
            layout.candidates_dir,
            layout.best_dir,
            layout.best_cu_path,
            layout.final_dir,
        ):
            assert layout.root in p.parents or p == layout.root

    def test_candidate_paths(self, tmp_path):
        layout = RunLayout(tmp_path, "run_xyz")
        cdir = layout.candidate_dir("candidate_002")
        assert cdir == layout.candidates_dir / "candidate_002"
        assert layout.candidate_file("candidate_002", "candidate.cu") == cdir / "candidate.cu"

    def test_baseline_per_d_paths(self, tmp_path):
        layout = RunLayout(tmp_path, "run_xyz")
        assert layout.baseline_reference_path("Y", 4096).name == "Y_d4096.pt"
        assert layout.baseline_input_path("W", 4096).name == "W_d4096.pt"


class TestRunLayoutMkdir:
    def test_mkdir_creates_full_skeleton(self, tmp_path):
        layout = RunLayout(tmp_path, "run_xyz")
        layout.mkdir()

        for p in (
            layout.root,
            layout.benchmark_specs_dir,
            layout.hardware_dir,
            layout.baseline_dir,
            layout.baseline_references_dir,
            layout.baseline_inputs_dir,
            layout.candidates_dir,
            layout.best_dir,
            layout.final_dir,
        ):
            assert p.is_dir()

    def test_mkdir_idempotent(self, tmp_path):
        layout = RunLayout(tmp_path, "run_xyz")
        layout.mkdir()
        layout.mkdir()  # should not raise
        assert layout.root.is_dir()


class TestArtifactExistence:
    def test_predicates_false_initially(self, tmp_path):
        layout = RunLayout(tmp_path, "run_xyz")
        layout.mkdir()
        assert not layout.has_hardware_profile()
        assert not layout.has_baseline()
        assert not layout.has_best()
        assert not layout.has_benchmark_spec("hardware")
        assert not layout.has_candidate_profile("candidate_000")

    def test_predicates_true_after_writing(self, tmp_path):
        layout = RunLayout(tmp_path, "run_xyz")
        layout.mkdir()

        layout.hardware_profile_path.write_text("{}")
        assert layout.has_hardware_profile()

        layout.baseline_path.write_text("{}")
        assert layout.has_baseline()

        layout.benchmark_spec_path("hardware").write_text("{}")
        assert layout.has_benchmark_spec("hardware")

        layout.best_cu_path.write_text("// stub")
        assert layout.has_best()

        cdir = layout.candidate_dir("candidate_000")
        cdir.mkdir(parents=True)
        layout.candidate_file("candidate_000", "profile.json").write_text("{}")
        assert layout.has_candidate_profile("candidate_000")


class TestRelpath:
    def test_relpath_under_workspace(self, tmp_path):
        layout = RunLayout(tmp_path, "run_xyz")
        rel = layout.relpath(layout.best_cu_path)
        assert rel == "runs/run_xyz/best/best.cu"

    def test_relpath_outside_workspace_returns_absolute(self, tmp_path):
        layout = RunLayout(tmp_path, "run_xyz")
        from pathlib import Path
        external = Path("/some/where/else.txt").resolve()
        # Falls back to str(p) when not under workspace_root.
        assert layout.relpath(external) == str(external)
