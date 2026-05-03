"""
tests/test_autodetect.py — Unit tests for _autodetect_env and related helpers.

All tests use mocking so they run without a GPU or profiler tools installed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.core.config import ExecutorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**kwargs) -> ExecutorConfig:
    return ExecutorConfig(**kwargs)


def _mock_subprocess_ok(stdout: str) -> MagicMock:
    m = MagicMock()
    m.returncode = 0
    m.stdout = stdout
    m.stderr = ""
    m.timed_out = False
    return m


def _mock_subprocess_fail() -> MagicMock:
    m = MagicMock()
    m.returncode = 1
    m.stdout = ""
    m.stderr = "error"
    m.timed_out = False
    return m


# ---------------------------------------------------------------------------
# _detect_arch_flags
# ---------------------------------------------------------------------------

class TestDetectArchFlags:
    def test_returns_arch_flag_from_compute_cap(self):
        from agents.tools.cuda_executor import _detect_arch_flags
        with patch("agents.tools.cuda_executor._run_subprocess", return_value=_mock_subprocess_ok("12.0\n")):
            result = _detect_arch_flags()
        assert result == "-arch=sm_120"

    def test_returns_none_on_nvidia_smi_failure(self):
        from agents.tools.cuda_executor import _detect_arch_flags
        with patch("agents.tools.cuda_executor._run_subprocess", return_value=_mock_subprocess_fail()):
            result = _detect_arch_flags()
        assert result is None

    def test_returns_none_on_non_numeric_output(self):
        from agents.tools.cuda_executor import _detect_arch_flags
        with patch("agents.tools.cuda_executor._run_subprocess", return_value=_mock_subprocess_ok("N/A\n")):
            result = _detect_arch_flags()
        assert result is None

    def test_strips_dot_correctly(self):
        from agents.tools.cuda_executor import _detect_arch_flags
        with patch("agents.tools.cuda_executor._run_subprocess", return_value=_mock_subprocess_ok("8.6\n")):
            result = _detect_arch_flags()
        assert result == "-arch=sm_86"


# ---------------------------------------------------------------------------
# _autodetect_env — arch flags
# ---------------------------------------------------------------------------

class TestAutodetectEnvArch:
    def test_adds_arch_flag_when_not_set(self):
        from agents.tools.cuda_executor import _autodetect_env
        cfg = _make_cfg()
        with patch("agents.tools.cuda_executor._run_subprocess", return_value=_mock_subprocess_ok("12.0\n")):
            updated, notes = _autodetect_env(cfg)
        assert "-arch=sm_120" in updated.nvcc_default_flags
        assert any("sm_120" in n for n in notes)

    def test_does_not_add_arch_flag_if_already_set(self):
        from agents.tools.cuda_executor import _autodetect_env
        cfg = _make_cfg(nvcc_default_flags=["-arch=sm_86"])
        with patch("agents.tools.cuda_executor._run_subprocess", return_value=_mock_subprocess_ok("12.0\n")):
            updated, notes = _autodetect_env(cfg)
        arch_flags = [f for f in updated.nvcc_default_flags if f.startswith("-arch")]
        assert len(arch_flags) == 1
        assert arch_flags[0] == "-arch=sm_86"   # existing flag preserved

    def test_warns_when_nvidia_smi_fails(self):
        from agents.tools.cuda_executor import _autodetect_env
        cfg = _make_cfg()
        with patch("agents.tools.cuda_executor._run_subprocess", return_value=_mock_subprocess_fail()):
            updated, notes = _autodetect_env(cfg)
        assert not any(f.startswith("-arch") for f in updated.nvcc_default_flags)
        assert any("AGENT_NVCC_FLAGS" in n for n in notes)


# ---------------------------------------------------------------------------
# _autodetect_env — ccbin (Windows only)
# ---------------------------------------------------------------------------

class TestAutodetectEnvCcbin:
    def test_does_not_override_existing_ccbin(self):
        from agents.tools.cuda_executor import _autodetect_env
        cfg = _make_cfg(nvcc_ccbin="C:/MSVC/bin/x64")
        with patch("agents.tools.cuda_executor._run_subprocess", return_value=_mock_subprocess_ok("12.0\n")):
            updated, notes = _autodetect_env(cfg)
        assert updated.nvcc_ccbin == "C:/MSVC/bin/x64"

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_detects_msvc_on_windows(self, tmp_path):
        from agents.tools.cuda_executor import _autodetect_env, _detect_msvc_ccbin
        # Create a fake cl.exe
        cl_dir = tmp_path / "VC" / "Tools" / "MSVC" / "14.38.0" / "bin" / "Hostx64" / "x64"
        cl_dir.mkdir(parents=True)
        (cl_dir / "cl.exe").touch()

        fake_vs_path = tmp_path

        def fake_subprocess(cmd, timeout_s):
            if "vswhere" in str(cmd):
                return _mock_subprocess_ok(str(fake_vs_path) + "\n")
            return _mock_subprocess_ok("12.0\n")

        cfg = _make_cfg()
        with patch("agents.tools.cuda_executor._run_subprocess", side_effect=fake_subprocess), \
             patch("pathlib.Path.exists", return_value=True):
            detected = _detect_msvc_ccbin()
        assert detected is not None
        assert "x64" in detected


# ---------------------------------------------------------------------------
# _autodetect_env — ncu / nsys tool paths
# ---------------------------------------------------------------------------

class TestAutodetectEnvToolPaths:
    def test_ncu_not_overridden_if_on_path(self):
        from agents.tools.cuda_executor import _autodetect_env
        cfg = _make_cfg()
        with patch("agents.tools.cuda_executor._run_subprocess", return_value=_mock_subprocess_ok("12.0\n")), \
             patch("shutil.which", return_value="/usr/bin/ncu"):
            updated, notes = _autodetect_env(cfg)
        assert updated.ncu_bin == "ncu"   # stays at default; no override needed

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows search paths only")
    def test_ncu_detected_from_search_path(self, tmp_path):
        from agents.tools.cuda_executor import _autodetect_env
        fake_ncu = str(tmp_path / "ncu.exe")
        Path(fake_ncu).touch()

        cfg = _make_cfg()
        with patch("agents.tools.cuda_executor._run_subprocess", return_value=_mock_subprocess_ok("12.0\n")), \
             patch("shutil.which", return_value=None), \
             patch("agents.tools.cuda_executor._NCU_SEARCH_GLOBS_WIN", [fake_ncu]):
            updated, notes = _autodetect_env(cfg)
        assert updated.ncu_bin == fake_ncu

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows search paths only")
    def test_nsys_detected_from_search_path(self, tmp_path):
        from agents.tools.cuda_executor import _autodetect_env
        fake_nsys = str(tmp_path / "nsys.exe")
        Path(fake_nsys).touch()

        cfg = _make_cfg()
        with patch("agents.tools.cuda_executor._run_subprocess", return_value=_mock_subprocess_ok("12.0\n")), \
             patch("shutil.which", return_value=None), \
             patch("agents.tools.cuda_executor._NSYS_SEARCH_GLOBS_WIN", [fake_nsys]):
            updated, notes = _autodetect_env(cfg)
        assert updated.nsys_bin == fake_nsys

    def test_no_override_when_tool_not_found_anywhere(self):
        from agents.tools.cuda_executor import _autodetect_env
        cfg = _make_cfg()
        empty_globs_attr_ncu = (
            "_NCU_SEARCH_GLOBS_WIN" if sys.platform == "win32" else "_NCU_SEARCH_GLOBS_LIN"
        )
        empty_globs_attr_nsys = (
            "_NSYS_SEARCH_GLOBS_WIN" if sys.platform == "win32" else "_NSYS_SEARCH_GLOBS_LIN"
        )
        with patch("agents.tools.cuda_executor._run_subprocess", return_value=_mock_subprocess_ok("12.0\n")), \
             patch("shutil.which", return_value=None), \
             patch(f"agents.tools.cuda_executor.{empty_globs_attr_ncu}", []), \
             patch(f"agents.tools.cuda_executor.{empty_globs_attr_nsys}", []):
            updated, notes = _autodetect_env(cfg)
        assert updated.ncu_bin == "ncu"    # unchanged default
        assert updated.nsys_bin == "nsys"  # unchanged default
