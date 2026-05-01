"""
tests/test_executor.py — Unit tests for executor.py

Tests are split into two groups:
  - Pure-Python logic (no GPU/nvcc required): sandbox helpers, cache, error structs
  - CUDA tests (marked with @pytest.mark.cuda): require nvcc + GPU
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from agents.tools.cuda_executor import (
    ExecutorError,
    JobSpec,
    JobResult,
    SubResult,
    _Workspace,
    _JobCache,
    _safe_join,
    _check_binary,
    _run_subprocess,
)
from agents.core.config import ExecutorConfig


# ===========================================================================
# Helpers
# ===========================================================================

def _make_cfg(**overrides) -> ExecutorConfig:
    base = ExecutorConfig()
    for k, v in overrides.items():
        object.__setattr__(base, k, v)
    return base


# ===========================================================================
# _safe_join — path traversal guard
# ===========================================================================

class TestSafeJoin:
    def test_normal_relative_path(self, tmp_path):
        result = _safe_join(tmp_path, "src/foo.cu")
        assert result == tmp_path / "src" / "foo.cu"

    def test_absolute_path_raises(self, tmp_path):
        with pytest.raises(ExecutorError) as exc_info:
            _safe_join(tmp_path, "/etc/passwd")
        assert exc_info.value.kind == "path_escape"

    def test_traversal_dotdot_raises(self, tmp_path):
        with pytest.raises(ExecutorError) as exc_info:
            _safe_join(tmp_path, "../../etc/passwd")
        assert exc_info.value.kind == "path_escape"

    def test_nested_normal_path(self, tmp_path):
        result = _safe_join(tmp_path, "a/b/c.txt")
        assert str(result).startswith(str(tmp_path))


# ===========================================================================
# _check_binary — whitelist guard
# ===========================================================================

class TestCheckBinary:
    def test_not_whitelisted_raises(self):
        cfg = _make_cfg(allowed_binaries=["nvcc"])
        with pytest.raises(ExecutorError) as exc_info:
            _check_binary(cfg, "rm")
        assert exc_info.value.kind == "binary_not_whitelisted"

    def test_whitelisted_but_missing_raises(self):
        cfg = _make_cfg(allowed_binaries=["nvcc", "definitely_not_real_bin"])
        with pytest.raises(ExecutorError) as exc_info:
            _check_binary(cfg, "definitely_not_real_bin")
        assert exc_info.value.kind == "binary_not_found"

    def test_full_path_stem_used_for_whitelist(self, exec_cfg):
        """Full path to nvcc.exe should still pass the whitelist check."""
        import shutil
        nvcc_path = shutil.which(exec_cfg.nvcc_bin)
        if nvcc_path is None:
            pytest.skip("nvcc not in PATH")
        resolved = _check_binary(exec_cfg, nvcc_path)
        assert resolved is not None


# ===========================================================================
# _Workspace
# ===========================================================================

class TestWorkspace:
    def test_subdirs_created(self, tmp_path):
        ws = _Workspace(str(tmp_path))
        for sub in _Workspace.SUBDIRS:
            assert (ws.root / sub).is_dir()

    def test_write_creates_file(self, tmp_path):
        ws = _Workspace(str(tmp_path))
        p = ws.write("src/test.cu", "hello")
        assert p.read_text() == "hello"

    def test_write_bytes(self, tmp_path):
        ws = _Workspace(str(tmp_path))
        p = ws.write("src/test.bin", b"\x00\x01\x02")
        assert p.read_bytes() == b"\x00\x01\x02"

    def test_allocate_returns_unique_paths(self, tmp_path):
        ws = _Workspace(str(tmp_path))
        p1 = ws.allocate("ncu", ".csv")
        p2 = ws.allocate("ncu", ".csv")
        assert p1 != p2

    def test_rel_returns_posix(self, tmp_path):
        ws = _Workspace(str(tmp_path))
        p = ws.write("src/foo.cu", "x")
        rel = ws.rel(p)
        assert "/" in rel
        assert "\\" not in rel

    def test_cleanup_removes_dir(self, tmp_path):
        ws = _Workspace(str(tmp_path))
        assert ws.root.exists()
        ws.cleanup()
        assert not ws.root.exists()


# ===========================================================================
# _JobCache
# ===========================================================================

class TestJobCache:
    def _make_result(self, name="test") -> JobResult:
        return JobResult(
            job_id="abc123",
            backend="cuda_probe",
            name=name,
            status="done",
            summary={"stdout": "ok"},
            artifact_refs={},
            cache_hit=False,
            elapsed_s=1.0,
            started_at="2026-01-01T00:00:00Z",
        )

    def test_miss_returns_none(self):
        cache = _JobCache()
        assert cache.get("nonexistent") is None

    def test_put_then_get(self):
        cache = _JobCache()
        r = self._make_result()
        cache.put("key1", r)
        assert cache.get("key1") is r

    def test_different_keys_independent(self):
        cache = _JobCache()
        r1 = self._make_result("a")
        r2 = self._make_result("b")
        cache.put("k1", r1)
        cache.put("k2", r2)
        assert cache.get("k1").name == "a" # type: ignore
        assert cache.get("k2").name == "b" # type: ignore


# ===========================================================================
# JobSpec.cache_key
# ===========================================================================

class TestJobSpecCacheKey:
    def test_same_spec_same_key(self):
        s = JobSpec(backend="cuda_probe", name="foo", payload={"a": 1})
        assert s.cache_key() == s.cache_key()

    def test_different_payload_different_key(self):
        s1 = JobSpec(backend="cuda_probe", name="foo", payload={"a": 1})
        s2 = JobSpec(backend="cuda_probe", name="foo", payload={"a": 2})
        assert s1.cache_key() != s2.cache_key()

    def test_gpu_tag_affects_key(self):
        s = JobSpec(backend="cuda_probe", name="foo", payload={})
        assert s.cache_key("gpu_a") != s.cache_key("gpu_b")

    def test_key_is_16_hex_chars(self):
        s = JobSpec(backend="cuda_probe", name="foo", payload={})
        key = s.cache_key()
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)


# ===========================================================================
# JobResult serialization
# ===========================================================================

class TestJobResult:
    def _make(self) -> JobResult:
        return JobResult(
            job_id="abc",
            backend="cuda_probe",
            name="test",
            status="done",
            summary={"stdout": "42", "stderr": ""},
            artifact_refs={"csv": "ncu/a.csv"},
            cache_hit=True,
            elapsed_s=3.14,
            started_at="2026-01-01T00:00:00Z",
        )

    def test_to_tool_result_includes_status(self):
        r = self._make()
        d = r.to_tool_result()
        assert d["status"] == "done"
        assert d["cache_hit"] is True
        assert "stdout" in d

    def test_to_log_dict_excludes_summary(self):
        r = self._make()
        d = r.to_log_dict()
        assert "summary" not in d
        assert d["job_id"] == "abc"
        assert d["elapsed_s"] == 3.14

    def test_to_tool_result_is_json_serializable(self):
        r = self._make()
        json.dumps(r.to_tool_result())  # must not raise


# ===========================================================================
# Executor.run_cuda_probe — requires CUDA  (marked)
# ===========================================================================

@pytest.mark.cuda
class TestRunCudaProbe:
    HELLO_SRC = r"""
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k() {}
int main() {
    k<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("hello_cuda\n");
    return 0;
}
"""
    BAD_SRC = "this is not valid CUDA code {"

    def test_smoke_compile_and_run(self, executor):
        result = executor.run_cuda_probe(source=self.HELLO_SRC, probe_name="smoke")
        assert result["status"] == "done"
        assert "hello_cuda" in result.get("stdout", "")

    def test_compile_error_returns_error_status(self, executor):
        result = executor.run_cuda_probe(source=self.BAD_SRC, probe_name="bad")
        assert result["status"] == "error"
        assert result.get("error") == "compile_failed"
        # LLM must see the actual error message
        assert len(result.get("stderr", "")) > 0
        # error_class must be present so LLM knows whether to fix code or env
        assert "error_class" in result
        assert result["error_class"] in ("user_code", "infrastructure", "timeout")
        # cmd must NOT be present — it caused LLM to misread arch flags as the error
        assert "cmd" not in result

    def test_cache_hit_on_second_call(self, executor):
        result1 = executor.run_cuda_probe(source=self.HELLO_SRC, probe_name="cache_test")
        result2 = executor.run_cuda_probe(source=self.HELLO_SRC, probe_name="cache_test")
        assert result1["status"] == "done"
        assert result2["cache_hit"] is True

    def test_on_job_complete_callback(self, exec_cfg):
        import tempfile
        from agents.tools.cuda_executor import Executor

        completed = []
        cfg = exec_cfg.__class__(
            **{**exec_cfg.__dict__, "workspace_root": tempfile.mkdtemp()}
        )
        exc = Executor(cfg, on_job_complete=lambda r: completed.append(r))
        exc.run_cuda_probe(source=self.HELLO_SRC, probe_name="cb_test")
        assert len(completed) == 1
        assert completed[0].status == "done"

    def test_stdout_truncate(self, exec_cfg):
        """stdout_truncate_bytes is respected."""
        import tempfile
        from agents.tools.cuda_executor import Executor

        cfg = exec_cfg.__class__(
            **{**exec_cfg.__dict__,
               "workspace_root": tempfile.mkdtemp(),
               "stdout_truncate_bytes": 50}
        )
        exc = Executor(cfg)
        # Generate a lot of output
        big_src = r"""
#include <cstdio>
int main() {
    for (int i = 0; i < 1000; ++i) printf("AAAAAAAAAA\n");
    return 0;
}
"""
        result = exc.run_cuda_probe(source=big_src, probe_name="trunc_test")
        # Status may be done or error depending on whether nvcc treats this as CUDA
        if result["status"] == "done":
            assert len(result.get("stdout", "")) <= 200  # truncated + marker overhead


# ===========================================================================
# _run_subprocess helpers
# ===========================================================================

class TestRunSubprocess:
    def test_captures_stdout(self):
        result = _run_subprocess(["python", "-c", "print('hi')"], timeout_s=10)
        assert "hi" in result.stdout
        assert result.returncode == 0

    def test_captures_stderr(self):
        result = _run_subprocess(
            ["python", "-c", "import sys; sys.stderr.write('err\\n')"],
            timeout_s=10,
        )
        assert "err" in result.stderr

    def test_nonzero_returncode(self):
        result = _run_subprocess(["python", "-c", "raise SystemExit(42)"], timeout_s=10)
        assert result.returncode == 42

    def test_timeout_sets_flag(self):
        result = _run_subprocess(
            ["python", "-c", "import time; time.sleep(10)"],
            timeout_s=1,
        )
        assert result.timed_out is True
        assert result.returncode == -1

    def test_stdout_truncated(self):
        result = _run_subprocess(
            ["python", "-c", "print('A' * 200)"],
            timeout_s=10,
            truncate_bytes=50,
        )
        assert len(result.stdout.encode()) <= 100  # truncate marker adds bytes


class TestCompileErrorParsing:
    """Tests for _extract_nvcc_errors and _classify_compile_error helpers."""

    def test_extract_removes_nvcc_invocation_line(self):
        from agents.tools.executor.classifiers import _extract_nvcc_errors
        combined = (
            "nvcc.EXE -ccbin C:/MSVC/bin -arch=sm_120 -o foo.exe src/foo.cu\n"
            "src/foo.cu(5): error: 'clockRate' is not a member of 'cudaDeviceProp'\n"
            "1 error detected in compilation of 'src/foo.cu'\n"
        )
        result = _extract_nvcc_errors(combined)
        assert "nvcc.EXE" not in result
        assert "clockRate" in result

    def test_extract_fallback_when_no_diagnostics(self):
        from agents.tools.executor.classifiers import _extract_nvcc_errors
        combined = "something weird with no diagnostic keywords"
        result = _extract_nvcc_errors(combined)
        assert len(result) > 0   # fallback returns something

    def test_extract_respects_max_chars(self):
        from agents.tools.executor.classifiers import _extract_nvcc_errors
        combined = "error: " + "x" * 5000
        result = _extract_nvcc_errors(combined, max_chars=100)
        assert len(result) <= 100

    def test_classify_user_code_for_syntax_error(self):
        from agents.tools.executor.classifiers import _classify_compile_error
        combined = "src/foo.cu(10): error: expected a ';'\n1 error detected"
        assert _classify_compile_error(combined) == "user_code"

    def test_classify_infrastructure_for_ccbin_missing(self):
        from agents.tools.executor.classifiers import _classify_compile_error
        combined = "nvcc -ccbin C:/missing/path: cannot find compiler\n1 error"
        assert _classify_compile_error(combined) == "infrastructure"

    def test_classify_infrastructure_for_command_not_found(self):
        from agents.tools.executor.classifiers import _classify_compile_error
        combined = "nvcc: command not found"
        assert _classify_compile_error(combined) == "infrastructure"

    def test_executor_error_has_error_class_field(self):
        from agents.tools.cuda_executor import ExecutorError
        err = ExecutorError("compile_failed", error_class="user_code", returncode=1, stderr="oops")
        assert err.error_class == "user_code"
        assert err.kind == "compile_failed"

    def test_executor_error_default_error_class_is_infrastructure(self):
        from agents.tools.cuda_executor import ExecutorError
        err = ExecutorError("binary_not_found", name="ncu")
        assert err.error_class == "infrastructure"

    def test_executor_error_hint_field(self):
        from agents.tools.cuda_executor import ExecutorError
        err = ExecutorError("binary_not_found", error_class="infrastructure",
                            hint="Set AGENT_NCU_BIN", name="ncu")
        assert err.hint == "Set AGENT_NCU_BIN"
