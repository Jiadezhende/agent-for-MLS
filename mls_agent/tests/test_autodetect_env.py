"""Unit tests for _autodetect_env environment-variable side effects.

Specifically: when GPU arch is detected via nvidia-smi, the function must pin
TORCH_CUDA_ARCH_LIST so subprocess `cpp_extension.load` skips the multi-arch
fatbin that defaults when this env var is unset.
"""
from __future__ import annotations

import os

import pytest

from mls_agent.tools.cuda import cuda_executor
from mls_agent.tools.cuda.config import ExecutorConfig


def test_autodetect_pins_torch_cuda_arch_list_when_arch_detected(monkeypatch):
    """If nvidia-smi reports compute_cap=8.6, env var should be set to '8.6'."""
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr(cuda_executor, "_detect_arch_flags", lambda: "-arch=sm_86")
    monkeypatch.setattr(cuda_executor, "_detect_msvc_ccbin", lambda: None)
    monkeypatch.setattr(cuda_executor, "_detect_tool_path_glob", lambda *a, **kw: None)
    monkeypatch.setattr(cuda_executor.shutil, "which", lambda _: None)

    cfg = ExecutorConfig(workspace_root="/tmp")
    _, notes = cuda_executor._autodetect_env(cfg)

    assert os.environ.get("TORCH_CUDA_ARCH_LIST") == "8.6"
    assert any("TORCH_CUDA_ARCH_LIST=8.6" in n for n in notes)


def test_autodetect_does_not_overwrite_existing_torch_cuda_arch_list(monkeypatch):
    """User explicit setting wins."""
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "7.5;8.0+PTX")
    monkeypatch.setattr(cuda_executor, "_detect_arch_flags", lambda: "-arch=sm_86")
    monkeypatch.setattr(cuda_executor, "_detect_msvc_ccbin", lambda: None)
    monkeypatch.setattr(cuda_executor, "_detect_tool_path_glob", lambda *a, **kw: None)
    monkeypatch.setattr(cuda_executor.shutil, "which", lambda _: None)

    cfg = ExecutorConfig(workspace_root="/tmp")
    cuda_executor._autodetect_env(cfg)

    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "7.5;8.0+PTX"


def test_autodetect_skips_pin_when_arch_undetectable(monkeypatch):
    """nvidia-smi unavailable → no env var, no crash."""
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr(cuda_executor, "_detect_arch_flags", lambda: None)
    monkeypatch.setattr(cuda_executor, "_detect_msvc_ccbin", lambda: None)
    monkeypatch.setattr(cuda_executor, "_detect_tool_path_glob", lambda *a, **kw: None)
    monkeypatch.setattr(cuda_executor.shutil, "which", lambda _: None)

    cfg = ExecutorConfig(workspace_root="/tmp")
    cuda_executor._autodetect_env(cfg)

    assert "TORCH_CUDA_ARCH_LIST" not in os.environ


@pytest.mark.parametrize(
    "arch_str,expected",
    [("-arch=sm_75", "7.5"), ("-arch=sm_86", "8.6"), ("-arch=sm_90", "9.0")],
)
def test_autodetect_arch_dotted_format(monkeypatch, arch_str, expected):
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr(cuda_executor, "_detect_arch_flags", lambda: arch_str)
    monkeypatch.setattr(cuda_executor, "_detect_msvc_ccbin", lambda: None)
    monkeypatch.setattr(cuda_executor, "_detect_tool_path_glob", lambda *a, **kw: None)
    monkeypatch.setattr(cuda_executor.shutil, "which", lambda _: None)

    cfg = ExecutorConfig(workspace_root="/tmp")
    cuda_executor._autodetect_env(cfg)

    assert os.environ.get("TORCH_CUDA_ARCH_LIST") == expected
