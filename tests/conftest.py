"""
conftest.py — Shared pytest fixtures.
"""
from __future__ import annotations

import os
import sys
import tempfile
import uuid
import pytest
from pathlib import Path
from dotenv import load_dotenv

# Load .env so ExecutorConfig.from_env() has AGENT_NVCC_CCBIN etc.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
_PYTEST_TMP = _REPO_ROOT / ".tmp" / "pytest"
_PYTEST_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMP", str(_PYTEST_TMP))
os.environ.setdefault("TEMP", str(_PYTEST_TMP))
os.environ.setdefault("TMPDIR", str(_PYTEST_TMP))
tempfile.tempdir = str(_PYTEST_TMP)

load_dotenv(_REPO_ROOT / ".env")


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "cuda: marks tests that require a working CUDA toolchain (nvcc + GPU)",
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_workspace(tmp_path):
    """A fresh _Workspace rooted in pytest's tmp_path."""
    from agents.tools.cuda_executor import _Workspace
    return _Workspace(str(tmp_path))


@pytest.fixture()
def exec_cfg():
    """ExecutorConfig loaded from the real .env (so CCBIN / arch flags are set)."""
    from agents.core.config import ExecutorConfig
    return ExecutorConfig.from_env()


@pytest.fixture()
def executor(exec_cfg):
    """A live Executor instance (workspace in a temp dir)."""
    import tempfile
    from agents.tools.cuda_executor import Executor
    cfg = exec_cfg.__class__(
        **{**exec_cfg.__dict__, "workspace_root": tempfile.mkdtemp()}
    )
    return Executor(cfg)


@pytest.fixture()
def agent_ctx():
    """A minimal AgentContext for recording-tool tests."""
    from agents.core.types import AgentContext, MemoryStore, Task
    task = Task(
        id=str(uuid.uuid4()),
        type="hardware_probe",
        description="test task",
        payload={"targets": []},
        constraints={},
    )
    return AgentContext(task=task, memory=MemoryStore())
