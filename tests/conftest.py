"""
conftest.py — Shared pytest fixtures.
"""
from __future__ import annotations

import shutil
import uuid
import pytest
from pathlib import Path
from dotenv import load_dotenv

# Load .env so ExecutorConfig.from_env() has AGENT_NVCC_CCBIN etc.
load_dotenv(Path(__file__).parent.parent / ".env")


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    """Skip CUDA integration tests unless nvcc and nvidia-smi are available."""
    has_cuda_toolchain = shutil.which("nvcc") is not None and shutil.which("nvidia-smi") is not None
    if has_cuda_toolchain:
        return

    skip_cuda = pytest.mark.skip(reason="requires nvcc and nvidia-smi")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)


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
