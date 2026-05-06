from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from mls_agent.tools.cuda.exceptions import ExecutorError


def _safe_join(root: Path, rel: str) -> Path:
    """Resolve rel relative to root and reject path traversal / absolute paths."""
    if Path(rel).is_absolute():
        raise ExecutorError(
            "path_escape",
            error_class="infrastructure",
            rel=rel,
            reason="absolute path not allowed",
        )
    resolved = (root / rel).resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ExecutorError(
            "path_escape",
            error_class="infrastructure",
            rel=rel,
            reason="path escapes workspace",
        )
    return resolved


class _Workspace:
    """Per-run temporary directory with deterministic sub-paths."""

    SUBDIRS = ("src", "bin", "ncu", "nsys", "logs")

    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve() / "exec"
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in self.SUBDIRS:
            (self.root / sub).mkdir(exist_ok=True)

    def allocate(self, kind: str, suffix: str) -> Path:
        """Return a unique path inside the workspace (file not yet created)."""
        uid = uuid.uuid4().hex[:6]
        return self.root / kind / f"{kind}_{uid}{suffix}"

    def write(self, rel: str, content: str | bytes) -> Path:
        """Write content to a workspace-relative path (creates parent dirs)."""
        target = _safe_join(self.root, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            target.write_text(content, encoding="utf-8")
        else:
            target.write_bytes(content)
        return target

    def cleanup(self, keep: bool = False) -> None:
        if not keep and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def rel(self, path: Path) -> str:
        """Return POSIX-style path relative to workspace root."""
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()
