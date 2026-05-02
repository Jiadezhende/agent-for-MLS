"""pipeline/tools/_script_runner.py — shared helpers for tools that drive
``executor.profile_with_torch`` to evaluate candidates / generate baselines.

Each tool builds a Python script string with embedded absolute paths into the
RunLayout, prints a delimited JSON blob, and parses it back here.
"""
from __future__ import annotations

import json
from typing import Any


def parse_marked_json(stdout: str, marker: str) -> Any | None:
    """Find ``marker`` in ``stdout`` and return the next JSON line.

    Tolerates extra blank lines or non-JSON lines after the marker — picks the
    first parseable line. Returns None if the marker is absent or no JSON is
    found.
    """
    if not stdout or marker not in stdout:
        return None
    after = stdout.rsplit(marker, 1)[1]
    for line in after.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            # could be a stack trace line or something — keep scanning
            continue
    return None


def collect_stdout(output: dict) -> str:
    """Return whatever stdout-shaped content the executor returned.

    profile_with_torch normally returns either ``{"stdout": ...}`` or
    ``{"stdout_head", "stdout_tail"}``; we concatenate whichever pieces are
    present so parse_marked_json sees the marker.
    """
    parts = []
    for key in ("stdout", "stdout_head", "stdout_tail"):
        val = output.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
    return "\n".join(parts)


def run_python_script(
    executor: Any,
    *,
    code: str,
    op_name: str,
    timeout_s: int = 120,
) -> dict:
    """Wrap profile_with_torch with a uniform error envelope.

    Returns:
      - on success: the raw dict from profile_with_torch (stdout / stdout_tail / ...)
      - on executor exception: ``{"_executor_error": "<class>: <msg>"}``
    """
    try:
        return executor.profile_with_torch(code, op_name=op_name, timeout_s=timeout_s)
    except Exception as e:  # noqa: BLE001 — the tool decides how to report
        return {"_executor_error": f"{type(e).__name__}: {e}"}
