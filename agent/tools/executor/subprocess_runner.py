from __future__ import annotations

import locale
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STDOUT_TAIL_BYTES = 8_192
STDOUT_HEAD_BYTES = 1_024


@dataclass
class SubResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


def _run_subprocess(
    cmd: list[str],
    timeout_s: int,
    cwd: Path | None = None,
    env: dict | None = None,
    truncate_bytes: int | None = 64_000,
    encoding: str | None = None,
) -> SubResult:
    """Run cmd and return a SubResult. Never raises; timeouts kill the process tree."""
    # .bat files on Windows are not directly executable; wrap with cmd /c
    if sys.platform == "win32" and cmd and Path(cmd[0]).suffix.lower() == ".bat":
        cmd = ["cmd", "/c"] + cmd

    trunc_marker = b"\n[... output truncated ...]\n"
    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "cwd": cwd,
        "env": env,
    }
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True

    timed_out = False
    timer: threading.Timer | None = None

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)

        def _kill_tree() -> None:
            nonlocal timed_out
            timed_out = True
            try:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                        timeout=10,
                    )
                else:
                    import signal
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        if timeout_s and timeout_s > 0:
            timer = threading.Timer(timeout_s, _kill_tree)
            timer.daemon = True
            timer.start()

        stdout_b, stderr_b = proc.communicate()
        returncode = -1 if timed_out else proc.returncode
    except FileNotFoundError:
        stdout_b = b""
        stderr_b = f"command not found: {cmd[0]}".encode()
        timed_out = False
        returncode = 127
    except Exception as exc:
        stdout_b = b""
        stderr_b = str(exc).encode(_subprocess_encoding(), errors="replace")
        returncode = -1 if timed_out else 127
    finally:
        if timer is not None:
            timer.cancel()

    if truncate_bytes is not None and len(stdout_b) > truncate_bytes:
        stdout_b = stdout_b[:truncate_bytes] + trunc_marker
    if truncate_bytes is not None and len(stderr_b) > truncate_bytes:
        stderr_b = stderr_b[:truncate_bytes] + trunc_marker

    enc = encoding or _subprocess_encoding()
    return SubResult(
        stdout=stdout_b.decode(enc, errors="replace"),
        stderr=stderr_b.decode(enc, errors="replace"),
        returncode=returncode,
        timed_out=timed_out,
    )


def _subprocess_encoding() -> str:
    """Encoding for compiler/runtime output, independent of redirected stdout."""
    if sys.platform == "win32":
        return locale.getpreferredencoding(False) or "gbk"
    return "utf-8"


def _reduce_probe_output(stdout: str | bytes, encoding: str | None = None) -> dict:
    """Return compact LLM-facing stdout evidence, preserving tail measurements."""
    enc = encoding or _subprocess_encoding()
    stdout_b: bytes = stdout.encode(enc, errors="replace") if isinstance(stdout, str) else stdout
    total = len(stdout_b)
    if total <= STDOUT_TAIL_BYTES + STDOUT_HEAD_BYTES:
        stdout_s = stdout_b.decode(enc, errors="replace")
        return {
            "stdout": stdout_s,
            "stdout_tail": stdout_s,
            "stdout_total_bytes": total,
        }

    head_b = stdout_b[:STDOUT_HEAD_BYTES]
    tail_b = stdout_b[-STDOUT_TAIL_BYTES:]

    head_nl = head_b.rfind(b"\n")
    if head_nl > 0:
        head_b = head_b[:head_nl + 1]
    tail_nl = tail_b.find(b"\n")
    if tail_nl >= 0 and tail_nl + 1 < len(tail_b):
        tail_b = tail_b[tail_nl + 1:]

    return {
        "stdout_head": head_b.decode(enc, errors="replace"),
        "stdout_tail": tail_b.decode(enc, errors="replace"),
        "stdout_total_bytes": total,
    }
