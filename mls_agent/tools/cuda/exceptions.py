"""Executor-level exceptions.

``ExecutorError`` carries a structured taxonomy via the ``error_class``
attribute so tool wrappers can map executor failures to the
``ToolErrorCode`` codes recognized by the runtime.

  ``user_code``      — CUDA source or arguments are wrong; the LLM should fix code.
  ``infrastructure`` — binary missing or env misconfigured; the LLM should NOT retry.
  ``timeout``        — execution exceeded the time limit; the LLM may reduce workload.
"""
from __future__ import annotations

from typing import Any


class ExecutorError(Exception):
    """Raised for expected executor-level failures (compile error, bad path, …)."""

    def __init__(
        self,
        kind: str,
        error_class: str = "infrastructure",
        hint: str | None = None,
        **details: Any,
    ) -> None:
        self.kind = kind
        self.error_class = error_class
        self.hint = hint
        self.details = details
        super().__init__(f"ExecutorError({kind}): {details}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "error",
            "error": self.kind,
            "error_class": self.error_class,
            "hint": self.hint,
            **self.details,
        }
