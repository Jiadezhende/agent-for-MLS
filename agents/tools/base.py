"""
agents/tools/base.py — Tool protocol (structural interface for all tools).
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """Structural type that all tool callables satisfy."""
    name: str
    schema: dict

    def __call__(self, **kwargs: Any) -> Any: ...
