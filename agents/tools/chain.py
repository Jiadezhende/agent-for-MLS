"""
agents/tools/chain.py — ToolChain: compose multiple tools in sequence.
"""
from __future__ import annotations

from typing import Any


class ToolChain:
    """Compose multiple tool callables into a sequential pipeline.

    Each tool receives the output dict of the previous tool merged into
    its input kwargs. Use this to build multi-step tool pipelines without
    writing custom orchestration logic.
    """

    def __init__(self, tools: list) -> None:
        self._tools = tools

    def run(self, initial_input: dict[str, Any]) -> dict[str, Any]:
        state = dict(initial_input)
        for tool in self._tools:
            result = tool(**state)
            if isinstance(result, dict):
                state.update(result)
        return state
