"""
agents/tools/base.py — Tool ABC and ToolParameter.

Adapted from HelloAgents hello_agents/tools/base.py.
ToolParameter is kept identical to the HelloAgents version (5 fields).
Tools with complex OpenAI schemas (enum, oneOf, nullable, minItems) override
to_openai_schema() directly rather than encoding those constraints here.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel

from agents.tools.response import ToolErrorCode, ToolResponse


class ToolParameter(BaseModel):
    """Single parameter definition — same contract as HelloAgents."""
    name:        str
    type:        str    # "string"|"integer"|"number"|"boolean"|"array"|"object"
    description: str
    required:    bool = True
    default:     Any  = None


class Tool(ABC):
    """Base class for all MLS tools.

    Every tool is a self-contained object that defines its own parameter schema
    and handles its own execution.  The registry calls run() (or run_with_timing())
    and receives a ToolResponse; it never needs to know the tool's internal type.
    """

    def __init__(self, name: str, description: str) -> None:
        self.name        = name
        self.description = description

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        """Execute the tool and return a standardised ToolResponse."""

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """Return the list of accepted parameters.

        The default to_openai_schema() builds an OpenAI function-calling schema
        from this list.  Tools with complex constraints (enum, oneOf, nullable,
        minItems, min/max) should override to_openai_schema() instead of trying
        to encode those in ToolParameter.
        """

    # ------------------------------------------------------------------
    # Schema generation (override for complex schemas)
    # ------------------------------------------------------------------

    def to_openai_schema(self) -> Dict[str, Any]:
        """Generate the OpenAI function-calling schema for this tool.

        Default implementation derives the schema from get_parameters().
        Override this method when the parameters require enum, oneOf, nullable,
        minItems, or other JSON Schema features not expressible via ToolParameter.
        """
        properties: Dict[str, Any] = {}
        required:   List[str]      = []

        for p in self.get_parameters():
            prop: Dict[str, Any] = {"type": p.type, "description": p.description}
            if p.type == "array":
                prop["items"] = {"type": "string"}
            if p.default is not None:
                prop["description"] = f"{p.description} (default: {p.default})"
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name":        self.name,
                "description": self.description,
                "parameters": {
                    "type":       "object",
                    "properties": properties,
                    "required":   required,
                },
            },
        }

    # ------------------------------------------------------------------
    # Timed execution
    # ------------------------------------------------------------------

    def run_with_timing(self, parameters: Dict[str, Any]) -> ToolResponse:
        """Call run() and add stats.time_ms.  Catches unhandled exceptions."""
        t0 = time.perf_counter()
        try:
            response = self.run(parameters)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Unhandled exception in {self.name}: {exc}",
                stats={"time_ms": elapsed_ms},
            )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if response.stats is None:
            response.stats = {}
        response.stats.setdefault("time_ms", elapsed_ms)
        return response

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"
