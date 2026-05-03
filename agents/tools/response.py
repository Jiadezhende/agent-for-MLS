"""
agents/tools/response.py — Standardised tool response protocol.

Adapted from HelloAgents hello_agents/tools/response.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ToolStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"   # result available but with caveats (timeout, truncation)
    ERROR   = "error"


@dataclass
class ToolResponse:
    """Standardised tool response.

    Fields:
        status      — overall outcome
        text        — human-readable summary the LLM reads first
        data        — structured payload (full executor output, counts, etc.)
        error_info  — {code, message} only when status=ERROR
        stats       — runtime stats (elapsed_s, cache_hit, time_ms, …)
    """
    status:     ToolStatus
    text:       str
    data:       Dict[str, Any]          = field(default_factory=dict)
    error_info: Optional[Dict[str, str]] = None
    stats:      Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status": self.status.value,
            "text":   self.text,
            "data":   self.data,
        }
        if self.error_info:
            result["error"] = self.error_info
        if self.stats:
            result["stats"] = self.stats
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def success(
        cls,
        text: str,
        data: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> ToolResponse:
        return cls(status=ToolStatus.SUCCESS, text=text, data=data or {}, stats=stats)

    @classmethod
    def partial(
        cls,
        text: str,
        data: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> ToolResponse:
        return cls(status=ToolStatus.PARTIAL, text=text, data=data or {}, stats=stats)

    @classmethod
    def error(
        cls,
        code: str,
        message: str,
        stats: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> ToolResponse:
        return cls(
            status=ToolStatus.ERROR,
            text=message,
            data=data or {},
            error_info={"code": code, "message": message},
            stats=stats,
        )


class ToolErrorCode:
    """Standard error codes for all tools."""
    UNKNOWN_TOOL    = "unknown_tool"
    INVALID_ARGS    = "invalid_args"
    CIRCUIT_OPEN    = "circuit_open"
    EXECUTION_ERROR = "execution_error"
    EMPTY_EVIDENCE  = "empty_evidence"
    INVALID_NAME    = "invalid_name"
    SKILL_NOT_FOUND = "skill_not_found"
    INTERNAL_ERROR  = "internal_error"
