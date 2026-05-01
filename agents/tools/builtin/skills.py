"""
agents/tools/builtin/skills.py — Read measurement strategy documents from skills/.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from agents.tools.base import Tool, ToolParameter
from agents.tools.response import ToolErrorCode, ToolResponse

_SKILLS_DIR = Path(__file__).parent.parent.parent.parent / "skills"
_MAX_BYTES  = 32 * 1024
_NAME_RE    = re.compile(r"^[a-zA-Z0-9_]+$")


class ListSkillsTool(Tool):
    """List all available measurement strategy documents."""

    def __init__(self) -> None:
        super().__init__(
            name="list_skills",
            description=(
                "List all available measurement strategy documents (skills). "
                "Call this first to discover what strategies exist, then use "
                "read_skill to load the one most relevant to your target metric."
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return []

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        if not _SKILLS_DIR.exists():
            return ToolResponse.success(text="No skills directory found.", data={"skills": []})

        skills = []
        for path in sorted(_SKILLS_DIR.glob("*.md")):
            if path.name.startswith("_") or path.stem[0].isupper():
                continue
            skills.append({"name": path.stem, "summary": _extract_summary(path)})

        return ToolResponse.success(
            text=f"Found {len(skills)} skill(s): {[s['name'] for s in skills]}",
            data={"skills": skills},
        )


class ReadSkillTool(Tool):
    """Read the full content of a measurement strategy document."""

    def __init__(self) -> None:
        super().__init__(
            name="read_skill",
            description=(
                "Read the full content of a measurement strategy document. "
                "Each skill describes when to use a particular approach, what "
                "CUDA kernel pattern to apply, how to interpret the output, "
                "and what anomalies to watch for."
            ),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="name",
                type="string",
                description=(
                    "Skill name (filename without .md extension). "
                    "Use list_skills to get valid names."
                ),
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        name = parameters.get("name", "")

        if not _NAME_RE.match(name):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_NAME,
                message=(
                    f"Skill name '{name}' is not valid. "
                    "Use only letters, digits, and underscores."
                ),
            )

        path = _SKILLS_DIR / f"{name}.md"
        if not path.exists():
            available = [p.stem for p in _SKILLS_DIR.glob("*.md") if not p.name.startswith("_")]
            return ToolResponse.error(
                code=ToolErrorCode.SKILL_NOT_FOUND,
                message=f"Skill '{name}' not found.",
                stats={"available": available},
            )

        raw = path.read_bytes()
        truncated = len(raw) > _MAX_BYTES
        content = raw[:_MAX_BYTES].decode("utf-8", errors="replace")
        if truncated:
            content += "\n\n[... content truncated at 32 KB ...]"

        return ToolResponse.success(
            text=content,
            data={"content": content, "truncated": truncated},
        )


def _extract_summary(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                return stripped[:120]
    except Exception:
        pass
    return "(no summary)"
