"""
agents/tools/builtin/skills.py — Read measurement strategy documents from skills/.

Skills are discovered via YAML frontmatter (name + description fields) embedded in
each .md file. A module-level SkillRegistry scans the directory once at import time;
ListSkillsTool and ReadSkillTool both query the registry instead of re-scanning disk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from agent.tools.base import Tool, ToolParameter
from agent.tools.response import ToolErrorCode, ToolResponse

_SKILLS_DIR = Path(__file__).parent.parent.parent.parent / "skills"
_MAX_BYTES   = 32 * 1024
_NAME_RE     = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_/]*$")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class SkillMeta:
    name: str          # relative stem, e.g. "operators/lora_matmul"
    description: str   # from frontmatter
    path: Path         # absolute path to .md file


def _parse_frontmatter(path: Path) -> dict | None:
    """Return {'name': ..., 'description': ...} from leading YAML block, or None."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    if not text.startswith("---"):
        return None

    end = text.find("\n---", 3)
    if end == -1:
        return None

    block = text[3:end]
    result: dict = {}
    for line in block.splitlines():
        for key in ("name", "description"):
            prefix = f"{key}:"
            if line.startswith(prefix):
                value = line[len(prefix):].strip().strip('"').strip("'")
                if value:
                    result[key] = value
    return result if "description" in result else None


def _strip_frontmatter(text: str) -> str:
    """Remove leading YAML block (--- ... ---) from skill content."""
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    return text[end + 4:].lstrip("\n")


class SkillRegistry:
    def __init__(self, skills_dir: Path) -> None:
        self._skills: dict[str, SkillMeta] = {}
        if skills_dir.exists():
            self._load(skills_dir)

    def _load(self, skills_dir: Path) -> None:
        for path in sorted(skills_dir.rglob("*.md")):
            if path.name.startswith("_") or path.name[0].isupper():
                continue
            stem = path.relative_to(skills_dir).with_suffix("").as_posix()
            meta = _parse_frontmatter(path)
            if meta is None:
                continue
            self._skills[stem] = SkillMeta(
                name=stem,
                description=meta["description"],
                path=path,
            )

    def list(self) -> list[SkillMeta]:
        return list(self._skills.values())

    def get(self, name: str) -> SkillMeta | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return list(self._skills.keys())


_REGISTRY = SkillRegistry(_SKILLS_DIR)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

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
        skills = _REGISTRY.list()
        if not skills:
            return ToolResponse.success(text="No skills found.", data={"skills": []})

        entries = [{"name": s.name, "description": s.description} for s in skills]
        return ToolResponse.success(
            text=f"Found {len(entries)} skill(s): {[e['name'] for e in entries]}",
            data={"skills": entries},
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
                    "Skill name as returned by list_skills "
                    "(e.g. 'memory_hierarchy' or 'operators/lora_matmul')."
                ),
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        name = parameters.get("name", "")

        if not _NAME_RE.match(name) or ".." in name:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_NAME,
                message=(
                    f"Skill name '{name}' is not valid. "
                    "Use only letters, digits, underscores, and forward slashes "
                    "(e.g. 'memory_hierarchy' or 'operators/lora_matmul')."
                ),
            )

        meta = _REGISTRY.get(name)
        if meta is None:
            return ToolResponse.error(
                code=ToolErrorCode.SKILL_NOT_FOUND,
                message=f"Skill '{name}' not found.",
                stats={"available": _REGISTRY.names()},
            )

        raw = meta.path.read_bytes()
        truncated = len(raw) > _MAX_BYTES
        content = raw[:_MAX_BYTES].decode("utf-8", errors="replace")
        if truncated:
            content += "\n\n[... content truncated at 32 KB ...]"

        content = _strip_frontmatter(content)

        return ToolResponse.success(
            text=content,
            data={"content": content, "truncated": truncated},
        )
