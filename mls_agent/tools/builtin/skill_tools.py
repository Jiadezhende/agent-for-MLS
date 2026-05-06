"""Discover and read measurement-strategy markdown documents.

Skills are discovered via YAML frontmatter (``name`` + ``description`` fields)
embedded in each ``.md`` file. ``SkillRegistry`` walks the directory once at
construction; ``ListSkillsTool`` and ``ReadSkillTool`` query the registry
instead of touching disk on every call.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mls_agent.tools.base import Tool
from mls_agent.tools.response import ToolErrorCode, ToolResponse


_MAX_BYTES = 32 * 1024
_NAME_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_/]*$")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillMeta:
    name: str          # relative stem, e.g. "operators/lora_matmul"
    description: str   # from frontmatter
    path: Path         # absolute path to .md file


def _parse_frontmatter(path: Path) -> dict | None:
    """Extract ``{name, description}`` from the leading ``---``-fenced block."""
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
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    return text[end + 4:].lstrip("\n")


class SkillRegistry:
    """In-memory index of skill markdown files under a single root directory."""

    def __init__(self, skills_dir: str | Path) -> None:
        self._skills: dict[str, SkillMeta] = {}
        self._root = Path(skills_dir)
        if self._root.is_dir():
            self._load(self._root)

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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class ListSkillsTool(Tool):
    NAME = "list_skills"
    DESCRIPTION = (
        "List all available skill documents. Call first to see what strategies "
        "exist, then use read_skill to load the relevant document."
    )

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        skills = self._registry.list()
        if not skills:
            return ToolResponse.success(text="no skills found.", data={"skills": []})
        entries = [{"name": s.name, "description": s.description} for s in skills]
        return ToolResponse.success(
            text=f"Found {len(entries)} skill(s): {[e['name'] for e in entries]}",
            data={"skills": entries},
        )


class ReadSkillTool(Tool):
    NAME = "read_skill"
    DESCRIPTION = (
        "Read the full content of a skill document. Each skill describes when "
        "to apply a strategy, what kernel pattern to use, how to interpret the "
        "output, and what anomalies to watch for."
    )

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Skill name as listed by list_skills "
                        "(e.g. 'memory_hierarchy' or 'operators/lora_matmul')."
                    ),
                }
            },
            "required": ["name"],
        }

    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        name = parameters["name"]

        if not _NAME_RE.match(name) or ".." in name:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message=(
                    f"skill name {name!r} is invalid. "
                    "Use only letters, digits, underscores, and forward slashes."
                ),
            )

        meta = self._registry.get(name)
        if meta is None:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_ARGS,
                message=f"skill {name!r} not found. Available: {self._registry.names()}",
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_skill_tools(skills_dir: str | Path) -> tuple[Tool, ...]:
    """Construct (ListSkillsTool, ReadSkillTool) sharing one ``SkillRegistry``."""
    registry = SkillRegistry(skills_dir)
    return (ListSkillsTool(registry), ReadSkillTool(registry))
