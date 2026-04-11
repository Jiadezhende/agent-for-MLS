"""
tools/skills.py — Read measurement strategy documents from skills/.

These tools do NOT go through the Executor; they are pure filesystem reads.
"""
from __future__ import annotations

import re
from pathlib import Path

# Resolve the skills directory relative to this file
_SKILLS_DIR = Path(__file__).parent.parent / "skills"

_MAX_BYTES = 32 * 1024   # 32 KB per skill
_NAME_RE   = re.compile(r"^[a-zA-Z0-9_]+$")


def list_skills() -> dict:
    """Return a list of available skill names and one-line summaries."""
    skills = []
    if not _SKILLS_DIR.exists():
        return {"skills": []}

    for path in sorted(_SKILLS_DIR.glob("*.md")):
        # Exclude: internal templates (start with _) and
        # documentation files (start with uppercase, e.g. README.md)
        if path.name.startswith("_") or path.stem[0].isupper():
            continue
        name = path.stem
        summary = _extract_summary(path)
        skills.append({"name": name, "summary": summary})

    return {"skills": skills}


def read_skill(name: str) -> dict:
    """Return the full content of a skill document."""
    if not _NAME_RE.match(name):
        return {
            "error": "invalid_name",
            "detail": (
                f"Skill name '{name}' is not valid. "
                "Use only letters, digits, and underscores."
            ),
        }

    path = _SKILLS_DIR / f"{name}.md"
    if not path.exists():
        available = [p.stem for p in _SKILLS_DIR.glob("*.md") if not p.name.startswith("_")]
        return {
            "error": "skill_not_found",
            "name": name,
            "available": available,
        }

    raw = path.read_bytes()
    truncated = len(raw) > _MAX_BYTES
    content = raw[:_MAX_BYTES].decode("utf-8", errors="replace")
    if truncated:
        content += "\n\n[... content truncated at 32 KB ...]"

    return {"content": content, "truncated": truncated}


def _extract_summary(path: Path) -> str:
    """Read the first meaningful line from a markdown file as its summary."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                return stripped[:120]
    except Exception:
        pass
    return "(no summary)"
