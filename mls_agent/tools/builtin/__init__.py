"""Provider-agnostic builtin tools shipped with mls_agent.

Two families:

* ``skill_tools`` — discover and read measurement-strategy markdown documents
  from a ``skills/`` directory.
* ``side_effect_tools`` — let the agent record events / measurements via the
  ``ToolResponse`` side-effect channels (``events`` / ``measurements``) so the
  runtime appends them to ``AgentContext`` after dispatch.

Each module exposes a small factory function that wraps construction details:

>>> from mls_agent.tools.builtin import make_skill_tools, make_side_effect_tools
>>> skill_pair = make_skill_tools(skills_dir="./skills")
>>> tools = {t.NAME: t for t in (*skill_pair, *make_side_effect_tools())}
"""
from mls_agent.tools.builtin.skill_tools import (
    ListSkillsTool,
    ReadSkillTool,
    SkillRegistry,
    make_skill_tools,
)
from mls_agent.tools.builtin.side_effect_tools import (
    FlagEventTool,
    RecordMeasurementTool,
    make_side_effect_tools,
)

__all__ = [
    "FlagEventTool",
    "ListSkillsTool",
    "ReadSkillTool",
    "RecordMeasurementTool",
    "SkillRegistry",
    "make_side_effect_tools",
    "make_skill_tools",
]
