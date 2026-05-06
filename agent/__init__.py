"""agents — shared executor / LLM / tool layer.

The legacy planner/critic + per-agent plugin registry was removed when the
pipeline architecture replaced them; concrete agents now live in
``pipeline/agents/`` and are wired up explicitly in ``main.py``.

This package still owns:
  - ``agents.core``: Config, LLMClient, AgentLoop, Memory/Event types.
  - ``agents.tools``: Executor, ToolRegistry, builtin tools (recording, skills).
"""
