"""Exception hierarchy for mls_agent."""
from __future__ import annotations


class MlsAgentError(Exception):
    """Root exception type for the mls_agent framework."""


class AgentBackendError(MlsAgentError):
    """An LLM backend call failed irrecoverably (after retries)."""

    def __init__(self, original: Exception):
        super().__init__(f"{type(original).__name__}: {original}")
        self.original = original
