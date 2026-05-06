"""Runtime configuration for an Agent run."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """Per-run agent configuration.

    Tuning these does not change the framework contract — only the
    iteration ceiling, nudge policy, and circuit-breaker thresholds.
    """

    max_iterations: int = 40
    max_consecutive_no_tool_call: int = 2
    circuit_breaker_threshold: int = 3
    circuit_breaker_half_open_s: float = 60.0
    truncate_arg_log_at: int = 120

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError(
                f"max_iterations must be > 0, got {self.max_iterations}"
            )
        if self.max_consecutive_no_tool_call <= 0:
            raise ValueError(
                "max_consecutive_no_tool_call must be > 0, got "
                f"{self.max_consecutive_no_tool_call}"
            )
        if self.circuit_breaker_threshold <= 0:
            raise ValueError(
                "circuit_breaker_threshold must be > 0, got "
                f"{self.circuit_breaker_threshold}"
            )
        if self.circuit_breaker_half_open_s < 0:
            raise ValueError(
                "circuit_breaker_half_open_s must be >= 0, got "
                f"{self.circuit_breaker_half_open_s}"
            )
        if self.truncate_arg_log_at <= 0:
            raise ValueError(
                f"truncate_arg_log_at must be > 0, got {self.truncate_arg_log_at}"
            )
