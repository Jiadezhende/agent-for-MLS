"""pipeline/stage_agent.py — abstract base class for all StageAgent implementations.

Every StageAgent declares the Stage it serves and the exact set of tools it's
allowed to call. The stage_runner enforces this declaration: agents that ask
for unknown tools fail to build, and the StageResult schema must be honoured.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .state import Stage, StageResult

if TYPE_CHECKING:
    from .stage_runner import StageContext


class StageAgent(ABC):
    """Base class for every pipeline stage agent.

    Subclasses (BenchmarkSpecAgent, KernelTuningAgent, ...) must:
      - set the class attribute ``stage`` to a Stage enum value
      - set ``allowed_tools`` to the exhaustive list of tools they may call
      - implement ``run(StageContext) -> StageResult``

    The stage_runner uses ``allowed_tools`` to build a per-stage ToolRegistry
    so that calls to undeclared tools fail at dispatch time (not at audit time).
    """

    # Subclass overrides --------------------------------------------------
    stage: Stage  # required
    allowed_tools: tuple[str, ...] = ()  # required (declare even if empty)

    # Lifecycle -----------------------------------------------------------
    @abstractmethod
    def run(self, context: "StageContext") -> StageResult:
        """Execute the stage and return a StageResult.

        The stage_runner will:
          - validate the result's stage matches ``self.stage``
          - validate the result against the StageResult schema
          - convert exceptions to a ``failed`` StageResult with the error in caveats
        """
        raise NotImplementedError
