"""pipeline/agents/_base.py — LLMStageAgent base class.

Every concrete stage agent inherits from LLMStageAgent and overrides:
  - ``stage`` (Stage enum value)
  - ``allowed_tools`` (tuple[str, ...])
  - ``SYSTEM_PROMPT`` (str)
  - ``build_user_message(context) -> str``
  - optionally ``max_iterations`` (default 20)

The base class handles ReAct loop construction, exception → failed-StageResult
conversion, and StageResult retrieval via the agent_loop_signal helpers.
"""
from __future__ import annotations

from typing import Any

from agents.core.types import AgentContext, MemoryStore
from agents.tools.circuit_breaker import CircuitBreaker

from ..agent_loop_signal import pop_stage_result
from ..stage_agent import StageAgent
from ..stage_runner import StageContext
from ..state import StageResult


class LLMStageAgent(StageAgent):
    """Common scaffolding for every LLM-driven stage agent."""

    # --- subclass overrides ---------------------------------------------
    SYSTEM_PROMPT: str = ""
    max_iterations: int = 20

    def __init__(self, llm: Any, *, agent_cfg: Any = None, verbose: bool = False):
        self.llm = llm
        self.agent_cfg = agent_cfg
        self.verbose = verbose

    # --- subclass must implement ----------------------------------------
    def build_user_message(self, context: StageContext) -> str:
        raise NotImplementedError

    # --- run() — delegates to AgentLoop ---------------------------------
    def run(self, context: StageContext) -> StageResult:
        # Lazy import keeps test isolation cheap.
        from agents.core.loop import AgentLoop

        agent_ctx = AgentContext(
            memory=MemoryStore(),
            circuit_breaker=CircuitBreaker(
                threshold=getattr(self.agent_cfg, "circuit_breaker_threshold", 3),
                half_open_timeout_s=getattr(self.agent_cfg, "half_open_timeout_s", 60.0),
            ),
        )

        max_iter = self.max_iterations
        cfg_max = getattr(self.agent_cfg, "max_iterations", None)
        if isinstance(cfg_max, int) and cfg_max > 0:
            max_iter = min(max_iter, cfg_max)

        try:
            AgentLoop(
                llm=self.llm,
                registry=context.tools,
                ctx=agent_ctx,
                max_iterations=max_iter,
                verbose=context.verbose or self.verbose,
                worker_id=f"sub_{self.stage.value.lower()}",
                system_prompt=self.SYSTEM_PROMPT,
                user_message=self.build_user_message(context),
            ).run()
        except RuntimeError as e:
            return self._failed_result(f"agent_loop_aborted: {e}", agent_ctx)

        result = pop_stage_result(agent_ctx)
        if result is None:
            return self._failed_result("no_submit_called", agent_ctx)
        return result

    def _failed_result(self, reason: str, agent_ctx: AgentContext) -> StageResult:
        return StageResult(
            stage=self.stage.value,
            status="failed",
            artifacts={},
            metrics={"reasoning_log_len": len(agent_ctx.reasoning_log)},
            confidence=0.0,
            caveats=[reason],
        )
