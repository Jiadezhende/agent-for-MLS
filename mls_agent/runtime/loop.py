"""ReAct loop — the explicit Thought / Validate / Act / Observe / Apply / Decide
state machine.

One ``run()`` call corresponds to one ReAct trajectory: alternating LLM
calls and tool dispatches until a terminate signal, the iteration
ceiling, the no-tool-call streak limit, or an unrecoverable LLM error.

Termination mechanics:
  * A ``ToolResponse`` with ``terminate=True`` ends the loop with
    reason="completed", carrying the summary + payload through to
    ``AgentResult``. This is the *only* sanctioned termination channel
    for tools — no exception control flow.
  * Other terminations are produced by the loop itself
    (``max_iterations``, ``no_tool_call``, ``llm_error``).
"""
from __future__ import annotations

from typing import Sequence

from mls_agent.llm.backend import LLMBackend
from mls_agent.llm.types import Message
from mls_agent.runtime.config import AgentConfig
from mls_agent.runtime.context import AgentContext
from mls_agent.runtime.observer import AgentObserver, NullObserver
from mls_agent.runtime.result import AgentResult
from mls_agent.runtime.state import ReActPhase
from mls_agent.tools.registry import ToolRegistry
from mls_agent.tools.response import Event, ToolResponse


_NUDGE_TEXT = (
    "You must call a tool. Review your task and continue, or call your "
    "designated submit/finalize tool to terminate when done."
)


class ReActLoop:
    """Run one ReAct trajectory.

    The loop is ``stateful`` only with respect to ``AgentContext`` (built
    fresh in ``run()``); the loop object itself is reusable across runs.
    """

    def __init__(
        self,
        backend: LLMBackend,
        registry: ToolRegistry,
        system_prompt: str,
        config: AgentConfig,
        observer: AgentObserver | None = None,
    ) -> None:
        self._backend = backend
        self._registry = registry
        self._system_prompt = system_prompt
        self._cfg = config
        self._obs: AgentObserver = observer or NullObserver()

    # ------------------------------------------------------------------

    def run(self, user_message: str) -> AgentResult:
        ctx = AgentContext()
        ctx.messages.append(Message.system(self._system_prompt))
        ctx.messages.append(Message.user(user_message))
        self._obs.on_run_start(ctx)

        no_tool_call_streak = 0

        for i in range(self._cfg.max_iterations):
            ctx.iteration = i
            self._obs.on_iteration_start(i, ctx)

            # ───── THOUGHT ────────────────────────────────────────────
            self._obs.on_phase_transition(None, ReActPhase.THOUGHT)
            self._obs.on_llm_call(tuple(ctx.messages))
            try:
                response = self._backend.chat(
                    list(ctx.messages),
                    self._registry.schemas(),
                )
            except Exception as exc:  # noqa: BLE001 — surface as result, not crash
                self._obs.on_error(exc, ReActPhase.THOUGHT)
                ctx.events.append(
                    Event(type="llm_error", severity="error", detail=str(exc))
                )
                return AgentResult(
                    reason="llm_error",
                    summary=None,
                    payload=None,
                    iterations=i,
                    context=ctx,
                )
            self._obs.on_llm_response(response)
            ctx.messages.append(response.message)

            # ───── VALIDATE ───────────────────────────────────────────
            self._obs.on_phase_transition(ReActPhase.THOUGHT, ReActPhase.VALIDATE)
            if not response.message.tool_calls:
                no_tool_call_streak += 1
                if no_tool_call_streak >= self._cfg.max_consecutive_no_tool_call:
                    return AgentResult(
                        reason="no_tool_call",
                        summary=response.message.content,
                        payload=None,
                        iterations=i + 1,
                        context=ctx,
                    )
                ctx.messages.append(Message.user(_NUDGE_TEXT))
                continue
            no_tool_call_streak = 0

            # ───── ACT + OBSERVE + APPLY + DECIDE ─────────────────────
            self._obs.on_phase_transition(ReActPhase.VALIDATE, ReActPhase.ACT)
            terminate_result: AgentResult | None = None
            for call in response.message.tool_calls:
                tool_response = self._registry.dispatch(call.name, call.arguments)
                self._obs.on_tool_call(call, tool_response)

                # OBSERVE — text the LLM gets back as the tool message
                ctx.messages.append(
                    Message.tool_result(
                        tool_call_id=call.id,
                        content=_format_tool_result(tool_response),
                    )
                )

                # APPLY — land side effects in the order the tool requested
                self._obs.on_phase_transition(ReActPhase.OBSERVE, ReActPhase.APPLY)
                ctx.events.extend(tool_response.events)
                ctx.measurements.extend(tool_response.measurements)

                # DECIDE — terminate if any tool requested it
                if tool_response.terminate:
                    terminate_result = AgentResult(
                        reason="completed",
                        summary=tool_response.terminate_summary,
                        payload=tool_response.terminate_payload,
                        iterations=i + 1,
                        context=ctx,
                    )
                    break

            if terminate_result is not None:
                self._obs.on_terminate(terminate_result)
                return terminate_result

        # Iteration ceiling reached without a terminate signal.
        result = AgentResult(
            reason="max_iterations",
            summary=None,
            payload=None,
            iterations=self._cfg.max_iterations,
            context=ctx,
        )
        self._obs.on_terminate(result)
        return result


def _format_tool_result(response: ToolResponse) -> str:
    """Produce the string the LLM sees as the tool message content.

    We pass the ``ToolResponse.to_dict()`` JSON through verbatim — the
    LLM already understands the ``status / text / data / error`` shape
    from previous turns and from tool descriptions. Side-effect fields
    (events / measurements / terminate*) are deliberately NOT shown to
    the LLM: they are framework-internal.
    """
    import json

    return json.dumps(response.to_dict(), default=str, ensure_ascii=False)
