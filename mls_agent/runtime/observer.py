"""Observer protocol — external hooks into ReActLoop execution.

Replaces the ad-hoc ``print`` calls scattered across the legacy loop.
Default implementations: ``NullObserver`` (do-nothing) and
``StdoutObserver`` (formatted lines on stderr, with large-arg truncation).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any, Protocol, Sequence, runtime_checkable

from mls_agent.llm.types import ChatResponse, Message, ToolCall
from mls_agent.runtime.context import AgentContext
from mls_agent.runtime.result import AgentResult
from mls_agent.runtime.state import ReActPhase
from mls_agent.tools.response import ToolResponse, ToolStatus


@runtime_checkable
class AgentObserver(Protocol):
    """Hooks fired at every notable step of one Agent.run() call.

    All hook implementations should be cheap, exception-safe, and
    side-effect-free with respect to the agent. The runtime does not
    catch exceptions raised from observers — observe at your own risk.
    """

    def on_run_start(self, ctx: AgentContext) -> None: ...
    def on_iteration_start(self, iteration: int, ctx: AgentContext) -> None: ...
    def on_phase_transition(
        self, prev: ReActPhase | None, next: ReActPhase
    ) -> None: ...
    def on_llm_call(self, messages: Sequence[Message]) -> None: ...
    def on_llm_response(self, response: ChatResponse) -> None: ...
    def on_tool_call(self, call: ToolCall, response: ToolResponse) -> None: ...
    def on_terminate(self, result: AgentResult) -> None: ...
    def on_error(self, exc: Exception, phase: ReActPhase) -> None: ...


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


class NullObserver:
    """No-op observer — all hooks are pass-through."""

    def on_run_start(self, ctx: AgentContext) -> None: pass
    def on_iteration_start(self, iteration: int, ctx: AgentContext) -> None: pass
    def on_phase_transition(self, prev, next) -> None: pass
    def on_llm_call(self, messages) -> None: pass
    def on_llm_response(self, response) -> None: pass
    def on_tool_call(self, call, response) -> None: pass
    def on_terminate(self, result) -> None: pass
    def on_error(self, exc, phase) -> None: pass


class StdoutObserver:
    """Human-readable output to stderr.

    Replaces the verbose ``print`` calls in the legacy AgentLoop. Long
    string arguments are truncated to ``truncate_arg_log_at`` characters
    in the displayed JSON (the full value still goes to the LLM).
    """

    def __init__(
        self,
        *,
        prefix: str = "",
        stream=None,
        truncate_arg_log_at: int = 120,
    ) -> None:
        self._prefix = prefix
        self._stream = stream if stream is not None else sys.stderr
        self._truncate = truncate_arg_log_at

    # ------------------------------------------------------------------

    def _emit(self, text: str) -> None:
        self._stream.write(f"{self._prefix}{text}\n")
        self._stream.flush()

    def _summarize_args(self, args: dict[str, Any] | None) -> str:
        if not args:
            return "(no args)"
        compacted: dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, str) and len(v) > self._truncate:
                compacted[k] = f"<{len(v)} chars>"
            else:
                compacted[k] = v
        return json.dumps(compacted, separators=(",", ":"), default=str)

    def _summarize_response(self, response: ToolResponse) -> str:
        if response.status == ToolStatus.ERROR:
            code = (response.error_info or {}).get("code", "?")
            return f"status=error  code={code}  {response.text[:80]}"
        if response.status == ToolStatus.PARTIAL:
            return f"status=partial  {response.text[:80]}"
        # SUCCESS
        stats = response.stats or {}
        elapsed = stats.get("elapsed_s", "")
        cache = " [cache_hit]" if stats.get("cache_hit") else ""
        timing = f"  elapsed={elapsed}s{cache}" if elapsed else ""
        terminate = "  TERMINATE" if response.terminate else ""
        return f"status=success{timing}{terminate}  {response.text[:200]}"

    # ------------------------------------------------------------------
    # Observer hooks
    # ------------------------------------------------------------------

    def on_run_start(self, ctx: AgentContext) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._emit(f"[{ts}] run start  ({len(ctx.messages)} initial messages)")

    def on_iteration_start(self, iteration: int, ctx: AgentContext) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._emit(f"\n[{ts}] ── iter {iteration + 1} {'─' * 50}")

    def on_phase_transition(self, prev, next) -> None:
        # Quiet on phase transitions by default — too noisy.
        return

    def on_llm_call(self, messages: Sequence[Message]) -> None:
        return

    def on_llm_response(self, response: ChatResponse) -> None:
        msg = response.message
        if msg.content:
            self._emit(f"  {msg.content}")
        if response.finish_reason not in ("stop", "tool_calls"):
            self._emit(f"  finish_reason={response.finish_reason}")

    def on_tool_call(self, call: ToolCall, response: ToolResponse) -> None:
        self._emit(f"  call: {call.name}  {self._summarize_args(call.arguments)}")
        self._emit(f"  result: {call.name}  {self._summarize_response(response)}")

    def on_terminate(self, result: AgentResult) -> None:
        self._emit(
            f"  agent finished — reason={result.reason} "
            f"iterations={result.iterations}"
        )
        if result.summary:
            self._emit(f"  summary: {result.summary[:160]}")

    def on_error(self, exc: Exception, phase: ReActPhase) -> None:
        self._emit(f"  ERROR in phase {phase.value}: {type(exc).__name__}: {exc}")
