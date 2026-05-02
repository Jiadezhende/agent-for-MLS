"""
agents/core/loop.py — The main agent loop: LLM ↔ tool dispatch ↔ AgentContext.

One iteration = one LLM API call. Within a single call the LLM may return
multiple parallel tool calls; we dispatch them sequentially.
"""
from __future__ import annotations

import json
import sys

from agents.core.llm import LLMClient
from agents.core.types import AgentContext
from agents.tools.registry import ToolRegistry, _Terminated
from agents.tools.response import ToolResponse, ToolStatus


_LARGE_TEXT_ARGS: frozenset[str] = frozenset({"source", "source_or_path", "python_code"})
_ARG_TRUNCATE_AT: int = 120


def _summarize_args(args: dict | None) -> str:
    if not args:
        return "(no args)"
    compacted: dict = {}
    for k, v in args.items():
        if k in _LARGE_TEXT_ARGS and isinstance(v, str) and len(v) > _ARG_TRUNCATE_AT:
            compacted[k] = f"<{len(v)} chars>"
        else:
            compacted[k] = v
    return json.dumps(compacted, separators=(",", ":"), default=str)


def _summarize_result(response: ToolResponse) -> str:
    if response.status == ToolStatus.ERROR:
        code = (response.error_info or {}).get("code", "?")
        return f"status=error  code={code}  {response.text[:80]}"
    if response.status == ToolStatus.PARTIAL:
        return f"status=partial  {response.text[:80]}"
    # SUCCESS
    stats   = response.stats or {}
    elapsed = stats.get("elapsed_s", "")
    cache   = " [cache_hit]" if stats.get("cache_hit") else ""
    timing  = f"  elapsed={elapsed}s{cache}" if elapsed else ""
    return f"status=success{timing}  {response.text[:60]}"


class AgentLoop:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        ctx: AgentContext,
        max_iterations: int = 40,
        verbose: bool = False,
        worker_id: int | str | None = None,
        system_prompt: str | None = None,
        user_message: str = "",
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.ctx = ctx
        self.max_iterations = max_iterations
        self.verbose = verbose
        self._prefix = f"[W{worker_id}] " if worker_id is not None else ""
        self._system_prompt = system_prompt or ""
        self._initial_user_message = user_message

    def _emit(self, *args: object) -> None:
        if self.verbose:
            if args and isinstance(args[0], str):
                print(self._prefix + args[0], *args[1:], file=sys.stderr, flush=True)
            else:
                print(self._prefix, *args, file=sys.stderr, flush=True)

    def run(self) -> AgentContext:
        if not self.ctx.messages:
            self.ctx.messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user",   "content": self._initial_user_message},
            ]
        messages = self.ctx.messages
        nudged = False

        for i in range(self.max_iterations):
            self.ctx.iteration = i

            try:
                resp = self.llm.chat(messages, tools=self.registry.schemas())
            except Exception as exc:
                self.ctx.events.append({"type": "llm_error", "detail": str(exc), "iteration": i})
                raise

            messages.append(resp.to_openai_message())
            self.ctx.reasoning_log.append({
                "iteration": i,
                "content": resp.content,
                "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in resp.tool_calls],
            })

            self._emit(f"\n── iter {i + 1}/{self.max_iterations} {'─' * 40}")
            if resp.content:
                self._emit(f"  {resp.content}")

            if not resp.tool_calls:
                self._emit("  (no tool calls — sending nudge)")
                if nudged:
                    raise RuntimeError(
                        "LLM replied without a tool call twice in a row. Aborting."
                    )
                nudge_content = "You must call a tool. Review your task and continue, or call submit_results if done."
                messages.append({"role": "user", "content": nudge_content})
                nudged = True
                continue

            nudged = False

            for tc in resp.tool_calls:
                self._emit(f"  call: {tc.name}  {_summarize_args(tc.arguments)}")
                try:
                    result = self.registry.dispatch(tc.name, tc.arguments, self.ctx)
                except _Terminated as t:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"ok": True, "summary": t.summary}, default=str),
                    })
                    self.ctx.memory.set("run", "summary", t.summary)
                    self._emit(f"  result: {tc.name}  agent finished — {t.summary[:80]}")
                    return self.ctx

                self._emit(f"  result: {tc.name}  {_summarize_result(result)}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result.to_dict(), default=str),
                })

        raise RuntimeError(
            f"max_iterations={self.max_iterations} exhausted without a submit_results call. "
            "Partial results have been saved."
        )
