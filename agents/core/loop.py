"""
agents/core/loop.py — The main agent loop: LLM ↔ tool dispatch ↔ AgentContext.

One iteration = one LLM API call. Within a single call the LLM may return
multiple parallel tool calls; we dispatch them sequentially.
"""
from __future__ import annotations

import json
import sys

from agents.core.llm import LLMClient
from agents.core.prompts import build_user_message
from agents.core.types import AgentContext
from agents.tools.registry import ToolRegistry, _Terminated


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


def _summarize_result(result: dict) -> str:
    status = result.get("status")
    if status == "error":
        err = result.get("error", "?")
        detail = result.get("detail") or result.get("stderr", "")
        snippet = str(detail)[:80] if detail else ""
        return f"status=error  error={err}  {snippet}"
    if status == "circuit_open":
        return f"status=circuit_open  kinds={result.get('open_error_kinds')}"
    if status in ("done", "timed_out"):
        elapsed = result.get("elapsed_s", "?")
        cache_tag = " [cache_hit]" if result.get("cache_hit") else ""
        return f"status={status}  elapsed={elapsed}s{cache_tag}"
    if result.get("ok") is True:
        extra = {k: v for k, v in result.items() if k != "ok"}
        return "ok  " + json.dumps(extra, separators=(",", ":"), default=str)
    if "error" in result:
        return f"error={result['error']}  " + str(result.get("detail", ""))[:80]
    raw = json.dumps(result, separators=(",", ":"), default=str)
    return (raw[:100] + "...") if len(raw) > 100 else raw


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
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.ctx = ctx
        self.max_iterations = max_iterations
        self.verbose = verbose
        self._prefix = f"[W{worker_id}] " if worker_id is not None else ""
        self._system_prompt = system_prompt or ""

    def _emit(self, *args: object) -> None:
        if self.verbose:
            if args and isinstance(args[0], str):
                print(self._prefix + args[0], *args[1:], file=sys.stderr, flush=True)
            else:
                print(self._prefix, *args, file=sys.stderr, flush=True)

    def _missing_targets(self) -> list[str]:
        """Return targets not yet recorded in ctx.results."""
        recorded = {r.metric for r in self.ctx.results}
        return [t for t in self.ctx.task.payload.get("targets", []) if t not in recorded]

    def run(self) -> AgentContext:
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": build_user_message(self.ctx.task.payload)},
        ]
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
                missing = self._missing_targets()
                if missing:
                    nudge_content = (
                        f"You must call a tool. Still unrecorded: {missing}. "
                        "Call record_measurement for each measured target, "
                        "then call submit_results when all are done."
                    )
                else:
                    nudge_content = "All targets recorded. Call submit_results now."
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
                    "content": json.dumps(result, default=str),
                })

        raise RuntimeError(
            f"max_iterations={self.max_iterations} exhausted without a submit_results call. "
            "Partial results have been saved."
        )
