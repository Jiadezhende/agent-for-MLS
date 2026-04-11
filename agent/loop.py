"""
agent/loop.py — The main agent loop: LLM ↔ tool dispatch ↔ AgentContext.

One iteration = one LLM API call.  Within a single call the LLM may return
multiple parallel tool calls; we dispatch them sequentially.
"""
from __future__ import annotations

import json

from agent.prompts import SYSTEM_PROMPT, build_user_message
from agent.tool_registry import ToolRegistry, _Terminated
from agent.types import AgentContext
from llm.client import LLMClient


class AgentLoop:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        ctx: AgentContext,
        max_iterations: int = 40,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.ctx = ctx
        self.max_iterations = max_iterations

    def run(self) -> AgentContext:
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(self.ctx.task.payload)},
        ]
        nudged = False  # track whether we already sent the nudge message

        for i in range(self.max_iterations):
            self.ctx.iteration = i

            # --- LLM call ---------------------------------------------------
            try:
                resp = self.llm.chat(messages, tools=self.registry.schemas())
            except Exception as exc:
                self.ctx.events.append({
                    "type": "llm_error",
                    "detail": str(exc),
                    "iteration": i,
                })
                raise

            # Append assistant message (includes tool_calls for the API's next turn)
            messages.append(resp.to_openai_message())

            # Log reasoning regardless of outcome
            self.ctx.reasoning_log.append({
                "iteration": i,
                "content": resp.content,
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in resp.tool_calls
                ],
            })

            # --- No tool call -----------------------------------------------
            if not resp.tool_calls:
                if nudged:
                    raise RuntimeError(
                        "LLM replied without a tool call twice in a row. "
                        "Possible causes: model does not support tool_choice, "
                        "or context is too long. Aborting."
                    )
                messages.append({
                    "role": "user",
                    "content": (
                        "You must call a tool to proceed. "
                        "When all metrics are recorded, call submit_results."
                    ),
                })
                nudged = True
                continue

            nudged = False

            # --- Dispatch tool calls ----------------------------------------
            for tc in resp.tool_calls:
                try:
                    result = self.registry.dispatch(tc.name, tc.arguments, self.ctx)
                except _Terminated as t:
                    # submit_results was called — record the final tool-result
                    # message, store summary, and exit the loop.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(
                            {"ok": True, "summary": t.summary}, default=str
                        ),
                    })
                    self.ctx.memory.set("run", "summary", t.summary)
                    return self.ctx

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })

        raise RuntimeError(
            f"max_iterations={self.max_iterations} exhausted without "
            "a submit_results call. Partial results have been saved."
        )
