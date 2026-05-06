"""Standalone demo: a two-step calculator agent.

Demonstrates the full mls_agent framework with no network and no GPU:
a fake backend yields scripted tool calls, two tools (one pure-function
calculator, one stateful "ledger" with an injected dependency), and a
``submit`` tool that terminates with a structured payload.

Run::

    python -m mls_agent.examples.calculator_demo
"""
from __future__ import annotations

import sys

from mls_agent import (
    Agent,
    AgentConfig,
    ChatResponse,
    LLMBackend,
    Message,
    StdoutObserver,
    Tool,
    ToolCall,
    ToolParameter,
    ToolRegistry,
    ToolResponse,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class CalculatorTool(Tool):
    """Pure-function tool — no external dependencies."""

    NAME = "calc"
    DESCRIPTION = "Evaluate a basic arithmetic expression like '2 + 3 * 4'."

    def parameters_schema(self):
        return Tool.schema_from_parameters([
            ToolParameter(
                name="expression",
                type="string",
                description="Arithmetic expression using + - * / and parentheses.",
            ),
        ])

    def run(self, parameters):
        expr = parameters["expression"]
        # Restrict to safe characters before evaluating.
        if not all(c in "0123456789+-*/(). " for c in expr):
            return ToolResponse.error(
                code="invalid_expr",
                message=f"Disallowed characters in {expr!r}",
            )
        try:
            value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 (sanitized)
        except Exception as e:  # noqa: BLE001
            return ToolResponse.error(
                code="eval_failed",
                message=f"{type(e).__name__}: {e}",
            )
        return ToolResponse.success(text=str(value), data={"value": value})


class _Ledger:
    """An injected dependency (think Executor / DB / MCP handle)."""

    def __init__(self):
        self.entries: list[float] = []

    def add(self, value: float) -> int:
        self.entries.append(value)
        return len(self.entries)


class RecordTool(Tool):
    """Tool that holds an external dependency — the framework still
    treats it identically to a pure-function tool."""

    NAME = "record"
    DESCRIPTION = "Append a numeric value to the running ledger."

    def __init__(self, ledger: _Ledger):
        self._ledger = ledger

    def parameters_schema(self):
        return Tool.schema_from_parameters([
            ToolParameter(
                name="value",
                type="number",
                description="Value to append to the ledger.",
            ),
        ])

    def run(self, parameters):
        n = self._ledger.add(float(parameters["value"]))
        return ToolResponse.success(
            text=f"appended; ledger size now {n}",
            data={"size": n},
        )


class SubmitTool(Tool):
    """Termination tool — returns a payload that becomes AgentResult.payload."""

    NAME = "submit"
    DESCRIPTION = "Finalize and submit the agent's findings."

    def parameters_schema(self):
        return Tool.schema_from_parameters([
            ToolParameter(name="answer", type="number", description="final answer"),
            ToolParameter(name="summary", type="string", description="summary"),
        ])

    def run(self, parameters):
        return ToolResponse.terminate_with(
            summary=parameters["summary"],
            payload={"answer": parameters["answer"]},
        )


# ---------------------------------------------------------------------------
# Scripted backend (so the demo runs offline)
# ---------------------------------------------------------------------------


class ScriptedBackend(LLMBackend):
    def __init__(self, responses):
        self._responses = list(responses)

    def chat(self, messages, tools):
        if not self._responses:
            raise RuntimeError("scripted backend exhausted")
        return self._responses.pop(0)


def _call(call_id, name, arguments, content=None) -> ChatResponse:
    tc = ToolCall(id=call_id, name=name, arguments=arguments)
    return ChatResponse(
        message=Message.assistant(content=content, tool_calls=(tc,)),
        finish_reason="tool_calls",
    )


# ---------------------------------------------------------------------------
# Demo entry
# ---------------------------------------------------------------------------


def main() -> int:
    ledger = _Ledger()

    backend = ScriptedBackend([
        _call("c1", "calc", {"expression": "12 * 3 + 4"}, content="Let me compute."),
        _call("c2", "record", {"value": 40}, content="Record the result."),
        _call(
            "c3",
            "submit",
            {"answer": 40, "summary": "12 * 3 + 4 = 40; recorded."},
        ),
    ])

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(RecordTool(ledger))
    registry.register(SubmitTool())

    agent = Agent(
        backend=backend,
        registry=registry,
        system_prompt=(
            "You are an arithmetic agent. Use the calc tool, then record the "
            "result, then call submit when done."
        ),
        config=AgentConfig(max_iterations=10),
        observer=StdoutObserver(prefix="[demo] "),
    )

    result = agent.run("Compute 12 * 3 + 4 and record the answer.")

    print("=" * 60)
    print(f"reason     : {result.reason}")
    print(f"iterations : {result.iterations}")
    print(f"summary    : {result.summary}")
    print(f"payload    : {result.payload}")
    print(f"ledger     : {ledger.entries}")
    print("=" * 60)

    if result.reason != "completed" or result.payload != {"answer": 40}:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
