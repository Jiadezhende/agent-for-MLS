# mls_agent

Standardized ReAct agent framework. Independent of `pipeline/` and the
old `agent/` module — designed to be imported and reused as-is.

## Layers

```
┌─────────────────────────────────────────────────────────────┐
│  runtime/   Agent · ReActLoop · AgentContext · Observer      │
└──────────────────┬─────────────────────────┬─────────────────┘
                   │                         │
        ┌──────────▼──────────┐  ┌───────────▼──────────────┐
        │  llm/                │  │  tools/                  │
        │  Message · Backend   │  │  Tool · Registry         │
        │  OpenAIBackend       │  │  ToolResponse · Breaker  │
        └──────────────────────┘  └──────────────────────────┘
```

* `llm/` — provider-agnostic `Message`/`ChatResponse` types and the
  `LLMBackend` protocol. Only `openai_backend.py` imports `openai`;
  every other module sees normalized types.
* `tools/` — `Tool` ABC, `ToolResponse` (with side-effect channels),
  `ToolRegistry`, and `CircuitBreaker`. Tools are **stateless w.r.t.
  `AgentContext`** but may carry their own external dependencies.
* `runtime/` — `Agent` entry point and `ReActLoop` with explicit
  `THOUGHT → VALIDATE → ACT → OBSERVE → APPLY → DECIDE` phases.

## Quick start

```python
from mls_agent import (
    Agent, AgentConfig, OpenAIBackend, LLMConfig,
    Tool, ToolParameter, ToolRegistry, ToolResponse, StdoutObserver,
)


class WeatherTool(Tool):
    NAME = "get_weather"
    DESCRIPTION = "Get the current weather for a city."

    def parameters_schema(self):
        return Tool.schema_from_parameters([
            ToolParameter(name="city", type="string", description="city name"),
        ])

    def run(self, parameters):
        return ToolResponse.success(text=f"sunny in {parameters['city']}")


class SubmitTool(Tool):
    NAME = "submit"
    DESCRIPTION = "Submit final answer."

    def parameters_schema(self):
        return Tool.schema_from_parameters([
            ToolParameter(name="answer", type="string", description="answer"),
        ])

    def run(self, parameters):
        return ToolResponse.terminate_with(
            summary="done", payload={"answer": parameters["answer"]},
        )


backend = OpenAIBackend(LLMConfig.from_env())
registry = ToolRegistry()
registry.register(WeatherTool())
registry.register(SubmitTool())

agent = Agent(
    backend=backend,
    registry=registry,
    system_prompt="You are a helpful assistant.",
    config=AgentConfig(max_iterations=10),
    observer=StdoutObserver(prefix="[demo] "),
)

result = agent.run("What's the weather in Tokyo?")
print(result.reason)    # "completed"
print(result.payload)   # {"answer": "..."}
```

## Tools with external dependencies

Tools that delegate to executors / HTTP clients / MCP servers are
indistinguishable from pure-function tools as far as the framework is
concerned — they just take their dependency in `__init__`:

```python
class CudaCompileTool(Tool):
    NAME = "cuda_compile"
    DESCRIPTION = "Compile a CUDA source via the executor."

    def __init__(self, executor):
        self._executor = executor

    def parameters_schema(self):
        return {"type": "object",
                "properties": {"source": {"type": "string"}},
                "required": ["source"]}

    def run(self, parameters):
        result = self._executor.compile(parameters["source"])
        return ToolResponse.success(text=str(result), data=result)
```

The framework never reaches into the tool's state. Concurrency control,
timeouts, caching, etc. are the executor's job.

## Termination

A tool requests termination by returning a `ToolResponse` with
`terminate=True`:

```python
return ToolResponse.terminate_with(
    summary="all done",
    payload={"score": 0.95, "trace": [...]},
)
```

The loop returns `AgentResult(reason="completed", payload=...)` to the
caller. Other termination reasons (`max_iterations`, `no_tool_call`,
`llm_error`) are produced by the loop itself.

## Side effects

Tools cannot reach `AgentContext` directly. To record events or
measurements, they include them in the `ToolResponse`:

```python
return ToolResponse.success(
    text="profiled",
    events=(Event(type="clock_locked", severity="warn", detail="..."),),
    measurements=(
        Measurement(metric="dram_bw", value=400, unit="GB/s",
                    confidence=0.9, method="bandwidthTest",
                    evidence=("stdout: 400 GB/s",)),
    ),
)
```

The loop applies these to `ctx.events` / `ctx.measurements` after
dispatch.

## Tests

```
pytest mls_agent/tests/         # 168 tests, ~0.1s
python -m mls_agent.examples.calculator_demo
```

The demo runs offline with a scripted backend; no API key needed.
