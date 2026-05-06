"""Agent — top-level entry point that wires backend + registry + loop.

Typical use::

    from mls_agent.llm import LLMConfig
    from mls_agent.llm.openai_backend import OpenAIBackend
    from mls_agent.tools import ToolRegistry
    from mls_agent.runtime import Agent, AgentConfig, StdoutObserver

    backend = OpenAIBackend(LLMConfig.from_env())
    registry = ToolRegistry()
    registry.register(MyTool())

    agent = Agent(
        backend=backend,
        registry=registry,
        system_prompt="You are a helpful assistant.",
        config=AgentConfig(max_iterations=10),
        observer=StdoutObserver(prefix="[demo] "),
    )
    result = agent.run("What's the weather in Tokyo?")
    print(result.reason, result.summary, result.payload)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mls_agent.llm.backend import LLMBackend
from mls_agent.runtime.config import AgentConfig
from mls_agent.runtime.loop import ReActLoop
from mls_agent.runtime.observer import AgentObserver, NullObserver
from mls_agent.runtime.result import AgentResult
from mls_agent.tools.registry import ToolRegistry


@dataclass
class Agent:
    backend: LLMBackend
    registry: ToolRegistry
    system_prompt: str
    config: AgentConfig = field(default_factory=AgentConfig)
    observer: AgentObserver = field(default_factory=NullObserver)

    def run(self, user_message: str) -> AgentResult:
        loop = ReActLoop(
            backend=self.backend,
            registry=self.registry,
            system_prompt=self.system_prompt,
            config=self.config,
            observer=self.observer,
        )
        return loop.run(user_message)
