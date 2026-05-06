"""Tool ABC and parameter helpers.

Subclasses declare ``NAME`` and ``DESCRIPTION`` as class attributes and
implement ``run`` + ``parameters_schema``. The default ``to_openai_schema``
assembles those into the OpenAI function-calling envelope; tools needing
``oneOf`` / ``nullable`` / ``minItems`` etc. simply return a complete
schema from ``parameters_schema``.

Tools are stateless w.r.t. AgentContext but may carry their own
constructor-injected dependencies (executors, HTTP clients, MCP handles).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from mls_agent.tools.response import ToolResponse


@dataclass(frozen=True)
class ToolParameter:
    """Lightweight parameter description.

    Used by tools whose schema is simple enough that the default
    ``to_openai_schema`` builder can derive it from a parameter list.
    For complex schemas, override ``parameters_schema`` directly.
    """

    name: str
    type: str  # "string" | "integer" | "number" | "boolean" | "array" | "object"
    description: str
    required: bool = True
    default: Any = None
    enum: tuple[Any, ...] | None = None
    items: dict[str, Any] | None = None  # for type="array"

    def __post_init__(self) -> None:
        valid_types = ("string", "integer", "number", "boolean", "array", "object")
        if self.type not in valid_types:
            raise ValueError(
                f"ToolParameter.type must be one of {valid_types}, got {self.type!r}"
            )
        if not self.name:
            raise ValueError("ToolParameter.name must be non-empty")


class Tool(ABC):
    """Base class for all tools.

    Subclasses MUST declare ``NAME`` and ``DESCRIPTION`` as class
    attributes and implement ``run`` and ``parameters_schema``.

    Stateless w.r.t. AgentContext: the framework will not pass a context
    object in. Side effects are returned through ``ToolResponse`` fields
    (events / measurements / terminate).

    Tool instances may carry their own dependencies::

        class CudaCompileTool(Tool):
            NAME = "cuda_compile"
            DESCRIPTION = "Compile a CUDA source via the executor."

            def __init__(self, executor: CudaExecutor):
                self._executor = executor

            def parameters_schema(self):
                return {
                    "type": "object",
                    "properties": {"source": {"type": "string"}},
                    "required": ["source"],
                }

            def run(self, parameters):
                result = self._executor.compile(parameters["source"])
                return ToolResponse.success(text=..., data=result)
    """

    NAME: ClassVar[str] = ""
    DESCRIPTION: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Allow private/intermediate ABCs to skip the check by starting with "_".
        if cls.__name__.startswith("_"):
            return
        # ABCMeta has not yet set ``__abstractmethods__`` when
        # ``__init_subclass__`` runs, so detect abstract members manually
        # by walking the inherited methods.
        for name in ("run", "parameters_schema"):
            attr = getattr(cls, name, None)
            if getattr(attr, "__isabstractmethod__", False):
                return  # still abstract; defer the NAME/DESCRIPTION check
        if not getattr(cls, "NAME", ""):
            raise TypeError(
                f"{cls.__name__} must define class attribute NAME (non-empty string)"
            )
        if not getattr(cls, "DESCRIPTION", ""):
            raise TypeError(
                f"{cls.__name__} must define class attribute DESCRIPTION "
                "(non-empty string)"
            )

    @abstractmethod
    def run(self, parameters: dict[str, Any]) -> ToolResponse:
        """Execute the tool. Must return a ToolResponse; may raise on internal bugs."""

    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """Return JSON Schema describing the ``parameters`` object."""

    # ------------------------------------------------------------------
    # Schema generation
    # ------------------------------------------------------------------

    def to_openai_schema(self) -> dict[str, Any]:
        """Wrap ``parameters_schema`` with the OpenAI function-calling envelope."""
        return {
            "type": "function",
            "function": {
                "name": self.NAME,
                "description": self.DESCRIPTION,
                "parameters": self.parameters_schema(),
            },
        }

    @staticmethod
    def schema_from_parameters(
        parameters: list[ToolParameter] | tuple[ToolParameter, ...],
    ) -> dict[str, Any]:
        """Helper for simple tools: derive a JSON Schema object from a parameter list."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.type == "array":
                prop["items"] = p.items or {"type": "string"}
            if p.enum is not None:
                prop["enum"] = list(p.enum)
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.NAME!r})"
