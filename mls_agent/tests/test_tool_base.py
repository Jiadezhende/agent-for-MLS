"""Unit tests for mls_agent.tools.base."""
from __future__ import annotations

import pytest

from mls_agent.tools.base import Tool, ToolParameter
from mls_agent.tools.response import ToolResponse


# ---------------------------------------------------------------------------
# ToolParameter
# ---------------------------------------------------------------------------


class TestToolParameter:
    def test_basic(self):
        p = ToolParameter(name="x", type="string", description="d")
        assert p.name == "x"
        assert p.required is True

    def test_invalid_type_rejected(self):
        with pytest.raises(ValueError, match="type"):
            ToolParameter(name="x", type="banana", description="d")

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            ToolParameter(name="", type="string", description="d")


# ---------------------------------------------------------------------------
# Tool subclass declaration enforcement
# ---------------------------------------------------------------------------


class TestToolDeclaration:
    def test_subclass_without_name_rejected(self):
        with pytest.raises(TypeError, match="NAME"):
            class Bad(Tool):
                DESCRIPTION = "x"

                def run(self, parameters):
                    return ToolResponse.success("ok")

                def parameters_schema(self):
                    return {"type": "object"}

    def test_subclass_without_description_rejected(self):
        with pytest.raises(TypeError, match="DESCRIPTION"):
            class Bad(Tool):
                NAME = "x"

                def run(self, parameters):
                    return ToolResponse.success("ok")

                def parameters_schema(self):
                    return {"type": "object"}

    def test_proper_subclass_works(self):
        class Good(Tool):
            NAME = "good"
            DESCRIPTION = "good tool"

            def run(self, parameters):
                return ToolResponse.success("ok")

            def parameters_schema(self):
                return {"type": "object"}

        t = Good()
        assert t.NAME == "good"
        assert isinstance(t.run({}), ToolResponse)

    def test_intermediate_abstract_subclass_skips_check(self):
        # An intermediate ABC that doesn't implement the abstract methods
        # is itself still abstract — the check should not fire on it.
        class IntermediateABC(Tool):
            pass

        # Concrete subclass without NAME should still error.
        with pytest.raises(TypeError, match="NAME"):
            class StillBad(IntermediateABC):
                DESCRIPTION = "x"

                def run(self, parameters):
                    return ToolResponse.success("ok")

                def parameters_schema(self):
                    return {"type": "object"}

    def test_underscore_prefixed_subclass_skips_check(self):
        # Convention: leading-underscore subclass names skip the strict check.
        class _Helper(Tool):
            def run(self, parameters):
                return ToolResponse.success("ok")

            def parameters_schema(self):
                return {"type": "object"}


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------


class _Echo(Tool):
    NAME = "echo"
    DESCRIPTION = "Echo input back to the caller."

    def parameters_schema(self):
        return Tool.schema_from_parameters(
            [ToolParameter(name="text", type="string", description="text to echo")]
        )

    def run(self, parameters):
        return ToolResponse.success(parameters["text"])


class TestSchema:
    def test_to_openai_schema_envelope(self):
        s = _Echo().to_openai_schema()
        assert s["type"] == "function"
        assert s["function"]["name"] == "echo"
        assert s["function"]["description"] == "Echo input back to the caller."
        assert s["function"]["parameters"]["type"] == "object"
        assert "text" in s["function"]["parameters"]["properties"]

    def test_schema_from_parameters_required_list(self):
        schema = Tool.schema_from_parameters([
            ToolParameter(name="x", type="string", description="d", required=True),
            ToolParameter(name="y", type="integer", description="d", required=False),
        ])
        assert schema["required"] == ["x"]
        assert "x" in schema["properties"]
        assert "y" in schema["properties"]

    def test_schema_from_parameters_array_default_items(self):
        schema = Tool.schema_from_parameters([
            ToolParameter(name="xs", type="array", description="list of strings"),
        ])
        assert schema["properties"]["xs"]["items"] == {"type": "string"}

    def test_schema_from_parameters_array_custom_items(self):
        schema = Tool.schema_from_parameters([
            ToolParameter(
                name="xs",
                type="array",
                description="list of ints",
                items={"type": "integer"},
            ),
        ])
        assert schema["properties"]["xs"]["items"] == {"type": "integer"}

    def test_schema_from_parameters_enum(self):
        schema = Tool.schema_from_parameters([
            ToolParameter(
                name="severity",
                type="string",
                description="severity",
                enum=("info", "warn", "error"),
            ),
        ])
        assert schema["properties"]["severity"]["enum"] == ["info", "warn", "error"]


class TestRunInjection:
    """An injected dependency should be reachable from run()."""

    def test_dependency_injection(self):
        class Dep:
            def __init__(self):
                self.calls = 0

            def do(self):
                self.calls += 1
                return "result"

        class WithDep(Tool):
            NAME = "with_dep"
            DESCRIPTION = "demo"

            def __init__(self, dep: Dep):
                self._dep = dep

            def parameters_schema(self):
                return {"type": "object"}

            def run(self, parameters):
                value = self._dep.do()
                return ToolResponse.success(value)

        dep = Dep()
        t = WithDep(dep)
        r = t.run({})
        assert r.text == "result"
        assert dep.calls == 1
