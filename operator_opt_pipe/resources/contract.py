"""OperatorContract — operator-agnostic schema (Python-defined, code-only).

The contract is the single source of truth for tensor shapes, dtypes,
reference formula, forward signature, and tolerances. Both the
deterministic resources (baseline / benchmark / evaluation) and the
agent prompts read it. Concrete contract instances live in
``operator_opt_pipe.operators.<name>`` modules; ``load_contract`` is a
dict lookup re-exported from ``operator_opt_pipe.operators``.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


_DEFAULT_RTOL = 1e-4
_DEFAULT_ATOL = 1e-4


# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorSpec:
    """Shape entries are str (variable name) or int (constant)."""

    name: str
    shape: tuple[Any, ...]
    dtype: str

    def __post_init__(self) -> None:
        for item in self.shape:
            if not isinstance(item, (str, int)):
                raise ValueError(
                    f"TensorSpec {self.name!r} shape entries must be str (variable) or int "
                    f"(constant); got {item!r}"
                )

    def render_shape(self) -> str:
        """``"d, d"`` / ``"d, 16"`` — for use as Python tuple args."""
        return ", ".join(str(s) for s in self.shape)

    def torch_dtype(self) -> str:
        return f"torch.{self.dtype}"


@dataclass(frozen=True)
class OperatorContract:
    """Frozen view of the operator schema.

    Driven entirely by code; everything below is operator-agnostic — the
    only operator-specific facts live in
    ``inputs / output / reference_pytorch / forward_args``.
    """

    name: str
    inputs: tuple[TensorSpec, ...]
    output: TensorSpec
    reference_pytorch: str
    forward_args: tuple[str, ...]
    shape_param: str
    shape_param_range: tuple[int, int]
    rtol: float = _DEFAULT_RTOL
    atol: float = _DEFAULT_ATOL

    def __post_init__(self) -> None:
        input_names = {t.name for t in self.inputs}
        unknown = [a for a in self.forward_args if a not in input_names]
        if unknown:
            raise ValueError(
                f"forward_args references undeclared inputs: {unknown}; "
                f"declared: {sorted(input_names)}"
            )
        lo, hi = self.shape_param_range
        if lo > hi:
            raise ValueError(f"shape_param_range min > max: {lo} > {hi}")

    # ------------------------------------------------------------------
    # Convenience derived properties
    # ------------------------------------------------------------------

    @property
    def dtype(self) -> str:
        return self.output.dtype

    @property
    def device(self) -> str:
        return "cuda"

    @property
    def shape_param_min(self) -> int:
        return self.shape_param_range[0]

    @property
    def shape_param_max(self) -> int:
        return self.shape_param_range[1]

    def default_shape_grid(self) -> tuple[int, ...]:
        """Three-point grid: lo, mid, hi. Min 3 points to spot d-sensitive bottlenecks."""
        lo = self.shape_param_min
        hi = self.shape_param_max
        mid = (lo + hi) // 2
        return (lo, mid, hi)

    def forward_signature_text(self) -> str:
        return f"forward({', '.join(self.forward_args)})"

    def summary_for_prompt(self) -> str:
        tensor_lines = [
            f"  {t.name}: shape=[{t.render_shape()}] dtype={t.dtype}"
            for t in self.inputs
        ]
        out = (
            f"  {self.output.name}: shape=[{self.output.render_shape()}] "
            f"dtype={self.output.dtype}"
        )
        lo, hi = self.shape_param_range
        return (
            f"=== Operator contract: {self.name} ===\n"
            f"Reference formula: {self.output.name} = {self.reference_pytorch}\n"
            f"Inputs:\n" + "\n".join(tensor_lines) + "\n"
            f"Output:\n{out}\n"
            f"Shape parameter: {self.shape_param} ∈ [{lo}, {hi}]\n"
            f"Tolerance: rtol={self.rtol}, atol={self.atol}\n"
            f"CUDA forward signature: {self.forward_signature_text()}"
        )


# ---------------------------------------------------------------------------
# Shape evaluation (controlled mini-DSL)
# ---------------------------------------------------------------------------


_ALLOWED_NODES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod,
    ast.UnaryOp, ast.USub,
)


def eval_shape(spec: Any, **vars_: int) -> int:
    """Resolve a shape entry: int constant, bare variable, or simple integer arithmetic.

    Function calls / attribute access / comparisons are rejected.
    """
    if isinstance(spec, int):
        return spec
    if isinstance(spec, str):
        if spec in vars_:
            return int(vars_[spec])
        try:
            tree = ast.parse(spec, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"cannot parse shape expression {spec!r}: {exc}") from exc
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise ValueError(
                    f"forbidden ast node in shape expression {spec!r}: "
                    f"{type(node).__name__}"
                )
            if isinstance(node, ast.Name) and node.id not in vars_:
                raise ValueError(
                    f"unknown variable {node.id!r} in shape expression {spec!r}; "
                    f"known: {sorted(vars_)}"
                )
        result = eval(  # noqa: S307 — controlled by node-type whitelist above
            compile(tree, "<shape>", "eval"), {"__builtins__": {}}, dict(vars_)
        )
        if not isinstance(result, int):
            raise ValueError(f"shape expression {spec!r} must evaluate to int, got {result!r}")
        return result
    raise ValueError(f"unsupported shape entry type {type(spec).__name__}: {spec!r}")


def shape_id(contract: OperatorContract, **shape_values: int) -> str:
    """Stable filename component for a particular shape configuration."""
    parts = []
    for var in (contract.shape_param,) + tuple(
        sorted(k for k in shape_values if k != contract.shape_param)
    ):
        if var in shape_values:
            parts.append(f"{var}{int(shape_values[var])}")
    return "_".join(parts)


__all__ = [
    "OperatorContract",
    "TensorSpec",
    "eval_shape",
    "shape_id",
]
