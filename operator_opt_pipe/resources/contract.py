"""OperatorContract — operator-agnostic schema parsed from skill frontmatter.

The contract is the single source of truth for tensor shapes, dtypes,
reference formula, forward signature, and tolerances. Both the deterministic
resources (baseline / benchmark / evaluation) and the agent prompts read
it. Adding a new operator means writing a ``skills/operators/<name>.md``
with the right frontmatter — no code changes required here.

Frontmatter schema (single shape variable + constants is supported; multi-
variable shapes like attention need a richer schema and are out of scope):

    name: operators/<name>
    shape_param: d
    shape_param_range: [min, max]
    inputs:
      - {name: W, shape: [d, d], dtype: float32}
      - ...
    output: {name: Y, shape: [d, d], dtype: float32}
    reference_pytorch: "W @ X + A @ (B.T @ X)"   # RHS only
    forward_args: [W, X, A, B]
    correctness: {rtol: 1.0e-4, atol: 1.0e-4}    # optional, defaults shown
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_FRONTMATTER_FENCE = "---"

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

    @classmethod
    def from_dict(cls, raw: dict) -> "TensorSpec":
        try:
            name = str(raw["name"])
            shape = tuple(raw["shape"])
            dtype = str(raw["dtype"])
        except KeyError as e:
            raise ValueError(f"TensorSpec missing field {e}; got {raw!r}") from None
        for item in shape:
            if not isinstance(item, (str, int)):
                raise ValueError(
                    f"TensorSpec '{name}' shape entries must be str (variable) or int "
                    f"(constant); got {item!r}"
                )
        return cls(name=name, shape=shape, dtype=dtype)

    def render_shape(self) -> str:
        """``"d, d"`` / ``"d, 16"`` — for use as Python tuple args."""
        return ", ".join(str(s) for s in self.shape)

    def torch_dtype(self) -> str:
        return f"torch.{self.dtype}"


@dataclass(frozen=True)
class OperatorContract:
    """Frozen view of the operator schema.

    Driven entirely by the skill markdown's frontmatter. Everything below
    is operator-agnostic — the only operator-specific facts live in
    ``inputs / output / reference_pytorch / forward_args``.
    """

    name: str                                      # "operators/lora_matmul"
    inputs: tuple[TensorSpec, ...]                 # length == len(forward_args)
    output: TensorSpec
    reference_pytorch: str                         # RHS expression only
    forward_args: tuple[str, ...]                  # mod.forward(*forward_args)
    shape_param: str                               # "d"
    shape_param_range: tuple[int, int]             # (3584, 4608)
    rtol: float = _DEFAULT_RTOL
    atol: float = _DEFAULT_ATOL

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
        """Three-point grid covering both ends of the range plus the midpoint.

        Used by orchestrator + benchmark when the caller has not supplied
        a custom grid. Three points is the minimum that lets us notice
        d-sensitive bottlenecks (low-rank correction is dominant for small d,
        compute-bound matmul for large d).
        """
        lo = self.shape_param_min
        hi = self.shape_param_max
        mid = (lo + hi) // 2
        return (lo, mid, hi)

    # ------------------------------------------------------------------
    # Source-rendering helpers (used by baseline + evaluation subprocess
    # script generators). All take a ``shape_var`` (e.g. "d") and a
    # ``device_var`` so the same helpers work in both single-d and
    # multi-d loops.
    # ------------------------------------------------------------------

    def render_input_creation(self, *, device_var: str = "device") -> str:
        """``W = torch.randn(d, d, device=device, dtype=torch.float32)`` ..."""
        lines = []
        for t in self.inputs:
            lines.append(
                f"{t.name} = torch.randn({t.render_shape()}, "
                f"device={device_var}, dtype={t.torch_dtype()})"
            )
        return "\n".join(lines)

    def render_save_inputs(self, *, dir_var: str, shape_id_expr: str) -> str:
        """``torch.save(W.cpu(), os.path.join(INPUT_DIR, f"W_{shape_id}.pt"))`` ..."""
        lines = []
        for t in self.inputs:
            lines.append(
                f'torch.save({t.name}.cpu(), os.path.join({dir_var}, '
                f'f"{t.name}_{{{shape_id_expr}}}.pt"))'
            )
        return "\n".join(lines)

    def render_load_inputs(self, *, dir_var: str, shape_id_expr: str,
                           device_var: str = "device") -> str:
        lines = []
        for t in self.inputs:
            lines.append(
                f'{t.name} = torch.load(os.path.join({dir_var}, '
                f'f"{t.name}_{{{shape_id_expr}}}.pt"), map_location={device_var})'
            )
        return "\n".join(lines)

    def render_reference_compute(self) -> str:
        """``Y = <reference_pytorch>``."""
        return f"{self.output.name} = {self.reference_pytorch}"

    def render_save_reference(self, *, dir_var: str, shape_id_expr: str) -> str:
        return (
            f'torch.save({self.output.name}.cpu(), '
            f'os.path.join({dir_var}, f"{self.output.name}_{{{shape_id_expr}}}.pt"))'
        )

    def render_load_reference(self, *, dir_var: str, shape_id_expr: str,
                              var_name: str = "Y_ref",
                              device_var: str = "device") -> str:
        return (
            f'{var_name} = torch.load(os.path.join({dir_var}, '
            f'f"{self.output.name}_{{{shape_id_expr}}}.pt"), map_location={device_var})'
        )

    def render_forward_call(self, mod_var: str = "mod") -> str:
        """``mod.forward(W, X, A, B)`` — argument order from forward_args."""
        return f"{mod_var}.forward({', '.join(self.forward_args)})"

    def forward_signature_text(self) -> str:
        """``forward(W, X, A, B)`` — for prompts."""
        return f"forward({', '.join(self.forward_args)})"

    def summary_for_prompt(self) -> str:
        """Compact text block injected into agent user_message at runtime."""
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
# Loading
# ---------------------------------------------------------------------------


def load_contract(skills_root: str | Path, *, operator: str = "lora_matmul") -> OperatorContract:
    """Read ``skills/operators/<operator>.md`` and parse its frontmatter."""
    path = Path(skills_root) / "operators" / f"{operator}.md"
    if not path.is_file():
        raise FileNotFoundError(
            f"operator skill not found: {path} — create it under "
            f"skills/operators/ with the required frontmatter (inputs, output, "
            f"reference_pytorch, forward_args, shape_param, shape_param_range)."
        )
    text = path.read_text(encoding="utf-8")
    fm = _extract_frontmatter(text)
    if fm is None:
        raise ValueError(f"skill {path} has no YAML frontmatter")
    return _from_frontmatter(operator, fm)


def _from_frontmatter(operator: str, fm: dict) -> OperatorContract:
    required = ("inputs", "output", "reference_pytorch", "forward_args",
                "shape_param", "shape_param_range")
    missing = [k for k in required if k not in fm]
    if missing:
        raise ValueError(
            f"operator {operator!r} frontmatter missing fields: {missing}"
        )

    inputs = tuple(TensorSpec.from_dict(d) for d in fm["inputs"])
    output = TensorSpec.from_dict(fm["output"])
    forward_args = tuple(str(a) for a in fm["forward_args"])

    input_names = {t.name for t in inputs}
    unknown = [a for a in forward_args if a not in input_names]
    if unknown:
        raise ValueError(
            f"operator {operator!r} forward_args references undeclared inputs: {unknown}; "
            f"declared inputs: {sorted(input_names)}"
        )

    rng = fm["shape_param_range"]
    if not (isinstance(rng, list) and len(rng) == 2):
        raise ValueError(f"operator {operator!r} shape_param_range must be [min, max]")
    lo, hi = int(rng[0]), int(rng[1])
    if lo > hi:
        raise ValueError(
            f"operator {operator!r} shape_param_range min > max: {lo} > {hi}"
        )

    correctness = fm.get("correctness") or {}
    rtol = float(correctness.get("rtol", _DEFAULT_RTOL))
    atol = float(correctness.get("atol", _DEFAULT_ATOL))

    return OperatorContract(
        name=str(fm.get("name", operator)),
        inputs=inputs,
        output=output,
        reference_pytorch=str(fm["reference_pytorch"]),
        forward_args=forward_args,
        shape_param=str(fm["shape_param"]),
        shape_param_range=(lo, hi),
        rtol=rtol,
        atol=atol,
    )


def _extract_frontmatter(text: str) -> dict | None:
    if not text.startswith(_FRONTMATTER_FENCE):
        return None
    end = text.find("\n" + _FRONTMATTER_FENCE, len(_FRONTMATTER_FENCE))
    if end == -1:
        return None
    block = text[len(_FRONTMATTER_FENCE):end]
    parsed = yaml.safe_load(block)
    if not isinstance(parsed, dict):
        return None
    return parsed


# ---------------------------------------------------------------------------
# Shape evaluation (controlled mini-DSL)
# ---------------------------------------------------------------------------


_ALLOWED_NODES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod,
    ast.UnaryOp, ast.USub,
)


def eval_shape(spec: Any, **vars_: int) -> int:
    """Resolve a single shape entry: int constant or named variable expression.

    ``spec`` is one entry from a TensorSpec.shape — usually just a literal
    int or a single variable name like ``"d"``. We also tolerate small
    arithmetic expressions like ``"d * 2"`` or ``"d + 16"`` because they
    cost nothing to support and make multi-block tensors expressible.

    Only integer arithmetic is allowed; function calls / attribute access /
    comparisons are rejected.
    """
    if isinstance(spec, int):
        return spec
    if isinstance(spec, str):
        # Cheap path for the common case: bare variable name.
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
    """Stable filename component for a particular shape configuration.

    For LoRA's single-variable schema this is just ``"d3584"``. For a future
    two-variable algo it would be ``"d3584_h64"``.
    """
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
    "load_contract",
    "shape_id",
]
