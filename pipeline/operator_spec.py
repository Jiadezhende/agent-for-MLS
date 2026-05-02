"""pipeline/operator_spec.py — machine-readable operator schema.

The pipeline state machine is operator-agnostic, but the runtime templates that
generate baseline / candidate evaluation Python subprocess scripts used to
hard-code LoRA-specific tensors and formulas. This module abstracts those
templates by parsing a structured schema from the operator skill markdown's
YAML frontmatter.

Each ``skills/operators/<name>.md`` file's frontmatter must contain:

    inputs:           list of {name, shape, dtype} dicts
    output:           single {name, shape, dtype} dict
    reference_pytorch: Python expression for the reference computation (RHS of Y = ...)
    forward_args:     list of input tensor names defining mod.forward(...) call order
    shape_param:      string name of the variable shape dimension (e.g. "d")
    shape_param_range: [min, max] inclusive bounds for the shape parameter

The shape model is "single variable + constants" — each tensor's shape is a
list of either the shape_param (str) or fixed integers. This covers LoRA,
plain GEMM, LayerNorm, and most element-wise / fusion operators. Multi-variable
shapes (attention, conv) are out of scope and would need a richer schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_FRONTMATTER_FENCE = "---"


@dataclass(frozen=True)
class TensorSpec:
    """Shape uses str entries (variable, e.g. ``"d"``) and int entries (constant, e.g. ``16``)."""
    name: str
    shape: tuple[Any, ...]   # tuple of str | int
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
        """Render shape tuple as Python-source comma-separated args (e.g. ``"d, d"``, ``"d, 16"``)."""
        return ", ".join(str(s) for s in self.shape)

    def torch_dtype(self) -> str:
        """Map dtype string to ``torch.<dtype>`` source token."""
        return f"torch.{self.dtype}"


@dataclass(frozen=True)
class OperatorSpec:
    """Machine-readable operator schema parsed from skills/operators/<name>.md frontmatter."""
    name: str
    inputs: tuple[TensorSpec, ...]
    output: TensorSpec
    reference_pytorch: str
    forward_args: tuple[str, ...]
    shape_param: str
    shape_param_range: tuple[int, int]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load_from_skill(cls, skills_root: str | Path, operator: str) -> "OperatorSpec":
        """Load ``skills/operators/<operator>.md`` and parse its frontmatter."""
        path = Path(skills_root) / "operators" / f"{operator}.md"
        if not path.is_file():
            raise FileNotFoundError(
                f"operator skill not found: {path} — create it under skills/operators/ "
                "with the required frontmatter (inputs, output, reference_pytorch, "
                "forward_args, shape_param, shape_param_range)"
            )
        text = path.read_text(encoding="utf-8")
        fm = _extract_frontmatter(text)
        if fm is None:
            raise ValueError(f"skill {path} has no YAML frontmatter")
        return cls.from_frontmatter(operator, fm)

    @classmethod
    def from_frontmatter(cls, operator: str, fm: dict) -> "OperatorSpec":
        required = ("inputs", "output", "reference_pytorch", "forward_args",
                    "shape_param", "shape_param_range")
        missing = [k for k in required if k not in fm]
        if missing:
            raise ValueError(f"operator '{operator}' frontmatter missing fields: {missing}")

        inputs = tuple(TensorSpec.from_dict(d) for d in fm["inputs"])
        output = TensorSpec.from_dict(fm["output"])
        forward_args = tuple(str(a) for a in fm["forward_args"])

        # Validate forward_args references known input names.
        input_names = {t.name for t in inputs}
        unknown = [a for a in forward_args if a not in input_names]
        if unknown:
            raise ValueError(
                f"operator '{operator}' forward_args references undeclared inputs: {unknown}; "
                f"declared inputs: {sorted(input_names)}"
            )

        rng = fm["shape_param_range"]
        if not (isinstance(rng, list) and len(rng) == 2):
            raise ValueError(f"operator '{operator}' shape_param_range must be [min, max]")
        lo, hi = int(rng[0]), int(rng[1])
        if lo > hi:
            raise ValueError(f"operator '{operator}' shape_param_range min > max: {lo} > {hi}")

        return cls(
            name=operator,
            inputs=inputs,
            output=output,
            reference_pytorch=str(fm["reference_pytorch"]),
            forward_args=forward_args,
            shape_param=str(fm["shape_param"]),
            shape_param_range=(lo, hi),
        )

    # ------------------------------------------------------------------
    # Source-rendering helpers used by baseline / candidate script templates
    # ------------------------------------------------------------------

    def render_input_creation(self, *, device_var: str = "device") -> str:
        """Lines like ``W = torch.randn(d, d, device=device, dtype=torch.float32)``."""
        lines = []
        for t in self.inputs:
            lines.append(
                f"{t.name} = torch.randn({t.render_shape()}, "
                f"device={device_var}, dtype={t.torch_dtype()})"
            )
        return "\n".join(lines)

    def render_save_inputs(self, *, dir_var: str, d_var: str) -> str:
        """Lines like ``torch.save(W.cpu(), os.path.join(INPUT_DIR, f"W_d{d}.pt"))``."""
        lines = []
        for t in self.inputs:
            lines.append(
                f'torch.save({t.name}.cpu(), os.path.join({dir_var}, f"{t.name}_d{{{d_var}}}.pt"))'
            )
        return "\n".join(lines)

    def render_load_inputs(self, *, dir_var: str, d_var: str, device_var: str = "device") -> str:
        """Lines like ``W = torch.load(os.path.join(INPUT_DIR, f"W_d{d}.pt"), map_location=device)``."""
        lines = []
        for t in self.inputs:
            lines.append(
                f'{t.name} = torch.load(os.path.join({dir_var}, f"{t.name}_d{{{d_var}}}.pt"), '
                f"map_location={device_var})"
            )
        return "\n".join(lines)

    def render_reference_compute(self) -> str:
        """``Y = <reference_pytorch>``."""
        return f"{self.output.name} = {self.reference_pytorch}"

    def render_save_reference(self, *, dir_var: str, d_var: str) -> str:
        return (
            f'torch.save({self.output.name}.cpu(), '
            f'os.path.join({dir_var}, f"{self.output.name}_d{{{d_var}}}.pt"))'
        )

    def render_load_reference(self, *, dir_var: str, d_var: str, var_name: str = "Y_ref",
                              device_var: str = "device") -> str:
        return (
            f'{var_name} = torch.load(os.path.join({dir_var}, '
            f'f"{self.output.name}_d{{{d_var}}}.pt"), map_location={device_var})'
        )

    def render_forward_call(self, mod_var: str = "mod") -> str:
        """``mod.forward(W, X, A, B)`` — with the user-defined argument order."""
        return f"{mod_var}.forward({', '.join(self.forward_args)})"

    def forward_signature_text(self) -> str:
        """Human-readable forward signature for prompts: ``forward(W, X, A, B)``."""
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
            f"CUDA forward signature: {self.forward_signature_text()}"
        )


# ---------------------------------------------------------------------------
# Frontmatter extraction
# ---------------------------------------------------------------------------

def _extract_frontmatter(text: str) -> dict | None:
    """Parse a ``---\\n<YAML>\\n---`` leading block; return the parsed dict (or None)."""
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
