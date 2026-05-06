"""LoRA operator contract — derived from skills/operators/lora_matmul.md frontmatter.

Pure-Python module — no torch / CUDA dependency. The contract is the single
source of truth for tensor shapes, dtype, formula, and tolerances; both the
benchmark/baseline scripts (next-PR work) and the agent prompts read it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Reuse the operator-spec parser already shipped with the legacy pipeline. It
# lives in ``pipeline/operator_spec.py`` and has its own tests; importing it
# here keeps the YAML-frontmatter contract in one place. When the legacy
# pipeline is removed, this module will be moved into operator_opt_pipe/.
from pipeline.operator_spec import OperatorSpec


@dataclass(frozen=True)
class LoRAContract:
    """Frozen view of the LoRA operator contract.

    The tolerances are LoRA-specific and reflect float32 GEMM accumulation
    error at d ∈ [3584, 4608]; tighter bounds break under reordering. Phase-2
    evaluation uses these same defaults.
    """

    operator: str
    d_range: tuple[int, int]                  # inclusive (3584, 4608)
    r: int                                    # LoRA rank — 16
    dtype: str                                # "float32"
    device: str                               # "cuda"
    forward_args: tuple[str, ...]             # ("W", "X", "A", "B")
    reference_pytorch: str                    # "W @ X + A @ (B.T @ X)"
    output_name: str                          # "Y"
    tolerance_atol: float = 1e-2
    tolerance_rtol: float = 1e-2

    @property
    def shape_param_min(self) -> int:
        return self.d_range[0]

    @property
    def shape_param_max(self) -> int:
        return self.d_range[1]


def load_contract(skills_root: str | Path, *, operator: str = "lora_matmul") -> LoRAContract:
    """Construct a ``LoRAContract`` by reading skill markdown frontmatter."""
    op_spec = OperatorSpec.load_from_skill(skills_root, operator)

    # Detect the LoRA rank by inspecting input shapes: r is the constant
    # appearing alongside the variable shape param in the A/B inputs.
    rank = _infer_rank(op_spec)
    dtypes = {t.dtype for t in op_spec.inputs} | {op_spec.output.dtype}
    if len(dtypes) != 1:
        raise ValueError(
            f"contract requires a single dtype across inputs+output, got {sorted(dtypes)}"
        )

    return LoRAContract(
        operator=op_spec.name,
        d_range=op_spec.shape_param_range,
        r=rank,
        dtype=next(iter(dtypes)),
        device="cuda",
        forward_args=op_spec.forward_args,
        reference_pytorch=op_spec.reference_pytorch,
        output_name=op_spec.output.name,
    )


def _infer_rank(op_spec: OperatorSpec) -> int:
    """Find the constant integer appearing in any input alongside the shape param."""
    sp = op_spec.shape_param
    constants: set[int] = set()
    for tensor in op_spec.inputs:
        for entry in tensor.shape:
            if isinstance(entry, int):
                constants.add(entry)
            elif entry != sp:
                # Multi-variable shapes (e.g. attention) aren't supported in v1.
                raise ValueError(
                    f"contract loader only supports a single shape variable; "
                    f"tensor {tensor.name!r} has shape entry {entry!r}"
                )
    if not constants:
        raise ValueError(
            f"contract loader expected at least one constant integer in input shapes "
            f"to infer LoRA rank; got none in {op_spec.inputs}"
        )
    if len(constants) > 1:
        raise ValueError(
            f"contract loader expected exactly one rank constant, got {sorted(constants)}"
        )
    return next(iter(constants))
