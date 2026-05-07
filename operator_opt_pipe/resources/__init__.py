"""Operator-agnostic deterministic resources.

These modules are NOT exposed as LLM tools — they are imported directly by
the orchestrator. The split marks the operator-coupling boundary: every
function takes an ``OperatorContract`` and reads ``contract.inputs /
output / reference_pytorch / forward_args`` to drive its behavior. Adding
a new operator means writing a single ``skills/operators/<name>.md`` —
no Python changes here.
"""

from operator_opt_pipe.resources.contract import (
    OperatorContract,
    TensorSpec,
    eval_shape,
    load_contract,
    shape_id,
)

__all__ = [
    "OperatorContract",
    "TensorSpec",
    "eval_shape",
    "load_contract",
    "shape_id",
]
