"""Plain GEMM smoke-test operator: ``Y = W @ X``."""
from __future__ import annotations

from operator_opt_pipe.resources.contract import OperatorContract, TensorSpec


CONTRACT = OperatorContract(
    name="operators/plain_matmul",
    inputs=(
        TensorSpec(name="W", shape=("d", "d"), dtype="float32"),
        TensorSpec(name="X", shape=("d", "d"), dtype="float32"),
    ),
    output=TensorSpec(name="Y", shape=("d", "d"), dtype="float32"),
    reference_pytorch="W @ X",
    forward_args=("W", "X"),
    shape_param="d",
    shape_param_range=(1024, 4096),
)
