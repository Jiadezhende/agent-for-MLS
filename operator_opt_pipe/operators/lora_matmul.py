"""LoRA-fused MATMUL contract: ``Y = W @ X + A @ (B^T @ X)``."""
from __future__ import annotations

from operator_opt_pipe.resources.contract import OperatorContract, TensorSpec


CONTRACT = OperatorContract(
    name="operators/lora_matmul",
    inputs=(
        TensorSpec(name="W", shape=("d", "d"), dtype="float32"),
        TensorSpec(name="X", shape=("d", "d"), dtype="float32"),
        TensorSpec(name="A", shape=("d", 16), dtype="float32"),
        TensorSpec(name="B", shape=("d", 16), dtype="float32"),
    ),
    output=TensorSpec(name="Y", shape=("d", "d"), dtype="float32"),
    reference_pytorch="W @ X + A @ (B.transpose(0, 1).contiguous() @ X)",
    forward_args=("W", "X", "A", "B"),
    shape_param="d",
    shape_param_range=(3584, 4608),
    rtol=1e-4,
    atol=1e-4,
)
