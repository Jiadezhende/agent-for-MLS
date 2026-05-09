"""LoRA-fused MATMUL contract: ``Y = W @ X + A @ (B^T @ X)``."""
from __future__ import annotations

from typing import Any

import torch

from operator_opt_pipe.operators._base import OperatorOps
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


class LoraMatmulOps(OperatorOps):
    contract = CONTRACT

    def make_inputs(
        self, d: int, *, device: torch.device, generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        f32 = torch.float32
        return {
            "W": torch.randn((d, d), device=device, generator=generator, dtype=f32),
            "X": torch.randn((d, d), device=device, generator=generator, dtype=f32),
            "A": torch.randn((d, 16), device=device, generator=generator, dtype=f32),
            "B": torch.randn((d, 16), device=device, generator=generator, dtype=f32),
        }

    def reference(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        W, X, A, B = inputs["W"], inputs["X"], inputs["A"], inputs["B"]
        return W @ X + A @ (B.transpose(0, 1).contiguous() @ X)

    def forward_call(self, mod: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return mod.forward(inputs["W"], inputs["X"], inputs["A"], inputs["B"])

    def reference_doc(self) -> str:
        return "Y = W @ X + A @ (B^T @ X)   # LoRA-fused matmul, r=16"


OPS = LoraMatmulOps()
