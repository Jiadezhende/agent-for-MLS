"""Plain GEMM smoke-test operator: ``Y = W @ X``."""
from __future__ import annotations

from typing import Any

import torch

from operator_opt_pipe.operators._base import OperatorOps
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


class PlainMatmulOps(OperatorOps):
    contract = CONTRACT

    def make_inputs(
        self, d: int, *, device: torch.device, generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        f32 = torch.float32
        return {
            "W": torch.randn((d, d), device=device, generator=generator, dtype=f32),
            "X": torch.randn((d, d), device=device, generator=generator, dtype=f32),
        }

    def reference(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return inputs["W"] @ inputs["X"]

    def forward_call(self, mod: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return mod.forward(inputs["W"], inputs["X"])

    def reference_doc(self) -> str:
        return "Y = W @ X   # plain GEMM"


OPS = PlainMatmulOps()
