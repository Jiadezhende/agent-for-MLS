"""Operator execution interface.

Each operator module exports a concrete ``OperatorOps`` subclass instance
as ``OPS``. The class binds the structural ``CONTRACT`` to the executable
behaviour (input synthesis, reference compute, candidate forward call).

Baseline / evaluation call directly into these methods instead of
generating subprocess scripts via string templates.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from operator_opt_pipe.resources.contract import OperatorContract


class OperatorOps(ABC):
    """Abstract base for per-operator execution."""

    contract: OperatorContract

    @property
    def name(self) -> str:
        return self.contract.name

    @property
    def short_name(self) -> str:
        return self.contract.name.split("/")[-1]

    # ------------------------------------------------------------------
    # Operator-specific behaviour — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def make_inputs(
        self, d: int, *, device: torch.device, generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        """Materialize all input tensors at shape parameter ``d``.

        Returns a dict keyed by tensor name (matching ``contract.inputs``).
        """

    @abstractmethod
    def reference(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the PyTorch reference output (oracle) for the given inputs."""

    @abstractmethod
    def forward_call(self, mod: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Invoke the candidate CUDA module with the right argument order."""

    def reference_doc(self) -> str:
        """One-line human-readable formula, used in LLM prompts. Override."""
        return f"{type(self).__name__}.reference"

    # ------------------------------------------------------------------
    # Operator-agnostic helpers — concrete defaults
    # ------------------------------------------------------------------

    def shape_id(self, d: int) -> str:
        """Stable filename component for a single-shape config."""
        return f"{self.contract.shape_param}{int(d)}"

    def save_inputs(
        self, inputs: dict[str, torch.Tensor], dir_: Path, shape_id: str,
    ) -> None:
        for spec in self.contract.inputs:
            t = inputs[spec.name]
            torch.save(t.detach().cpu(), Path(dir_) / f"{spec.name}_{shape_id}.pt")

    def load_inputs(
        self, dir_: Path, shape_id: str, *, device: torch.device,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for spec in self.contract.inputs:
            out[spec.name] = torch.load(
                Path(dir_) / f"{spec.name}_{shape_id}.pt", map_location=device,
            )
        return out

    def save_oracle(self, Y: torch.Tensor, dir_: Path, shape_id: str) -> None:
        out_name = self.contract.output.name
        torch.save(Y.detach().cpu(), Path(dir_) / f"{out_name}_{shape_id}.pt")

    def load_oracle(
        self, dir_: Path, shape_id: str, *, device: torch.device,
    ) -> torch.Tensor:
        out_name = self.contract.output.name
        return torch.load(
            Path(dir_) / f"{out_name}_{shape_id}.pt", map_location=device,
        )


__all__ = ["OperatorOps"]
