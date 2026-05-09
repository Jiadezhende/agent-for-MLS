"""Operator registry.

Each module under this package defines:

* ``CONTRACT`` — frozen ``OperatorContract`` describing shapes / dtypes /
  forward signature / tolerances.
* ``OPS`` — concrete ``OperatorOps`` instance carrying the executable
  behaviour (``make_inputs`` / ``reference`` / ``forward_call``).

Adding a new operator: write ``operator_opt_pipe/operators/<name>.py``
exporting both, then register it in ``OPERATORS`` / ``OPS_REGISTRY``
below. The skill markdown under ``skills/operators/<name>.md`` only
holds tuning narrative (read by the LLM via ``read_skill``); it carries
no machine-readable contract.
"""
from __future__ import annotations

from operator_opt_pipe.operators import lora_matmul as _lora_matmul
from operator_opt_pipe.operators import plain_matmul as _plain_matmul
from operator_opt_pipe.operators._base import OperatorOps
from operator_opt_pipe.resources.contract import OperatorContract


OPERATORS: dict[str, OperatorContract] = {
    _lora_matmul.CONTRACT.name.split("/")[-1]: _lora_matmul.CONTRACT,
    _plain_matmul.CONTRACT.name.split("/")[-1]: _plain_matmul.CONTRACT,
}


OPS_REGISTRY: dict[str, OperatorOps] = {
    _lora_matmul.OPS.short_name: _lora_matmul.OPS,
    _plain_matmul.OPS.short_name: _plain_matmul.OPS,
}


def load_contract(operator: str) -> OperatorContract:
    """Look up an ``OperatorContract`` by short name (e.g. ``"lora_matmul"``)."""
    try:
        return OPERATORS[operator]
    except KeyError:
        known = sorted(OPERATORS)
        raise FileNotFoundError(
            f"unknown operator {operator!r}; known: {known}. "
            f"Add a module under operator_opt_pipe/operators/ to register a new one."
        ) from None


def load_ops(operator: str) -> OperatorOps:
    """Look up an ``OperatorOps`` instance by short name."""
    try:
        return OPS_REGISTRY[operator]
    except KeyError:
        known = sorted(OPS_REGISTRY)
        raise FileNotFoundError(
            f"unknown operator {operator!r}; known: {known}. "
            f"Add a module under operator_opt_pipe/operators/ to register a new one."
        ) from None


__all__ = [
    "OPERATORS",
    "OPS_REGISTRY",
    "OperatorOps",
    "load_contract",
    "load_ops",
]
