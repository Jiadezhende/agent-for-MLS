"""Operator contract registry.

Each module under this package defines exactly one ``OperatorContract``
instance and exports it as ``CONTRACT``. Adding a new operator =
new ``operator_opt_pipe/operators/<name>.py`` + new entry in ``OPERATORS``
below. The skill markdown under ``skills/operators/<name>.md`` only
holds tuning narrative (read by the LLM via ``read_skill``); it carries
no machine-readable contract.
"""
from __future__ import annotations

from operator_opt_pipe.operators import lora_matmul as _lora_matmul
from operator_opt_pipe.operators import plain_matmul as _plain_matmul
from operator_opt_pipe.resources.contract import OperatorContract


OPERATORS: dict[str, OperatorContract] = {
    _lora_matmul.CONTRACT.name.split("/")[-1]: _lora_matmul.CONTRACT,
    _plain_matmul.CONTRACT.name.split("/")[-1]: _plain_matmul.CONTRACT,
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


__all__ = ["OPERATORS", "load_contract"]
