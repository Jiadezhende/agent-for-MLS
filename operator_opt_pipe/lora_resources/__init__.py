"""LoRA-specific deterministic resources.

These modules are NOT exposed to the LLM — they are imported directly by the
orchestrator and ``RoundRunner``. Splitting them into a sub-package marks the
operator-coupling boundary: when a second operator lands, this is the only
sub-tree that needs to be parameterized or duplicated.
"""

from operator_opt_pipe.lora_resources.contract import LoRAContract, load_contract

__all__ = ["LoRAContract", "load_contract"]
