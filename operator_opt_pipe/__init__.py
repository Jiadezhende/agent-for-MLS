"""operator_opt_pipe — autonomous CUDA-operator optimization pipeline.

The pipeline runs a fixed state machine over an operator contract (currently
LoRA-fused matmul) and produces ``./optimized_lora.cu`` for the Phase-2
evaluation harness. The framework is built directly on top of ``mls_agent``;
LLM agents propose / repair candidates and diagnose bottlenecks, while
deterministic Python (``lora_resources/`` + ``RoundRunner``) owns the
benchmark, baseline, evaluation, and best-promotion path.
"""

from operator_opt_pipe.state import (
    ROUND_STEP_ANALYZE,
    ROUND_STEP_OPTIMIZE,
    RunLayout,
    RunState,
    SCHEMA_VERSION,
    Stage,
    check_submit_payload,
    load_blackboard,
    save_blackboard,
)
from operator_opt_pipe.transitions import MIN_TUNING_SLICE_S, next_stage

__all__ = [
    "MIN_TUNING_SLICE_S",
    "ROUND_STEP_ANALYZE",
    "ROUND_STEP_OPTIMIZE",
    "RunLayout",
    "RunState",
    "SCHEMA_VERSION",
    "Stage",
    "check_submit_payload",
    "load_blackboard",
    "next_stage",
    "save_blackboard",
]
