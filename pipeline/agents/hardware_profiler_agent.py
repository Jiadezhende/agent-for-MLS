"""pipeline/agents/hardware_profiler_agent.py — HARDWARE_PROFILE stage agent."""
from __future__ import annotations

import json

from ..stage_runner import StageContext
from ..state import Stage
from ._base import LLMStageAgent


_SYSTEM_PROMPT = """You are the HardwareProfilerAgent. Your job is to characterize the GPU so \
KernelTuningAgent can choose tile sizes and pick a memory-bound vs compute-bound \
optimization strategy.

REQUIRED METRICS (best effort — partial is acceptable)
  dram_bandwidth_gbps     peak DRAM read GB/s
  boost_clock_mhz         sustained clock under compute load
  sm_count                effective number of SMs
  l2_cache_size_mb        L2 capacity in MB
  dram_latency_cycles     pointer-chasing DRAM latency in clock cycles
  l2_latency_cycles       L2 round-trip latency in clock cycles

WORKFLOW
  1. Read skills/probes_*.md for tested microbenchmark patterns (use list_skills + read_skill).
  2. Use run_cuda_probe to compile and run small CUDA C kernels measuring each metric.
  3. Use profile_with_ncu to confirm DRAM bandwidth / clock when available.
  4. When all (or as many as possible) metrics are gathered, call \
submit_hardware_profile with a metrics dict and a confidence in [0, 1].

KEEP IT BOUNDED
  - This is advisory data. If a metric is hard to measure on this GPU, log it \
under caveats and submit the rest. Don't burn the whole budget on one number.
  - 1e10 cycles or > 5 GHz boost are obviously wrong — flag and retry.
"""


class HardwareProfilerAgent(LLMStageAgent):
    stage = Stage.HARDWARE_PROFILE
    allowed_tools = (
        "list_skills",
        "read_skill",
        "run_cuda_probe",
        "profile_with_ncu",
        "profile_with_nsys",
        "record_measurement",
        "flag_event",
        "submit_hardware_profile",
        "write_workspace_file",
        "probe_environment",
    )
    max_iterations = 30
    SYSTEM_PROMPT = _SYSTEM_PROMPT

    def build_user_message(self, context: StageContext) -> str:
        spec_path = context.layout.benchmark_spec_path("hardware")
        spec_blob = ""
        if spec_path.is_file():
            try:
                spec_blob = json.dumps(
                    json.loads(spec_path.read_text(encoding="utf-8")), indent=2
                )
            except json.JSONDecodeError:
                spec_blob = spec_path.read_text(encoding="utf-8")
        return (
            "Profile the GPU according to the hardware benchmark spec below. "
            "Submit hardware_profile.json when done.\n\n"
            f"=== hardware spec ===\n{spec_blob or '(spec not found)'}\n"
        )
