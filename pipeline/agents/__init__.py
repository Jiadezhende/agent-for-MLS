"""pipeline.agents — concrete StageAgent implementations.

Importing this package registers nothing globally; the orchestrator wires
agents up explicitly in main.py via ``default_stage_agents()``.
"""
from .benchmark_spec_agent import BenchmarkSpecAgent
from .hardware_profiler_agent import HardwareProfilerAgent
from .baseline_agent import BaselineAgent
from .kernel_tuning_agent import KernelTuningAgent
from .profile_analysis_agent import ProfileAnalysisAgent
from .summary_agent import SummaryAgent

__all__ = [
    "BenchmarkSpecAgent",
    "HardwareProfilerAgent",
    "BaselineAgent",
    "KernelTuningAgent",
    "ProfileAnalysisAgent",
    "SummaryAgent",
]
