"""pipeline/tools/ — Stage-specific tools.

Distinct from agents/tools/builtin/ on purpose: these tools are scoped to the
new pipeline (per-stage allowed_tools whitelist, layout-aware, no cross-stage
state). The old builtin tools (record_measurement, flag_event, run_cuda_probe,
profile_with_torch, etc.) are still reused — see pipeline.tool_factory.
"""
