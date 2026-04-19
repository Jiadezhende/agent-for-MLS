"""
agent/tasks/hardware_probe/critic_rules.py — Critic system prompt and tool
schema for cross-validating hardware_probe results.

Moved from agent/critic.py.
"""

CRITIC_SYSTEM_PROMPT = """\
You are a GPU benchmark result auditor. You receive results from multiple
parallel workers that each measured different GPU hardware parameters.

Your job:
1. Cross-validate related metrics for physical consistency:
   - DRAM bandwidth ≈ bus_width × clock_rate; latency and bandwidth are
     inversely related. Flag if values are physically implausible.
   - L1 latency < L2 latency < DRAM latency always holds for real GPUs.
   - Boost clock and base clock: boost should be higher than base.
2. Detect suspicious confidence scores — a worker reporting confidence=0.99
   for something that normally has high variance should be scrutinised.
3. Identify duplicate metrics — if the same metric was measured by two
   workers, flag discrepancies > 20%.
4. Adjust confidence DOWN for anomalous results; leave well-supported
   results unchanged.

Call audit_results exactly once with your findings.
Only include metrics in confidence_adjustments when you actually want to
change the score; omit metrics you consider reliable.\
"""

AUDIT_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "audit_results",
        "description": "Return cross-validation findings for all worker results.",
        "parameters": {
            "type": "object",
            "properties": {
                "confidence_adjustments": {
                    "type": "object",
                    "description": (
                        "metric_name → new_confidence (0.0–1.0). "
                        "Only include metrics you want to adjust."
                    ),
                    "additionalProperties": {"type": "number"},
                },
                "anomaly_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of metric names that appear anomalous.",
                },
                "overall_assessment": {
                    "type": "string",
                    "description": "2-4 sentence summary of result quality.",
                },
            },
            "required": ["confidence_adjustments", "anomaly_flags", "overall_assessment"],
        },
    },
}
