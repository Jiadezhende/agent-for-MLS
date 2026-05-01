---
name: <skill_name>
description: <one-line summary, < 120 chars>
---

# [Skill Name]

## Purpose

What hardware property or workload characteristic does this skill measure?

## When to use

Which target metrics (metric names from target_spec.json) should trigger this
skill? What kind of analysis question does this answer?

## Kernel strategy

Describe the CUDA or Python code pattern to use.

For CUDA kernels:
- Array size considerations (must exceed which cache tier?)
- Access pattern (linear, random, pointer-chasing, strided?)
- Self-timing approach (clock64() before and after the measured region)
- Output format expected in stdout (e.g. `latency_cycles: 512`)

For Python/PyTorch:
- Which torch API to exercise
- How to ensure GPU execution (device placement, synchronization)

## Expected output

What does the stdout look like on a successful run? Show a sample.

```
metric_value: 442.5
unit: cycles
```

## Cross-verification

Which ncu metrics can confirm or contradict this measurement?
E.g., `l2__throughput.avg.pct_of_peak_sustained_elapsed` for bandwidth.

## Anomaly signals

- Value < X → likely cache residency (array too small)
- Value variance > 20% → clock instability; flag_event "clock_unstable"
- Measured clock vs cudaGetDeviceProperties().clockRate differs > 10% → flag_event "clock_locked"
