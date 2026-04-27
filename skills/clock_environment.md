# Clock and Environment Probing

## Purpose

Measure the actual GPU operating environment rather than trusting advertised
specifications or potentially virtualized CUDA API fields.

## When to use

Use this skill for targets such as:

- `actual_boost_clock_mhz`, `actual_clock_mhz`, `gpu_clock_mhz`
- `sm_count`, `effective_sm_count`, `active_sm_count`
- requests that mention clock locking, SM masking, throttling, or API spoofing

## Primary CUDA strategies

For clock, launch a sustained compute kernel and measure both GPU cycles and
wall time:

```text
actual_clock_mhz = (clock64_end - clock64_start) / wall_seconds / 1e6
```

Use CUDA events for wall time and `clock64()` from inside the measured kernel.
Run long enough to reduce launch overhead, usually 100 ms or more. Repeat two or
three times if the value looks unstable.

For effective SM count, use a persistent kernel with one block per possible SM
and atomic counters that record concurrently resident blocks. Treat API-reported
SM count as advisory only.

For API spoofing checks, compare direct measurement against API or driver
queries. If they disagree materially, record the measured value and flag the
disagreement.

## Output fields

```text
gpu_cycles=125991000
wall_ms=123.20
actual_clock_mhz=1022.7
api_clock_mhz=1410.0
```

```text
effective_sm_count=12
api_sm_count=36
sm_mask_detected=1
```

## Metric mapping

| stdout field | target key | unit |
| --- | --- | --- |
| `actual_clock_mhz` | `actual_boost_clock_mhz` | MHz |
| `actual_clock_mhz` | `actual_clock_mhz` | MHz |
| `effective_sm_count` | `effective_sm_count` | count |
| `effective_sm_count` | `sm_count` | count |

## Cross-verification

For clock sanity checks, profile with:

- `sm__cycles_elapsed.avg.per_second`
- `sm__cycles_elapsed.avg`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`

For environment anomalies, compare measured results against `nvidia-smi` or CUDA
properties only as secondary evidence. Never replace measured values with spec
sheet values.

## Anomaly signals

- Measured clock differs from API or driver clock by more than 10 percent:
  flag `clock_locked` or `clock_throttled`.
- Large run-to-run clock variance: flag `clock_unstable` and report median.
- Effective SM count is much lower than API count: flag `sm_masked`.
- API values contradict measured launch limits or counters: flag `api_spoofed`.

## Failure fallback

- On `timeout`, reduce loop iterations and repeat.
- On `user_code`, remove unsupported API calls first; `clock64()` is the core
  requirement.
- On `infrastructure`, avoid static lookup tables. Record `unavailable` with
  low confidence only when no executable probe can run.
