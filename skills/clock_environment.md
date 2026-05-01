---
name: clock_environment
description: Measure actual GPU boost clock and validate environment constraints via CUDA clock64() kernels.
---

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

For clock, the **primary method** is combining `clock64()` cycle counts with
CUDA-event wall time:

```text
actual_clock_mhz = (clock64_end - clock64_start) / wall_seconds / 1e6
```

Use CUDA events for wall time and `clock64()` from inside the measured kernel.
Run long enough to reduce launch overhead, usually 100 ms or more. Repeat two or
three times if the value looks unstable.

### Anti-optimization: use a clock64 busy-wait loop

Empty `for` loops and arithmetic-only loops are silently eliminated by the
compiler; the kernel finishes in nanoseconds and the clock estimate becomes
nonsensically high. The only loop the compiler cannot remove is one whose
condition reads `clock64()` directly — the compiler cannot prove `clock64()` is
pure, so every iteration must execute:

```text
while (clock64() - start < target_cycles) {}
```

Always warm up with a short spin before the timed run; the first kernel launch
often executes at a lower frequency before boost engages.

On architectures where inline PTX is available, `mov.u64 %globaltimer` can be
used as an independent nanosecond-scale wall-clock cross-check. Keep the primary
reported clock based on direct in-kernel cycles divided by measured elapsed time.

`cudaDeviceProp.clockRate` and driver/API queries are **secondary evidence only**.
Use them to cross-check, not as the primary measurement. On some drivers they
report the TDP boost ceiling, not the sustained operating clock.

**Nsight Compute (`profile_with_ncu`) may perturb boost clocks.** The profiler
replays kernels under instruction-level sampling, which can force the GPU into
a lower power state. Always run the microbenchmark first with `run_cuda_probe`,
record the clock from that run, and use `profile_with_ncu` only as a
cross-check — never let the ncu result replace the probe result for
`actual_boost_clock_mhz`.

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

Nsight Compute clock metrics are validation context, not the primary clock
answer, unless direct timing cannot run.

## Anomaly signals

- Measured clock differs from API or driver clock by more than 10 percent:
  flag `clock_locked` or `clock_throttled`.
- Large run-to-run clock variance: flag `clock_unstable` and report median.
- Effective SM count is much lower than API count: flag `sm_masked`.
- API values contradict measured launch limits or counters: flag `api_spoofed`.
- `cudaDeviceProp.clockRate` may be missing in newer CUDA versions; do not use
  it as a required field in probe templates.
- `clock64()` returning zero is usually a launch/setup problem. Check launch
  errors and architecture flags before concluding that clock access is disabled.

## Failure fallback

- On `timeout`, reduce loop iterations and repeat.
- On `user_code`, remove unsupported API calls first; `clock64()` is the core
  requirement.
- On `infrastructure`, avoid static lookup tables. Record `unavailable` with
  low confidence only when no executable probe can run.
