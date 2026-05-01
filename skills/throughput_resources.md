---
name: throughput_resources
description: Measure DRAM/L2 bandwidth, SM count, shared-memory capacity, and register file size via CUDA microbenchmarks.
---

# Throughput and Resource Probing

## Purpose

Measure effective bandwidth and resource limits by generating kernels that
saturate a specific memory path or stress a specific resource constraint.

## When to use

Use this skill for targets such as:

- `peak_dram_bandwidth_GBps`, `global_memory_bandwidth_GBps`
- `peak_shmem_bandwidth_TBps`, `shared_memory_bandwidth_TBps`
- `max_shmem_per_block_kb`, `max_shared_memory_per_block_kb`
- `bank_conflict_penalty_cycles`, `shared_memory_bank_conflict_penalty`
- requests that mention bandwidth, shared memory capacity, or bank conflict cost

## Primary CUDA strategies

For global memory bandwidth, use a large linear streaming kernel with coalesced
loads and stores. Use many blocks, enough iterations to exceed 50 ms, and CUDA
events for wall time. Report bytes moved divided by elapsed seconds.

For shared memory bandwidth, run a repeated shared-memory load/store loop inside
each block, time with `clock64()` or CUDA events, and compute effective bytes per
second. Keep global memory traffic outside the timed region when possible.

For maximum shared memory per block, compile one kernel that requests dynamic
shared memory and sweep launch sizes. If launches fail, binary search the
largest successful byte count. Prefer measured launch success over
`cudaGetDeviceProperties`.

For bank conflict penalty, compare two nearly identical shared-memory kernels:
one conflict-free pattern and one conflicting pattern. Report both latencies and
their difference or ratio.

## Output fields

Use parseable stdout:

```text
peak_dram_bandwidth_GBps=301.62
bytes_moved=8589934592
elapsed_ms=28.48
```

```text
peak_shmem_bandwidth_TBps=4.173
conflict_free_cycles=32
conflict_cycles=71
bank_conflict_penalty_cycles=39
max_shmem_per_block_kb=96
```

## Metric mapping

| stdout field | target key | unit |
| --- | --- | --- |
| `peak_dram_bandwidth_GBps` | `peak_dram_bandwidth_GBps` | GB/s |
| `peak_dram_bandwidth_GBps` | `global_memory_bandwidth_GBps` | GB/s |
| `peak_shmem_bandwidth_TBps` | `peak_shmem_bandwidth_TBps` | TB/s |
| `max_shmem_per_block_kb` | `max_shmem_per_block_kb` | KB |
| `bank_conflict_penalty_cycles` | `bank_conflict_penalty_cycles` | cycles |

## Cross-verification

For global bandwidth, profile with:

- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`
- `l2__throughput.avg.pct_of_peak_sustained_elapsed`

For bank conflicts and shared memory behavior, profile with:

- `l1tex__data_bank_conflicts_pipe_lsu.sum`
- `l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum`
- `l1tex__t_sectors_pipe_lsu_mem_shared_op_st.sum`

Use ncu as a sanity check, not as the primary numeric answer, because the judge
expects values measured in the active environment.

## Anomaly signals

- Measured DRAM bandwidth greater than physical peak is usually a byte-count bug.
- Very low bandwidth with high variance may indicate thermal throttling or too
  short a timed region.
- Bank conflict penalty near zero means the conflict pattern was optimized away
  or did not map lanes to the same bank.
- Shared memory limit from API and measured launch success may differ; report
  measured launch success and flag `api_spoofed` if needed.

## Failure fallback

- On `timeout`, reduce array size or iteration count but keep enough work for a
  stable timing window.
- On `user_code`, simplify the kernel and avoid templates or unsupported CUDA
  features.
- On `infrastructure`, record unavailable evidence only if no CUDA path works.
- On ncu permission failure, keep the CUDA microbenchmark result and flag
  `ncu_unavailable`.
