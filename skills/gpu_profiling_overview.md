# GPU Profiling Overview

This file is the routing index for the hardware-probe skill library. Read it
first, then read the one domain skill that matches the requested target.

## Skill routing

| Target family | Examples | Read this skill |
| --- | --- | --- |
| Memory latency and cache size | `dram_latency_cycles`, `l2_latency_cycles`, `l2_cache_capacity_bytes` | `memory_hierarchy` |
| Bandwidth and shared resources | `peak_dram_bandwidth_GBps`, `peak_shmem_bandwidth_TBps`, `max_shmem_per_block_kb`, `bank_conflict_penalty_cycles` | `throughput_resources` |
| Clock and environment state | `actual_boost_clock_mhz`, `effective_sm_count`, clock lock or SM masking checks | `clock_environment` |

## General method

1. Prefer `run_cuda_probe` with a self-timed CUDA C microbenchmark.
2. Use `profile_with_ncu` only after a successful CUDA probe and pass the
   returned `binary_path`.
3. Record direct stdout evidence in every `record_measurement` call.
4. Use `flag_event` for strategy choices, suspicious values, clock locking,
   SM masking, API spoofing, ncu permission failures, and circuit breaker events.
5. Report measured values from the active environment, not online specifications.

## Anti-hacking checklist

- Treat `cudaGetDeviceProperties`, `nvidia-smi`, and spec sheets as secondary
  evidence only.
- If measured clock differs from reported clock by more than 10 percent, flag
  `clock_locked` or `clock_throttled` and report the measured value.
- If measured SM count is lower than API count, flag `sm_masked`.
- If DRAM latency is below 200 cycles, increase the working set before recording.
- If bandwidth exceeds plausible physical peak, fix byte accounting before
  recording.

## Completion rule

Call `submit_results` exactly once after every requested target has either a
measured result or an explicit low-confidence unavailable result with evidence.
