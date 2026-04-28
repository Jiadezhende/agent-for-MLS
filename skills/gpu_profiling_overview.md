# GPU Profiling Overview

This file is the routing index for the hardware-probe skill library. Read it
first, then read the one domain skill that matches the requested target.

## Skill routing

| Target family | Examples | Read this skill |
| --- | --- | --- |
| Memory latency and cache size | `dram_latency_cycles`, `l2_latency_cycles`, `l2_cache_capacity_bytes` | `memory_hierarchy` |
| Bandwidth and shared resources | `peak_dram_bandwidth_GBps`, `peak_shmem_bandwidth_TBps`, `max_shmem_per_block_kb`, `bank_conflict_penalty_cycles` | `throughput_resources` |
| Clock and environment state | `actual_boost_clock_mhz`, `effective_sm_count`, clock lock or SM masking checks | `clock_environment` |

## Preflight rule

Do **not** hardcode `-arch`, `--gpu-architecture`, or any `sm_NNN` flag in
CUDA source or compiler arguments. The Executor auto-detects the GPU
architecture via `nvidia-smi` and injects the correct flag through
`AGENT_NVCC_FLAGS`. Hardcoding an arch flag can override detection and
silently produce wrong code (e.g., `clock64()` is unavailable below sm_70
and requires sm_120 on Blackwell).

## General method

Choose the primary tool by metric family:

| Metric family | Examples | Primary tool | Fallback |
| --- | --- | --- | --- |
| Throughput / utilization / clock | `peak_dram_bandwidth_GBps`, `actual_boost_clock_mhz`, `sm__throughput.*`, `gpu__compute_memory_throughput.*` | `profile_with_ncu` with explicit `--metrics` | `run_cuda_probe` if ncu returns `infrastructure` error |
| Latency | `dram_latency_cycles`, `l2_latency_cycles`, `l1_latency_cycles` | `run_cuda_probe` pointer-chasing kernel | `profile_with_ncu` latency metrics to cross-verify |
| CPU-GPU timeline / launch overhead | kernel launch latency, stream concurrency | `profile_with_nsys` | — |

Rules:

1. For throughput/clock metrics, call `profile_with_ncu` first with the exact ncu
   metric names. Do not write a CUDA microbenchmark for what ncu can measure directly.
2. For latency metrics, write a pointer-chasing `run_cuda_probe` kernel first, then
   call `profile_with_ncu` latency metrics to cross-verify.
3. If ncu returns `error_class="infrastructure"` (e.g. `ERR_NVGPUCTRPERM`), switch to
   `run_cuda_probe` self-timed kernels. Do NOT retry ncu.
4. Record direct stdout / ncu output as evidence in every `record_measurement` call.
5. Use `flag_event` for strategy choices, suspicious values, clock locking,
   SM masking, API spoofing, ncu permission failures, and circuit breaker events.
6. Report measured values from the active environment, not online specifications.

## CUDA preflight rules

- Do not hardcode `-arch` or `--gpu-architecture` in `compile_flags` unless a
  prior probe proves the Executor's detected architecture is wrong. The Executor
  or `AGENT_NVCC_FLAGS` is the source of truth for the active GPU architecture.
- If a kernel appears not to execute, device writes stay zero, `clock64()` returns
  zero, or ncu sees no kernel, first check `cudaGetLastError()` and
  `cudaDeviceSynchronize()` for launch errors. Treat architecture or toolchain
  mismatch as the leading suspect before blaming timers or compiler optimization.
- For Blackwell / compute capability 12.0 GPUs, old flags such as `sm_75` can
  compile but fail at launch with PTX/toolchain errors. Rebuild with the detected
  native architecture before continuing measurement work.

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
