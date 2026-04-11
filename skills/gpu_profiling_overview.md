# GPU Profiling Overview

This document provides a high-level map of the measurement strategies
available in this skill library.

## Metric categories

### Memory latency hierarchy

Use pointer-chasing kernels to measure access latency for each cache tier.
The key is to size the working set to force cache misses at the tier you
want to measure, and use a random (non-sequential) access pattern so the
prefetcher cannot hide latency.

- L1 cache latency: array fits entirely in L1 (~32 KB for most GPUs)
- L2 cache latency: array fits in L2 but not L1 (~4–40 MB)
- DRAM latency: array is much larger than L2 (>100 MB)

Skill to use: `memory_latency` (Phase 2)

### Memory bandwidth

Use streaming read/write kernels to saturate memory bandwidth. The array
must be large enough to exceed all cache levels, and the access pattern
must be sequential (coalesced) for global memory bandwidth.

Skill to use: `memory_bandwidth` (Phase 2)

### L2 cache capacity

Sweep the working set size across a range (e.g. 1 MB to 64 MB in steps)
and plot the measured latency. The point where latency jumps sharply
indicates the L2 boundary.

Skill to use: `cache_capacity` (Phase 2)

### Actual boost clock frequency

Run a compute-intensive kernel and measure elapsed wall time against
elapsed GPU clock cycles using `clock64()`. This gives the true operating
frequency regardless of what nvidia-smi or cudaGetDeviceProperties report.

Skill to use: `clock_measurement` (Phase 2)

### Bank conflict penalty

Compare shared memory access time with stride=1 (conflict-free) versus
stride=32 (every thread hits the same bank). The ratio gives the conflict
penalty in cycles.

Skill to use: `bank_conflict` (Phase 2)

## Anti-hacking checklist

Before finalizing any measurement:
1. Compare measured clock against cudaGetDeviceProperties().clockRate.
   If difference > 10%, flag_event "clock_locked" and report the measured value.
2. Verify L2 latency is in the expected range for the GPU family.
   If DRAM latency < 200 cycles, the working set may still be in L2.
3. Check that measured peak bandwidth is lower than theoretical max.
   If measured > theoretical, the measurement likely contains artifacts.

## Recommended sequence for hardware_probe tasks

1. list_skills → read_skill for each relevant strategy
2. run_cuda_probe with a self-timed kernel → primary measurement
3. profile_with_ncu with relevant counter metrics → cross-verification
4. flag_event for any discrepancy > 10%
5. record_measurement with confidence reflecting variance across runs
6. submit_results with methodology summary
