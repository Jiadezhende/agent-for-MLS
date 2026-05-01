# Operator: [Name]

## Formula

Describe the operator's mathematical formula and parameter shapes.

Example:
```
Y = f(X, W, ...)
```

- Parameter shapes, dtypes
- Constraints on shapes (e.g. d must be a multiple of 128)

## Input Specification

- Tensor shapes, dtypes, storage format (.pt files? random tensors?)
- Variable range (e.g., d ∈ [3584, 4608])
- Constraints: the implementation must handle any value in the specified range, not just one fixed shape.

## Optimization Goal

What to minimize: latency? memory bandwidth? FLOPs? Some combination?

State the primary metric for benchmarking and the comparison baseline (e.g., PyTorch default).

## Required Hardware Measurements

List the hardware parameters the Planner should measure via hardware_probe:

- `dram_bandwidth_gbps` — peak DRAM read bandwidth
- `boost_clock_mhz` — actual GPU boost clock under load
- `sm_count` — effective SM count
- `l2_cache_size_mb` — L2 cache size
- `dram_latency_cycles` — pointer-chasing DRAM latency
- (add/remove as needed)

## Baseline Profiling (op_profiler)

When the `op_profiler` agent is available:

Describe how to measure the naive/PyTorch baseline:
- Which torch calls to benchmark
- What metrics to collect (latency per shape, achieved bandwidth %, SM occupancy)
- Which d/shape values to test

## Bottleneck Analysis (bottleneck_analyst)

When the `bottleneck_analyst` agent is available:

Describe how to determine the compute/memory bound nature:
- Arithmetic intensity formula for this operator
- Roofline model parameters to use
- Decision rule: if AI < ridge_point → memory-bound, else compute-bound

## Success Criteria

The Critic evaluates against ALL of the following. Check each box before calling
`mark_ready_for_critic`:

1. Hardware parameters measured (all items in Required Hardware Measurements)
2. Bottleneck identified (compute-bound or memory-bound) with quantitative evidence
3. Optimized implementation produced (CUDA kernel or fused PyTorch code)
4. Correctness verified: output matches torch reference for ≥ 3 distinct shape values
   in the specified range, with max absolute difference < [threshold]
5. Performance benchmarked: measured latency and speedup vs. baseline for ≥ 3 shapes

## Potential Strategies

List optimization strategies ranked by expected impact:

**If memory-bound:**
- Fuse operations to reduce round-trips to DRAM
- Use shared-memory tiling to improve data reuse
- Use 128-bit vectorized loads (float4)
- Exploit L2 cache by choosing tile sizes that fit

**If compute-bound:**
- Increase instruction-level parallelism (ILP)
- Use tensor cores / WMMA intrinsics if available
- Tune tile shapes (BLOCK_M, BLOCK_N, BLOCK_K) for the target GPU

**General:**
- Exploit low-rank structure if present (e.g., reduce intermediate dimensions)
- Handle variable input sizes without recompilation (use runtime checks or templates)

## Anomaly Signals

- Correctness failure on any shape → stop and report; do not claim success
- Speedup > 20× without a clear algorithmic reason → likely a measurement error
- Benchmark variance > 15% → report unstable timing; use median of ≥ 5 runs
