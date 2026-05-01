---
name: operators/lora_matmul
description: Optimization target for LoRA-fused MATMUL — formula, hardware requirements, success criteria, and optimization strategies.
---

# Operator: LoRA-fused MATMUL

## Formula

```
Y = W X + A (B^T X)
```

Where:
- W ∈ ℝ^(d×d) — base weight matrix
- X ∈ ℝ^(d×d) — input activation matrix
- A ∈ ℝ^(d×r) — LoRA up-projection
- B ∈ ℝ^(d×r) — LoRA down-projection (applied as B^T)
- r = 16 (fixed low rank)
- All tensors: float32

This is a LoRA (Low-Rank Adaptation) fused matmul. The second term `A(B^T X)` is a
low-rank correction to the base matrix multiply `WX`.

## Input Specification

- All four tensors (W, X, A, B) are stored as `.pt` files, loaded with `torch.load`.
- Hidden dimension d is chosen from **d ∈ [3584, 4608]** at evaluation time.
- The implementation must produce correct results for **any integer d in this range**,
  not just a single fixed value. Do not hard-code d = 4096.
- dtype: float32 for all tensors.
- Device: CUDA GPU.

## Optimization Goal

**Minimize end-to-end latency** of computing Y = WX + A(B^T X) on the target GPU.

Baseline: naive sequential PyTorch:
```python
Y = W @ X + A @ (B.T @ X)
```

Target: a custom CUDA kernel or fused implementation that is measurably faster
than the PyTorch baseline for representative d values in [3584, 4608].

## Required Hardware Measurements

Use `hardware_probe` to measure:

- `dram_bandwidth_gbps` — peak achievable DRAM read bandwidth (GB/s)
- `boost_clock_mhz` — actual GPU boost clock under sustained compute load
- `sm_count` — effective number of streaming multiprocessors
- `l2_cache_size_mb` — L2 cache capacity in MB
- `dram_latency_cycles` — pointer-chasing DRAM latency in clock cycles
- `l2_latency_cycles` — L2 cache round-trip latency in clock cycles

These parameters are needed to build a roofline model and choose tile sizes.

## Baseline Profiling (op_profiler)

When `op_profiler` is available:

Profile the PyTorch baseline using `profile_with_torch`. Required output format:

```
shape=3584 torch_ms=<median>
shape=4096 torch_ms=<median>
shape=4608 torch_ms=<median>
```

Template Python script for `profile_with_torch`:

```python
import torch, statistics

device = torch.device("cuda")
shapes = [3584, 4096, 4608]
r = 16

for d in shapes:
    W = torch.randn(d, d, device=device, dtype=torch.float32)
    X = torch.randn(d, d, device=device, dtype=torch.float32)
    A = torch.randn(d, r, device=device, dtype=torch.float32)
    B = torch.randn(d, r, device=device, dtype=torch.float32)
    # warm-up
    for _ in range(10):
        _ = W @ X + A @ (B.T @ X)
    torch.cuda.synchronize()
    # timed runs
    times = []
    for _ in range(50):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        Y = W @ X + A @ (B.T @ X)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    print(f"shape={d} torch_ms={statistics.median(times):.4f}", flush=True)
```

Record as `torch_baseline_ms_d<d>` (unit: "ms") for each shape.

## Bottleneck Analysis (bottleneck_analyst)

When `bottleneck_analyst` is available:

**Arithmetic intensity of WX (d×d @ d×d):**
```
FLOPs = 2 × d³
Bytes = (d² + d² + d²) × 4   (read W, X; write Y)
AI    = 2d³ / (12d²) = d/6
```
For d = 4096: AI ≈ 683 FLOP/Byte — this is typically compute-bound on modern GPUs.

**Arithmetic intensity of A(B^T X) — the low-rank term:**
```
Step 1: T = B^T X    (r×d @ d×d → r×d)    FLOPs = 2rd²,  Bytes ≈ (rd + d² + rd) × 4
Step 2: A T          (d×r @ r×d → d×d)    FLOPs = 2rd²,  Bytes ≈ (dr + rd + d²) × 4
```
With r = 16 and d = 4096: AI for each step ≈ 2×16×d / (32×d + d²) ≈ 0.5 FLOP/Byte.
The low-rank term is **strongly memory-bound**.

**Decision rule:**
- If measured DRAM bandwidth >> compute TFLOPS / AI → memory-bound
- Fusing W X and A(B^T X) into one kernel can save memory round-trips

## Success Criteria

All of the following must be satisfied before calling `mark_ready_for_critic`:

1. **Hardware measured**: dram_bandwidth_gbps, boost_clock_mhz, sm_count, l2_cache_size_mb,
   dram_latency_cycles, l2_latency_cycles — all recorded with confidence ≥ 0.70.
2. **Bottleneck identified**: compute-bound or memory-bound determination with
   arithmetic intensity estimate and comparison to roofline ridge point.
3. **Optimized implementation**: a custom CUDA kernel or fused implementation
   that applies at least one optimization strategy based on the bottleneck analysis.
4. **Correctness verified**: output Y matches `torch.float32` reference
   (W @ X + A @ (B.T @ X)) for **at least 3 distinct d values** in [3584, 4608],
   with max absolute difference < 1e-2 (float32 accumulation tolerance).
5. **Performance benchmarked**: measured latency (ms) and speedup vs. PyTorch
   baseline for at least 3 d values; speedup > 1.0 required for a meaningful result.

## Potential Strategies

**Fusion (highest priority — eliminates memory round-trips):**
- Fuse `WX` and `A(B^T X)` into a single kernel pass to avoid writing/reading
  intermediate results to DRAM.
- Key insight: the low-rank term is memory-bound; WX is compute-bound.
  A fused kernel can hide the memory latency of the low-rank term behind the
  compute-intensive WX.

**If the combined operator is memory-bound (lower d, bandwidth-limited GPU):**
- Use shared-memory tiling to improve data reuse for W and X.
- Vectorized loads (float4 = 128-bit) for coalesced global memory access.
- Choose tile sizes that fit in L2: tile_size² × 4 bytes ≤ L2_size / 4.

**If compute-bound (higher d, high-bandwidth GPU):**
- Increase ILP: unroll inner loops, keep more live values in registers.
- Use tensor cores (WMMA or `mma.sync` PTX) if the GPU supports them
  (sm_80+ for TF32, sm_70+ for FP16). Note: float32 tensor cores are sm_80+.
- Tune BLOCK_M / BLOCK_N / BLOCK_K to maximize L2 reuse.

**Low-rank structure exploitation:**
- The term `A(B^T X)` only needs `d×r` intermediate storage (r=16 << d).
  This intermediate fits in shared memory or L1 for all relevant d values.
- Strategy: compute `T = B^T X` (r×d) in registers/smem, then `A T` (d×d)
  as a batched outer product — avoids all DRAM traffic for the intermediate.

**Handling variable d:**
- Use template parameters or runtime dispatch on tile sizes.
- Ensure d is padded to a multiple of WARP_SIZE (32) or tile width if needed.
- Use bounds-checked loads for non-multiple d; or pad the matrices before the call.

## Anomaly Signals

- Correctness failure (max diff > 1e-2) on any d value → stop; do not report success.
- Speedup > 15× → likely a measurement error (clock mismatch, caching effect); verify.
- Benchmark variance > 15% across runs → use median of ≥ 10 runs; flag_event "unstable_timing".
- If ncu/nsys tools are unavailable (infrastructure error) → use self-timed CUDA kernels
  with cudaEventRecord for timing; do not keep retrying ncu.
