---
name: operators/lora_matmul_tuning
description: Tuning navigation guide for LoRA-fused MATMUL — realistic performance ceiling, profiling-driven exploration, search space dimensions, and tile size guidance.
---

# LoRA MATMUL — Tuning Navigation Guide

Companion to `operators/lora_matmul`. Use this during TUNING_LOOP to decide what
to build next. Start by reading the hardware blackboard; let profiling evidence
drive every decision rather than following a fixed recipe.

> **Dependency constraint (official eval environment)**
> Allowed headers: `<torch/extension.h>`, `<cuda_runtime.h>`, `<mma.h>` (for
> tensor cores), standard C/C++ headers. **Do NOT use** CUTLASS, direct
> cuBLAS/cuDNN calls, Thrust, or any header not in the CUDA 12 toolkit. No
> extra source files beyond `optimized_lora.cu`. No `extra_ldflags`.

## Realistic Performance Ceiling

The PyTorch baseline calls cuBLAS for both GEMM terms — cuBLAS is already
highly tuned. Understand the ceiling before writing a kernel:

**WX term** (d×d @ d×d): arithmetic intensity ≈ d/6 FLOP/Byte.
For d ≥ 3584 this exceeds the roofline crossover on any modern GPU — WX is
**deeply compute-bound**. A hand-written CUDA core kernel will not beat cuBLAS
on this term alone. Savings come from kernel-launch fusion and shared memory
reuse, not raw FLOP throughput.

**A(B^T X) term** (low-rank, r=16): arithmetic intensity ≈ 0.5 FLOP/Byte.
This term is **strongly memory-bound**. Fusing it into the WX tile loop
eliminates one DRAM round-trip for the intermediate T = B^T X — this is where
the real gain comes from.

**Practical speedup range**: The achievable range depends heavily on which
correctness constraint applies. With atol=1e-4 (Phase-2), the WX term must
use cuBLAS-equivalent precision (e.g. `at::mm`); optimization space is limited
to the low-rank correction and overhead reduction, which historically yields
~1.05–1.10×. If a looser tolerance is acceptable, a fully fused FP32 kernel
can potentially reach 1.3–2.0×. Before committing to a strategy, estimate the
actual benefit from the available headroom rather than assuming the higher end.
Anything above 2× warrants timing verification; above 3× is almost certainly
a measurement artifact.

**Starting point warning**: a kernel that calls `torch::matmul` for all GEMMs
and adds a custom elementwise kernel will often be *slower* than the PyTorch
baseline due to the extra tensor allocation and the custom kernel replacing
PyTorch's fused `+`. Verify that the very first candidate is at least at parity
before spending iterations on further tuning.

## Step 0 — Read Hardware Before Writing Any Kernel

Call `read_blackboard("hardware")` first. Use the measured values to anchor
every subsequent decision:

| Field | How to use it |
|-------|--------------|
| `dram_bw_gbps` | Memory bandwidth ceiling; `memory_ceiling_TBs = dram_bw_gbps / 1000` |
| `sm_count` | Warp concurrency ceiling; occupancy target = blocks_per_sm × sm_count |
| `l2_kb` | Working-set guidance for tile sizes |
| `compute_capability` | cc ≥ 8.0 → WMMA tensor cores available; cc < 8.0 → CUDA cores only |
| `peak_clock_mhz` | FP32 ceiling ≈ sm_count × 128 × 2 × peak_clock_mhz × 1e6 / 1e12 TFLOPS |

Roofline crossover (FLOP/Byte) = FP32_ceiling_TFLOPS × 1e3 / dram_bw_gbps.
Compare WX arithmetic intensity (≈ d/6) against crossover to confirm whether
WX is compute-bound or memory-bound on **this specific machine**.

## Exploration Space

Each dimension is a direction to explore, not a fixed prescription. Profile
first (Step 2), then pick the dimension most likely to address the observed
bottleneck.

**Kernel architecture**
- `split`: two separate kernel launches (one for WX, one for A(B^TX)). Simpler
  to write and debug. Two DRAM round-trips for X.
- `fused`: one kernel, both terms share smem tiles of X. Eliminates the second
  X load from DRAM. Preferred direction once correctness is confirmed.

**Low-rank path** (r = 16, fixed)
- `sequential`: compute T = B^T X to a temporary DRAM buffer, then A @ T.
  Two extra DRAM writes/reads for the 16×d intermediate.
- `smem_side`: accumulate the low-rank contribution inside the WX k-tile loop.
  T lives in registers (or smem if BM > 32). Eliminates all intermediate DRAM.
  This is almost always the better choice; explore it early.

**Memory access pattern**
- `naive`: one float per thread per load.
- `smem_tile`: W and X tiles staged through shared memory; enables coalescing
  and data reuse across threads.
- `float4`: 128-bit vectorized loads (4 floats at once). Effective when
  columns are 128-bit aligned; use with `__ldg` for read-only inputs.

**Compute unit**
- `cuda_cores`: standard FP32 FMA. Good baseline.
- `tensor_cores`: **NOT usable for the WX term on sm86 (RTX 3090).** On
  Ampere, `wmma::mma_sync` with float32 fragments automatically uses TF32
  (10-bit mantissa). The oracle is generated with TF32 disabled (FP32 mode),
  so WMMA results differ from the oracle by ~0.1 — far above atol=1e-4.
  Do not attempt WMMA for the WX term.

**Tile shape BM × BN × BK**
- Drives shared memory use: `smem_bytes = (BM + BN + 16) × BK × 4` (the +16
  accounts for the B slice).
- Hard ceiling: `cudaDeviceProp.sharedMemPerBlock` (typically 48–100 KB).
- Starting point: BM = BN = 64, BK = 16. Adjust based on occupancy data from
  ncu — if occupancy is low because smem is the limiter, reduce BM/BN; if
  occupancy is fine but compute is low, increase BM/BN.
- Register pressure: keeping t_tile (the low-rank accumulator) in registers
  requires 16 × BM floats. Use BM ≤ 32 to stay within register budget, or
  spill t_tile to smem for larger BM.

## Step 1 — Correctness First

### FP32 precision constraint for d≥3584

The oracle is generated by cuBLAS with TF32 disabled (pure FP32 mode).
cuBLAS uses parallel reduction trees internally (error ~ sqrt(d) × ε ≈ 7e-6
for d=3584). A hand-written sequential FP32 accumulation has a larger
rounding error ~ d × ε × scale ≈ 2–3e-3 for d=3584, which exceeds atol=1e-4.

This is a mathematical property of FP32 non-associativity, not a kernel logic
bug. Being aware of this tradeoff shapes every architectural decision:

| Approach | Precision vs oracle | Throughput implication |
|----------|--------------------|-----------------------|
| Sequential FP32 k-loop | ~2–3e-3 (fails atol=1e-4) | Full FP32 speed |
| FP64 accumulation, cast to FP32 | ~7e-6 (passes) | ~1/64 of FP32 on RTX 3090 |
| `at::mm` via `<torch/extension.h>` | ~0 (bit-identical) | cuBLAS speed, no fusion |
| Custom tiled FP32 + parallel reduction | ~1e-5 if matching cuBLAS order | Hard to implement correctly |

Note on WMMA: on sm86 (Ampere), `wmma::mma_sync` with float32 fragments
uses TF32 in hardware. The oracle is FP32, so WMMA introduces ~0.1 error —
worse than sequential FP32. See the Compute unit note in Exploration Space.

**Diagnostic signal**: if a simple naive kernel and a complex tiled kernel
produce the *same* max_abs_err, and a pure-PyTorch forward gives 0 error,
you are looking at an oracle precision mismatch, not a logic bug. See
`cuda_kernel_debug` → "Special case: oracle precision mismatch" for details.

Fix guide for compile/correctness failures:
- `compile_ok=False` → Check `PYBIND11_MODULE`, `forward` signature, headers.
- `correctness_ok=False` with large identical error across kernel variants →
  precision mismatch; reconsider the accumulation strategy.
- `correctness_ok=False` with error that varies by kernel → logic bug; use
  `cuda_kernel_debug` checklist (B^T indexing, boundary guards, syncthreads).

## Step 2 — Profile Before the Next Iteration

After a correct candidate exists, the analyst stage runs ncu. Read
`read_blackboard("latest_diagnosis")` before writing the next candidate.

Key ncu metrics and what they tell you:

| Metric | Low value means | High value means |
|--------|----------------|-----------------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | under-utilized SMs | compute-bound |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | low DRAM pressure | memory-bound |
| `l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum` | little smem use | smem-heavy |

Direction from evidence:
- DRAM util high, compute util low → **memory-bound**: improve access patterns
  (float4, coalescing, larger tiles for reuse)
- Compute util high, DRAM util low → **compute-bound**: try tensor cores or
  unroll the inner K loop for ILP
- Both low → occupancy or launch overhead: check smem allocation vs. hardware
  limit; reduce BM/BN to fit more blocks per SM

## Step 3 — One Variable at a Time

Change one dimension per candidate. Keep the prior `candidate_id` as a
reference so the speedup delta is attributable to the single change.
Record your hypothesis in the `submit_candidate` call.

## Speedup Signals

| Observed speedup | Interpretation |
|-----------------|---------------|
| < 1.0 | Access pattern regression or bank conflict — profile for root cause |
| 1.0–1.3 | Modest gain; continue profiling to find the dominant bottleneck |
| 1.3–1.8 | Meaningful improvement; look at the next ncu bottleneck |
| 1.8–2.0 | Near the realistic ceiling; verify timing stability before submitting |
| > 2.0 | Very strong — double-check `torch.cuda.synchronize()` placement and cudaEvent recording before claiming |

## Low-rank Exploitation (r = 16)

The `smem_side` pattern eliminates DRAM traffic for the intermediate T = B^T X.
Pseudocode for the fused tile loop:

```cuda
for (int kk = 0; kk < d; kk += BK) {
    // Load tiles into shared memory
    smem_W[BM][BK] <- W[row][kk:kk+BK]   // base weight slice
    smem_X[BK][BN] <- X[kk:kk+BK][col]   // input slice (shared with low-rank)
    smem_B[BK][16] <- B[kk:kk+BK][0:16]  // B slice (r=16 cols)

    __syncthreads();

    // Base GEMM contribution
    acc_W += dot(smem_W[ty][0:BK], smem_X[0:BK][tx]);

    // Low-rank contribution: accumulate A @ (B^T @ X) in registers
    for (int k = 0; k < BK; k++) {
        float x_val = smem_X[k][tx];
        for (int r = 0; r < 16; r++)
            t_reg[r] += smem_B[k][r] * x_val;  // t_reg = B^T @ X column
    }
    __syncthreads();
}
// Load A tile once, compute low-rank output
smem_A[BM][16] <- A[row][0:16]
float acc_lr = dot(smem_A[ty][0:16], t_reg[0:16]);
Y[row][col] = acc_W + acc_lr;
```

Register pressure: `t_reg[16]` is 16 floats per thread — safe for any BM.
The accumulated `smem_W / smem_X` tiles are the larger cost; BM=BN=64 with
BK=16 uses 64×4 + 64×4 + 16×4 = 576 bytes of smem per warp-width element.
