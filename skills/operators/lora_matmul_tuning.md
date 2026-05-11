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

**Practical speedup range**: correctness is judged against a reference
recomputed *online* by `ops.reference(inputs)` inside the same subprocess /
cuBLAS state as the candidate (see [evaluation.py](operator_opt_pipe/resources/evaluation.py)
— `_build_quick_script` / `_build_bench_script`). PyTorch's default
`allow_tf32 = True` is in effect, so the reference itself uses Tensor Core TF32
on Ampere+. A candidate that also dispatches to cuBLAS / TF32-backed Tensor
Core ops is therefore expected to be near bit-exact. A fully fused FP32 kernel
can potentially reach 1.3–2.0×; combining cuBLAS-equivalent `at::mm` for WX
with a fused low-rank epilogue typically yields ~1.05–1.30×. Estimate the
benefit from the available headroom rather than assuming the higher end.
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
- `tensor_cores`: on Ampere, `wmma::mma_sync` with float32 fragments runs in
  TF32 (10-bit mantissa). The correctness reference is recomputed online with
  PyTorch defaults (`allow_tf32 = True`), so the reference matmul itself uses
  TF32 — a TF32 WMMA path no longer carries an automatic ~0.1 error penalty
  against the gate. Still verify: WMMA reduction order differs from cuBLAS,
  so per-element drift can reach `~atol + rtol·|y_ref|` for large-magnitude
  elements at d=3584. Profile `max_abs_err` and `rel_l2_err` rather than
  assuming pass/fail in advance.

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

### How correctness is judged (read this before chasing precision)

The gate is `torch.allclose(Y, Y_ref, rtol=1e-4, atol=1e-4)`, where `Y_ref` is
recomputed *online* in the candidate's own subprocess via `ops.reference(inputs)`
(see [evaluation.py](operator_opt_pipe/resources/evaluation.py) —
`_build_quick_script` / `_build_bench_script`). Two consequences agents
repeatedly miss:

1. **`allclose` is element-wise**, not a `max_abs_err` threshold. The
   condition is `|Δᵢ| ≤ atol + rtol·|y_refᵢ|` per element. For LoRA matmul at
   d=3584, `Y` elements have magnitude `~√d ≈ 60`, so large-magnitude
   elements get an effective tolerance of `1e-4 + 1e-4 × 60 ≈ 6e-3` — far
   above atol. A reported `max_abs_err = 7e-4` can already be passing if the
   offending element is in the high-magnitude regime. **Do not treat
   `max_abs_err < atol` as the pass condition.** Read `correctness_ok` from
   the tool response.
2. **The reference uses PyTorch defaults, including `allow_tf32 = True`.**
   So `Y_ref` itself goes through Tensor Core TF32 paths on Ampere. There
   is no on-disk oracle — the reference is rebuilt in-process for every
   evaluation. A candidate that dispatches to cuBLAS / `at::mm` is
   therefore expected to be near bit-exact, and TF32-backed WMMA is no
   longer ruled out by precision alone.

### FP32 reduction-order drift

A hand-written sequential FP32 k-loop accumulates left-to-right and produces
error ~ `d × ε × scale ≈ 2–3e-3` at d=3584. Whether this passes the gate
depends on the element-wise rule above and per-shape statistics — sometimes
it passes thanks to the `rtol·|y_ref|` budget at large d, sometimes not.
Treat it as something to *measure*, not as an a priori veto.

| Approach | Typical max_abs_err | Throughput implication |
| -------- | ------------------- | ---------------------- |
| Sequential FP32 k-loop | ~2–3e-3 (may or may not pass `allclose`) | Full FP32 speed |
| FP64 accumulation, cast to FP32 | ~7e-6 | ~1/64 of FP32 on RTX 3090 (rarely worth it) |
| `at::mm` via `<torch/extension.h>` | ~0 (cuBLAS-identical) | cuBLAS speed, no custom fusion |
| Tiled FP32 with parallel reduction matching cuBLAS | ~1e-5 if correctly tuned | Hard to implement |
| TF32 WMMA (`wmma::mma_sync` with float fragments) | similar order to TF32 cuBLAS path | Tensor Core speed; verify per-shape |

**Diagnostic shortcut**: if two consecutive custom kernels with different
precision/accumulator strategies (e.g. FP32 → FP64) produce `max_abs_err`
within 5% of each other *and both fail the gate*, the residual mismatch is
**reduction order** against `Y_ref`'s cuBLAS path, not raw precision. Don't
try a third precision tweak — switch the matmul reduction to `at::mm` /
`torch::mm` and only fuse the LoRA epilogue. `write_candidate` prints a
`hint:` line on the third such attempt, but recognizing the pattern after
two iterations saves a wasted round.

Fix guide for compile/correctness failures:
- `compile_ok=False` → Check `PYBIND11_MODULE`, `forward` signature, headers.
- `correctness_ok=False` with `max_abs_err` ≥ ~1e-1 and a clear shape-dependent
  pattern → logic bug; use `cuda_kernel_debug` checklist (B^T indexing,
  boundary guards, syncthreads).
- `correctness_ok=False` with `max_abs_err` in the `1e-3 – 1e-2` range that
  barely moves across precision variants → reduction-order drift against
  cuBLAS; route the matmul through `at::mm`.

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
