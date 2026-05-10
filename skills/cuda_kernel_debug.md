---
name: cuda_kernel_debug
description: Systematic checklist for diagnosing CUDA kernel correctness failures — error magnitude triage, isolation strategy, and common bug patterns.
---

# CUDA Kernel Correctness Debug Guide

Use this when `write_candidate` returns `correctness_ok=False`. Work through
the checklist top-to-bottom; stop at the first confirmed root cause.

## Step 0 — Read the error magnitude first

| `max_abs_err` magnitude | What it almost certainly means |
|------------------------|-------------------------------|
| > 100 | Logic bug — wrong index, unguarded OOB read, missing `__syncthreads__` |
| 1e-1 – 100 | Accumulation bug — wrong sign, accumulated into wrong slot, partial sum |
| 1e-4 – 1e-1 | Precision / algorithm error — wrong formula, B vs B^T transposition |
| < 1e-4 | Fine (within rtol/atol=1e-4). `correctness_ok` should be True |

**If error > 1**: do NOT tune tile sizes. Fix the logic first.

## Step 1 — Isolate with the simplest possible kernel

Reduce to the smallest failing unit before adding any optimization back.

1. Strip the kernel to **plain `Y = W @ X`** (no LoRA term, no shared memory,
   one float per thread). If this is wrong, the problem is in basic indexing.
2. Once plain GEMM is correct, add the **low-rank term only** (no tiling).
3. Only after both are correct independently, reintroduce tiling / vectorization.

Use a small shape (d=64 or d=128) during isolation — faster compile-test cycles.

## Step 2 — Index formula checklist

For a row-major matrix `M[rows][cols]` stored contiguously:
```
element M[r][c] → M_ptr[r * cols + c]
```

Common mistakes:

| Bug | Symptom | Fix |
|-----|---------|-----|
| Row/col swapped (`c * rows + r`) | Large error, pattern looks transposed | Swap indices |
| Using `d` as stride when shape is not square | Large error on non-square shapes | Use actual col count |
| B accessed as `B[col * r + rank]` instead of `B[row * r + rank]` | LoRA term wrong | Check which dimension B is indexed on |
| B^T: should be `B[k * r + rank_idx]`, not `B[rank_idx * d + k]` | Low-rank term wrong | B is stored d×r; B^T access is `B[k][rank]` = `B_ptr[k * r + rank]` |

## Step 3 — Tensor stride / contiguity

PyTorch tensors passed from Python may **not** be contiguous (e.g., after
`.t()`, `.permute()`, slicing). A non-contiguous tensor has non-unit strides.

Symptoms: large error that is shape-dependent or appears only on certain inputs.

Fix options (choose one):
- In Python before `forward()` call: force contiguous with `.contiguous()`
- In the kernel: use actual strides instead of assuming `cols` as stride:
  ```cuda
  // Instead of:  ptr[row * d + col]
  // Use:         ptr[row * stride_row + col * stride_col]
  // where strides are passed as extra int args
  ```

The `write_candidate` quick-check uses freshly constructed tensors that are
contiguous, so stride bugs may not show there but will show in multi-shape
benchmark. When in doubt, force `.contiguous()` in the pybind forward wrapper.

## Step 4 — Boundary guards

Every thread that computes `row = blockIdx.y * BM + threadIdx.y` must guard
against out-of-bounds **before** any load or store:

```cuda
if (row >= d || col >= d) return;
```

Missing guard → threads read garbage from adjacent memory → large error.

For tiled kernels, also guard partial tiles at the edge of the matrix:
```cuda
// When loading smem tile, pad with 0 for out-of-bound positions
float val = (global_row < d && global_col < d) ? ptr[global_row * d + global_col] : 0.0f;
```

## Step 5 — Shared memory synchronization

Every `__syncthreads()` must appear in **all threads of the block**, not inside
a conditional branch. Missing syncs cause race conditions that produce
non-deterministic large errors.

Required sync points:
1. After loading tiles into smem → before computing with smem data.
2. After computing with smem data → before overwriting smem with the next tile.

```cuda
// Correct pattern
load_smem_W(smem_W, ...);
load_smem_X(smem_X, ...);
__syncthreads();          // ← all threads must reach this
compute_dot(acc, smem_W, smem_X);
__syncthreads();          // ← before next tile load overwrites smem
```

## Step 6 — Grid / block dimension formula

```cuda
// Correct ceil-division for grid:
dim3 grid(
    (d + BN - 1) / BN,   // cols
    (d + BM - 1) / BM    // rows
);
dim3 block(BN, BM);
```

Off-by-one here leaves the last tile uncomputed → zeros in output → large error
at matrix edges.

## Step 7 — Accumulator initialization

```cuda
float acc = 0.0f;       // correct
// NOT: float acc;      // uninitialized — UB, large random error
```

For the low-rank register array:
```cuda
float t_reg[16] = {0.0f};   // zero-initialize all 16 slots
```

## Quick isolation script (Python-side)

If `write_candidate` correctness is hard to read, add a temporary printf
inside the kernel for a single thread to print intermediate values:

```cuda
if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    printf("acc after k=%d: %f\\n", kk, acc);
}
```

Remove before final submission — printf in hot loops will tank performance.

## Escalation path

If all above checks pass and error is still > 1e-3:
1. Read `read_blackboard("hardware")` → check compute_capability. If cc < 7.0,
   some WMMA instructions are not available — avoid `<mma.h>`.
2. Check that the pybind forward signature matches the kernel parameter order
   exactly. A swapped `W` / `X` argument produces a transposed result.
3. Reduce to d=32 (fits in one warp) and use a single-block, single-warp kernel
   with printf to trace the full computation path.

## Special case: oracle precision mismatch (NOT a kernel logic bug)

**Symptom**: naive single-thread kernel and a complex tiled kernel have the
**exact same** `max_abs_err`. A pure PyTorch forward implementation gives
`max_abs_err=0.000`.

**Diagnosis**: This is a FP32 accumulation-order mismatch against the oracle,
not a logic bug. The oracle is computed by cuBLAS FP32, which uses a parallel
reduction tree (error ~ sqrt(d) × ε ≈ 7e-6 for d=3584). Any hand-written
sequential FP32 accumulation has error ~ d × ε × scale ≈ 2.8e-3 for d=3584 —
mathematically impossible to fix by changing kernel logic.

**Why WMMA does NOT fix this**: On sm86 (RTX 3090), `wmma::mma_sync` with
float32 fragments automatically uses TF32 (10-bit mantissa). The oracle is
computed with TF32 disabled. WMMA error vs oracle ≈ 0.1, which is worse than
sequential FP32.

**Why FP64 accumulation does NOT give speedup**: RTX 3090 FP64 throughput =
FP32 / 64 = 556 GFLOPS. Using FP64 for the K-loop makes the kernel ~60× slower
than cuBLAS, eliminating any speedup.

**Precision tradeoffs** — each path has a different cost:

| Approach | Error vs oracle | Throughput cost |
|----------|----------------|-----------------|
| Sequential FP32 k-loop | ~2–3e-3 (fails atol=1e-4) | none |
| FP64 accumulation, cast to FP32 | ~7e-6 (passes) | ~64× slower on RTX 3090 |
| `at::mm` (via `<torch/extension.h>`) | ~0, bit-identical | cuBLAS speed, no custom fusion |
| Tiled FP32 matching cuBLAS reduction order | ~1e-5 if correct | difficult to achieve |

Note: on sm86 (Ampere), WMMA mma_sync with float32 fragments uses TF32
internally — error vs FP32 oracle ≈ 0.1, worse than sequential FP32.

The right path depends on what you are optimizing for. Consider the speedup
headroom available from each approach before deciding.
