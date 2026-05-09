---
name: operators/lora_matmul_tuning
description: Tuning navigation guide for LoRA-fused MATMUL — search space, result-driven optimization decisions, low-rank exploitation, and tile size reference.
---

# LoRA MATMUL — Tuning Navigation Guide

Companion to `operators/lora_matmul`. Use this during TUNING_LOOP to decide what
to build next given your current evaluation results.

> **Dependency constraint (official eval environment)**
> Allowed headers: `<torch/extension.h>`, `<cuda_runtime.h>`, `<mma.h>` (for
> tensor cores), standard C/C++ headers. **Do NOT use** CUTLASS, direct
> cuBLAS/cuDNN calls, Thrust, or any header not in the CUDA 12 toolkit. No
> extra source files beyond `optimized_lora.cu`. No `extra_ldflags`.

## Search Space

| Dimension | Options | Notes |
|-----------|---------|-------|
| Kernel architecture | `split` / `fused` | `split` = separate kernels for W@X and A@(B^T@X); `fused` = single kernel handles both paths sharing smem |
| Low-rank path | `sequential` / `smem_side` | `sequential` = B^T@X then A@T as two passes; `smem_side` = accumulate low-rank contribution in smem while streaming WX tiles, no intermediate DRAM write |
| Memory access | `naive` / `smem_tile` / `float4` | Float4 (128-bit) vectorized loads are almost always worth doing for W and X |
| Compute unit | `cuda_cores` / `tensor_cores` | Tensor cores (WMMA/mma.sync) need sm_80+; use TF32 accumulation for float32 inputs |
| Tile shape BM × BN × BK | runtime-tuned | Drives smem usage and occupancy; see reference table below |

## Navigation Rules

Apply the **first matching rule** top-to-bottom after each `evaluate_candidate` result:

| Signal | Diagnosis | Next step |
|--------|-----------|-----------|
| `compile_ok=False` | Syntax / API error | Fix PYBIND11_MODULE binding, `forward` signature, and `#include <torch/extension.h>` |
| `correctness_ok=False` | Algorithm bug | Check: (a) row/col index order in matmul loops, (b) B.T vs B in the low-rank term, (c) accumulation into output (+=) not overwrite (=) |
| speedup < 1.0, fused kernel | smem bank conflict or occupancy too low | Add `#pragma unroll`; check smem layout to avoid bank conflicts on the BK dimension; try a smaller BK=8 |
| speedup < 1.0, split/naive kernel | Memory access pattern suboptimal | Switch to float4 vectorized loads; ensure `blockDim.x` is a multiple of 32 |
| speedup 1.0–1.5 | Likely memory-bound (DRAM bottleneck) | Increase BM/BN for better data reuse; use float4; consider fusing both terms to eliminate one DRAM write/read round-trip |
| speedup 1.5–3.0 | Good. Likely compute-bound | Try tensor cores (WMMA API); unroll inner K loop; increase ILP via register blocking |
| speedup 3.0–8.0 | Excellent. Confirm measurement | Run `mode="confirm"`; if `variance_pct < 15%` submit `best_update` |
| speedup > 8.0 | Possible measurement artifact | Verify `torch.cuda.synchronize()` placement and Event recording; re-run confirm before claiming best |
| `variance_pct > 15%` on confirm | Timing unstable across d values | Try d-adaptive launch config (different tile per d); submit `strategy_guidance` and note config |
| per-d speedup spread > 2× | Tile config mismatched to some d | Add runtime dispatch: smaller BM/BN for d values that aren't multiples of the tile width |

## Low-rank Exploitation (r = 16, fixed)

The intermediate `T = B^T X` has shape (16, d) — tiny enough for on-chip storage.
This is the main handle for eliminating DRAM traffic in the low-rank term.

**Recommended pattern** — compute the low-rank contribution inside the WX tile loop:

```
for k_tile in range(K // BK):
    smem_X[BM, BK]  ← load from global X          // input tile
    smem_W[BM, BK]  ← load from global W          // base weight tile
    smem_B[16, BK]  ← load from global B          // all 16 rows of B slice

    acc_W  += smem_W  @ smem_X.T                   // base GEMM contribution
    t_tile  = smem_B  @ smem_X.T                   // low-rank T slice (16 × BM)
    // t_tile lives in registers (16 × BM ≤ 512 floats for BM=32)

load smem_A[BM, 16] from global A
acc_lr = smem_A @ t_full                           // (BM × 16) @ (16 × BN)
Y[BM, BN] = acc_W + acc_lr
```

**Register pressure**: for BM=64, t_tile needs 16×64 = 1024 floats (4 KB), which
exceeds the register budget. Use BM ≤ 32 when keeping t_tile in registers, or spill
t_tile to shared memory.

## Tile Size Reference

Shared memory per block: `smem_bytes = (BM + BN) × BK × 4`.
Keep `smem_bytes < L2_mb × 1024² / 4` to fit working set in L2.

| GPU L2 cache | Recommended BM × BN × BK | smem / block |
|--------------|--------------------------|--------------|
| 4 MB         | 64 × 64 × 16             | 64 KB        |
| 8 MB         | 128 × 64 × 16            | 96 KB        |
| 16 MB        | 128 × 128 × 16           | 128 KB       |
| 32 MB        | 128 × 128 × 32           | 256 KB       |

Check `cudaDeviceProp.sharedMemPerMultiprocessor` — most GPUs cap smem per block
at 48–100 KB. Use 128×128 tiles only if the device supports ≥ 128 KB per block.

## Anomaly Signals

- `max_abs_diff > 1e-2` on any d → do not submit `best_update`; fix the kernel first.
- speedup > 8× on quick mode → verify with `mode="confirm"` before claiming `best_update`.
- `variance_pct > 15%` → submit `strategy_guidance`, note tile config and d values affected.
- Compiler / driver errors → `compile_ok=False` will be set; analyze `compile_error` field.
