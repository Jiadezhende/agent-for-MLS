---
name: operators/lora_matmul
description: Operator contract for LoRA-fused MATMUL — formula, tensor specification, and bottleneck analysis.
shape_param: d
shape_param_range: [3584, 4608]
inputs:
  - {name: W, shape: [d, d], dtype: float32}
  - {name: X, shape: [d, d], dtype: float32}
  - {name: A, shape: [d, 16], dtype: float32}
  - {name: B, shape: [d, 16], dtype: float32}
output: {name: Y, shape: [d, d], dtype: float32}
reference_pytorch: "W @ X + A @ (B.T @ X)"
forward_args: [W, X, A, B]
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

## Bottleneck Analysis

**Arithmetic intensity of WX (d×d @ d×d):**

```
FLOPs = 2 × d³
Bytes = (d² + d² + d²) × 4   (read W, X; write Y)
AI    = 2d³ / (12d²) = d/6
```

For d = 4096: AI ≈ 683 FLOP/Byte — compute-bound on modern GPUs.

**Arithmetic intensity of A(B^T X) — the low-rank term:**

```
Step 1: T = B^T X    FLOPs = 2rd²,  Bytes ≈ (rd + d² + rd) × 4
Step 2: A T          FLOPs = 2rd²,  Bytes ≈ (dr + rd + d²) × 4
```

With r = 16 and d = 4096: AI ≈ 0.5 FLOP/Byte — **strongly memory-bound**.

Fusing WX and A(B^T X) into one kernel eliminates the intermediate DRAM round-trip
and is the highest-priority optimization direction.
