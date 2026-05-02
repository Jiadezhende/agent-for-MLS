---
name: operators/plain_matmul
description: Plain GEMM Y = W @ X — minimal smoke-test operator for the operator-spec abstraction (not part of regular pipeline runs).
shape_param: d
shape_param_range: [1024, 4096]
inputs:
  - {name: W, shape: [d, d], dtype: float32}
  - {name: X, shape: [d, d], dtype: float32}
output: {name: Y, shape: [d, d], dtype: float32}
reference_pytorch: "W @ X"
forward_args: [W, X]
---

# Operator: Plain MATMUL

Smoke-test operator used to validate that the operator-spec abstraction is not
LoRA-specific. The pipeline does not need to optimize this — only baseline
input/reference generation must work end-to-end.
