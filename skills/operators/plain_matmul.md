---
name: operators/plain_matmul
description: Plain GEMM Y = W @ X — minimal smoke-test operator for the operator-spec abstraction (not part of regular pipeline runs).
---

# Operator: Plain MATMUL

Smoke-test operator used to validate that the operator-spec abstraction is not
LoRA-specific. The pipeline does not need to optimize this — only baseline
input/reference generation must work end-to-end.
