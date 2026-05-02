# Phase 2 任务需求提炼：Agentic Optimization of a LoRA Operator

本文档整理 Phase 2 的核心任务、调优对象、性能要求，以及当前需要向课程 staff 明确的问题。重点是把项目从 Phase 1 的 GPU 参数测量 agent，转向 Phase 2 的 CUDA operator 优化 agent。

## 1. 核心任务

Phase 2 要提交的不是一个单独手写 CUDA kernel，而是一个能自动优化 LoRA-style operator 的 agent 系统。

官方评测会进入提交根目录并执行：

```bash
bash run.sh
```

随后读取同一目录下的：

```text
./optimized_lora.cu
```

因此，`run.sh` 应负责启动优化流程；agent 的核心职责是持续维护当前最优、可编译、正确的 `optimized_lora.cu`。

运行过程中必须尽早生成一份可用实现，不能等到 30 分钟预算结束前才第一次写出 `optimized_lora.cu`。

## 2. 调优对象

目标算子为：

```text
Y = W X + A(B^T X)
```

张量形状和类型：

```text
W: d x d
X: d x d
A: d x 16
B: d x 16
Y: d x d
d in [3584, 4608]
dtype: float32
```

低秩维度固定为 `r = 16`。隐藏测试会选择多个 `d`，因此实现不能只针对单一尺寸硬编码。

从计算量看，`W @ X` 是绝对主成本：

```text
W @ X:              O(d^3)
B^T @ X:            O(16 * d^2)
A @ (B^T @ X):      O(16 * d^2)
```

以 `d = 4096` 粗略估算，`W @ X` 约为 137 GFLOPs，而整个 LoRA correction 约为 1.07 GFLOPs，不到主 GEMM 的 1%。所以优化重点不是简单把全部逻辑写进一个巨大自定义 GEMM，而是：

- 保持 `W @ X` 的高性能，不能明显输给 PyTorch/cuBLAS 的大 GEMM。
- 低成本计算并加入 LoRA correction。
- 减少 `d x d` 临时矩阵、global memory 往返和 kernel launch。
- 对 `d in [3584, 4608]` 的多个尺寸保持稳定正确和稳定性能。

## 3. 正确性与性能要求

正确性是硬门槛。官方参考实现等价于：

```python
Y_ref = W @ X + A @ (B.transpose(0, 1).contiguous() @ X)
```

检查条件：

```python
torch.allclose(Y_student, Y_ref, rtol=1e-4, atol=1e-4)
```

如果 correctness 不通过，性能分为 0。

性能分数基于 speedup：

```text
speedup = PyTorch reference median runtime / optimized_lora.cu median runtime
```

benchmark 使用 CUDA event、warmup 和多次重复后的 median latency。

因此本地 agent 搜索时需要同时测：

- 候选实现是否可编译。
- 候选输出是否通过 PyTorch reference correctness。
- 候选 runtime。
- PyTorch reference runtime。
- 候选相对 PyTorch reference 的 speedup。

## 4. Baseline 的角色

建议同时维护两类 baseline：

### PyTorch reference baseline

PyTorch reference 应作为 correctness oracle 和 speedup denominator。官方环境文档明确包含：

```text
Python 3.10.12
PyTorch 2.3.0a0+6ddf5cf85e.nv24.04
CUDA 12.4
GCC 11.4.0
```

官方 harness 也依赖 `torch` 和 `torch.utils.cpp_extension.load`，所以 agent 内部使用 PyTorch 生成 synthetic inputs、编译候选、验证正确性、benchmark reference 是合理的。

### CUDA baseline candidate

agent 应自己生成或维护一份保守 CUDA baseline candidate，作为第一份可提交实现和后续搜索起点。它的目标是：

- 保证 `optimized_lora.cu` 一开始就存在且可编译。
- 作为后续 candidate mutation 的 anchor。
- 在优化失败、timeout 或候选不正确时提供 fallback。

需要避免把最终优化答案伪装成固定大字符串直接 dump 出来。baseline 应是朴素、稳定、通用的起点，而不是绕过 agentic optimization 的隐藏最终答案。

## 5. Agent 应具备的优化循环

有效的 Phase 2 agent 至少应实现以下闭环：

```text
generate candidate CUDA implementation
compile candidate
test correctness on synthetic d values
benchmark candidate and PyTorch reference
compare against current best
promote better valid candidate to optimized_lora.cu
record candidate history and decisions
repeat within 30-minute budget
```

推荐 synthetic 测试覆盖多个代表尺寸，例如：

```text
3584, 3840, 4096, 4352, 4608
```

最终 `optimized_lora.cu` 必须是单文件、自包含、可由官方 harness 直接编译，并导出：

```cpp
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B);
```

同时必须通过 `PYBIND11_MODULE` 暴露 `forward`。

## 6. 需要答疑确认的问题

以下问题会影响 baseline 和候选搜索空间，需要尽早确认：

1. `optimized_lora.cu` 的 `forward` 内部是否允许调用 ATen/PyTorch ops，例如 `at::matmul` 或 `torch::matmul`？
2. `optimized_lora.cu` 是否允许直接使用 cuBLAS 或 cuBLASLt？它们是否属于允许依赖？
3. 官方 benchmark 是否每个 hidden `d` 单独编译，还是一个编译好的 module 连续测试多个 `d`？
4. hidden tensors 是否保证为 CUDA contiguous float32 tensors？
5. `run.sh` 执行期间是否允许 agent 写中间候选文件、日志和 benchmark records？
6. 30 分钟预算是否包含依赖安装、extension compile 和 agent search 的全部时间？
7. agent methodology 评分是否会检查日志文件或 candidate history？如果会，期望的文件名或格式是什么？

## 7. 当前设计结论

Phase 2 的关键不是处理 `.pt` 输入输出本身。`.pt` 输入加载和 hidden harness 属于官方评测流程，agent 不应把 hidden path 或官方 I/O contract 写死到核心逻辑里。

我们的核心工程目标应是：

- `run.sh` 启动 agent。
- agent 使用 synthetic tensors 在公开尺寸范围内搜索实现。
- agent 使用 PyTorch reference 作为正确性和性能基线。
- agent 始终维护一个最新可用的 `optimized_lora.cu`。
- 最终性能优化围绕“大 GEMM + 低秩修正”展开，而不是盲目重写整个 GEMM。
