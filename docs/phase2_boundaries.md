# Phase 2 边界说明：optimized_lora.cu 与 Agent 的职责

本文档按项目要求明确 Phase 2 的实现边界。核心原则：最终被官方 benchmark 的对象是 `./optimized_lora.cu`，它必须是单文件、自包含、可由官方 harness 直接编译的 CUDA extension。

## 1. 最终文件边界

最终提交根目录必须包含：

```text
run.sh
optimized_lora.cu
```

官方流程是：

```bash
bash run.sh
```

之后读取：

```text
./optimized_lora.cu
```

因此：

- `run.sh` 负责启动 agent 优化流程。
- agent 负责生成、测试、比较候选 CUDA 实现。
- `optimized_lora.cu` 必须始终是当前 best valid implementation。
- 不能等最后才第一次写出 `optimized_lora.cu`。

最终 benchmark 只依赖 `optimized_lora.cu`。它不能依赖额外的 `.cu`、`.cuh`、`.h`、`.cpp` 或其他提交侧源文件。

## 2. optimized_lora.cu 必须提供的接口

`optimized_lora.cu` 必须导出：

```cpp
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B);
```

并通过：

```cpp
PYBIND11_MODULE(...)
```

暴露 `forward`，使官方 harness 可以调用：

```python
module.forward(W, X, A, B)
```

`optimized_lora.cu` 不负责读取 `.pt` 文件，也不负责处理 hidden input 路径。hidden tensors 由官方 Python harness 用 `torch.load` 加载后传入 `forward`。

## 3. optimized_lora.cu 的允许依赖

项目明确允许：

- standard CUDA headers
- standard C/C++ library headers
- standard PyTorch extension headers already available in the system environment

据此，默认可使用：

```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>
```

以及标准 C/C++ 头文件。

PyTorch extension headers 的角色应限于：

- 接收 `torch::Tensor` 参数。
- 检查 tensor shape、dtype、device、contiguous。
- 分配输出 tensor，例如 `torch::empty_like` 或 `torch::empty`。
- 获取 raw pointer，例如 `data_ptr<float>()`。
- 通过 pybind 暴露 `forward`。

## 4. 不应放进 optimized_lora.cu 的内容

按项目意图，最终 `optimized_lora.cu` 应是 generated CUDA implementation，而不是 PyTorch op wrapper。因此默认不应在最终计算路径中调用：

```cpp
at::matmul(...)
torch::matmul(...)
at::mm(...)
torch::mm(...)
```

也不应在 `optimized_lora.cu` 内部调用 Python、读取 `.pt`、启动 agent、执行 benchmark、访问 hidden input 目录。

cuBLAS/cuBLASLt 也不应作为默认最终路线，除非 staff 明确确认允许。原因：

- 题面没有显式列出 cuBLAS/cuBLASLt。
- 官方 harness 示例只传 `extra_cuda_cflags=["-O3"]`，没有显式 `extra_ldflags=["-lcublas"]`。
- 直接依赖 cuBLAS 可能在链接阶段失败，且可能偏离“生成 CUDA 实现”的评分意图。

因此默认实现边界是：`optimized_lora.cu` 使用自定义 CUDA kernels 完成核心计算。

## 5. Agent 可以使用 PyTorch 做什么

项目 practical advice 明确鼓励：

- compile and test candidates automatically
- verify correctness against a local PyTorch reference
- benchmark repeatedly after warmup

所以 agent 运行期间可以使用 PyTorch 作为外部评测工具：

```python
Y_ref = W @ X + A @ (B.transpose(0, 1).contiguous() @ X)
```

PyTorch 在 agent 中的合法职责：

- 生成 synthetic tensors。
- 编译 candidate extension。
- 计算 reference output。
- 检查 `torch.allclose(..., rtol=1e-4, atol=1e-4)`。
- benchmark PyTorch reference runtime。
- benchmark candidate runtime。
- 计算 speedup。

但 PyTorch reference 是 agent 的评测基线，不是最终 `optimized_lora.cu` 的计算实现。

## 6. 算子与形状边界

最终实现必须计算：

```text
Y = W X + A(B^T X)
```

输入约束：

```text
W: d x d
X: d x d
A: d x 16
B: d x 16
d in [3584, 4608]
dtype: float32
```

低秩维度固定为 `16`，但 `d` 不固定。实现可以根据 runtime `d` 做分支或选择不同 kernel 参数，但不能只支持单个 hardcoded shape。

正确性对齐官方 PyTorch reference：

```python
torch.allclose(Y_student, Y_ref, rtol=1e-4, atol=1e-4)
```

正确性不通过则没有性能分。

## 7. Agentic 行为边界

禁止：

- 只提交静态 kernel。
- 在 agent 中硬编码完整最终 `optimized_lora.cu`，运行时直接 dump。
- 让最终 measured implementation 依赖 `optimized_lora.cu` 之外的源文件。
- 破坏 `bash run.sh` 和 `./optimized_lora.cu` 的官方 I/O contract。

应实现：

- 生成多个 CUDA candidate。
- 自动编译、测试、benchmark。
- 比较候选结果。
- 将通过 correctness 且更快的候选提升为 `optimized_lora.cu`。
- 记录 candidate history、benchmark records 和优化决策。

一句话边界：agent 可以用 PyTorch 来评测；最终 `optimized_lora.cu` 应用 PyTorch extension 做接口，用自定义 CUDA kernels 做计算。
