# 算子评测流程技术文档

本文档描述 `agent-for-MLS` 内部的算子评测流程：候选 CUDA kernel 是怎么从一段源码走到「编译 → 正确性 → 性能 → 是否被 promote 为新最优」的。

**适用范围**：流水线内部的所有「跑一段 candidate」类操作，包括 `write_candidate` 的快速反馈、`benchmark_on_grid` 的多 shape 计时、以及 `BENCHMARK_BASELINE` 阶段的 PyTorch 基线测量。

**不在本文档范围**：Agent ReAct 循环、Orchestrator 状态机、硬件 profile。这些走 `operator_opt_pipe/orchestrator.py` 和 `operator_opt_pipe/agents.py`，与评测正交。

## 1. 设计目标与约束

评测流程要同时满足三个约束，**任何改动都不能破坏其中之一**：

| 约束 | 含义 | 实现位置 |
|---|---|---|
| **算子无关** | 流程代码不能出现 `W @ X` / `A @ B` 这类公式字面量 | 所有算子语义走 `OperatorOps` 接口 |
| **Phase-2 同构** | 本地评测出来的「最优」必须等于评测端给出的「最优」 | 数值/编译/语义三层对齐，见 §5 |
| **CUDA-context 安全** | 候选 kernel 的崩溃 / OOM 不能拖死整条流水线 | 候选编译 + 跑全部丢到子进程，见 §6 |

## 2. 核心抽象 `OperatorOps`

`OperatorOps`（[operator_opt_pipe/operators/_base.py](../operator_opt_pipe/operators/_base.py)）是评测流程与算子之间唯一的接缝。每个算子在 `operator_opt_pipe/operators/<name>.py` 里导出一个具体子类的单例 `OPS`，整条评测链只跟 `ops.*` 打交道。

接口分两组：

### 2.1 算子必须实现的 3 个 abstract 方法

| 方法 | 输入 | 输出 | 调用方 |
|---|---|---|---|
| `make_inputs(d, *, device, generator)` | shape 参数 d、cuda device、seeded generator | `dict[str, torch.Tensor]`，键名匹配 `contract.inputs` | `baseline.build_correctness_fixtures` |
| `reference(inputs)` | 输入张量 dict | 参考输出 `Y`（PyTorch 实现） | baseline 测时序、子进程现场算 `Y_ref` |
| `forward_call(mod, inputs)` | 已编译的 cpp_extension 模块 + 输入 | 候选 kernel 的输出 `Y` | 子进程对每个 candidate 调用 |

### 2.2 默认实现的辅助方法（一般不重写）

| 方法 | 作用 |
|---|---|
| `shape_id(d)` | 把 shape 参数转成稳定的文件名片段（如 `d3584`） |
| `save_inputs / load_inputs` | 用 `torch.save / torch.load` 把输入张量落盘 / 重载，按 `contract.inputs` 顺序遍历 |
| `reference_doc()` | 给 LLM 看的人类可读公式，写到 `baseline.json` 里 |

### 2.3 配套的 `OperatorContract`（数据）

[operator_opt_pipe/resources/contract.py](../operator_opt_pipe/resources/contract.py) 定义的 frozen dataclass，描述 shape / dtype / 容差 / shape 范围。评测器从中读：

- `inputs / output` — 决定要落多少个 `.pt`、形状怎么实例化
- `shape_param_range` — `BenchmarkSpec.for_contract(contract)` 用它生成 shape grid
- `rtol / atol` — 子进程脚本把这两个值拼进 `torch.allclose(...)`

> **关键不变量**：评测代码绝不直接读 `reference_pytorch` 字符串去 `eval()` —— 那串只是给 prompt 看的。所有"怎么算 reference"的逻辑都在 `ops.reference` 方法里。

## 3. 评测的三个入口

按调用频率与代价从低到高排序：

```
BENCHMARK_BASELINE 阶段 ─→ build_correctness_fixtures + measure_pytorch_latency
                            (一次性，只跑 PyTorch，不跑候选)

write_candidate 工具 ──────→ compile_and_check_quick
                            (每次 agent 提交源码，子进程，1 shape)

orchestrator 的 promote ──→ benchmark_on_grid
                            (候选通过 quick check 后，子进程，多 shape + 计时)
```

### 3.1 `build_correctness_fixtures` — 输入落盘

**位置**：[operator_opt_pipe/resources/baseline.py](../operator_opt_pipe/resources/baseline.py)

**职责**：对 `spec.shape_grid` 里每个 shape，用 `ops.make_inputs` 生成输入张量，落盘到 `inputs/`。**不**算参考输出。

**关键细节**：
- 使用单个 seeded `torch.Generator`，跨 shape 顺序推进，保证可复现
- 在 main process 运行（不开子进程），因为只调 `torch.randn` 和 `torch.save`
- 失败时抛 `RuntimeError("build_correctness_fixtures: cuda_unavailable")` —— Orchestrator 会捕获并记 `stage_failed`

**调用方**：`PipelineOrchestrator._run_benchmark_baseline`。

### 3.2 `measure_pytorch_latency` — 基线时延

**位置**：同上。

**职责**：对每个 shape 加载已落盘的输入，跑 `spec.warmup` 次 warmup + `spec.samples` 次 cudaEvent 计时，median/min/max 全部记录。

**输出**：`BaselineResult`，包含 `per_shape[sid].ms_median` —— 这是后续每个候选算 speedup 的**分母**。

**关键细节**：
- 在 main process 运行（同上，不需要子进程隔离）
- 用 `torch.cuda.Event(enable_timing=True)` + 每次迭代后 `synchronize()`，与 Phase-2 评测端同款
- 取 median 而不是 mean，抗 outlier
- `ms_median_overall` = 所有 shape median 的 median（一个粗略汇总值）

### 3.3 `compile_and_check_quick` — 单 shape 快速反馈

**位置**：[operator_opt_pipe/resources/evaluation.py](../operator_opt_pipe/resources/evaluation.py)

**职责**：给 `write_candidate` 工具用。子进程编译候选 → 在最小 shape 上验证 `torch.allclose` → 返回 `QuickEvalResult`。

**编译 flag**：`["-O0", "--threads", "4"]`（见 §5.2）

**超时**：600s（`profile_with_torch(..., timeout_s=600)`）

**返回结构** `QuickEvalResult`：
- `compile_ok: bool` + `compile_log: str`（编译失败时含 nvcc 错误）
- `correctness_ok: bool`
- `max_abs_err / rel_l2_err`（参考诊断指标，**不是 gate**）
- `shape_id`（这次跑的是哪个 shape）
- `diagnostics`（任何运行时异常字符串）

**用途分工**：返回值原样落到 `candidates/<cid>/correctness_quick.json`；`text` 字段给 LLM 阅读，`compile_log` 末 1500 字符直接拼进去让 LLM 修。

### 3.4 `benchmark_on_grid` — 多 shape 性能 + 正确性

**位置**：同上。

**职责**：候选通过 quick check 并被 `submit_candidate` 提交后，Orchestrator 调它跑全 shape grid。

**编译 flag**：`["-O3"]`（与 Phase-2 评测端一致，见 §5.2）

**超时**：900s

**单 shape 流程**（在子进程内串行做）：
1. `ops.load_inputs(...)` 加载该 shape 的输入
2. `ops.reference(inputs)` 现场算 `Y_ref`
3. `ops.forward_call(mod, inputs)` 跑候选 → `Y`
4. `torch.allclose(Y, Y_ref, rtol, atol)` 判正确性
5. **仅当正确**才进入计时阶段：warmup + samples 次 cudaEvent
6. 记录 `ms_median / ms_min / ms_max / max_abs_err / rel_l2_err`

**返回结构** `BenchmarkResult`：
- `compile_ok: bool` + `all_correct: bool`
- `per_shape[sid]` — 时序与误差
- `speedup_geomean / speedup_worst / speedup_best` — 跨 shape 汇总（`baseline_ms / candidate_ms`）
- `correctness_per_shape` — 逐 shape 的 pass/fail
- `correctness_ok` 属性 = `all_correct`
- `speedup` 属性 = `speedup_geomean`（promotion 比较用的单数字）

**Speedup 汇总用几何平均**而不是算术平均，理由：跨 shape 的 speedup 是比值，几何平均才与"复合性能改善"语义一致；同时对极端值更鲁棒（一个 shape 上 5× 不会掩盖另一个 shape 上 0.5× 的回归）。

**Agent 看不到 `BenchmarkResult`** —— 这是 Orchestrator 唯一能信赖的性能数据来源，promote 决策 100% 由 deterministic code 做。Agent 在下一轮通过 `read_blackboard("history")` 看到 promote 后的精简版 entry。

## 4. 子进程脚本模板

### 4.1 为什么必须开子进程

候选 kernel 是 LLM 写的，可能：
- segfault → 拖死整条 Python 进程
- 死循环 / launch 风暴 → 占满 CUDA context、阻塞后续候选
- 内存泄漏 → 累积下去耗尽 VRAM
- 改 cuBLAS / cuDNN 全局状态

子进程方案：每次评测开新 Python 进程跑候选，崩了由 `Executor.profile_with_torch` 捕获 timeout / non-zero exit，原 pipeline 继续。

### 4.2 模板结构

`_build_quick_script` / `_build_bench_script` 都是 f-string 模板，**结构固定**：

```python
import json, sys, traceback
sys.path.insert(0, r"<PROJECT_ROOT>")  # 让子进程能 import operator_opt_pipe

import torch
from torch.utils.cpp_extension import load
from operator_opt_pipe.operators import load_ops

# 常量从外层模板拼入：路径、shape、容差、cflags
CU_PATH = r"..."
INPUT_DIR = r"..."
SHORT_NAME = "lora_matmul"
RTOL, ATOL = 1e-4, 1e-4

result = {...}  # 默认全 False/None

if not torch.cuda.is_available():
    print("=== QUICK_RESULT ===")
    print(json.dumps(result))
    sys.exit(0)

ops = load_ops(SHORT_NAME)              # ← 子进程重新拿 OperatorOps

try:
    mod = load(name="cand_<id>", sources=[CU_PATH], extra_cuda_cflags=[...])
    result["compile_ok"] = True
except Exception as exc:
    result["compile_log"] = traceback.format_exception_only(...)
    print("=== QUICK_RESULT ==="); print(json.dumps(result)); sys.exit(0)

try:
    inputs = ops.load_inputs(INPUT_DIR, SHAPE_ID, device=device)
    Y_ref = ops.reference(inputs)
    Y     = ops.forward_call(mod, inputs)
    result["max_abs_err"] = float((Y - Y_ref).abs().max())
    result["correctness_ok"] = bool(torch.allclose(Y, Y_ref, rtol=RTOL, atol=ATOL))
except Exception as exc:
    result["diagnostics"]["error"] = ...

print("=== QUICK_RESULT ===")
print(json.dumps(result))
```

`bench` 模板多一段：跨 `SHAPE_GRID` 循环、warmup + samples 计时。

### 4.3 父子进程通信：marker + JSON

子进程在 stdout 打印一行 `=== QUICK_RESULT ===` 或 `=== BENCH_RESULT ===` 作为分隔符，紧跟一行 JSON。父进程通过 `parse_marked_output(job, marker)`（[resources/benchmark.py](../operator_opt_pipe/resources/benchmark.py)）抓出来反序列化。

为什么用 marker：子进程 stdout 还会混进 nvcc 的 warning / PyTorch 的 UserWarning，简单的 `json.loads(stdout)` 会爆。marker 划分"正常输出"与"结构化结果"。

## 5. Phase-2 评测对齐

本地评测必须与 Phase-2 官方评测端**字面一致**，否则本地"最优"和评测端"最优"不是一个东西。三层对齐：

### 5.1 数值对齐：现场算 reference（不预存 oracle）

Phase-2 官方评测脚本结构：

```python
y_student = module.forward(W, X, A, B)
y_ref     = reference_impl(W, X, A, B)   # ← 评测进程内现场算
torch.allclose(y_student, y_ref, rtol=1e-4, atol=1e-4)
```

本地子进程脚本结构（`_build_quick_script` / `_build_bench_script`）：

```python
mod   = load(...)                         # ← 编译候选
Y     = ops.forward_call(mod, inputs)     # mod.forward(W, X, A, B)
Y_ref = ops.reference(inputs)             # ← 子进程内现场算
torch.allclose(Y, Y_ref, rtol=RTOL, atol=ATOL)
```

**关键：reference 不预存到磁盘**。理由：
- TF32 状态、cuBLAS handle / algo 选择会随进程不同而漂移
- 预存 oracle 与候选用不同 cuBLAS state 算 → 有几率漂出 atol → 临界候选误判（heisenbug）
- 现场重算成本可忽略（LoRA d=3584 一次 ~ms 级）

**历史教训**：早期版本预存了 strict-FP32 oracle（baseline.py 把 `allow_tf32 = False`），与 Phase-2 默认 TF32 不对齐，导致正确候选大量误判。已彻底移除，详见 [operator_autotuning_pitfalls.md §1](operator_autotuning_pitfalls.md)。

### 5.2 编译对齐：bench 用 `-O3`，quick 可用 `-O0`

| 路径 | flags | 理由 |
|---|---|---|
| `compile_and_check_quick` | `["-O0", "--threads", "4"]` | 只验证逻辑，纯快速反馈，`-O0` 编译 ~10-20s 而 `-O3` 60-90s |
| `benchmark_on_grid` | `["-O3"]` | **对齐 Phase-2 评测端** |

`cpp_extension.load` 按 (sources, flags) 哈希做缓存，所以 quick 阶段编出的 `-O0` `.so` 不会污染 bench 阶段——bench 第一次跑会触发 `-O3` 重编一次，等价于"只为最终候选付一次优化代价"。

另一个隐含约束：父 main.py 启动时 `os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "<arch>")`（来自 `_autodetect_env`），避免子进程默认按所有可见 GPU 出 fatbin（多花 30-50s 编译）。

### 5.3 输出契约对齐：`./optimized_lora.cu`

Phase-2 评测端只读这一个文件。Orchestrator 在三个时刻把 `best/best.cu` 复制到这里：
1. `INITIAL_CANDIDATE` 阶段第一个 `compile_ok=True, all_correct=True` 的候选 —— 保底
2. 每次 `_promote_to_best` 接受了更快的候选
3. `FINALIZE` 收尾再 sync 一次（保险）

**约束**：`best/` 目录只能由 Orchestrator 写，agent 永远写到 `candidates/candidate_NNN/`。`submit_candidate` 工具会校验 `.cu` 文件存在再 terminate，不让 broken 候选进 `best/`。

## 6. Workspace 上的产物布局

`workspace/runs/<run_id>/` 下评测相关的部分：

```text
inputs/                              ← build_correctness_fixtures 写入
    W_d3584.pt, X_d3584.pt, ...      所有 contract.inputs × 所有 shape

baseline.json                        ← measure_pytorch_latency 写入
                                     {ms_median_overall, per_shape, reference_pytorch}

candidates/candidate_NNN/            ← agent 通过 write_candidate 写入
    candidate.cu                     候选源码
    compile.json                     {compile_ok, compile_log}
    correctness_quick.json           ← QuickEvalResult.to_dict()

benchmark/candidate_NNN.json         ← orchestrator 通过 benchmark_on_grid 写入
                                     ← BenchmarkResult.to_dict()

best/                                ★ orchestrator 独占
    best.cu                          当前最快通过的候选源码
    best_result.json                 {candidate_id, speedup, promoted_at}

build/                               cpp_extension.load 的 build_directory
                                     候选 .so 落在这里，run 结束后可全删
```

**显式：reference 输出 `Y` 不落盘**。早期版本曾在 `oracle/` 目录下保存 `Y_<sid>.pt`，已彻底删除（包括 `RunLayout.oracle_dir` 属性、`OperatorOps.save_oracle` 方法）。

## 7. 加新算子的步骤

完整流程：

### 7.1 写 `operator_opt_pipe/operators/<new_op>.py`

模仿 [lora_matmul.py](../operator_opt_pipe/operators/lora_matmul.py)，导出 `CONTRACT` + `OPS`：

```python
from operator_opt_pipe.operators._base import OperatorOps
from operator_opt_pipe.resources.contract import OperatorContract, TensorSpec

CONTRACT = OperatorContract(
    name="operators/<new_op>",
    inputs=(TensorSpec(name="X", shape=("d", "d"), dtype="float32"), ...),
    output=TensorSpec(name="Y", shape=("d", "d"), dtype="float32"),
    reference_pytorch="<RHS 表达式，仅给 LLM 看>",
    forward_args=("X", ...),
    shape_param="d",
    shape_param_range=(<lo>, <hi>),
    rtol=1e-4, atol=1e-4,
)

class <NewOp>Ops(OperatorOps):
    contract = CONTRACT
    def make_inputs(self, d, *, device, generator): ...
    def reference(self, inputs): ...
    def forward_call(self, mod, inputs): ...
    def reference_doc(self): return "<人类可读公式>"

OPS = <NewOp>Ops()
```

### 7.2 在 `operator_opt_pipe/operators/__init__.py` 注册 2 行

```python
from operator_opt_pipe.operators import <new_op> as _<new_op>
...
OPERATORS = {... , _<new_op>.CONTRACT.name.split("/")[-1]: _<new_op>.CONTRACT}
OPS_REGISTRY = {... , _<new_op>.OPS.short_name: _<new_op>.OPS}
```

### 7.3（可选）写 `skills/operators/<new_op>_tuning.md`

调优经验/方向指南。Agent 通过 `read_skill` 加载；**没有结构化字段，纯文本**。

### 7.4 跑

```bash
python main.py --operator <new_op> --time-budget 1800 --output ./<new_op>.cu
```

**评测代码 0 改动**。

### 7.5 当前已知扩展边界

[operator_evaluation_pipeline.md 评测器扩展性边界 — TODO 视实际需要再扩]：

- `BenchmarkSpec.shape_grid` 假设 shape 是「单整数列表」。多维 shape（attention、conv2d）需要重新设计 `shape_grid` 元素与 `ops.shape_id` 接口
- `extra_cuda_cflags` 在 `_build_*_script` 里硬编码。需要 `--use_fast_math` 等额外 flag 时要改 evaluation.py
- `samples / warmup / timeout` 是 LoRA 经验值，慢算子可能要放大

## 8. 关键文件索引

| 文件 | 角色 |
|---|---|
| [operator_opt_pipe/operators/_base.py](../operator_opt_pipe/operators/_base.py) | `OperatorOps` ABC + 默认 save/load_inputs |
| [operator_opt_pipe/operators/__init__.py](../operator_opt_pipe/operators/__init__.py) | `OPERATORS` / `OPS_REGISTRY` 字典 + `load_contract` / `load_ops` |
| [operator_opt_pipe/operators/lora_matmul.py](../operator_opt_pipe/operators/lora_matmul.py) | LoRA 算子（contract + ops） |
| [operator_opt_pipe/resources/contract.py](../operator_opt_pipe/resources/contract.py) | `OperatorContract`、`TensorSpec`、`shape_id` mini-DSL |
| [operator_opt_pipe/resources/benchmark.py](../operator_opt_pipe/resources/benchmark.py) | `BenchmarkSpec` + `parse_marked_output`（子进程结果解析） |
| [operator_opt_pipe/resources/baseline.py](../operator_opt_pipe/resources/baseline.py) | `build_correctness_fixtures` + `measure_pytorch_latency` |
| [operator_opt_pipe/resources/evaluation.py](../operator_opt_pipe/resources/evaluation.py) | `compile_and_check_quick` + `benchmark_on_grid` + 子进程模板 |
| [operator_opt_pipe/state.py](../operator_opt_pipe/state.py) | `RunLayout`：所有评测产物的路径定义 |
| [operator_opt_pipe/orchestrator.py](../operator_opt_pipe/orchestrator.py) | 调用上述函数、做 promote 决策、sync `./optimized_lora.cu` |
| [operator_opt_pipe/tools.py](../operator_opt_pipe/tools.py) | `WriteCandidateTool` 是 agent 触达 `compile_and_check_quick` 的唯一通道 |

## 9. 测试覆盖

```bash
pytest tests/operator_opt_pipe/    # 89 个测试，无需 GPU
pytest -m cuda                     # GPU 集成测试（需要 nvcc + CUDA + torch）
```

评测相关的关键测试：

- `test_resources.py::test_quick_script_uses_O0_for_fast_correctness_check` — quick 编译策略
- `test_resources.py::test_bench_script_keeps_O3_for_realistic_perf` — bench 编译策略
- `test_resources.py::test_quick_script_uses_online_reference` — 反向回归：oracle 路径不会偷偷回来
- `test_resources.py::test_bench_script_uses_online_reference` — 同上，bench 侧
- `test_state.py::test_runlayout_input_paths` — RunLayout 路径契约
- `test_tools.py::test_write_candidate_writes_file_and_returns_quick_result` — `WriteCandidateTool` → `compile_and_check_quick` 链路
- `test_orchestrator.py::*` — promote 决策与多 shape benchmark 集成（用 fake runner 注入）

## 10. 相关文档

- [operator_autotuning_pitfalls.md](operator_autotuning_pitfalls.md) — 评测流程演进过程中踩过的坑（含为何移除 oracle、`max_abs_err` 不是 gate 等关键背景）
- [phase2_requirements.md](phase2_requirements.md) — Phase-2 官方评测约束
- [../CLAUDE.md](../CLAUDE.md) — 项目总览，含完整 stage 状态机
