# 算子自动调优踩坑记录

记录 agent-for-MLS 在 LoRA matmul 自动调优过程中遇到的非显然问题，每条都附**症状 / 根因 / 修复 / 提交**，避免下次重蹈覆辙。

---

## 1. Oracle 与 Phase-2 评测的精度不对称（最大坑）

**症状**：candidate 一直卡在 `max_abs_err ≈ 7e-4` fail，agent 反复试 FP64 累加 / 禁 TF32 / WMMA 都无效。手写 `torch::mm` 全包候选才 pass。本地通过率极低，agent 浪费 4-6 个 iter 在伪精度问题上。

**根因**：[baseline.py:73-77](operator_opt_pipe/resources/baseline.py#L73-L77) 生成 oracle 时强制 `allow_tf32 = False`（strict FP32）；但 Phase-2 评测脚本里 `y_ref = W @ X + ...` 是**评测进程现场算**的，TF32 状态走 torch 默认。两边算 reference 的 cuBLAS 配置不对称，本地等于跟比 Phase-2 严的标准比较。一个调 `at::mm` 的正确 student 跟 strict-FP32 oracle 差 ~1e-3，跟现场 cuBLAS ref 几乎 bit-exact。

**修复**：[evaluation.py 的 `_build_quick_script` / `_build_bench_script`](operator_opt_pipe/resources/evaluation.py) 改为现场调 `ops.reference(inputs)`，与 student 在同一 subprocess 同一 cuBLAS state 下比较，与 Phase-2 完全同构。oracle 文件保留给 analyst 离线诊断用，不再当 correctness gate。

**提交**：`58a6044` — fix: 用现场重算 ref 替代 strict-FP32 oracle

**衍生教训**：评测语义对齐 > 数值"严"。本地 gate 跟 Phase-2 不一致时，伪阴性会成倍消耗 iter 预算。任何"我把标准设严一点更安全"的直觉，需要先确认严的方向跟最终评测方向一致。

---

## 2. `torch.allclose` 的 element-wise 语义被误读为 max_abs_err 阈值

**症状**：agent 看到 `max_abs_err: 7.32e-04`、`atol=1e-4` 就判定"失败"，进而追求 `max_abs_err = 0`，绕远路走 `at::mm` 全包路线、放弃所有 fusion 机会。

**根因**：`torch.allclose(y, y_ref, rtol=1e-4, atol=1e-4)` 对每个元素判 `|Δ| ≤ atol + rtol·|y_ref|`。LoRA matmul d=3584，Y 元素量级 ~√d ≈ 60，对大量级元素阈值 ≈ `1e-4 + 1e-4 × 60 = 6e-3`，远大于 atol。`max_abs_err` 仅 7e-4 完全可能是某个**小量级元素**触发 atol-bound，**绝大多数元素早就在容差内**。

**修复**：在 [skills/operators/lora_matmul_tuning.md](skills/operators/lora_matmul_tuning.md) 与 `cuda_kernel_debug.md` 加诊断说明，明确"`max_abs_err < atol` 不是必要条件"。tools 输出层面继续给 `max_abs_err` + `prev` delta，但语义上 agent 应理解 pass/fail 由 allclose 决定。

**衍生教训**：评测函数的语义文档要白纸黑字写出"对每个元素的判据"，不能只列阈值数字让 agent 自己脑补。

---

## 3. Quick check 编译开销吃掉一半预算

**症状**：单 iter 平均 ~110s，其中 60-90s 是 nvcc 冷编译。1800s budget 里 INITIAL_CANDIDATE 12 iter ≈ 22 min，TUNING_LOOP 直接没机会跑。

**根因（双重）**：
1. quick check 用 `extra_cuda_cflags=["-O3"]`，但 quick 只验证单 shape 正确性，`-O3` 优化纯属浪费——`-O0` 编译时间约 10-20s。
2. 没设 `TORCH_CUDA_ARCH_LIST`，subprocess 的 `cpp_extension.load` 默认按所有可见 GPU 的 arch 出 fatbin，每个 arch 一份 SASS，编译时间额外 +30-50s 且伴随 UserWarning。

**修复**：
- `_build_quick_script` 改用 `["-O0", "--threads", "4"]`；bench 仍 `-O3`（torch 按 flags 哈希做缓存，promote 后只 bench 一次会触发一次 `-O3` 重编，等价于"只为最终入选候选付一次优化代价"）。
- `_autodetect_env` 检测到 GPU arch 后 `os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")`。

**提交**：`c12ee02` — perf: 砍掉每 iter 编译开销 + 给 agent 加 delta 反馈

**衍生教训**：
- 可观测性必须区分**正确性验证**和**性能验证**——这两件事的编译策略应当不同。
- 评测前路径要走全套 `-O3 + 全 arch fatbin`，不代表搜索路径也得这么走。

---

## 4. Agent 重复测试同一失败假设

**症状**：candidate_001~004 max_abs_err 从 7.32e-4 → 6.87e-4 → 6.87e-4 → 6.87e-4，agent 每次都换一个"精度策略"（FP64 累加 / 禁 TF32 / WMMA…），但**误差几乎不变**。agent 没意识到「误差不变 = 不是精度问题，是 reduction 顺序问题」，要到 iter 12 才转向。

**根因**：
1. agent 看不到上一个 candidate 的 max_abs_err（要回滚自己的 context 对比），缺乏主动信号。
2. INITIAL_CANDIDATE 不写 `blackboard["history"]`（只 TUNING_LOOP 写），agent 也无法 `read_blackboard("history")` 回顾。
3. `cuda_kernel_debug.md` 里其实有"Special case: oracle precision mismatch"段落明确说明这个模式，但 agent 直到 iter 10 才主动 `read_skill`。

**修复**：[tools.py:WriteCandidateTool](operator_opt_pipe/tools.py) 输出加：
- `prev_max_abs_err: 7.32e-04 (delta: -4.6e-05, -6%)` 一行
- 连续 3 次 correctness 失败且 `|delta| < 5%` → 触发 `hint:` 行，明确建议切换 reduction-order 假设、停止调精度

**提交**：`c12ee02`

**衍生教训**：
- **被动 skill** 不如**主动 hint**。agent 不会在每次失败后都重新 `list_skills`，文档式知识必须在工具响应里也露面。
- 把"识别失败模式"的工作从 agent 转移给 deterministic code——规则简单（连续 N 次同向小 delta）就别让 LLM 推理。

---

## 5. Preflight smoke test 自身的 bug 把环境检测搞瘫

**症状（多次踩）**：preflight 报 `load_inline: FAILED — load_inline() missing 1 required positional argument: 'cpp_sources'`，提示"verify nvcc and g++ versions are compatible"，让人以为是工具链问题，浪费时间排查 nvcc / g++ / CUDA runtime 兼容性。

**根因**：smoke test 只传了 `cuda_sources`，但 `cpp_sources` 是 `load_inline()` 的必需位置参数。修了之后又踩第二个：`is_python_module=True`（默认）需要 `PYBIND11_MODULE` 导出 `PyInit_*` 符号，空 `cpp_sources` 编出来的 .so 没有，加载时 `dynamic module does not define module export function`。

**修复**：[cuda_executor.py:_autodetect_env](mls_agent/tools/cuda/cuda_executor.py)
```python
load_inline(
    name="autodetect_nop",
    cpp_sources="",          # 必需位置参数
    cuda_sources=["__global__ void _nop_() {}"],
    is_python_module=False,  # 走 ctypes.CDLL 加载，不要求 PyInit
    ...
)
```

**提交**：`a8ed5db` — fix: 修复编译链检查 / `5a71e74` — fix: preflight

**衍生教训**：
- preflight 是**降低后续诊断成本**的工具，自身有 bug 时反而**放大诊断成本**。任何 preflight 改动都应在干净 + 故障两类环境下各跑一次，确认能区分"我自己挂了"和"目标系统挂了"。
- 错误提示别把所有失败统一归因到一个最常见原因（如"nvcc/g++ 版本"），要把 smoke test 自身错误和环境错误分开报。

---

## 6. INITIAL_CANDIDATE 阶段没有时间预算保护

**症状**：optimizer_cold agent 一直在 ReAct 循环里调 `write_candidate`，编译 + 正确性反复迭代。一旦掉进上述 #1 / #4 的坑，单一 INITIAL_CANDIDATE 阶段可以吃光全部 1800s。

**根因**：`_run_initial_candidate` 把 agent 的 `AGENT_MAX_ITERATIONS`（默认 30）当唯一硬上限，没有"该阶段最多花 X% 总预算"的软约束。

**修复**：暂未实现。候选思路：
- 在 INITIAL_CANDIDATE 阶段为 agent 注入 system prompt 提醒「此阶段目标只是产出**任意**编译通过 + 正确的候选作为 floor，不需要追求性能」
- 或在 orchestrator 层面限制：`elapsed_in_stage > 0.4 * time_budget_s` 时强制让 agent 提交当前最佳尝试

**衍生教训**：流水线时间预算应当**按阶段分摊**，而不是只有全局 ceiling。INITIAL_CANDIDATE 烧掉 95% 预算等于让 TUNING_LOOP 不存在。

---

## 7. nvcc / ncu 工具链发现的"鸡生蛋"问题

**症状**：HARDWARE_PROFILE 阶段 agent 第一次调 `run_cuda_probe` 直接 `nvcc_infrastructure_failure`（实为 g++ 缺失），agent 看不懂错误码，反复 `probe_environment` / `profile_with_torch` / `apt-get install g++`，烧 ~10 个 iter 才把工具链装齐。

**根因**：
- `nvcc_infrastructure_failure` 错误码笼统，不区分"nvcc 没装" / "g++ 没装" / "权限不足"。
- preflight 当时还没生效（这次坑发生在 preflight 修复之前）。

**修复**：preflight 修好之后这个坑应该不会再现——启动期一次性 fail-fast 报清楚到底缺什么。但仍需观察实际效果。

**衍生教训**：环境探测应当**集中在启动期，由确定性代码做，并区分错误类型**；不要让 agent 在 LLM 循环里推理 "为什么这个 CUDA 编译失败"——这是 LLM 最不擅长的事，且每次推理至少耗 10-30s。

---

## 总览：iter 预算流向

| 时段 | 主要消耗 | 状态 |
|------|---------|------|
| HARDWARE_PROFILE | 工具链探测 / g++ 安装 | 已修（preflight） |
| INITIAL_CANDIDATE | 编译 + 伪精度 fail | 已修（quick `-O0` + 现场 ref + delta hint） |
| TUNING_LOOP | 真正的性能搜索 | 之前几乎没机会跑；现在期望能拿到主要预算 |

**决策原则**：每发现一个新 iter 浪费点，问三个问题：
1. 是否可以**确定性代码**搞定（preflight、编译 flag、history 写入）？能就别让 LLM 来。
2. 是否可以**一次性给 agent 信号**（hint 行、skill shortcut）替代多次试错？能就在工具响应里推。
3. 是否本地 / 评测**语义对齐**？不对齐就修对齐方向，不对就修阈值方向。
