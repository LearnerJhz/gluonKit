# GluonKit Agent 指南

使用 `gk` CLI 开发并调优 `KERNEL_PATH` 指定的 Gluon GPU kernel（如 `examples/gemm/gemm.py`），在保证数值正确的前提下提升性能。

> 以本文件与 `.venv/bin/gk --help` 为唯一真值来源；`README.md`、`architecture.md`、`plan.md` 可能包含旧版本描述，请勿参考其中的旧命令或字段。

## 环境

- CLI 入口：`.venv/bin/gk`（Python 3.12 + PyTorch 2.14.1+cu130 + Triton 3.8.0）
- Triton 缓存：`gk` 将 `TRITON_CACHE_DIR` 固定设为当前目录下的 `.gluonkit/triton_cache`（沙盒内可直接运行，无需额外配置）。

## CLI 命令与输出契约

所有命令固定向 `stdout` 输出单行 JSON：`{"result": "success" | "failed", "info": ...}`（退出码 `0` 为成功，`1` 为失败；失败时 `info` 包含完整 traceback 与捕获的输出）。

| 命令 | 用途 | 关键返回（`info`） |
|---|---|---|
| `gk probe` | 查询本机 GPU 架构、SM 数、opt-in 共享内存上限、版本 | `device_name`, `arch`, `sm_count`, `smem_per_block_optin` |
| `gk layout "<expr>" --shape 64,64 [--hardware]` | 离线查看线程/warp 与张量映射（当前支持 `BlockedLayout` / 嵌套 `SliceLayout`） | 格式化映射文本 |
| `gk check <eval.py>` | 触发 JIT 编译并校验与 `ref()` 数值一致（默认 `atol=rtol=1e-2`） | `null` 或失败详情 |
| `gk bench <eval.py> [--note "..."] [--parent <id>] [--trace] [--no-baseline]` | 预热 10 次、测量 100 次耗时分布，并将结果与源码快照写入 `.gluonkit/` | `history_id`, `device_time_ms`, `std_ms`, `min_ms`, `median_ms`, `baseline_time_ms`, `speedup_vs_baseline`, `tflops`, `gbps`, `roofline_pct`, `bound_by`, `kernels` |
| `gk dump-info <eval.py>` | 运行一次，导出 `ttgir`/`llir`/`ptx` 到 `.gluonkit/dump_info/<eval名>/<kernel名>/` | `artifacts`（文件路径）, `summary`（`n_regs`, `n_spills`, `n_max_threads`, `shared_bytes`, `allocated_shared_bytes`, `num_warps`, `blocks_per_sm`, `occupancy_pct`） |

## 硬规则

1. **禁止篡改评测条件**：调优期间只修改算子文件（如 `examples/gemm/gemm.py`），禁止修改 `eval_*.py` 中的输入、参考实现 `ref`、容差 `atol`/`rtol`、`flops`/`bytes_moved` 或 `peaks`。
2. **先 `check` 后 `bench`**：`gk bench` 当前不会自动跑正确性校验。每次修改算子后**必须先通过 `gk check`**，再执行 `gk bench`；未过 `check` 的测速结果视为无效。
3. **禁止盲猜 Gluon API**：写代码前查 `gluon-api` skill，遇到未覆盖的原语直接读 `.venv/lib/python3.12/site-packages/triton/experimental/gluon/` 源码核实签名。
4. **单变量迭代与版本保留**：每次修改只验证一个明确假设；确认更快且通过校验后保留（建议配合 `git` 或 `--note` 记录）；退化或失败版本及时回退。
5. **警惕单 shape 过拟合**：保留完整的边界 mask（处理不能被 tile 整除的行列与 K 维尾块），不要为了当前评测尺寸写死特化而破坏通用性。

## Skills

- `gluon-api`（`.agents/skills/gluon-api/SKILL.md`）：Triton 3.8.0 Gluon API 速查、layout 约束与编译排错表。
- `kernel-tuning`（`.agents/skills/kernel-tuning/SKILL.md`）：硬件瓶颈诊断、IR/资源解读、噪声阈值与调优决策树。
