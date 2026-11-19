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
| `gk probe` | 查询本机 GPU 架构、SM 数、opt-in 共享内存上限、版本，以及本设备已标定的峰值 | `device_name`, `arch`, `sm_count`, `smem_per_block_optin`, `peaks`（`values` 为 `null` 表示未标定） |
| `gk layout "<expr>" --shape 64,64 [--hardware]` | 离线查看线程/warp 与张量映射（当前支持 `BlockedLayout` / 嵌套 `SliceLayout`） | 格式化映射文本 |
| `gk bank-conflicts --register-layout "<expr>" --shared-layout "<expr>" --shape 64,32 --dtype float16` | 离线调用 `gl.bank_conflicts()`（无需 GPU；共享布局支持 `SwizzledSharedLayout`） | `bank_conflicts_count`（每个 wavefront 的额外访问次数，0 表示无冲突） |
| `gk check <eval.py>` | 触发 JIT 编译并校验与 `ref()` 数值一致（默认 `atol=rtol=1e-2`） | `null` 或失败详情 |
| `gk bench <eval.py> [--note "..."] [--parent <id>] [--trace] [--no-baseline]` | 预热 10 次、测量 100 次耗时分布，并将结果与源码快照写入 `.gluonkit/` | `history_id`, `trace_path`, `kernel`, `metrics`, `baseline`, `peaks` |
| `gk dump-info <eval.py>` | 运行一次，导出 `ttgir`/`llir`/`ptx` 到 `.gluonkit/dump_info/<eval名>/<kernel名>/` | `artifacts`（文件路径）, `summary`（`n_regs`, `n_spills`, `n_max_threads`, `shared_bytes`, `allocated_shared_bytes`, `num_warps`, `blocks_per_sm`, `occupancy_pct`） |
| `gk history [<kernel_or_eval>] [--limit N]` | 查看已记录的 `bench` 实验摘要与资源占用 | `total`, `records`（`id`, `parent_id`, `trace_path`, `note`, `avg_ms`, `speedup`, `tflops`, `gbps`, `roofline`, `summary`） |
| `gk restore <id>` | 按历史快照恢复对应的 `kernel` 文件（支持唯一 ID 前缀） | `id`, `kernel_path`, `kernel_source_hash`, `previous_hash`, `changed` |

`kernel` 包含 `name` 和毫秒计的 `avg`、`median`、`min`、`std`。`metrics` 包含 `speedup`、`tflops`、`gbps`、`roofline`（峰值利用率百分比，数据不足时为 `null`）和 `bound_by`（`compute` / `memory` / `null`）。`baseline` 是由后端填充的字典。NVIDIA 后端只返回 `ref()` 实测的 `kernel_name` 和 `avg_ms`。启用 `--trace` 时只导出待测算子的 `trace.json`，路径见顶层 `trace_path`。关闭基准或缺少 `ref` 时，`baseline` 为 `null`。

`peaks` 是本次 roofline 实际使用的峰值 `{"tflops", "gbps", "source"}`：评测脚本显式给出的 `peaks` 字段优先（`source` 为 `task`），缺省字段取 `.gluonkit/peaks.json` 中当前设备的标定值（`source` 为 `calibrated:<项>`）；计算峰值由评测的 `"compute_mode"`（如 `"fp16"`、`"bf16"`、`"tf32"`、`"fp32"`，对应标定项 `tflops_<mode>`）选择；未声明 `compute_mode` 时 `tflops` 峰值为 `null`（不猜测），`roofline` 随之为 `null`（仅声明 `bytes_moved`、未声明 `flops` 的纯访存算子不受影响，按 `gbps` 单侧计算）。未标定且评测未给出时同样为 `null`。标定由一次性脚本 `.venv/bin/python scripts/calibrate_peaks.py [--items ...]` 完成（不属于 `gk` 命令，约 10 秒）。所有 GPU 命令固定使用 `cuda:0`。

## 硬规则

1. **禁止篡改评测条件**：调优期间只修改算子文件（如 `examples/gemm/gemm.py`），禁止修改 `eval_*.py` 中的输入、参考实现 `ref`、容差 `atol`/`rtol`、`flops`/`bytes_moved` 或 `peaks`/`compute_mode`。调优期间也不要重跑 `scripts/calibrate_peaks.py`（会改变 roofline 分母，使前后轮不可比）。
2. **先 `check` 后 `bench`**：`gk bench` 当前不会自动跑正确性校验。每次修改算子后**必须先通过 `gk check`**，再执行 `gk bench`；未过 `check` 的测速结果视为无效。
3. **禁止盲猜 Gluon API**：写代码前查 `gluon-api` skill，遇到未覆盖的原语直接读 `.venv/lib/python3.12/site-packages/triton/experimental/gluon/` 源码核实签名。
4. **单变量迭代与版本保留**：每次修改只验证一个明确假设。通过校验且确认更快后保留。退化或失败时用 `gk restore <id>` 撤销本轮算子改动，保留工作区已有修改。`bench --parent <id>` 用于关联实验记录，可用 `gk history` 回顾多轮指标与资源变化。
5. **按算子输入约定处理边界**：任务明确固定 shape、dtype 和布局时，可据此特化，并用原评测的 `gk check` 校验数值。选定 tile 能整除的维度可省去对应边界 mask，有余数的维度仍须处理尾块。向量化或异步拷贝还需满足实际地址与步长的对齐要求。若接口承诺支持多种尺寸，则保留相应边界处理，按支持范围补测非整除尺寸或 `K < BK`。

## Skills

仓库维护的 skill 位于 `.agents/skills/`，`.claude/skills` 是指向同一目录的符号链接。

- `gluon-api`（`.agents/skills/gluon-api/SKILL.md`）：Triton 3.8.0 Gluon API 速查、layout 约束与编译排错表。
- `kernel-tuning`（`.agents/skills/kernel-tuning/SKILL.md`）：硬件瓶颈诊断、IR/资源解读、噪声阈值与调优决策树。
