# GluonKit Agent 指南

使用 `gk` CLI 开发并调优与评测脚本 `eval_<name>.py` 同目录配对的 Gluon GPU kernel `<name>.py`（如 `examples/gemm/gemm.py`），在保证数值正确的前提下提升性能。

> 以本文件、[ARCHITECTURE.md](ARCHITECTURE.md) 与 `.venv/bin/gk --help` 为真值来源。

## Gluon API 知识库

Gluon 是 Triton 的底层 GPU 编程接口。每个 program 处理一个 tile，layout 描述元素在线程和寄存器上的分布。本仓库按 Triton 3.8.0 整理了两类资料，按当前任务选读即可。[上游 API 参考](https://triton-lang.org/main/gluon/api/index.html)

- **A. 通用核心功能**。写计算、归约、布局、访存、同步或编译启动逻辑时，读[通用核心功能与逐行求和示例](.knowledge/gluon-api/gluon-api-common.md)。
- **B. 平台扩展**。使用硬件专有能力时，到 `.knowledge/gluon-api/<平台>/` 查。NVIDIA 的资料在 [.knowledge/gluon-api/nvidia/](.knowledge/gluon-api/nvidia/)，从[NVIDIA 专用扩展](.knowledge/gluon-api/nvidia/nvidia.md)或 [sm86 FP16 GEMM 实例](.knowledge/gluon-api/nvidia/nvidia-sm86.md)进入。接入新平台时，在同级创建该平台目录并从这里链接。

遇到编译或运行错误时，按错误消息查[排错表](.knowledge/gluon-api/troubleshooting.md)。未收录的原语或版本差异，先用 `.venv/bin/gk api list|show [--full]` 查询，或沿本机安装包的导入路径核实定义，具体 dtype 和 layout 组合用目标设备校验。

## Skills

有独立触发场景和操作步骤的任务放在 `.agents/skills/`，`.claude/skills` 指向同一目录。新增 API、平台差异和报错案例时，更新知识库里对应的参考页。

- 编写、修改 Gluon kernel 或排查编译错误时，使用 [gluon-api](.agents/skills/gluon-api/SKILL.md) 选择参考页并核实接口。
- 做性能实验、比较结果或回退版本时，使用 [kernel-tuning](.agents/skills/kernel-tuning/SKILL.md)。

## 环境与后端注册

- CLI 入口：`.venv/bin/gk`
- 后端注册：[src/gluonkit/backends/registry.py](src/gluonkit/backends/registry.py) 静态注册了 `"nvidia"`（`NvidiaBackend`）与 `"metax"`（`MetaxBackend`），通过 `DEFAULT_BACKEND` 指定当前机器的确定后端（在沐曦机器上改为 `"metax"`）。
- Triton 缓存：`gk` 将 `TRITON_CACHE_DIR` 固定设为当前目录下的 `.gluonkit/triton_cache`（沙盒内可直接运行，无需额外配置）。

## CLI 命令与输出契约

除 `gk api` 输出纯文本外，所有命令固定向 `stdout` 输出单行 JSON：`{"result": "success" | "failed", "info": ...}`（失败时 `info` 包含完整 traceback 与捕获的输出）。

| 命令 | 用途 | 关键返回（`info`） |
|---|---|---|
| `gk probe` | 查询本机 GPU 架构、SM/CU 数、`warp_size`、共享内存上限、显存及软件版本 | `device_name`, `platform`, `arch`, `compiler_version`, `torch_version`, `warp_size`, `sm_count`, `smem_per_block_optin`, `vram_bytes` |
| `gk layout "<expr>" --shape 64,64 [--hardware]` | 离线查看线程/warp 与张量映射（按当前后端 `warp_size` 校验 `threads_per_warp` 乘积） | 格式化映射文本 |
| `gk bank-conflicts --register-layout "<expr>" --shared-layout "<expr>" --shape 64,32 --dtype float16` | 离线调用 `gl.bank_conflicts()`（无需 GPU；共享布局支持 `SwizzledSharedLayout`） | `bank_conflicts_count`（每个 wavefront 的额外访问次数，0 表示无冲突） |
| `gk check <eval.py>` | 触发 JIT 编译并校验与 `ref()` 数值一致（默认 `atol=rtol=1e-2`） | `passed`, `atol`, `rtol`, `max_abs_err`, `max_rel_err` |
| `gk baseline <eval.py> [--refresh]` | 若 `.gluonkit/baseline/baseline_<name>.json` 尚不存在则测量一次 `ref()` 并存入；已存在时校验 `eval_path`、`eval_source_hash` 与 `device_key` 后直接返回（`--refresh` 强制重测） | `file`, `eval_path`, `eval_source_hash`, `device_key`, `env`, `measurement_options`, `measured_at`, `baseline` |
| `gk bench <eval.py> [--note "..."] [--parent <id>] [--trace]` | 默认只对目标 kernel 预热 10 次、测量 100 次，读取已存基线算 `speedup` 并写入 `.gluonkit/` | `history_id`, `trace_path`, `kernel`, `metrics`, `baseline`, `peaks` |
| `gk api list [module]` / `gk api show [-m module] [--full] <symbol>...` | 离线查询本机 `triton.experimental.gluon` 的真实 API（无需 GPU）。**例外：输出纯文本，不是 JSON 信封**；`--full` 输出完整 docstring 与源码文件行号 | 纯文本 |
| `gk dump-info <eval.py>` | 运行一次，导出后端中间产物到 `.gluonkit/dump_info/<eval名>/<kernel名>/<hash>/` | `artifacts`（文件路径）, `summary`（`n_regs`, `n_spills`, `n_max_threads`, `shared_bytes`, `allocated_shared_bytes`, `num_warps`, `blocks_per_sm`, `occupancy_pct`, `limiting_factor`） |
| `gk history [<kernel_or_eval>] [--limit N]` | 查看已记录的 `bench` 实验摘要与资源占用。`.gluonkit/history.jsonl` 是 `gk bench` 生成的原始实验记录，可直接读取 | `total`, `records`（`id`, `parent_id`, `trace_path`, `note`, `avg_ms`, `median_ms`, `min_ms`, `p95_ms`, `std_ms`, `delta_vs_parent_pct`, `speedup`, `tflops`, `gbps`, `roofline`, `summary`） |
| `gk restore <id>` | 按历史快照恢复对应的 `kernel` 文件（支持唯一 ID 前缀） | `id`, `kernel_path`, `kernel_source_hash`, `previous_hash`, `changed` |

`kernel` 包含 `name` 和毫秒计的 `avg`、`median`、`min`、`p95`、`std`。`metrics` 包含 `speedup`、`tflops`、`gbps`、`roofline`（峰值利用率百分比，数据不足时为 `null`）、`bound_by`（`compute` / `memory` / `null`）和 `roofline_model`（`dual_roofline` / `memory_only` / `compute_only` / `null`）。

`peaks` 是本次 roofline 实际使用的峰值 `{"tflops", "gbps", "source"}`。标定由一次性脚本 `.venv/bin/python scripts/calibrate_peaks.py [--items ...]` 完成（不属于 `gk` 命令，约 10 秒）。所有 GPU 命令固定使用 `cuda:0`。

## 硬规则

1. **禁止篡改评测条件**：调优期间只修改算子文件（如 `examples/gemm/gemm.py`），禁止修改 `eval_*.py` 中的输入、参考实现 `ref`、容差 `atol`/`rtol`、`flops`/`bytes_moved` 或 `peaks`/`compute_mode`。调优期间也不要重跑 `scripts/calibrate_peaks.py`（会改变 roofline 分母，使前后轮不可比）。
2. **先 `baseline` / `check` 后 `bench`**：首次对某 `eval_<name>.py` 测速前，需先执行 `gk baseline <eval.py>` 生成 `.gluonkit/baseline/baseline_<name>.json`。每次修改算子后**必须先通过 `gk check`**，再执行 `gk bench`；未过 `check` 的测速结果视为无效。
3. **禁止盲猜 Gluon API**：写代码前查 `gluon-api` skill，遇到未覆盖的原语用 `.venv/bin/gk api list|show [--full]` 查询，或直接读 `.venv/lib/python3.12/site-packages/triton/experimental/gluon/` 源码核实签名。
4. **单变量迭代与版本保留**：每次修改只验证一个明确假设。通过校验且确认更快后保留。退化或失败时用 `gk restore <id>` 撤销本轮算子改动，保留工作区已有修改。`bench --parent <id>` 用于关联实验记录，可用 `gk history` 回顾多轮指标与资源变化。
5. **按算子输入约定处理边界**：任务明确固定 shape、dtype 和布局时，可据此特化，并用原评测的 `gk check` 校验数值。选定 tile 能整除的维度可省去对应边界 mask，有余数的维度仍须处理尾块。向量化或异步拷贝还需满足实际地址与步长的对齐要求。若接口承诺支持多种尺寸，则保留相应边界处理，按支持范围补测非整除尺寸或 `K < BK`。
