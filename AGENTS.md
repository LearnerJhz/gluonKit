# GluonKit Agent 指南

使用 `gk` CLI 开发并调优与评测脚本 `eval_<name>.py` 同目录配对的 Gluon GPU kernel `<name>.py`（如 `examples/gemm/gemm.py`），在保证数值正确的前提下提升性能。

> 以本文件、[ARCHITECTURE.md](ARCHITECTURE.md) 与 `.venv/bin/gk --help` 为真值来源。

## 一些高性能算子实现

仓库根目录下的 `bench-*/` 是维护者的历史开发记录，里面有一些算子的高性能实现，必要时可以参考。

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
- 沙箱与 `probe` 探测：部分环境下在沙箱内运行 `gk probe` 获取 GPU 硬件信息可能会因设备或驱动访问受限而报错。若确认报错由沙箱限制引起，可申请在沙箱外执行 `.venv/bin/gk probe`，或自行通过命令获取所需信息（如 `nvidia-smi` / 对应平台设备查询命令，以及用 `.venv/bin/python` 查询 `torch.__version__`、`triton.__version__` 与 `torch.cuda.get_device_properties(0)`）。

## CLI 命令与输出契约

除 `gk api` 输出纯文本外，所有命令固定向 `stdout` 输出单行 JSON：`{"result": "success" | "failed", "info": ...}`（失败时 `info` 包含完整 traceback 与捕获的输出）。

| 命令 | 用途 | 关键返回（`info`） |
|---|---|---|
| `gk probe` | 查询本机 GPU 架构、SM/CU 数、`warp_size`、共享内存上限、显存及软件版本（若在沙箱中报错，可改为沙箱外执行或自行通过命令获取） | `device_name`, `platform`, `arch`, `compiler_version`, `torch_version`, `warp_size`, `sm_count`, `smem_per_block_optin`, `vram_bytes` |
| `gk layout "<expr>" --shape 64,64 [--hardware]` | 离线查看线程/warp 与张量映射（按当前后端 `warp_size` 校验 `threads_per_warp` 乘积） | 格式化映射文本 |
| `gk bank-conflicts --register-layout "<expr>" --shared-layout "<expr>" --shape 64,32 --dtype float16` | 离线调用 `gl.bank_conflicts()`（无需 GPU；共享布局支持 `SwizzledSharedLayout`） | `bank_conflicts_count`（每个 wavefront 的额外访问次数，0 表示无冲突） |
| `gk check <eval.py>` | 触发 JIT 编译并校验与 `ref()` 数值一致（默认 `atol=rtol=1e-2`） | `passed`, `atol`, `rtol`, `max_abs_err`, `max_rel_err` |
| `gk baseline <eval.py>` | 采取固定基线原则：在测速与调优算子之前先跑一次，测量 `ref()` 并存入 `.gluonkit/baseline/baseline_<name>.json`；已存在时直接返回不重测，后续所有 `gk bench` 均在此固定基线之上跑并对比。**注意：基线仅按文件名 `<name>` 索引（不校验路径与内容哈希），不同评测任务须使用不同 `<name>`；若更换或修改了评测脚本，须先手动删除对应 `baseline_<name>.json` 再重跑。** 测量口径与 `gk bench` 一致：默认预热 10 次、每次迭代前清空 L2 Cache（`flush_l2=true`）后测量 100 次 | `file`, `device_key`, `measured_at`, `baseline` |
| `gk bench <eval.py> [--note "..."] [--parent <id>] [--trace]` | 与 `gk baseline` 同口径（默认预热 10 次、每次迭代前清空 L2 Cache，`flush_l2=true`，测量 100 次），读取已存基线算 `speedup` 并写入 `.gluonkit/` | `history_id`, `trace_path`, `kernel`, `metrics`, `baseline`, `peaks` |
| `gk api list [module]` / `gk api show [-m module] [--full] <symbol>...` | 离线查询本机 `triton.experimental.gluon` 的真实 API（无需 GPU）。**例外：输出纯文本，不是 JSON 信封**；`--full` 输出完整 docstring 与源码文件行号 | 纯文本 |
| `gk dump-info <eval.py>` | 运行一次，导出后端中间产物到 `.gluonkit/dump_info/<eval名>/<kernel名>/<hash>/` | `artifacts`（文件路径）, `summary`（`n_regs`, `n_spills`, `n_max_threads`, `shared_bytes`, `allocated_shared_bytes`, `num_warps`, `blocks_per_sm`, `occupancy_pct`, `limiting_factor`） |
| `gk history [<kernel_or_eval>] [--limit N]` | 查看已记录的 `bench` 实验摘要与资源占用。`.gluonkit/history.jsonl` 是 `gk bench` 生成的原始实验记录，可直接读取 | `total`, `records`（`id`, `parent_id`, `trace_path`, `note`, `avg_ms`, `median_ms`, `min_ms`, `p95_ms`, `std_ms`, `delta_vs_parent_pct`, `speedup`, `tflops`, `gbps`, `roofline`, `bound_by`, `roofline_model`, `summary`） |
| `gk restore <id>` | 按历史快照恢复对应的 `kernel` 文件（支持唯一 ID 前缀） | `id`, `kernel_path`, `kernel_source_hash`, `previous_hash`, `changed` |

`kernel` 包含 `name` 和毫秒计的 `avg`、`median`、`min`、`p95`、`std`。`metrics` 包含 `speedup`、`tflops`、`gbps`、`roofline`（相对硬件理论峰值的利用率百分比，不截断；若工作集命中 L2 Cache 可能大于 100%，数据不足时为 `null`）、`bound_by`（`compute` / `memory` / `null`）和 `roofline_model`（`dual_roofline` / `memory_only` / `compute_only` / `null`）。

`peaks` 是本次从 `.gluonkit/peaks.json` 读取并用于 roofline 计算的硬件理论峰值（`{"tflops": ..., "gbps": ...}`）。环境初始化时运行一次 `.venv/bin/python scripts/init_peaks.py` 即可按硬件规格计算写入（带宽：`mem_clock × 2 × bus_width / 8`；算力：`sm_count × flops_per_sm_per_clock × max_sm_clock`，不属于 `gk` 命令），也支持直接手写 `.gluonkit/peaks.json`（如沐曦 C500 或锁频场景）。所有 GPU 命令固定使用 `cuda:0`。

## 评测脚本契约（`eval_<name>.py`）

评测脚本须与待测算子 `<name>.py` 同目录配对、实际导入 `<name>.py`，并定义 `prepare(device)` 返回任务字典（不同评测任务的 `<name>` 须保持唯一，避免共用同一 `.gluonkit/baseline/baseline_<name>.json`；字段逻辑见 [checker.py](src/gluonkit/core/checker.py)、[roofline.py](src/gluonkit/core/roofline.py) 与 [examples/gemm/eval_gemm.py](examples/gemm/eval_gemm.py)）：

| 字段 | 是否必填 | 作用与写法建议 |
|---|---|---|
| `"run"` | **必填** | 无参 `lambda`（如 `lambda: gemm(a, b)`）。会在 `check` / `dump-info` / `bench` 里被反复调用，里面只放算子调用，不要放构造数据、`.to(device)` 等操作，也不要原地修改输入张量。 |
| `"ref"` | `check` / `baseline` 必填 | 无参 `lambda`（如 `lambda: torch.matmul(a, b)`），作为数值真值和 `speedup` 基准。返回值结构、shape、dtype 需与 `run()` 一致。 |
| `"atol"` / `"rtol"` | 可选（默认 `1e-2`） | 传给 `torch.testing.assert_close` 的绝对/相对误差容限（见 `check_task`）。`fp32` 算子可收紧到如 `1e-5`，低精度或大归约按需调整。 |
| `"flops"` | 可选 | 算子的理论浮点运算次数（如 GEMM 为 `2 * M * N * K`，乘加各算 1 次）。 |
| `"compute_mode"` | 写了 `flops` 时必填 | 决定去 `.gluonkit/peaks.json` 查哪一档算力峰值，预置：`"fp16"`、`"bf16"`、`"tf32"`、`"fp32"`。 |
| `"bytes_moved"` | 可选 | 理论最小全局显存读写字节数（读入输入 + 写出输出，元素个数 × 单元素字节数，例如 `float16` 乘 `2`、`float32` 乘 `4`）。纯访存算子可只填此项，走 `memory_only` 模型。 |

## 硬规则

1. **禁止篡改评测条件**：调优期间只修改算子文件（如 `examples/gemm/gemm.py`），禁止修改 `eval_*.py` 中的输入、参考实现 `ref`、容差 `atol`/`rtol`、`flops`/`bytes_moved` 或 `compute_mode`。调优期间也不要修改 `.gluonkit/peaks.json` 或重跑 `scripts/init_peaks.py`（保持前后轮 roofline 分母一致）。
2. **先 `baseline` / `check` 后 `bench`**：首次对某 `eval_<name>.py` 测速前，需先执行 `gk baseline <eval.py>` 生成 `.gluonkit/baseline/baseline_<name>.json`。每次修改算子后**必须先通过 `gk check`**，再执行 `gk bench`；未过 `check` 的测速结果视为无效。
3. **禁止盲猜 Gluon API**：写代码前查 `gluon-api` skill，遇到未覆盖的原语用 `.venv/bin/gk api list|show [--full]` 查询，或直接读 `.venv/lib/python3.12/site-packages/triton/experimental/gluon/` 源码核实签名。
4. **单变量迭代与版本保留**：每次修改只验证一个明确假设。通过校验且确认更快后保留。退化或失败时用 `gk restore <id>` 撤销本轮算子改动，保留工作区已有修改。`bench --parent <id>` 用于关联实验记录，可用 `gk history` 回顾多轮指标与资源变化。
5. **按算子输入约定处理边界**：任务明确固定 shape、dtype 和布局时，可据此特化，并用原评测的 `gk check` 校验数值。选定 tile 能整除的维度可省去对应边界 mask，有余数的维度仍须处理尾块。向量化或异步拷贝还需满足实际地址与步长的对齐要求。若接口承诺支持多种尺寸，则保留相应边界处理，按支持范围补测非整除尺寸或 `K < BK`。
