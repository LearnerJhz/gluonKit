# GluonKit (`gk`)

面向 AI Agent 与本地开发者使用的 Gluon GPU kernel 开发、分析与性能调优工具链。

> 本文档聚焦 `src/gluonkit` 的当前代码架构、模块职责与接口契约。Agent 调优使用的命令行规则与硬约束以 [AGENTS.md](AGENTS.md) 和 `.venv/bin/gk --help` 为准。

---

## 1. 目录结构

```text
gluonKit/
├── AGENTS.md                         # Agent 入口规则与 CLI 契约（SSOT）
├── README.md                         # 本文档：src/gluonkit 架构与模块说明
├── pyproject.toml                    # 包配置与 CLI 入口 gk = "gluonkit.cli:main"
├── scripts/
│   └── calibrate_peaks.py            # 一次性本机峰值标定脚本（写入 .gluonkit/peaks.json）
├── examples/gemm/
│   ├── gemm.py                       # 示例待调优 Gluon GEMM（FP16 矩阵乘、FP32 累加）
│   └── eval_gemm.py                  # 示例评测脚本（prepare 入口，同目录配对 gemm.py）
└── src/gluonkit/
    ├── __init__.py                   # __version__ = "0.1.0"
    ├── cli.py                        # argparse 入口、环境变量初始化与命令分发
    ├── output.py                     # stdio 重定向、traceback 格式化、浮点保留位与 JSON 输出
    ├── executor.py                   # 各命令调度、任务加载、资源摘要采集与源码快照校验
    ├── interfaces/
    │   ├── backend.py                # Backend Protocol 协议定义
    │   └── types.py                  # Output / ProbeResult / ProfileData / BenchResult
    ├── core/
    │   ├── gluon_api.py              # Gluon API 自省：resolve_module/resolve_symbol/list_module/show_symbols（纯文本，无需 GPU）
    │   ├── checker.py                # check_task：torch.testing.assert_close 校验
    │   ├── baseline.py               # BaselineStore（.gluonkit/baseline/*.json 读写与校验）
    │   ├── bencher.py                # measure_baseline / bench_task：独立基线测量与算子测速汇总
    │   ├── roofline.py               # validate_metadata / calculate_roofline：TFLOP/s、GB/s、Roofline
    │   ├── peaks.py                  # PeakStore（.gluonkit/peaks.json 读写）与 resolve_peaks
    │   └── history.py                # History：.gluonkit/ 实验记录、摘要读取与快照恢复
    └── backends/nvidia/
        ├── backend.py                # NvidiaBackend（实现 Backend 协议，惰性导入子模块）
        ├── probe.py                  # GPU 设备属性与 PyTorch/Triton 版本探测
        ├── layout.py                 # AST 白名单解析与 Triton layout 映射渲染
        ├── bank_conflicts.py         # 离线构建 Gluon IR 调用 gl.bank_conflicts
        ├── dump_info.py              # 拦截 CompiledKernel 导出 IR 与 occupancy 计算
        └── profiler.py               # torch.profiler 事件关联计时、统计与 trace 导出
```

---

## 2. 整体调用链路

```text
  Terminal / Agent
         │
         ▼
   cli.py (argparse + _configure_env + NvidiaBackend())
         │
         ▼
   output.capture_run() ──► executor.py (只返回业务数据或直接抛异常)
                                 │
                                 ├─► run_probe ──────────────────────────► NvidiaBackend.probe + core/peaks.py (PeakStore)
                                 ├─► run_layout / run_bank_conflicts / run_dump_info ──► NvidiaBackend
                                 ├─► gk api list/show（cli._run_api 直接输出纯文本，不经 capture_run）─► core/gluon_api.py
                                 ├─► run_check ──────────────────────────► core/checker.py
                                 ├─► run_bench
                                 │     ├─► core/peaks.py (PeakStore) + NvidiaBackend.probe
                                 │     ├─► core/bencher.py
                                 │     │     ├─► core/peaks.py (resolve_peaks)
                                 │     │     ├─► core/roofline.py
                                 │     │     └─► NvidiaBackend (profile / profile_baseline)
                                 │     ├─► _collect_kernel_summary ──────► NvidiaBackend.dump_info
                                 │     └─► core/history.py (.gluonkit/)
                                 └─► run_history / run_restore ──────────► core/history.py
         │
         ▼
   output.render_json() / emit_json() ──► 浮点保留位格式化 + stdout: {"result": "success" | "failed", "info": ...}
```

### 运行前环境初始化（[cli.py](src/gluonkit/cli.py)）

- `TRITON_CACHE_DIR` 固定设为 `<cwd>/.gluonkit/triton_cache`（确保沙盒内无需写入用户主目录）。
- `KINETO_LOG_LEVEL` 默认设为 `"6"`（必须在 `torch`/`triton` 初始化前设置，用于抑制 PyTorch Kineto 的 `USDT` stderr 噪音日志）。

---

## 3. 核心接口与数据结构

### 3.1 公共数据结构与输出层（[interfaces/types.py](src/gluonkit/interfaces/types.py)、[output.py](src/gluonkit/output.py)）

- **`Output(result: Literal["success", "failed"], info: Any = None)`**（`interfaces/types.py`）
  - 最外层输出信封；由 `output.py` 的 `render_json()` 使用 `allow_nan=False` 拒绝 `NaN`/`Inf`，并按字段名语义保留小数位（ms 耗时 `6` 位、`speedup` `3` 位、百分比 `2` 位、吞吐率 `>= 1` 保留 `2` 位 / `< 1` 保留 `4` 位、其余默认 `6` 位），不改变 `.gluonkit/history.jsonl` 中存储的原始精度。
- **`ProbeResult`**
  - `device_name`, `platform`, `arch`, `compiler_version`, `torch_version`, `smem_per_block_optin`, `sm_count`, `vram_bytes`
- **`ProfileData`**
  - 单次 `profile()` 结果：`kernel_name`, `avg_ms`, `std_ms`, `min_ms`, `median_ms`, `trace_path`
- **`BenchResult`**
  - 持有 `kernel_profile: ProfileData`、`metrics`（包含 `speedup`、`tflops`、`gbps`、`roofline`、`bound_by`）、`baseline: dict[str, Any] | None`、`peaks: dict[str, Any] | None`
  - `to_dict()` 序列化形态（`gk bench` 会在外层合入 `"history_id"`）：
    ```json
    {
      "trace_path": "...",
      "kernel": {
        "name": "...",
        "avg": 0.123,
        "median": 0.122,
        "min": 0.119,
        "std": 0.004
      },
      "metrics": {
        "speedup": 1.05,
        "tflops": 12.3,
        "gbps": 45.6,
        "roofline": 82.19,
        "bound_by": "compute"
      },
      "baseline": {
        "kernel_name": "...",
        "avg_ms": 0.129
      },
      "peaks": {
        "tflops": 17.07,
        "gbps": 157.06,
        "source": {"tflops": "calibrated:tflops_fp16", "gbps": "calibrated:gbps"}
      }
    }
    ```

### 3.2 Backend 协议（[interfaces/backend.py](src/gluonkit/interfaces/backend.py)）

```python
class Backend(Protocol):
    device: str

    def probe(self) -> ProbeResult: ...
    def layout(self, expression: str, shape: list[int], hardware: bool = False) -> str: ...
    def bank_conflicts(
        self, register_layout: str, shared_layout: str, shape: list[int], dtype: str,
    ) -> dict[str, int]: ...
    def dump_info(self, fn: Callable[[], object], output_dir: str | Path) -> dict[str, Any]: ...
    def profile(
        self, fn: Callable[[], object], iters: int, warmup: int, *,
        flush_l2: bool = False, trace_path: str | None = None,
    ) -> ProfileData: ...
    def profile_baseline(
        self, fn: Callable[[], object], iters: int, warmup: int,
    ) -> tuple[float, dict[str, Any]]: ...
```

**边界与当前约定**：`dump_info()` 返回的 `dict[str, Any]` 与 `profile_baseline()` 返回的详情字典由后端填充，但目前 `executor.py` 的 `_collect_kernel_summary()` 会读取 `dump_info()` 返回中的 `"summary"` 键，[history.py](src/gluonkit/core/history.py) 的 `_bench_metrics()` 会读取 `baseline` 字典中的 `"avg_ms"` 字段用于历史摘要。

---

## 4. CLI 命令与核心机制

| 命令 | 关键参数 | 说明 |
|---|---|---|
| `gk probe` | 无 | 返回平台、GPU 型号、SM 架构、SM 数、VRAM、opt-in 共享内存上限及 PyTorch/Triton 版本；附带 `.gluonkit/peaks.json` 中本设备的标定峰值与摘要 |
| `gk layout "<expr>" --shape 64,64` | `--hardware` | 离线渲染 `BlockedLayout` / 嵌套 `SliceLayout` 的线程/warp 映射；AST 白名单解析，不执行任意代码 |
| `gk bank-conflicts --register-layout ... --shared-layout ... --shape 64,32 --dtype float16` | 均为必选 | 离线构造轻量 `GluonOpBuilder` 模块并调用 `gl.bank_conflicts()`；支持 `BlockedLayout`、`SliceLayout`、`DotOperandLayout`、`NVMMADistributedLayout` 与 `SwizzledSharedLayout` |
| `gk api list [module]` / `gk api show [-m module] <symbol>...` | `-m` | 自省 `triton.experimental.gluon` 的公开 API；**纯文本输出，不走 JSON 信封**；显式 `-m` 时只在该模块内查找，未指定时按 `gl` → `gluon` 回退（命中根包会标注来源）；符号解析失败时 stderr 输出 `[ERROR]` 行并以退出码 1 结束 |
| `gk dump-info <eval.py>` | 无 | 临时 hook `CompiledKernel.launch_metadata`，按 `kernel.hash` 去重导出 `ttgir`/`llir`/`ptx` 并计算寄存器与 occupancy 资源摘要 |
| `gk check <eval.py>` | 无 | 运行一次 `run()` 与 `ref()`，用 `torch.testing.assert_close(..., atol=atol, rtol=rtol)` 校验数值（默认 `1e-2`） |
| `gk baseline <eval.py>` | 无 | 若 `.gluonkit/baseline/baseline_<name>.json` 不存在则测量一次 `ref()` 并写入，已存在时直接返回 |
| `gk bench <eval.py>` | `--note`, `--parent`, `--trace` | 仅测量待测 kernel，读取已存基线计算 `speedup`，写入历史与双源码快照 |
| `gk history [<kernel_or_eval>]` | `--limit N` | 列出已记录的 `bench` 实验摘要与资源 `summary`，支持按 `kernel_path` / `eval_path` 筛选与截断最近 N 条 |
| `gk restore <id>` | 无 | 按完整 ID 或唯一前缀查找记录，校验 `.gluonkit/sources/<sha256>.py` 完整性后将对应 `kernel` 源码恢复回 `kernel_path` |

### `src/gluonkit` 关键实现要点

1. **NVIDIA 精确计时（[profiler.py](src/gluonkit/backends/nvidia/profiler.py)）**
   - 预热后在 `torch.profiler.profile(CPU, CUDA)` 中用独立 `record_function` 包裹每轮 `fn() + torch.cuda.synchronize(device)`。
   - `_collect_kernels()` 仅采集本设备 `activity_type == "kernel"` 事件（自动跳过 memcpy/memset），通过 `linked_correlation_id`（PyTorch 算子）或 `event.id` 匹配 `cuda_runtime`/`cuda_driver`（Gluon launch），沿 `cpu_parent` 回溯归属到对应迭代（并排除 `flush_l2` 作用域）。
2. **IR 导出与 Occupancy（[dump_info.py](src/gluonkit/backends/nvidia/dump_info.py)、[executor.py](src/gluonkit/executor.py)）**
   - Triton 3.8.0 在 `launch_metadata` 之后访问 `kernel.run` 时才触发 `_init_handles()` 加载 cubin，因此先记录 `kernel` 引用，待 `call()` 返回后再读取 `n_regs`/`n_spills`/`n_max_threads`。
   - 通过 `cuda.bindings.driver` 调用 `cuOccupancyMaxActiveBlocksPerMultiprocessor(func, block_threads, dynamic_smem)` 计算 `blocks_per_sm` 与 `occupancy_pct`，并按 `static_smem + dynamic_smem + RESERVED_SHARED_MEMORY_PER_BLOCK` 及架构粒度（128B/256B）对齐得到 `allocated_shared_bytes`。`run_bench()` 测速完成后通过 `_collect_kernel_summary()` 在临时目录执行一次 `dump_info()`，将摘要写入 `history.jsonl` 的 `summary` 字段。
3. **Roofline 与峰值解析（[roofline.py](src/gluonkit/core/roofline.py)、[peaks.py](src/gluonkit/core/peaks.py)）**
   - `PeakStore` 以 `<platform>:<arch>:<device_name>` 为键原子读写 `.gluonkit/peaks.json`（标定由独立脚本 [calibrate_peaks.py](scripts/calibrate_peaks.py) 完成）。
   - `resolve_peaks()` 逐字段优先取评测显式 `peaks`，否则 `tflops` 按 `compute_mode` 取 `tflops_<mode>`、`gbps` 取 `gbps`（若当前设备在 `peaks.json` 中已有标定记录但缺少 `tflops_<compute_mode>` 则抛错；未标定或未声明 `compute_mode` 时不从标定表填充对应峰值）。`calculate_roofline()` 仅在已声明的工作量（`flops` / `bytes_moved`）均有对应峰值时才输出 `roofline_pct` 与 `bound_by`。
4. **实验历史与恢复（[history.py](src/gluonkit/core/history.py)）**
   - `gk bench` 将完整记录追加到 `.gluonkit/history.jsonl`，并将评测与算子源码按 SHA256 存入 `.gluonkit/sources/<sha256>.py`。
   - `History` 提供 `store_source()`、`save()`、`get()`（精确或唯一前缀）、`list()`（支持 `kernel` 或 `target` 筛选）与 `restore()`；`summarize_record()` 兼容历史平铺格式与现行嵌套 `BenchResult` 格式。
5. **独立基线存储（[baseline.py](src/gluonkit/core/baseline.py)）**
   - `BaselineStore` 按 `eval_<name>.py` 路径原子读写 `.gluonkit/baseline/baseline_<name>.json`，文件已存在时 `gk baseline` 直接返回而不重测；JSON 内容仅保留跨平台通用字段（`env`、`measurement_options`、`measured_at`、`baseline`），不强依赖特定平台的设备型号或架构字段。

---

## 5. 评测脚本约定

评测脚本命名为 `eval_<name>.py`，自动配对同目录下的算子文件 `<name>.py`。由 [executor.py](src/gluonkit/executor.py) 通过 `runpy.run_path` 在当前进程加载并调用 `prepare(device)`（算子源码不依赖 GluonKit）：

```python
def prepare(device: str) -> dict:
    return {
        "run": lambda: ...,          # 必需：无参待测调用
        "ref": lambda: ...,          # check 必需，bench 默认用作 baseline
        "atol": 1e-2,                # 可选，默认 1e-2
        "rtol": 1e-2,                # 可选，默认 1e-2
        "flops": ...,                # 可选正有限数，用于算 TFLOP/s
        "bytes_moved": ...,          # 可选正有限数，用于算 GB/s
        "peaks": {"tflops": ..., "gbps": ...},  # 可选，显式峰值，逐字段覆盖标定值
        "compute_mode": "fp16",                 # 声明 flops 时建议提供：fp16/bf16/tf32/fp32，选择 tflops_<mode>
    }
```

---

## 6. 常用操作与运行时目录

```bash
# 安装 / 可编辑模式开发（需要 Python >= 3.10, torch, triton==3.8.0）
pip install -e .

# 常用 CLI
.venv/bin/gk probe
.venv/bin/python scripts/calibrate_peaks.py   # 一次性标定本机峰值
.venv/bin/gk check examples/gemm/eval_gemm.py
.venv/bin/gk bench examples/gemm/eval_gemm.py --note "baseline"
.venv/bin/gk dump-info examples/gemm/eval_gemm.py
.venv/bin/gk history examples/gemm/eval_gemm.py --limit 5
.venv/bin/gk restore <history_id>
.venv/bin/gk api list [module]               # 列出 Gluon API
.venv/bin/gk api show max sum -m language    # 查询符号签名
```

运行时在工作目录下生成 `.gluonkit/`（已加入 `.gitignore`）：
- `.gluonkit/triton_cache/`：Triton 编译缓存
- `.gluonkit/history.jsonl`：每次 `gk bench` 的实验记录
- `.gluonkit/peaks.json`：按设备存储的标定峰值
- `.gluonkit/baseline/baseline_<name>.json`：`gk baseline <eval.py>` 存储的 `ref()` 基线测量结果
- `.gluonkit/sources/<sha256>.py`：评测脚本与算子源码内容寻址快照
- `.gluonkit/dump_info/<eval_stem>/<kernel>/<hash>/`：`kernel.ttgir`、`kernel.llir`、`kernel.ptx`
- `.gluonkit/traces/bench-*/trace.json`：`gk bench --trace` 导出的 Chrome trace

---

## 7. 已知局限与待完善项

- **单 kernel 假定**：`_collect_kernels()` 会累加同次迭代内所有设备 kernel 的耗时并记录首个 `kernel_name`，尚未对零 kernel 或多 kernel 情况做显式拦截或拆分。
- **`ref` trace 导出**：`--trace` 仅导出待测 `run()` 的 `trace.json`；`gk baseline` 存储 `kernel_name`、`avg_ms`、`std_ms`、`min_ms`、`median_ms` 等统计信息，`gk bench` 的 `baseline` 返回 `kernel_name` 与 `avg_ms`。
- **历史对比与恢复增强**：`gk history` 与 `gk restore` 尚未提供组内最优筛选（`best`）、两轮实验指标与源码差异（`diff`）以及 `restore` 覆盖前对未入库修改的自动备份。
- **测量、校验与测试补全**：`run_bench()` 暂未内置自动调用 `check_task` 或记录校验凭据；`--flush-l2` 暂未从 CLI 透传；`tests/` 目录与自动化测试集尚未建立。
