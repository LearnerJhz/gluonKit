# GluonKit 架构设计

> 本文档对齐当前仓库真实代码实现（`cli → output → executor → core / interfaces / backends`），作为内部开发与演进参考。Agent 使用规则与命令入口以 [AGENTS.md](AGENTS.md) 为准。

---

## 1. 目标与核心原则

GluonKit（CLI 命令 `gk`）是一套给 AI Agent 调用的 Gluon GPU kernel 开发与调优工具，核心目标是在保证数值正确的前提下持续提升算子性能。

- **进程内轻量执行**：工具直接通过 `runpy.run_path` 在当前进程加载评测脚本并调用 Python 函数，不引入子进程 IPC 或 `request.json` / `result.json` 中转文件。
- **统一 JSON 契约**：所有 CLI 子命令统一由 `output.py` 接管输出与异常捕获，固定向 `stdout` 输出单行 `{"result": "success" | "failed", "info": ...}`，隔离用户脚本打印与编译器杂音。
- **平台无关 Core 与硬件 Backend 解耦**：数值校验、性能汇总、Roofline 与实验历史保存在 `core/`，不直接依赖 CUDA；硬件探测、IR/资源导出与精细计时封装在 `backends/<平台>/`。
- **算子与评测分离**：算子文件（如 `examples/gemm/gemm.py`）保持纯粹、不依赖 GluonKit；评测脚本（如 `examples/gemm/eval_gemm.py`）统一负责输入构造、参考实现 `ref` 与计算/访存元数据。

---

## 2. 目录结构

```text
gluonKit/
├── AGENTS.md                         # Agent 入口指南与硬约束（命令/契约唯一真值）
├── CLAUDE.md                         # @AGENTS.md（供 Claude Code 自动加载）
├── .claude/skills/
│   ├── gluon-api/SKILL.md            # Triton 3.8.0 Gluon API 速查与报错排查
│   └── kernel-tuning/SKILL.md        # 瓶颈诊断与单变量调优闭环方法论
├── pyproject.toml                    # 包配置与 CLI 入口 gk = "gluonkit.cli:main"
├── examples/
│   └── gemm/
│       ├── gemm.py                   # 示例：SM86 FP16->FP32 Gluon GEMM（待调优算子）
│       └── eval_gemm.py              # 示例：固定评测脚本（KERNEL_PATH + prepare(device)）
├── src/gluonkit/
│   ├── __init__.py                   # 版本号（0.1.0）
│   ├── cli.py                        # 统一 CLI 入口（probe/layout/dump-info/check/bench）
│   ├── output.py                     # stdio 重定向、traceback 打包与单行 JSON 输出
│   ├── executor.py                   # 进程内任务加载与 dump-info/check/bench 调度
│   ├── interfaces/
│   │   ├── types.py                  # Output 信封与 Probe/Profile/Bench 结果数据结构
│   │   └── backend.py                # Backend Protocol 协议定义
│   ├── core/                         # 平台无关核心逻辑
│   │   ├── checker.py                # check_task：torch.testing.assert_close 数值校验
│   │   ├── bencher.py                # bench_task：warmup/iters 调度、baseline 与汇总
│   │   ├── roofline.py               # calculate_roofline：TFLOP/s、GB/s、Roofline 瓶颈
│   │   └── history.py                # History：.gluonkit/ 实验快照、best 分组与 diff
│   └── backends/
│       └── nvidia/                   # NVIDIA GPU 后端实现（cuda:0）
│           ├── backend.py            # NvidiaBackend（实现 Backend 协议，惰性加载子模块）
│           ├── probe.py              # 设备属性与 PyTorch/Triton 版本探测
│           ├── layout.py             # AST 白名单安全解析与 Triton layout 映射渲染
│           ├── dump_info.py          # 拦截 CompiledKernel 导出 IR 与资源摘要
│           └── profiler.py           # torch.profiler 事件关联计时与可选清 L2
└── tests/
    └── fixtures/                     # 测试夹具目录（自动化测试集待重建）
```

---

## 3. 整体调用链路

```text
  Agent / Terminal
         │
         ▼
   gk <subcommand>  (cli.py)
         │
         ├─► probe / layout ─────────────────────────┐
         │                                           ▼
         └─► dump-info / check / bench        NvidiaBackend (backends/nvidia/)
                   │                          ├─ probe.py      (设备/版本)
                   ▼                          ├─ layout.py     (离线 AST 映射)
             executor.py                      ├─ dump_info.py  (IR + 资源摘要)
             (load_task + _prepare_task)      └─ profiler.py   (GPU kernel 计时)
                   │                                 ▲
                   ├─► check ──► core/checker.py     │
                   └─► bench ──► core/bencher.py ────┘
                                   ├─► core/roofline.py
                                   └─► core/history.py (.gluonkit/)
         │
         ▼
   output.capture_run() + emit_json() ──► stdout: {"result": "...", "info": ...}
```

---

## 4. 接口与数据结构（`interfaces/`）

### 4.1 公共数据结构（`interfaces/types.py`）

统一使用普通 `dataclass` + `to_dict()` / `_to_json_value()` 进行 JSON 递归序列化：

- `Output(result: Literal["success", "failed"], info: Any = None)`
  - 最外层输出信封；成功 `Output.ok(info)`，失败 `Output.fail(info)`。
- `ProbeResult`
  - `device_name`, `platform`, `arch`, `compiler_version`, `torch_version`, `smem_per_block_optin`, `sm_count`, `vram_bytes`
- `ProfileKernel(name: str, avg_ms: float)`
- `ProfileData(device_time_ms: float, kernels: list[ProfileKernel], trace_path: str | None = None)`
- `BenchResult`
  - `device_time_ms`, `kernels`, `baseline_time_ms`, `speedup_vs_baseline`, `tflops`, `gbps`, `roofline_pct`, `bound_by`, `trace_path`

### 4.2 Backend 协议（`interfaces/backend.py`）

```python
class Backend(Protocol):
    name: str
    device: str

    def available(self) -> tuple[bool, str]: ...
    def probe(self) -> ProbeResult: ...
    def layout(self, expression: str, shape: list[int], hardware: bool = False) -> str: ...
    def dump_info(self, fn: Callable[[], object], output_dir: str | Path) -> dict[str, Any]: ...
    def profile(
        self, fn: Callable[[], object], iters: int, warmup: int, *,
        flush_l2: bool = False, trace_path: str | None = None,
    ) -> ProfileData: ...
    def flush_cache(self) -> None: ...
```

**关键边界**：`dump_info()` 返回 `dict[str, Any]`，由具体后端自行决定返回字段与摘要结构（例如 NVIDIA 返回 `artifacts` 与 `summary`），公共协议和 `executor.py` 不写死平台特定资源键名。

---

## 5. 各层实现说明

### 5.1 CLI 与输出层（`cli.py`、`output.py`）

- **`cli.py`**：当前提供 5 个子命令，全部直接实例 `NvidiaBackend()`，延迟导入 `executor`：
  - `gk probe`
  - `gk layout <expression> --shape <dims> [--hardware]`
  - `gk dump-info <eval_path>`
  - `gk check <eval_path>`
  - `gk bench <eval_path> [--note NOTE] [--parent PARENT] [--trace] [--no-baseline]`
- **`output.py`**：
  - `capture_run(fn)` 使用 `redirect_stdout` / `redirect_stderr` 捕获执行期所有输出；正常返回包装为 `Output.ok(value)`；捕获 `Exception` 后通过 `format_error()` 组装完整异常链 traceback 以及捕获到的 `[stdout]` / `[stderr]`，返回 `Output.fail(...)`。
  - `render_json()` 使用 `json.dumps(..., ensure_ascii=False, allow_nan=False)`，拒绝 `NaN`/`Inf` 静默落盘。
  - `emit_json()` 向 `stdout` 打印单行 JSON，按 `result` 返回退出码 `0` 或 `1`。

### 5.2 执行调度层（`executor.py`）

- **`load_task(eval_path, device)`**：将评测脚本所在目录加入 `sys.path`，通过 `runpy.run_path` 加载脚本并校验可调用入口 `prepare(device)`；执行一次 `prepare(device)` 并检查返回值为包含可调用 `"run"` 的 `dict`。
- **`_prepare_task(eval_path)`**：校验脚本文件存在、检查 `NvidiaBackend().available()`，再调用 `load_task`。
- **`run_check(eval_path)`**：调用 `_prepare_task` 后执行 `check_task(task)`，同时承担触发 JIT 编译与数值校验的门禁职责（注意：当前 `check_task` 在调用 `task["run"]()` 前先校验 `ref` 可调用）。
- **`run_dump_info(eval_path)`**：调用 `backend.dump_info(task["run"], Path("dumpInfo").resolve() / path.stem)` 并直接返回后端输出字典。
- **`run_bench(eval_path, *, trace=False, no_baseline=False, note=None, parent_id=None)`**：
  1. 读取评测文件字节并通过 AST 静态解析 `KERNEL_PATH` 字符串字面量，读取对应算子文件字节；校验可选 `parent_id` 存在。
  2. 调用 `_prepare_task` 并通过 `_module_loaded_from(kernel_file)` 验证评测脚本确实导入了该算子。
  3. 若开启 `trace=True`，在 `./runs/bench-*/trace.json` 创建临时目录路径（`bench_task` 异常时自动清理）。
  4. 调用 `bench_task(task, backend, ...)`（默认沿用 `DEFAULT_WARMUP=10`、`DEFAULT_ITERS=100`、`flush_l2=False`）。
  5. 再次比对评测与算子文件字节，确认测速期间源码未变化后，将硬件环境、测量口径、双源码 sha256 与 `BenchResult` 写入 `History`，最终返回 `{"history_id": record["id"], **result.to_dict()}`。

### 5.3 平台无关核心层（`core/`）

- **`checker.py` (`check_task`)**：读取可选 `atol`/`rtol`（默认均 `1e-2`），要求 `ref` 可调用，依次运行 `actual = task["run"]()` 与 `expected = ref()`，使用 `torch.testing.assert_close` 校对形状、类型、NaN/Inf 与数值误差。
- **`bencher.py` (`bench_task`)**：常量 `DEFAULT_ITERS = 100`、`DEFAULT_WARMUP = 10`。执行 GPU 测量前先调 `calculate_roofline(None, ...)` 校验元数据合法性；调用 `backend.profile` 测量 `run`，未设 `no_baseline` 且存在 `ref` 时以相同轮数测量 `ref`（不导出 trace）并计算 `speedup_vs_baseline`；最后合入 Roofline 指标返回 `BenchResult`。
- **`roofline.py` (`calculate_roofline`)**：校验 `flops`、`bytes_moved`、`peaks`（`{"tflops": ..., "gbps": ...}`）均为正有限数；按 `device_time_ms` 换算 `tflops`（`/1e12`）和 `gbps`（`/1e9`）；仅当 4 项元数据均齐备时才计算 `roofline_pct = max(compute_seconds, memory_seconds) / seconds * 100` 与 `bound_by`（`"compute"` / `"memory"`），缺项返回 `None`。
- **`history.py` (`History`)**：管理 `.gluonkit/history.jsonl` 与内容寻址快照 `.gluonkit/sources/<sha256>.py`，提供 `save`、`get`（完整 ID 或唯一前缀）、`list`、`best`（按 `kernel_path`、`platform`、`arch`、`env`、`eval_source_hash`、`measurement_options`、`device_name` 严格分组取 `device_time_ms` 最小记录，多组时报错提示 `--like`）与 `diff`（指标变化百分比及双源码 unified diff）。目前由 `run_bench` 写入记录，查询/对比接口暂未接为独立 CLI 子命令。

### 5.4 NVIDIA 后端实现（`backends/nvidia/`）

- **`probe.py`**：调用 `torch.cuda.get_device_properties(device)` 查询设备名称、`sm{major}{minor}`、SM 数、显存字节、`shared_memory_per_block_optin`（无则 `None`）及 `triton`/`torch` 版本。
- **`layout.py`**：用 `ast.parse(..., mode="eval")` 白名单解析构造器，只放行 `BlockedLayout`、嵌套 `SliceLayout`、整数字面量和 list/tuple；校验 2 的幂次、rank 匹配、order 排列、`prod(threads_per_warp) == 32` 且禁止 `cga_layout`；直接调用 Triton `format_tensor_view(shape)` / `format_hardware_view(shape)`，不依赖 GPU。
- **`dump_info.py`**：在执行 `call()` 期间临时替换 `triton.compiler.compiler.CompiledKernel.launch_metadata`（`finally` 恢复），按 `kernel.hash` 去重导出每个被调用 kernel（含缓存命中，同名不同特化加 `_2`/`_3` 后缀）的 `ttgir`/`llir`/`ptx` 到 `dumpInfo/<eval名>/<kernel名>/kernel.*`。由于 Triton 3.8.0 在 `launch_metadata` 之后访问 `kernel.run` 时才触发 `_init_handles()` 加载 cubin，实现中先记录 `kernel` 引用，待 `call()` 返回后再读取摘要并返回：
  ```json
  {
    "artifacts": { "<kernel>": { "ttgir": "...", "llir": "...", "ptx": "..." } },
    "summary": {
      "<kernel>": {
        "n_regs": 92,
        "n_spills": 0,
        "n_max_threads": 640,
        "shared_bytes": 6144,
        "num_warps": 4
      }
    }
  }
  ```
- **`profiler.py`**：
  - 预热 `warmup` 次并 `torch.cuda.synchronize(device)` 后，开启 `torch.profiler.profile(CPU, CUDA)` 循环 `iters` 次；每次用唯一 `record_function` 作用域包裹 `fn() + synchronize()`。
  - 若开启 `flush_l2`，在每次迭代作用域外先用独立清缓存标记写入 2 倍 L2 容量缓冲区（缺省 256 MiB）并同步。
  - `_collect_kernels()` 不按时间窗猜测归属，而是严格筛选本设备 `activity_type == "kernel"` 事件（自动跳过 `gpu_memcpy`、`gpu_memset`、`gpu_user_annotation`），通过 `linked_correlation_id`（PyTorch 算子）或 `event.id` 匹配 `cuda_runtime`/`cuda_driver`（Gluon 直接 driver launch），沿 `cpu_parent` 追溯迭代作用域或排除清缓存作用域；无法归属的 kernel 直接报 `RuntimeError`。
  - 返回单次调用总设备核耗时均值 `device_time_ms`（多流并发时为各 kernel 耗时之和）与按 kernel 聚合的单次启动均值 `ProfileKernel(name, avg_ms)`；支持导出 Chrome trace。

---

## 6. 评测脚本与算子约定（`examples/gemm/`）

评测脚本以普通 Python 编写，供 `executor` 加载：

```python
KERNEL_PATH = "gemm.py"  # gk bench 必需：相对当前评测脚本的待调优算子路径

def prepare(device: str) -> dict:
    return {
        "run": lambda: ...,          # 必需：无参待测调用（可重复执行）
        "ref": lambda: ...,          # check 必需，bench 默认用作 baseline
        "atol": 1e-2,                # 可选，默认 1e-2
        "rtol": 1e-2,                # 可选，默认 1e-2
        "flops": ...,                # 可选正有限数，用于算 TFLOP/s
        "bytes_moved": ...,          # 可选正有限数，用于算估算 GB/s
        "peaks": {"tflops": ..., "gbps": ...},  # 可选，用于算 roofline_pct / bound_by
    }
```

- [examples/gemm/gemm.py](examples/gemm/gemm.py)：使用 Triton Gluon 编写的 SM86 FP16 输入 / FP32 累加矩阵乘（`NVMMADistributedLayout`、`DotOperandLayout`、`SwizzledSharedLayout`、`ampere.async_copy`、`mma_v2`），不依赖 GluonKit。
- [examples/gemm/eval_gemm.py](examples/gemm/eval_gemm.py)：准备 `(1024, 512) @ (512, 1024)` FP16 张量与 `torch.matmul` 参考实现及 `flops` / `bytes_moved`。

---

## 7. 已知缺口与后续待演进项（TODO）

以下均为可在现有分层上平滑扩展的能力，当前刻意保持极简、未提前复杂化：

1. **测量前置校验与路径边界**：当前 `run_bench` 未内部调用 `check_task`（暂由 `AGENTS.md` 约束先 `check` 后 `bench`），且 `read_kernel_path` 尚未限制绝对路径/`..` 越界及异常时外层 `trace_dir` 清理。
2. **计时波动与缓存配置**：`profiler.py` 已有逐次迭代耗时，后续可顺带计算并输出 `std_ms`/`min_ms`/`median_ms` 噪声指标；`--flush-l2` 暂未从 CLI 透传；可增加 `~/.triton` 只读时自动 fallback 到可写 `TRITON_CACHE_DIR`。
3. **防止单 shape 过拟合与激活 Roofline**：可扩展多 shape 边界检查（不计入 `bench` 耗时），并为 `eval_gemm.py` 补充对应设备与精度的 `peaks`。
4. **实验查询入口**：`core/history.py` 的 `best`/`list`/`diff` 目前只保留在库层，视调优工作流决定是否接入 `gk history` 子命令，或简化为纯 Git 记录。
5. **工具可观测性增强**：`gk probe --gluon-api`（自省本机 Gluon 模块签名）、`layout.py` 白名单扩展更多 layout 类（如 `NVMMADistributedLayout`、`SwizzledSharedLayout`）。
6. **基础设施与文档**：重建 `tests/` 核心单测与真机回归集；视多平台需求再加 `registry`；若未来遇到严重 GPU hang/越界崩溃再在 `cli → executor` 间外挂带超时子进程隔离；精简过时的 `README.md` 与 `plan.md`。
