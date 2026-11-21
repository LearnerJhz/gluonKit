# GluonKit (`gk`) 多后端 SIMT 架构设计文档（NVIDIA + 沐曦 C500）

本文档定义 GluonKit 面向 **SIMT GPU 双后端（NVIDIA CUDA + 沐曦 MetaX C500 MACA）** 的重构目录结构、三层职责边界、数据结构契约与核心闭环机制。

---

## 1. 设计原则

1. **聚焦 SIMT GPU 共性，拒绝过度泛化**
   - 目标硬件为 **NVIDIA GPU（sm70/sm80/sm86/sm90）** 与 **沐曦 C500（MACA SIMT GPU）**。
   - 两者均遵循 SIMT 执行模型（CTA/Block、Warp/Wave、Shared Memory/LDS、VGPR/SGPR 寄存器、Occupancy 驻留与 Bank Conflict），且均使用 Triton / Gluon 编译栈与 `torch.cuda` 兼容运行时。
   - 因此保留**强类型的 SIMT 数据契约**，不退化为无模式的 `dict[str, Any]`，仅将硬件差异点（如 `warp_size=32 vs 64`、厂商专属 MMA Layout、IR 后缀、Occupancy 计算与计时降级）收敛为后端钩子。
2. **公共 SIMT 机制上移 `core/`，后端极简薄化**
   - AST Layout 解析与渲染、Gluon IR Bank Conflict 静态分析、`CompiledKernel.launch_metadata` 拦截与 IR 落盘、Kineto / Device Event 双模计时器全部提取到 `core/`。
   - 每个厂商后端（`backends/nvidia/`、`backends/metax/`）只保留约 120–150 行的硬件探测、特化 Layout 注册、Occupancy 估算与精度开关。
3. **加固 `core/` 实验控制面闭环**
   - **基线身份防串台**：基线绑定 `eval_path`（规范化相对路径）+ `eval_source_hash` + `device_key`，支持 `gk baseline --refresh`。
   - **校验与测速联动**：`gk check` 输出实际误差摘要（`max_abs_err` / `max_rel_err`）并写入 `.gluonkit/checks.json` 回执；`gk bench` 自动关联校验状态并支持 `--require-check` 强制门禁。
   - **诊断信息内聚**：`ProfileData` 直接输出 `p95_ms` 与 `iters`；`KernelSummary` 直接计算 `limiting_factor`（`regs` / `shared_memory` / `warps` / `blocks`）；`roofline` 明确区分 `dual_roofline`、`memory_only` 与 `compute_only`。

---

## 2. 目标目录结构

```text
gluonKit/
├── AGENTS.md                         # Agent 规则与 CLI 契约（唯一真值来源）
├── ARCHITECTURE.md                   # 本文档：多后端 SIMT 架构设计与改造蓝图
├── README.md                         # 代码实现与模块参考文档
├── pyproject.toml                    # 包配置与 CLI 入口 gk = "gluonkit.cli:main"
├── scripts/
│   └── calibrate_peaks.py            # 跨平台峰值标定脚本（通过 Backend 接口驱动，不硬编码 N 卡）
├── .knowledge/
│   └── gluon-api/
│       ├── gluon-api-common.md       # 跨平台通用 Gluon 核心原语
│       ├── troubleshooting.md        # 常见编译与运行排错表
│       ├── nvidia/                   # NVIDIA 平台扩展与 sm86 实例
│       └── metax/                    # 沐曦 C500 (Wave64 / MACA) 平台知识库预留目录
├── src/gluonkit/
│   ├── __init__.py
│   ├── cli.py                        # CLI 解析：支持 --backend (auto|nvidia|metax)、--refresh、--require-check、api show --full
│   ├── output.py                     # 统一 JSON 信封序列化與浮点数格式化
│   ├── executor.py                   # 命令编排层（完全平台无关）
│   ├── interfaces/
│   │   ├── backend.py                # Backend Protocol 定义
│   │   └── types.py                  # Output / ProbeResult / KernelSummary / ProfileData / CheckResult / BenchResult
│   ├── core/
│   │   ├── storage.py                # .gluonkit/ 原子 JSON 读写与路径规范化
│   │   ├── device_diag.py            # GPU 初始化失败结构化诊断（区分沙盒设备节点拦截、环境变量隐藏、驱动异常）
│   │   ├── gluon_api.py              # Gluon API 离线内省（支持 --full 完整 docstring 与源码行号定位）
│   │   ├── simt_layout.py            # [上移通用] 参数化 warp_size 的 Layout 渲染与 Bank Conflict 静态分析
│   │   ├── triton_dump.py            # [上移通用] CompiledKernel 拦截与 kernel.asm 全量文本 IR 导出
│   │   ├── torch_profiler.py         # [上移通用] Kineto 核函数计时引擎 + CUDA/MACA Event 计时引擎（含 P95 统计）
│   │   ├── checker.py                # 数值校验、误差统计与 .gluonkit/checks.json 校验回执管理
│   │   ├── baseline.py               # 绑定 eval_path + eval_source_hash + device_key 的基线存储
│   │   ├── bencher.py                # 算子测速、校验状态核验与历史快照落盘
│   │   ├── roofline.py               # TFLOP/s、GB/s 与 roofline_model (dual_roofline / memory_only / compute_only)
│   │   ├── peaks.py                  # .gluonkit/peaks.json 读写与峰值解析
│   │   └── history.py                # .gluonkit/history.jsonl 谱系、与父实验延迟 Δ 计算及源码恢复
│   └── backends/
│       ├── __init__.py
│       ├── base.py                   # SimtBackendBase：串联 core/simt_layout、triton_dump、torch_profiler 的公共基类
│       ├── registry.py               # 后端注册与自动探测（支持 --backend、GK_BACKEND 及运行时特征识别）
│       ├── nvidia/
│       │   ├── __init__.py
│       │   └── backend.py            # NvidiaBackend (warp_size=32, NVMMADistributedLayout, cuOccupancy + 架构受限因子计算)
│       └── metax/
│           ├── __init__.py
│           └── backend.py            # MetaxBackend (沐曦 C500: warp_size=64, Wave64 LDS/VGPR Occupancy 模型, Event/Kineto 自适应计时)
└── tests/
    └── test_architecture.py          # 覆盖多后端切换、Wave32/Wave64 Layout、基线防碰撞、Check 门禁、Roofline 的单元测试
```

---

## 3. 核心接口与数据契约（`interfaces/`）

### 3.1 `ProbeResult`（兼容旧字段，扩展 SIMT 通用属性）

```python
@dataclass
class ProbeResult:
    device_name: str
    platform: str                        # "nvidia" | "metax"
    arch: str                            # e.g. "sm86" | "maca_c500"
    compiler_version: str
    torch_version: str
    compute_unit_count: int              # SM 数 (NVIDIA) 或 CU/AP 数 (MetaX C500)
    sm_count: int                        # 与 compute_unit_count 同值（保持向后兼容）
    warp_size: int                       # 32 (NVIDIA) | 64 (MetaX C500)
    max_smem_per_block: int | None       # 单 Block 最大可用共享内存/LDS 字节数（含 opt-in）
    smem_per_block_optin: int | None     # 向后兼容字段
    default_smem_per_block: int | None   # 默认静态单 Block 共享内存上限
    l2_cache_bytes: int | None           # L2 Cache 容量
    vram_bytes: int
    supported_compute_modes: list[str]   # e.g. ["fp16", "bf16", "tf32", "fp32"]
    knowledge_docs: list[str]            # 当前平台推荐阅读的知识库文档路径
    extra: dict[str, Any] = field(default_factory=dict)
```

### 3.2 `KernelSummary`（`gk dump-info` 与 `gk bench` 资源摘要）

```python
@dataclass
class KernelSummary:
    n_regs: int | None
    n_spills: int | None
    n_max_threads: int | None
    shared_bytes: int | None
    allocated_shared_bytes: int | None
    num_warps: int | None
    warp_size: int
    blocks_per_cu: int | None
    blocks_per_sm: int | None            # 与 blocks_per_cu 同值（保持向后兼容）
    occupancy_pct: float | None
    limiting_factor: str | None = None   # "regs" | "shared_memory" | "warps" | "blocks"（支持并列如 "regs|warps"）
    occupancy_source: str | None = None  # "driver_api" | "arch_model"
    extra: dict[str, Any] = field(default_factory=dict)
```

### 3.3 `ProfileData` 与 `CheckResult`

- **`ProfileData`**：在原有 `kernel_name`, `avg_ms`, `std_ms`, `min_ms`, `median_ms`, `trace_path` 基础上，新增：
  - `p95_ms: float`：单次迭代耗时的 95 分位数（无需解析 Chrome trace 即可观测长尾抖动）。
  - `iters: int`：有效测量样本数。
  - `timing_method: str`：`"kineto_kernel"`（Kineto 关联设备核函数）或 `"device_event"`（双 Event 计时）。
- **`CheckResult`**：`gk check` 成功时不再返回 `null`，而是返回：
  - `passed`, `atol`, `rtol`, `max_abs_err`, `max_rel_err`, `eval_path`, `eval_source_hash`, `kernel_path`, `kernel_source_hash`, `checked_at`。

### 3.4 `Backend` 协议与 `SimtBackendBase`

```python
class Backend(Protocol):
    name: str
    device: str
    warp_size: int

    def probe(self) -> ProbeResult: ...
    def extra_layout_constructors(self) -> dict[str, type]: ...
    def summarize_kernel(self, kernel: Any) -> KernelSummary: ...
    def matmul_precision_context(self, compute_mode: str) -> ContextManager[None]: ...

    # 由 SimtBackendBase 统一实现并委派给 core/
    def layout(self, expression: str, shape: list[int], hardware: bool = False) -> str: ...
    def bank_conflicts(
        self, register_layout: str, shared_layout: str, shape: list[int], dtype: str,
    ) -> dict[str, int]: ...
    def dump_info(self, fn: Callable[[], object], output_dir: str | Path) -> dict[str, Any]: ...
    def profile(
        self, fn: Callable[[], object], iters: int, warmup: int, *,
        flush_l2: bool = False, trace_path: str | None = None,
    ) -> ProfileData: ...
```

---

## 4. NVIDIA 与沐曦 C500 后端差异对照表

| 能力维度 | `NvidiaBackend` (`backends/nvidia/backend.py`) | `MetaxBackend` (`backends/metax/backend.py`) |
|---|---|---|
| **`warp_size`** | `32` | `64`（Wave64） |
| **`arch` 标识** | `sm{major}{minor}`（如 `sm86`） | 优先取 `props.gcnArchName` 或识别为 `maca_c500` |
| **专属 Layout** | `NVMMADistributedLayout` | 自动探测注册 MACA / MFMA 相关 Distributed Layout |
| **IR 导出 (`dump-info`)** | `ttgir`, `llir`, `ptx` | `kernel.asm` 中全部文本 IR（`ttgir`, `llir`, `maca` / `isa` / `amdgcn` 等） |
| **Occupancy 与受限因子** | `cuda.bindings.driver` 实测 + SM 寄存器/共享内存/Warp 槽位理论分解推导 `limiting_factor` | 基于 C500 CU 硬件规格（Wave64、64KB LDS、VGPR 池、最大驻留 Wave/Block 数）计算 `blocks_per_cu`、`occupancy_pct` 与 `limiting_factor` |
| **计时策略 (`profile`)** | Kineto 设备核函数关联计时（`kineto_kernel`） | 自适应：优先 Kineto，若 `mcPyTorch` 环境缺少 correlation 字段则自动切换为 `torch.cuda.Event`（`device_event`） |

---

## 5. `core/` 实验控制面加固细节

1. **基线身份校验（`core/baseline.py`）**
   - `.gluonkit/baseline/baseline_<name>.json` 存储 `eval_path`、`eval_source_hash`、`device_key`。
   - `gk baseline <eval.py>` 命中已有文件时校验身份：
     - 身份匹配（或旧版文件自动迁移绑定同一路径与源码哈希）：返回 `"status": "reused"`。
     - 路径、源码哈希或设备不匹配：直接报错拒绝串用，提示使用 `gk baseline <eval.py> --refresh` 显式重测。
2. **校验回执与测速门禁（`core/checker.py`、`core/bencher.py`）**
   - `gk check <eval.py>` 通过后将 `(eval_path, eval_source_hash, kernel_source_hash)` 写入 `.gluonkit/checks.json`。
   - `gk bench <eval.py>` 自动核查 `.gluonkit/checks.json`，在返回结果和 `history.jsonl` 中记录 `"verified": true | false`；传入 `--require-check` 时，若当前算子源码未通过 `check` 则直接拒绝测速。
3. **Roofline 模型分类（`core/roofline.py`）**
   - `metrics` 新增 `"roofline_model": "dual_roofline" | "memory_only" | "compute_only" | null`，消除单侧流式带宽比值超过 `100%` 时的歧义。
