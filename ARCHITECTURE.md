# GluonKit (`gk`) 架构说明（SIMT 后端静态注册：NVIDIA → 沐曦 C500）

本文档说明 `src/gluonkit` 的目录结构、职责分层与从 NVIDIA 切换到沐曦 C500 的方式。

---

## 1. 设计定位

- **单机确定性平台**：工具每次只服务于当前运行机器上的确定平台（当前在 NVIDIA 上跑通全流程，未来部署到沐曦 C500 机器上运行）。
- **不做动态探测与过度抽象**：不引入复杂的运行时环境嗅探、多余的抽象基类或 CLI `--backend` 开关；通过 `backends/registry.py` 静态注册后端，迁移机器时只改一行 `DEFAULT_BACKEND = "metax"`。
- **保持 SIMT 模型一致**：NVIDIA 与沐曦 C500 同属 SIMT GPU，共用同一套 `Backend` 协议（`probe`、`layout`、`bank_conflicts`、`dump_info`、`profile`），仅在后端内部按硬件差异（如 `warp_size=32` vs `warp_size=64`、编译产物格式、Occupancy 计算）各自实现。

---

## 2. 目录结构

```text
gluonKit/
├── AGENTS.md                         # Agent 入口规则与 CLI 契约
├── ARCHITECTURE.md                   # 本文档：架构与目录结构说明
├── README.md                         # 模块职责与 CLI 使用说明
├── pyproject.toml                    # 包配置与 CLI 入口 gk = "gluonkit.cli:main"
├── scripts/
│   └── calibrate_peaks.py            # 本机峰值标定脚本（写入 .gluonkit/peaks.json）
└── src/gluonkit/
    ├── __init__.py
    ├── cli.py                        # CLI 参数解析与命令分发（通过 get_backend() 获取当前后端）
    ├── output.py                     # 统一 JSON 信封与浮点数格式化
    ├── executor.py                   # 各子命令调度与任务加载
    ├── interfaces/
    │   ├── backend.py                # Backend Protocol（probe / layout / bank_conflicts / dump_info / profile）
    │   └── types.py                  # Output / ProbeResult / ProfileData / BenchResult
    ├── core/
    │   ├── storage.py                # .gluonkit/ 目录与原子 JSON 读写
    │   ├── gluon_api.py              # Gluon API 自省（支持 gk api list / show [--full]）
    │   ├── checker.py                # check_task 数值校验与误差统计
    │   ├── baseline.py               # BaselineStore（.gluonkit/baseline/*.json 固定基线读写）
    │   ├── bencher.py                # 基线测量、算子测速与实验落盘
    │   ├── roofline.py               # .gluonkit/peaks.json 中枢读写、TFLOP/s、GB/s 与 Roofline 计算
    │   └── history.py                # History（.gluonkit/history.jsonl 读写与源码快照恢复）
    └── backends/
        ├── __init__.py
        ├── registry.py               # 后端注册表：注册 nvidia / metax，通过 DEFAULT_BACKEND 指定当前后端
        ├── nvidia/                   # NVIDIA 后端实现（warp_size = 32）
        │   ├── backend.py            # NvidiaBackend
        │   ├── probe.py              # 设备属性探测
        │   ├── layout.py             # Layout 解析与渲染（支持传入 warp_size）
        │   ├── bank_conflicts.py     # 离线 Bank Conflict 计算
        │   ├── dump_info.py          # 导出 ttgir/llir/ptx 与寄存器/Occupancy/受限因子摘要
        │   └── profiler.py           # torch.profiler 核函数计时（含 P95 统计）
        └── metax/                    # 沐曦 C500 后端实现（warp_size = 64）
            ├── __init__.py
            └── backend.py            # MetaxBackend（Wave64 布局校验、C500 IR 导出与 Occupancy、计时）
```

---

## 3. 核心接口与切换方式

### 3.1 后端注册表（[src/gluonkit/backends/registry.py](src/gluonkit/backends/registry.py)）

```python
BACKENDS: dict[str, type[Backend]] = {
    "nvidia": NvidiaBackend,
    "metax": MetaxBackend,
}

# 当前机器使用的后端与设备；部署到沐曦 C500 时将 DEFAULT_BACKEND 改为 "metax"
DEFAULT_BACKEND = "nvidia"
DEFAULT_DEVICE = "cuda:0"


def get_backend(name: str = DEFAULT_BACKEND, *, device: str = DEFAULT_DEVICE) -> Backend:
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; registered: {list(BACKENDS)}")
    return BACKENDS[name](device=device)
```

`cli.py` 直接调用 `backend = get_backend()`，无需每次传参或做环境探测。

### 3.2 `Backend` 协议（[src/gluonkit/interfaces/backend.py](src/gluonkit/interfaces/backend.py)）

```python
class Backend(Protocol):
    device: str
    warp_size: int

    def probe(self) -> ProbeResult: ...
    def layout(self, expression: str, shape: list[int], hardware: bool = False) -> str: ...
    def bank_conflicts(
        self, register_layout: str, shared_layout: str, shape: list[int], dtype: str,
    ) -> dict[str, int]: ...
    def dump_info(
        self, fn: Callable[[], object], output_dir: str | Path,
    ) -> dict[str, Any]: ...
    def profile(
        self, fn: Callable[[], object], iters: int, warmup: int, *,
        flush_l2: bool = False, trace_path: str | None = None,
    ) -> ProfileData: ...
```

### 3.3 数据结构（[src/gluonkit/interfaces/types.py](src/gluonkit/interfaces/types.py)）

- **`ProbeResult`**：`device_name`, `platform`, `arch`, `compiler_version`, `torch_version`, `warp_size`, `smem_per_block_optin`, `sm_count`, `vram_bytes`
- **`ProfileData`**：`kernel_name`, `avg_ms`, `std_ms`, `min_ms`, `median_ms`, `p95_ms`, `trace_path`
- **`BenchResult`**：`kernel_profile`, `metrics`（`speedup`, `tflops`, `gbps`, `roofline`, `bound_by`, `roofline_model`）, `baseline`, `peaks`
