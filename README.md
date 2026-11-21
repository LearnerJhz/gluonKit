# GluonKit (`gk`)

面向 AI Agent 与本地开发者使用的多后端 SIMT GPU（NVIDIA CUDA + 沐曦 MetaX C500 MACA）Gluon kernel 开发、分析与性能调优工具链。

> 本文档与 [ARCHITECTURE.md](ARCHITECTURE.md) 聚焦 `src/gluonkit` 的代码架构、模块职责与接口契约。Agent 调优使用的命令行规则与硬约束以 [AGENTS.md](AGENTS.md) 和 `.venv/bin/gk --help` 为准。

---

## 1. 目录结构

```text
gluonKit/
├── AGENTS.md                         # Agent 入口规则与 CLI 契约（SSOT）
├── ARCHITECTURE.md                   # 多后端 SIMT 架构设计文档（NVIDIA + 沐曦 C500）
├── README.md                         # 本文档：src/gluonkit 模块参考与使用说明
├── pyproject.toml                    # 包配置与 CLI 入口 gk = "gluonkit.cli:main"
├── scripts/
│   └── calibrate_peaks.py            # 跨平台本机峰值标定脚本（写入 .gluonkit/peaks.json）
├── examples/gemm/
│   ├── gemm.py                       # 示例待调优 Gluon GEMM（FP16 矩阵乘、FP32 累加）
│   └── eval_gemm.py                  # 示例评测脚本（prepare 入口，同目录配对 gemm.py）
├── src/gluonkit/
│   ├── __init__.py                   # __version__ = "0.1.0"
│   ├── cli.py                        # argparse 入口（含 --backend）、环境变量初始化与命令分发
│   ├── output.py                     # stdio 重定向、traceback 格式化、浮点保留位与 JSON 输出
│   ├── executor.py                   # 各命令调度、任务加载、校验回执管理与源码快照校验
│   ├── interfaces/
│   │   ├── backend.py                # Backend Protocol 协议定义
│   │   └── types.py                  # Output / ProbeResult / KernelSummary / ProfileData / CheckResult / BenchResult
│   ├── core/
│   │   ├── storage.py                # .gluonkit/ 原子 JSON 读写与路径规范化
│   │   ├── device_diag.py            # GPU 初始化失败结构化诊断（沙盒节点拦截 / 环境变量隐藏 / 驱动异常）
│   │   ├── gluon_api.py              # Gluon API 自省（支持 --full 完整 docstring 与源码位置，无需 GPU）
│   │   ├── simt_layout.py            # 参数化 warp_size 的 Layout 渲染与离线 Bank Conflict 静态分析
│   │   ├── triton_dump.py            # CompiledKernel 拦截、kernel.asm 全量文本 IR 导出与受限因子推导
│   │   ├── torch_profiler.py         # Kineto 核函数关联计时 + Device Event 计时双引擎（含 P95 统计）
│   │   ├── checker.py                # check_task 误差统计与 CheckStore（.gluonkit/checks.json 读写）
│   │   ├── baseline.py               # BaselineStore（绑定 eval_path + eval_source_hash + device_key）
│   │   ├── bencher.py                # measure_baseline / bench_task / run_benchmark
│   │   ├── roofline.py               # validate_metadata / calculate_roofline（区分 dual_roofline / memory_only / compute_only）
│   │   ├── peaks.py                  # PeakStore（.gluonkit/peaks.json 读写）与 resolve_peaks
│   │   └── history.py                # History：.gluonkit/ 实验记录、父实验 Δ 计算与快照恢复
│   └── backends/
│       ├── __init__.py
│       ├── base.py                   # SimtBackendBase：统一委派 layout / bank_conflicts / dump_info / profile
│       ├── registry.py               # 后端注册与自动探测（auto / nvidia / metax）
│       ├── nvidia/
│       │   └── backend.py            # NvidiaBackend（warp_size=32, NVMMADistributedLayout, cuOccupancy + 受限因子）
│       └── metax/
│           └── backend.py            # MetaxBackend（沐曦 C500: warp_size=64, Wave64 Occupancy 模型, Event/Kineto 自适应）
└── tests/
    └── test_architecture.py          # 多后端架构、Wave32/Wave64 Layout、基线防碰撞、Check 门禁等自动化测试
```

---

## 2. 常用操作与运行时目录

```bash
# 安装 / 可编辑模式开发（需要 Python >= 3.10, torch, triton==3.8.0）
pip install -e .

# 常用 CLI（默认自动探测后端，也可传 --backend nvidia 或 --backend metax）
.venv/bin/gk probe
.venv/bin/python scripts/calibrate_peaks.py          # 一次性标定本机峰值
.venv/bin/gk check examples/gemm/eval_gemm.py        # 数值校验并生成校验回执
.venv/bin/gk baseline examples/gemm/eval_gemm.py     # 建立（或 --refresh 刷新）参考基线
.venv/bin/gk bench examples/gemm/eval_gemm.py --note "baseline" --require-check
.venv/bin/gk dump-info examples/gemm/eval_gemm.py
.venv/bin/gk history examples/gemm/eval_gemm.py --limit 5
.venv/bin/gk restore <history_id>
.venv/bin/gk api list [module]                       # 列出 Gluon API
.venv/bin/gk api show --full load store              # 查询符号签名、完整参数说明与定义源码行号
```

运行时在工作目录下生成 `.gluonkit/`（已加入 `.gitignore`）：
- `.gluonkit/triton_cache/`：Triton 编译缓存
- `.gluonkit/history.jsonl`：每次 `gk bench` 的实验记录
- `.gluonkit/peaks.json`：按 `<platform>:<arch>:<device_name>` 存储的标定峰值
- `.gluonkit/checks.json`：`gk check` 记录的最新源码校验回执
- `.gluonkit/baseline/baseline_<name>.json`：`gk baseline <eval.py>` 存储的 `ref()` 基线测量结果（绑定 `eval_path`、`eval_source_hash`、`device_key`）
- `.gluonkit/sources/<sha256>.py`：评测脚本与算子源码内容寻址快照
- `.gluonkit/dump_info/<eval_stem>/<kernel>/<hash>/`：`kernel.ttgir`、`kernel.llir`、`kernel.ptx`（或目标平台 ISA）
- `.gluonkit/traces/bench-*/trace.json`：`gk bench --trace` 导出的 Chrome trace
