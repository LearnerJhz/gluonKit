# GluonKit (`gk`)

面向 AI Agent 与本地开发者使用的 SIMT GPU Gluon kernel 开发、分析与性能调优工具链。

> 本文档与 [ARCHITECTURE.md](ARCHITECTURE.md) 说明 `src/gluonkit` 的代码架构、模块职责与接口契约。Agent 调优使用的命令行规则与硬约束以 [AGENTS.md](AGENTS.md) 和 `.venv/bin/gk --help` 为准。

---

## 1. 目录结构

```text
gluonKit/
├── AGENTS.md                         # Agent 入口规则与 CLI 契约（SSOT）
├── ARCHITECTURE.md                   # 架构说明（SIMT 后端静态注册：NVIDIA → 沐曦 C500）
├── README.md                         # 本文档：src/gluonkit 模块参考与使用说明
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
    │   ├── storage.py                # .gluonkit/ 原子 JSON 读写与路径规范化
    │   ├── gluon_api.py              # Gluon API 自省（支持 --full，纯文本，无需 GPU）
    │   ├── checker.py                # check_task：torch.testing.assert_close 校验与误差统计
    │   ├── baseline.py               # BaselineStore（.gluonkit/baseline/*.json 固定基线读写）
    │   ├── bencher.py                # measure_baseline / bench_task：独立基线测量与算子测速汇总
    │   ├── roofline.py               # load_peaks / save_peaks / resolve_peaks / calculate_roofline：.gluonkit/peaks.json 读写与 Roofline 计算
    │   └── history.py                # History：.gluonkit/ 实验记录、摘要读取与快照恢复
    └── backends/
        ├── __init__.py
        ├── registry.py               # 后端静态注册表（DEFAULT_BACKEND = "nvidia"，切沐曦改 "metax"；DEFAULT_DEVICE = "cuda:0"）
        ├── nvidia/
        │   ├── backend.py            # NvidiaBackend（warp_size = 32）
        │   ├── probe.py              # GPU 设备属性与 PyTorch/Triton 版本探测
        │   ├── layout.py             # AST 白名单解析与 Triton layout 映射渲染（支持传入 warp_size）
        │   ├── bank_conflicts.py     # 离线构建 Gluon IR 调用 gl.bank_conflicts
        │   ├── dump_info.py          # 拦截 CompiledKernel 导出 IR、occupancy 与受限因子计算
        │   └── profiler.py           # torch.profiler 事件关联计时、P95 统计与 trace 导出
        └── metax/
            ├── __init__.py
            └── backend.py            # MetaxBackend（沐曦 C500：warp_size = 64）
```

---

## 2. 常用操作与运行时目录

```bash
# 安装 / 可编辑模式开发（需要 Python >= 3.10, torch, triton==3.8.0）
pip install -e .

# 常用 CLI
.venv/bin/gk probe
.venv/bin/python scripts/calibrate_peaks.py          # 环境初始化时一次性标定本机峰值
.venv/bin/gk check examples/gemm/eval_gemm.py
.venv/bin/gk baseline examples/gemm/eval_gemm.py     # 建立固定参考基线（已存在则直接返回）
.venv/bin/gk bench examples/gemm/eval_gemm.py --note "baseline"
.venv/bin/gk dump-info examples/gemm/eval_gemm.py
.venv/bin/gk history examples/gemm/eval_gemm.py --limit 5
.venv/bin/gk restore <history_id>
.venv/bin/gk api list [module]                       # 列出 Gluon API
.venv/bin/gk api show --full load store              # 查询符号签名、完整参数说明与定义源码行号
```

运行时在工作目录下生成 `.gluonkit/`（已加入 `.gitignore`）：
- `.gluonkit/triton_cache/`：Triton 编译缓存
- `.gluonkit/history.jsonl`：每次 `gk bench` 的实验记录
- `.gluonkit/peaks.json`：本机峰值配置（由 `calibrate_peaks.py` 标定或直接手填）
- `.gluonkit/baseline/baseline_<name>.json`：`gk baseline <eval.py>` 存储的 `ref()` 基线测量结果
- `.gluonkit/sources/<sha256>.py`：评测脚本与算子源码内容寻址快照
- `.gluonkit/dump_info/<eval_stem>/<kernel>/<hash>/`：`kernel.ttgir`、`kernel.llir`、`kernel.ptx`（或 `kernel.maca`）
- `.gluonkit/traces/bench-*/trace.json`：`gk bench --trace` 导出的 Chrome trace
