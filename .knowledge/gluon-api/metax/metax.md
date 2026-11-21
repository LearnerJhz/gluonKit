# 沐曦 (MetaX) C500 平台扩展与调优指南

沐曦 C500 采用 SIMT GPGPU 架构（MACA 软件栈，通过 `torch.cuda` 兼容接口与 `mcTriton` 运行），在 Gluon 中与 NVIDIA GPU 共享 `BlockedLayout`、`SliceLayout`、`SwizzledSharedLayout` 等 SIMT 核心抽象，但存在以下关键硬件差异：

## 1. 核心硬件参数差异（Wave64 vs Warp32）

- **`warp_size = 64`（Wave64）**：
  - 构造 `BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order)` 时，**`math.prod(threads_per_warp)` 必须等于 `64`**（而非 NVIDIA 的 `32`）。
  - 可使用 `.venv/bin/gk --backend metax layout "<BlockedLayout(...)>" --shape ... --hardware` 离线校验 Wave64 线程映射。
- **片上共享存储（LDS / Shared Memory）**：
  - 单 CU / Block 的 LDS 容量通常为 `64 KiB`（`65536 B`），不存在 NVIDIA Ampere 的 `101,376 B` 动态 `opt-in` 扩展空间。
  - 在多级流水线（`NUM_STAGES`）或大 Tile（如 `128x128`）设计时，需确保 `allocated_shared_bytes <= 65536`。
- **并发与驻留（Occupancy）**：
  - 单 CU 最大驻留 Wave 数为 `32`（即 `2048` 线程），单 CU 向量寄存器（VGPR）池为 `65536` 个 32-bit 寄存器（按 256 粒度对齐）。
  - 执行 `.venv/bin/gk dump-info <eval.py>` 时，`summary` 中的 `limiting_factor` 会直接指出当前驻留受限于 `regs`、`shared_memory`、`warps` 还是 `blocks`。

## 2. 编译产物与计时说明

- `gk dump-info` 会自动导出 `kernel.asm` 中的全部文本 IR（包括 `ttgir`、`llir` 以及 MACA 后端生成的目标汇编/ISA）。
- `gk bench` 默认优先尝试 Kineto 核函数级关联计时（`timing_method: "kineto_kernel"`）；若当前 `mcPyTorch` 环境未开启 Triton 自定义发射的 correlation 关联，会自动无缝降级为设备 Event 计时（`timing_method: "device_event"`）。
