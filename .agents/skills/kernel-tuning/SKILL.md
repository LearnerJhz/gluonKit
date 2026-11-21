---
name: kernel-tuning
description: Systematic GPU kernel performance tuning loop using gk probe, check, bench, and dump-info. Use when optimizing a Gluon kernel for speed while preserving numerical correctness.
---

# Gluon Kernel 性能调优闭环

当实验涉及更换归约、布局、矩阵指令或异步搬运原语时，按目标从 [AGENTS.md 的 Gluon API 知识库](../../../AGENTS.md#gluon-api-知识库) 选读对应平台页与通用页。

## 标准调优流程

1. **摸底（会话开始跑一次）**
   - `.venv/bin/gk probe`：记录 `arch`、`sm_count`、`warp_size` 与共享内存上限（如 NVIDIA 后端返回的 `smem_per_block_optin`）。
   - Roofline 峰值默认从预先标定好的 `.gluonkit/peaks.json` 读取（未生成该文件且评测未显式声明 `peaks` 时，`gk bench` 的 `roofline` 与 `bound_by` 为 `null`）；调优期间不要运行或重跑 `scripts/calibrate_peaks.py`。
2. **建基线**
   - `.venv/bin/gk check <eval.py>` -> `.venv/bin/gk baseline <eval.py>`（单独测一次 `ref()` 并存入 `.gluonkit/baseline/baseline_<name>.json`）-> `.venv/bin/gk bench <eval.py> --note "baseline"` -> `.venv/bin/gk dump-info <eval.py>`
   - 记录基线 `history_id`、`kernel.name`、`kernel.avg`、`kernel.std`、`kernel.min`、`kernel.median`、`baseline.avg_ms`、`metrics.speedup` 以及 `summary`（`n_regs`、`n_spills`、`shared_bytes`、`allocated_shared_bytes`、`num_warps`、`blocks_per_sm`、`occupancy_pct`；注意不要只看 `n_max_threads` 误判并发）。
   - 日常 `gk bench <eval.py>` 默认只测待测算子并直接读取已存基线算 `speedup`；基线文件已存在时 `gk baseline` 不会重测，若确实需要刷新基线，删除对应 `.gluonkit/baseline/baseline_<name>.json` 后重跑一次 `.venv/bin/gk baseline <eval.py>` 即可。
   - `gk` 将 Triton 编译缓存固定写入 `.gluonkit/triton_cache`（沙盒可直接写入）；首次编译略慢，后续仅在算子源码或编译常量变化时重编（需要强制重编可传 `TRITON_ALWAYS_COMPILE=1`）。
3. **提假设并单点改动**
   - 每次只改一个明确变量（例如只改 tile size，或只改 pipeline stage 数），记录“改了什么、预期改善什么”。
4. **先校验后测速（不可跳过）**
   - 每次修改后都用同一条 `.venv/bin/gk check <eval.py>` 验证约定的主用例，按原评测的容差与 `ref()` 比较。校验失败先修复或回退。
   - **按算子的输入约定决定是否补测。** 若 shape、dtype 和布局固定，主用例通过即可，无需另测奇数宽度。若算子承诺支持任意行宽，本轮又修改了尾块处理，就补测一个不能被 tile 宽度整除的行宽。若补测发现 bug，把能复现它的输入留作普通回归测试。
   - 按实际输入和本轮 tile 逐维判断边界。例如当 `K` 能被 `BK` 整除时，可省去对应 K 维尾块 mask，但流水线收尾仍要等待并消费最后已预取的 tile。调整 tile 后重新检查整除关系和访存对齐，对实际出现的尾块保留保护。
   - 再跑 `.venv/bin/gk bench <eval.py> --note "<改动说明>" --parent <上次id>`。
5. **噪声判断与留存**
   - 结合 `gk bench` 返回的 `kernel.std`、`kernel.min`、`kernel.median` 判断波动显著性（例如对比均值差与 `kernel.std`，以及 `kernel.median` 与 `kernel.avg` 是否存在长尾离群）。
   - **判断阈值：单次改动提速小于 2%–3% 或未显著超出测量标准差 `kernel.std` 时不算确认收益**；若接近临界值，连续复测 2–3 次观察稳定性。
   - 确认明显更快 -> 保留当前版本（以此次 `history_id` 作为后续 `--parent`）；变慢或无效 -> 用 `gk restore <id>` 回退单轮算子实验，切下一个假设。
6. **按需诊断**
   - 性能未按预期提升时跑 `gk dump-info <eval.py>` 检查资源摘要和后端导出的中间表示/汇编；需要看发射空隙、并发受限因子或执行轨迹时跑 `gk bench <eval.py> --trace`。

## 瓶颈诊断与优化杠杆

| 观测现象（来自 `bench` / `dump-info` / 中间产物） | 根因定位 | 优先尝试的优化手段 |
|---|---|---|
| `n_spills > 0`（或后端产物出现 local/scratch 溢出） | 寄存器溢出到慢速局部存储 | 减小 `BM`/`BN`/`BK`、减少同时活跃变量或 stage 数、消除冗余 `convert_layout` |
| `blocks_per_sm` / `occupancy_pct` 低 | 并发受寄存器（`n_regs`）、共享内存（`allocated_shared_bytes`）、warp 数或计算单元 block 上限限制（注：`summary` 暂不提供受限因子拆解，`n_max_threads` 也不含共享内存限制；可用 `gk bench --trace` 或后端信息进一步区分瓶颈） | 若受寄存器限制可减变量/调整 `num_warps`；若受共享内存限制可减小 tile 或 `NUM_STAGES`；若更大的 tile 复用收益更高则优先保留吞吐 |
| `shared_bytes` 接近设备单 block 上限 | 共享内存限制并发 block 或多级缓冲 | 收紧 `NUM_STAGES` 或调整 tile 形状，避免超出设备上限 |
| 中间 IR 中循环内存在多余 `convert_layout` | 寄存器布局不匹配触发隐藏 shared 中转或高开销指令 | 统一加载、计算、写回的 layout 链路，尽量把转换推到循环外（如 epilogue） |
| 多级预取在主循环每步等待全部传输完成 | 消费最早 tile 前过早排空了流水线 | 按提交与消费次序设置最小等待深度（如 NVIDIA Ampere 见 [Ampere 环形预取的等待深度](../../../.knowledge/gluon-api/nvidia/nvidia-sm86.md#ampere-环形预取的等待深度)） |
| 访存吞吐低 / shared memory bank conflict | 加载向量化不足或 shared 布局冲突 | 调整 `BlockedLayout` 连续维 `size_per_thread` 提升向量化宽度，调整 `SwizzledSharedLayout` 参数并用 `gk bank-conflicts` / `gl.bank_conflicts` 验证（如 [GEMM A tile 示例](../../../.knowledge/gluon-api/nvidia/nvidia-sm86.md#gemm-a-tile-的-bank-conflict)） |
| `metrics.tflops` 远低于基线且无 spill | Tile 过小、数据复用低或计算单元 wave 利用不均 | 增大 tile 并匹配 `warps_per_cta`；核对 grid 与设备计算单元数（`sm_count`）的 wave 匹配度 |

## 停止条件

- 达到目标性能（例如稳定超越基线 `metrics.speedup >= 1.0`），或核心杠杆（tile、multi-stage 流水、swizzle、warp 排布、epilogue）已探索充分且连续尝试无 >2–3% 净收益。
- 结束前务必确认 `eval_*.py` 未被改动且最终版本通过 `gk check`。
