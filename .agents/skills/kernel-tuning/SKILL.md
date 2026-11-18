---
name: kernel-tuning
description: Systematic GPU kernel performance tuning loop using gk probe, check, bench, and dump-info. Use when optimizing a Gluon kernel for speed while preserving numerical correctness.
---

# Gluon Kernel 性能调优闭环

## 标准调优流程

1. **摸底（会话开始跑一次）**
   - `.venv/bin/gk probe`：记录 `arch`、`sm_count`、`smem_per_block_optin`（例如 RTX 3050 Laptop：`sm86`、`20` SM、`101376` bytes）。
2. **建基线**
   - `.venv/bin/gk check <eval.py>` -> `.venv/bin/gk bench <eval.py> --note "baseline"` -> `.venv/bin/gk dump-info <eval.py>`
   - 记录基线 `history_id`、`device_time_ms`、`std_ms`、`min_ms`、`median_ms`、`baseline_time_ms`、`speedup_vs_baseline` 以及 `summary`（`n_regs`、`n_spills`、`shared_bytes`、`allocated_shared_bytes`、`num_warps`、`blocks_per_sm`、`occupancy_pct`；注意不要只看 `n_max_threads` 误判并发）。
   - `gk` 将 Triton 编译缓存固定写入 `.gluonkit/triton_cache`（沙盒可直接写入）；首次编译略慢，后续仅在算子源码或编译常量变化时重编（需要强制重编可传 `TRITON_ALWAYS_COMPILE=1`）。
3. **提假设并单点改动**
   - 每次只改一个明确变量（例如只改 tile size，或只改 pipeline stage 数），记录“改了什么、预期改善什么”。
4. **先校验后测速（不可跳过）**
   - 先跑 `.venv/bin/gk check <eval.py>`；失败立即修复或回退，禁止跳过 `check` 直接测速。
   - 再跑 `.venv/bin/gk bench <eval.py> --note "<改动说明>" --parent <上次id>`。
5. **噪声判断与留存**
   - 结合 `gk bench` 返回的 `std_ms`、`min_ms`、`median_ms` 判断波动显著性（例如对比均值差与 `std_ms`，以及 `median_ms` 与 `device_time_ms` 是否存在长尾离群）。
   - **判断阈值：单次改动提速小于 2%–3% 或未显著超出测量标准差 `std_ms` 时不算确认收益**；若接近临界值，连续复测 2–3 次观察稳定性。
   - 确认明显更快 -> 保留（建议用 `git` 记录版本）；变慢或无效 -> 立即回退，切下一个假设。
6. **按需诊断**
   - 性能未按预期提升时跑 `gk dump-info <eval.py>` 检查资源摘要和导出的 `ttgir`/`llir`/`ptx`；需要看硬件发射、流间空隙或确认并发受限因子（`limitingFactors`）时跑 `gk bench <eval.py> --trace`。

## 瓶颈诊断与优化杠杆

| 观测现象（来自 `bench` / `dump-info` / IR） | 根因定位 | 优先尝试的优化手段 |
|---|---|---|
| `n_spills > 0`（或 PTX 含 `ld.local`/`st.local`） | 寄存器溢出到 local memory | 减小 `BM`/`BN`/`BK`、减少同时活跃变量或 stage 数、消除冗余 `convert_layout` |
| `blocks_per_sm` / `occupancy_pct` 低 | 并发可能受寄存器（`n_regs`）、共享内存（`allocated_shared_bytes`，含约 1024B 保留开销）、warp 数或 SM block 上限限制（注：`summary` 暂不提供受限因子拆解，`n_max_threads` 也不含共享内存限制；需要精确区分瓶颈时结合 `n_regs` / `allocated_shared_bytes` 估算或跑 `gk bench --trace` 查看 `limitingFactors`） | 若受寄存器限制可减变量/调整 `num_warps`；若受共享内存限制可减小 tile 或 `NUM_STAGES`；若更大的 tile 复用收益更高则优先保留吞吐 |
| `shared_bytes` 接近 `smem_per_block_optin` | 共享内存限制并发 block 或多级缓冲 | 收紧 `NUM_STAGES` 或调整 tile 形状，避免超 `smem_per_block_optin` |
| `ttgir` 中循环内存在多余 `convert_layout` | 寄存器布局不匹配触发隐藏 smem 中转或高开销指令 | 统一加载、计算、写回的 layout 链路，尽量把转换推到循环外（如 epilogue） |
| 主循环每步都是 `wait_group(0)`（单级阻塞） | 全局访存延迟未掩盖（memory/latency bound） | 改用 `[NUM_STAGES, ...]` 共享内存环形 buffer + 预取（`wait_group(NUM_STAGES - 2)`，如 2–3 stages） |
| 访存吞吐低 / shared memory bank conflict | 加载向量化不足或 smem 布局冲突 | 调整 `BlockedLayout` 连续维 `size_per_thread`（如 128-bit = 8 个 fp16）；调整 `SwizzledSharedLayout` 参数（可用 `gl.bank_conflicts` 验证） |
| `tflops` 远低于参考实现且无 spill | Tile 过小、数据复用低或 SM wave 利用不均 | 增大 `BM`/`BN`/`BK`（如 `64x64x32`、`64x128x32`、`128x64x32`）并匹配 `warps_per_cta`；核对 grid 与 `sm_count`（20 SM）匹配度 |

## 停止条件

- 达到目标性能（例如稳定超越 cuBLAS 基线 `speedup_vs_baseline >= 1.0`），或核心杠杆（tile、multi-stage 流水、swizzle、warp 排布、epilogue）已探索充分且连续尝试无 >2–3% 净收益。
- 结束前务必确认 `eval_*.py` 未被改动（`git diff` 检查）且最终版本通过 `gk check`。
