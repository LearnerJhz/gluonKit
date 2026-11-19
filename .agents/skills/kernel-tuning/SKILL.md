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
| 多级预取在主循环每步都执行 `wait_group(0)` | 消费最早 tile 前等待了所有已提交的拷贝 | 按提交与消费的次序设置等待深度，见下方环形预取示例 |
| 访存吞吐低 / shared memory bank conflict | 加载向量化不足或 smem 布局冲突 | 调整 `BlockedLayout` 连续维 `size_per_thread`（如 128-bit = 8 个 fp16）；调整 `SwizzledSharedLayout` 参数（可用 `gl.bank_conflicts` 验证） |
| `tflops` 远低于参考实现且无 spill | Tile 过小、数据复用低或 SM wave 利用不均 | 增大 `BM`/`BN`/`BK`（如 `64x64x32`、`64x128x32`、`128x64x32`）并匹配 `warps_per_cta`；核对 grid 与 `sm_count`（20 SM）匹配度 |

## Ampere 环形预取的等待深度

以 `examples/gemm/gemm.py` 为例，每个 K tile 的 A、B 拷贝合并提交为一个 group。设 `S = NUM_STAGES >= 2`，`T = cdiv(K, BK)`，每一步按以下顺序消费 tile。

1. **预取**前 `min(T, S - 1)` 个有效 tile，每个 tile 提交一次。
2. **主循环**每轮先提交一个新 tile，再执行 `wait_group(S - 1)`，然后消费最早的 tile。此时最早 tile 后面有 `S - 1` 个更新的 group。双缓冲用 `wait_group(1)`，三缓冲用 `wait_group(2)`。
3. **收尾**用 `for d in gl.static_range(S - 1)` 展开。令 `i = T - (S - 1) + d`，仅在 `i >= 0` 时执行 `wait_group(S - 2 - d)` 并消费 tile `i`。这样等待参数是编译期常量，最后一块等待到 0。若 `T < S - 1`，主循环跳过，收尾只读已预取的 tile。

`wait_group(N)` 保证最近提交的 N 个 group 之前的所有 group 均已完成，空 group 在提交时即视为完成。推导等待深度时，按每个线程的提交顺序数出目标 tile 之后的 group。改变每块的提交次数或提交与等待的先后顺序时，需要重新推导。[PTX 异步拷贝等待语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all)

## GEMM A tile 的 bank conflict

对同一个 shared tile，分别用写入时的 `BlockedLayout` 和 MMA 读取时的 `DotOperandLayout` 检查。以下命令离线比较 `[BM, BK] = [64, 32]`、FP16（每行 64B）的两种 swizzle，仅改变 `per_phase`。

```bash
write_layout='BlockedLayout([1,8],[8,4],[4,1],[1,0])'
read_layout='DotOperandLayout(0,NVMMADistributedLayout([2,0],[2,2],[16,8]),2)'
for register_layout in "$write_layout" "$read_layout"; do
  for per_phase in 1 2; do
    .venv/bin/gk bank-conflicts --register-layout "$register_layout" \
      --shared-layout "SwizzledSharedLayout(8,$per_phase,4,[1,0])" \
      --shape 64,32 --dtype float16
  done
done
```

| A tile 访问路径 | `per_phase=1` | `per_phase=2` |
|---|---:|---:|
| 写入 shared memory | 0 | 0 |
| 从 shared memory 读取 MMA 左操作数 | 1 | 0 |

表中数值是 `bank_conflicts_count`，表示每个 wavefront 的额外访问次数。这个布局组合用 `per_phase=2` 可消除读取冲突。更换寄存器布局、tile 形状或 dtype 后，按新参数复算两条路径。

## 停止条件

- 达到目标性能（例如稳定超越 cuBLAS 基线 `speedup_vs_baseline >= 1.0`），或核心杠杆（tile、multi-stage 流水、swizzle、warp 排布、epilogue）已探索充分且连续尝试无 >2–3% 净收益。
- 结束前务必确认 `eval_*.py` 未被改动（`git diff` 检查）且最终版本通过 `gk check`。
