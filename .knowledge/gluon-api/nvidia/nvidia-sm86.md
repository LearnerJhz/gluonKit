# Ampere sm86 上的 FP16 GEMM

仓库示例让一个 CTA 计算一个输出 tile，在共享内存中预取 A、B，然后用 Tensor Core 做矩阵乘加。通用的 tile 索引、广播和共享内存描述符可单独查阅。[GEMM 源码](../../../examples/gemm/gemm.py) · [Gluon 通用核心功能](../gluon-api-common.md)

## 导入与布局

```python
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout, DotOperandLayout, NVMMADistributedLayout,
    NVMMASharedLayout, SliceLayout, SwizzledSharedLayout,
)
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2
```

sm86 上每个 warp 有 32 个线程。构造二维 `BlockedLayout` 时，`threads_per_warp` 两个分量之积为 32。可在仓库根目录用以下命令查看线程、warp 与元素的映射。

```bash
.venv/bin/gk layout "BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])" --shape 64,64 --hardware
```

| 数据 | 该示例的选择 |
|---|---|
| 累加器 | `NVMMADistributedLayout(version, warps_per_cta, instr_shape, cga_layout=[])`。示例为 `NVMMADistributedLayout([2, 0], [2, 2], [16, 8])`，用四个 warp 累加到 FP32 |
| A、B 乘数 | `DotOperandLayout(operand_index, parent, k_width)`。A 的 `operand_index=0`，B 为 1，两者的 `parent` 指向累加器布局，FP16 示例的 `k_width=2` |
| 共享 tile | 此例使用二维 `SwizzledSharedLayout`。若选择 NVIDIA MMA 专用的共享布局，可用 `NVMMASharedLayout(swizzle_byte_width, element_bitwidth, rank=2, transposed=False, fp4_padded=False, cga_layout=[])`。`NVMMASharedLayout.get_default_for(block_shape, dtype, transposed=False, fp4_padded=False, cga_layout=None)` 按每 CTA 连续维的字节数和 tile 形状选择 swizzle。例如给 A 的单个 tile 传入 `[BM, BK], gl.float16` |

## 异步拷贝与矩阵乘加

1. 用 `gl.program_id` 选取输出块，从 A、B 的加载布局派生 `SliceLayout` 构建 `rows`、`cols`、`ak` 和 `bk`。分别为 A、B 分配带 stage 维的共享内存，按累加器布局用 `gl.full` 初始化 `acc`。
2. 对当前 stage，用 `async_copy.async_load(smem, pointer, mask=None, cache_modifier='', eviction_policy='', volatile=False)` 从全局内存填充 A、B。目标共享描述符的形状须与广播后的指针张量一致。示例将同一个 K tile 的两次拷贝合为一组，用 `async_copy.commit_group()` 提交。
3. 在消费 stage 前调用 `async_copy.wait_group(num_outstanding=0)`。参数是编译期常量。值为 0 时等到全部已提交组完成，值为 N 时允许最近的至多 N 个组仍在执行。主循环按提交顺序推导等待深度，最后等待并消费所有已预取的有效 tile。
4. 对 A、B 分别用 `smem.index(stage).load(DotOperandLayout(...))` 读出乘数，再执行 `acc = mma_v2(a, b, acc, input_precision=None)`。收尾时把 `acc` 转成 FP16，用 `gl.convert_layout` 切换到输出布局并通过带 mask 的 `gl.store` 写回。

`async_load` 前端可解析的 `cache_modifier` 有 `""`、`".ca"`、`".cg"`、`".cs"`、`".cv"`，`eviction_policy` 有 `""`、`"evict_last"`、`"evict_first"`。Ampere PTX 的 `cp.async.ca` 支持每次拷贝 4、8、16 字节，`cp.async.cg` 要求 16 字节。给定指针、步长与 mask 能达到的连续宽度须满足目标指令的条件。需核对实际生成的指令时，运行 `.venv/bin/gk dump-info <eval.py>` 并查看产物。[PTX 异步拷贝指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)

同一共享 tile 的写入和读出可能有不同的 bank 冲突。`gl.bank_conflicts(distr_ty, shared_ty)` 估计给定类型组合每个 wavefront 的额外访问次数。用写入时的 `BlockedLayout` 和读出时的 `DotOperandLayout` 分别检查，详见 [GEMM A tile 的 bank conflict](#gemm-a-tile-的-bank-conflict)。

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

遇到拷贝宽度、乘数布局或共享内存资源报错时，按当前输入与编译消息检查。[编译与运行排错](../troubleshooting.md)
