# A. 通用核心功能

用 `from triton.experimental import gluon` 引入主机端入口，用 `from triton.experimental.gluon import language as gl` 在 kernel 中表达运算。“通用”指可以沿用同一套编程语义，但执行组宽度（warp/wavefront）、合法 dtype 与布局参数、共享内存容量和部分操作的作用域，仍由目标后端决定。

## 通用 API 全览（Triton 3.8.0）

| 功能域 | API 列表 | 关键语义与使用注意 |
|---|---|---|
| 运行时与编译期控制 | `@gluon.jit`、`gluon.constexpr_function`、`gluon.aggregate`、`gluon.must_use_result`；`gl.constexpr`、`gl.static_range`、`gl.static_assert`、`gl.static_print`；`gl.assume`、`gl.multiple_of`、`gl.max_constancy`、`gl.max_contiguous` | 主机通过 `kernel[grid](...)` 传递参数和编译期常量；`constexpr` 与 `static_range` 在编译期展开，不可在循环内对同名 `constexpr` 重复绑定 |
| 执行规模与坐标 | `gl.program_id`、`gl.num_programs`、`gl.num_warps`、`gl.num_ctas` | 获取当前 program 在 grid 中的坐标、总 grid 规模及当前执行上下文的 warp/CTA 数量 |
| 类型与描述符容器 | `gl.dtype`、`gl.void`、`gl.int1`/`8`/`16`/`32`/`64`、`gl.uint8`/`16`/`32`/`64`、`gl.float16`、`gl.bfloat16`、`gl.float32`、`gl.float64`、`gl.float8e5`/`e5b16`/`e4nv`/`e4b8`/`e4b15`；`gl.pointer_type`、`gl.tensor`、`gl.distributed_type`、`gl.block_type`、`gl.base_value`、`gl.base_type`、`gl.shared_memory_descriptor`、`gl.shared_memory_descriptor_type`、`gl.tuple`、`gl.tuple_type`、`gl.aggregate_replace` | 常用属性 `tensor.shape`、`tensor.dtype`、`tensor.type`；低精度格式（如 FP8）的合法组合依赖目标后端支持 |
| 整数、位运算与比较 | `gl.add`、`gl.sub`、`gl.mul`、`gl.umulhi`、`gl.cdiv`；tensor 运算符 `+`、`-`、`*`、`/`、`//`、`%`、位运算、移位、比较、取反、`logical_and`/`logical_or` | `gl.cdiv(x, div)` 为向上取整除法，常用于计算覆盖维度的 tile 数量 |
| 浮点基础与选择 | `gl.fma`、`gl.fdiv`、`gl.div_rn`、`gl.abs`、`gl.minimum`、`gl.maximum`、`gl.clamp`、`gl.where` | `maximum(x, y)` 逐元素取大值，区别于沿轴归约的 `gl.max`；`gl.where(cond, x, y)` 会先求值两分支，不替代访存 `mask` |
| 类型转换与解包 | `gl.cast`、`gl.fp4_to_fp`、`tensor.to(dtype, fp_downcast_rounding=None, bitcast=False)` | `bitcast=True` 按比特重解释；低精度向下转换可通过 `fp_downcast_rounding` 指定舍入模式 |
| 数学函数 | `gl.exp`、`gl.exp2`、`gl.log`、`gl.log2`、`gl.cos`、`gl.sin`、`gl.rsqrt`、`gl.sqrt`、`gl.sqrt_rn`、`gl.erf`、`gl.floor`、`gl.ceil` | 更多底层/高精度或特殊函数入口见下文 `gl.extra.libdevice` |
| 索引与张量构造 | `gl.arange`、`gl.full`、`gl.zeros`、`gl.full_like`、`gl.zeros_like`、`gl.to_tensor` | `arange`/`full`/`zeros` 构造分布式张量时需显式给出或可推导目标 `layout` |
| 形状与广播 | `gl.broadcast`、`gl.expand_dims`、`gl.permute`、`gl.reshape`、`gl.join`、`gl.split`、`gl.ravel`、`tensor.T`、切片/`None` 索引与对应 tensor 方法 | 涉及跨线程数据重排时会触发布局转换与对应通信开销 |
| 显式与自动布局 | `gl.BlockedLayout`、`gl.SliceLayout`、`gl.DistributedLinearLayout`、`gl.DotOperandLayout`、`gl.to_linear_layout`；`gl.AutoLayout`、`gl.CoalescedLayout`、`gl.set_auto_layout`；`gl.convert_layout(..., assert_trivial=False)` | 构造 `BlockedLayout` 时 `prod(threads_per_warp)` 须匹配目标 `warp_size`，`prod(warps_per_cta)` 须匹配 `num_warps`；`convert_layout` 可能引入寄存器重排、shuffle 或隐式 shared 中转 |
| 全局访存与原子 | `gl.load`、`gl.store`；`gl.atomic_add`、`gl.atomic_and`、`gl.atomic_cas`、`gl.atomic_max`、`gl.atomic_min`、`gl.atomic_or`、`gl.atomic_xchg`、`gl.atomic_xor` | `load`/`store` 对越界地址需传 `mask`（及 `other`）；全局原子支持 `sem`、`scope`、`mask` 参数 |
| 共享内存分配与访问 | `gl.allocate_shared_memory`、`gl.SwizzledSharedLayout`、`gl.PaddedSharedLayout`、`gl.SharedLinearLayout`、`gl.bank_conflicts`；`smem.load`、`smem.store`、`smem.gather`、`smem.scatter`、`smem.atomic_scatter_{add,max,min,and,or,xor,xchg}`；`smem.index`、`smem.slice`、`smem.permute`、`smem.reshape` | `smem.index`/`slice`/`permute`/`reshape` 为零拷贝描述符视图；`gl.bank_conflicts(distr_ty, shared_ty)` 为编译期冲突数查询，不生成运行时指令 |
| 归约、扫描、Gather | `gl.reduce`、`gl.sum`、`gl.max`、`gl.min`、`gl.reduce_or`、`gl.xor_sum`、`gl.associative_scan`、`gl.histogram`、`gl.gather` | `gl.sum` 等后端会按“线程内合并 -> 组内 shuffle -> 跨组 shared 暂存与 barrier”自动完成，源码无需手写通信屏障；注意 `gl.gather` 为分布式张量内部 gather，不同于 `smem.gather` |
| 通用计算、同步与扩展 | `gl.dot_fma`、`gl.map_elementwise`、`gl.barrier(cluster=False)`、`gl.warp_specialize`、`gl.inline_asm_elementwise`、`gl.extra.libdevice`、`gl.device_assert`、`gl.device_print`、`gl.expect_zero` | `gl.dot_fma(a, b, acc)` 是不依赖专用矩阵指令的通用 FMA 矩阵乘（`acc` 用 `BlockedLayout`，乘数用 `DotOperandLayout`）；`gl.extra.libdevice` 含 198 个底层数学/转换函数，可用 `.venv/bin/gk api list triton.language.extra.libdevice` 查询 |

## 通用与平台边界

- **顶层导出中的平台专用布局**：`gl.NVMMADistributedLayout` 和 `gl.NVMMASharedLayout` 虽从 `gl` 顶层导出，但语义属于 **NVIDIA 专用扩展**（见 [NVIDIA 专用扩展](nvidia/nvidia.md)）。非 NVIDIA 或通用路径应使用 `BlockedLayout`、`SwizzledSharedLayout`、`PaddedSharedLayout`、`SharedLinearLayout` 或 `DistributedLinearLayout`（以及 `AutoLayout` / `CoalescedLayout`）。
- **通用名下的能力门槛与后端绑定**：
  - `gl.barrier(cluster=True)` 与布局中的 `cga_layout` / `block_bases` 依赖目标硬件的跨 CTA cluster 能力；
  - `gl.warp_specialize(functions_and_args, worker_num_warps, worker_num_regs=None)` 中的 `worker_num_regs` 动态寄存器重分配依赖后端支持；
  - `gl.load`/`gl.store` 的 `cache_modifier`/`eviction_policy`、FP8/FP4 数据类型、特殊舍入模式与原子语义范围取决于具体后端；
  - `gl.inline_asm_elementwise(asm, constraints, ...)` 中的内联汇编体与寄存器约束直接绑定目标指令集（如 PTX）；
  - `gl.extra.libdevice` 同时包含通用数学入口和后端特定绑定（如在 NVIDIA 后端对应 CUDA libdevice）。

## 构造 tile 和处理边界

- 用 `gl.program_id(axis)` 取得当前 program 坐标，用 `gl.cdiv(x, div)` 计算覆盖整个维度所需的 tile 数。`gl.arange(start, end, layout=None)` 生成坐标，`gl.full(shape, value, dtype, layout=None)` 和 `gl.zeros(shape, dtype, layout=None)` 生成带显式布局的初值。
- `BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order, cga_layout=[])` 的前三项分别指定每线程、每执行组和每 CTA 在各维度的分布。`order=[1, 0]` 表示优先沿维度 1 分配相邻元素。
- `SliceLayout(dim, parent)` 移除父布局的一个轴。二维 tile 的行、列坐标可分别用 `SliceLayout(1, parent)` 和 `SliceLayout(0, parent)` 创建，再扩维广播到同一个二维布局。`gl.convert_layout(value, layout, assert_trivial=False)` 保留逻辑值并改变其在执行单元上的分布。
- 对 `gl.load(pointer, mask=..., other=...)` 和 `gl.store(pointer, value, mask=...)` 分别设置有效地址条件。分组映射最后一个不完整的 CTA 组时，可用 `gl.minimum(num_pid_m - first_pid_m, GROUP_M)` 限制组内行块数。
- 对 `gl.exp(x)`、`gl.max(input, axis=...)`、`gl.sum(input, axis=..., keep_dims=False, dtype=None)`，先确定归约轴和累加精度。对于 Softmax，先减去同一行的最大值，再算指数和分母；尾部填充值必须与数学运算的中性元相容。

## 共享内存描述符

`gl.allocate_shared_memory(element_ty, shape, layout, value=None)` 分配有布局的共享存储。使用 `[NUM_STAGES, BM, BK]` 组织流水线时，给单个 `[BM, BK]` tile 指定二维共享布局。通用 swizzle 可用 `SwizzledSharedLayout(vec, per_phase, max_phase, order, cga_layout=[])` 表达；需要显式填充或线性偏移映射时可用 `PaddedSharedLayout` 或 `SharedLinearLayout`。

`tile = smem.index(stage)` 沿首维选取 stage，支持运行时标量 `int32`，所得描述符形状为 `smem.shape[1:]` 并沿用原布局。`smem.slice(start, length, dim=0)` 在指定维取子区间并保留维数，三个参数均为编译期整数；此外还支持 `.permute(...)` 与 `.reshape(...)` 零拷贝视图。`tile.store(value)` 把分布式张量写入共享内存，`tile.load(layout)` 以指定寄存器布局读出；需要按索引读写或原子更新时使用 `.gather(indices, axis)`、`.scatter(values, indices, axis)` 与 `.atomic_scatter_*`。

## 逐行求和示例

假设输入是当前设备上一张连续 FP16 矩阵，行数大于零，列数为 1 到 256。每个 program 负责一行，列索引的 layout 由主机按目标执行组宽度构造。行宽为 127 时分配 128 个逻辑位置，额外位置读作 0，正好是求和的中性元。若把算子换成取最大值，有符号浮点数的尾块应填负无穷。

```python
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.runtime import driver


@gluon.jit
def _row_sum(X, Y, n, BLOCK: gl.constexpr, LAYOUT: gl.constexpr):
    row = gl.program_id(0)
    col = gl.arange(0, BLOCK, layout=LAYOUT)
    x = gl.load(X + row * n + col, mask=col < n, other=0)
    total = gl.sum(x.to(gl.float32), axis=0)
    gl.store(Y + row, total)


def row_sum(x):
    rows, cols = x.shape
    out = torch.empty((rows,), device=x.device, dtype=torch.float32)
    width = driver.active.get_current_target().warp_size
    num_warps = 4
    layout = gl.BlockedLayout([1], [width], [num_warps], [0])
    _row_sum[(rows,)](x, out, cols, triton.next_power_of_2(cols), layout,
                     num_warps=num_warps)
    return out
```

这里每行只有一个 program 写结果。线程间归约由 `gl.sum` 的通用 lowering（线程内合并 + 目标后端 shuffle + 跨 warp 共享暂存与同步）处理，源码里无需另加屏障。本例已在 sm86 上通过 `gk check`，与 `x.float().sum(dim=1)` 对照，覆盖 5 行、列数分别为 1、127、128、129、256 的 FP16 输入。

需要矩阵单元或显式异步搬运时，可按架构查找对应入口。[NVIDIA 专用扩展](nvidia/nvidia.md) · [返回 Gluon API 知识库](../../AGENTS.md#gluon-api-知识库)
