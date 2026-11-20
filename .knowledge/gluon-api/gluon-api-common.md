# A. 通用核心功能

用 `from triton.experimental import gluon` 引入主机端 JIT 入口，用 `from triton.experimental.gluon import language as gl` 在 kernel 中表达运算。“通用”指可以沿用同一套编程语义。执行组宽度、可接受的 dtype、合法布局参数、共享内存容量和部分操作的作用域，仍由目标后端决定。

| 功能 | 代表 API | 写算子和移植时看什么 |
|---|---|---|
| 计算 | `gl.exp`、`gl.maximum`、`gl.max`、`gl.sum`、`gl.reduce`、`gl.dot_fma` | `maximum(x, y)` 是逐元素取大值，`max(x, axis=...)` 沿轴归约。`gl.dot_fma(a, b, acc)` 是不依赖硬件矩阵单元的通用 FMA 矩阵乘（累加器用 `BlockedLayout`，乘数用匹配的 `DotOperandLayout`） |
| 布局 | `gl.BlockedLayout`、`gl.SliceLayout`、`gl.DistributedLinearLayout`、`gl.DotOperandLayout`、`gl.AutoLayout`、`gl.CoalescedLayout`、`gl.convert_layout`、`gl.set_auto_layout` | shape 给出逻辑尺寸，layout 给出元素归属。构造 `BlockedLayout` 时对齐目标的执行组宽度和启动的 `num_warps`；`AutoLayout`/`CoalescedLayout` 交由布局推导选择。布局转换可能引起跨线程数据交换 |
| 访存 | `gl.load`、`gl.store`、`gl.allocate_shared_memory`、`gl.SwizzledSharedLayout`、`gl.PaddedSharedLayout`、`gl.SharedLinearLayout` | 对越界地址在实际读写上加 mask。共享内存分配返回描述符，可用 `.store(value)`、`.load(layout)`、`.gather`/`.scatter`/`.atomic_scatter_*` 访问。检查步长、对齐和容量 |
| 同步 | `gl.barrier()`、`gl.barrier(cluster=True)` | 默认同步一个 CTA 内的线程，参与线程要到达同一屏障。多 CTA 的 cluster 同步依赖目标支持。显式启动的异步传输要按对应原语等待完成 |
| 编译与启动 | `@gluon.jit`、`gl.constexpr`、`kernel[grid](...)`、`gl.program_id` | 主机指定 grid 与编译期参数，program 用自己的 ID 定位 tile。首次启动或遇到新的特化时，JIT 按目标生成设备代码并通过驱动启动。可导出的中间产物由后端决定 |

## 通用与平台边界

- **顶层导出中的平台专用布局**：`gl.NVMMADistributedLayout` 和 `gl.NVMMASharedLayout` 虽从 `gl` 顶层导出，但语义上属于 **NVIDIA 专用扩展**（见 [NVIDIA 专用扩展](nvidia/nvidia.md)）；跨平台或非 NVIDIA 路径应使用 `BlockedLayout`、`SwizzledSharedLayout`、`PaddedSharedLayout`、`SharedLinearLayout` 或 `DistributedLinearLayout`。
- **通用名下的能力门槛与后端绑定**：
  - `gl.barrier(cluster=True)` 与 `cga_layout` 需要目标后端支持多 CTA cluster；
  - `gl.warp_specialize(functions_and_args, worker_num_warps, worker_num_regs=None)` 中的 `worker_num_regs` 动态寄存器重分配依赖后端支持；
  - `gl.load`/`gl.store` 的 `cache_modifier`/`eviction_policy`、FP8/FP4 与特殊舍入指令的有效取值与行为取决于目标后端；
  - `gl.inline_asm_elementwise(asm, constraints, ...)` 的汇编字符串与寄存器约束绑定具体目标指令集（如 PTX）；
  - `gl.extra.libdevice` 同时包含通用数学入口和后端特定设备函数绑定。

## 构造 tile 和处理边界

- 用 `gl.program_id(axis)` 取得当前 program 坐标，用 `gl.cdiv(x, div)` 计算覆盖整个维度所需的 tile 数。`gl.arange(start, end, layout=None)` 生成坐标，`gl.full(shape, value, dtype, layout=None)` 和 `gl.zeros(shape, dtype, layout=None)` 生成有指定布局的初值。编译期展开循环可用 `gl.static_range(arg1, arg2=None, step=None)`。
- `BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order, cga_layout=[])` 的前三项分别指定每线程、每执行组和每 CTA 在各维度的分布。对使用该布局的 program，`prod(threads_per_warp)` 匹配目标的执行组宽度，`prod(warps_per_cta)` 匹配启动的 `num_warps`。例如 `order=[1, 0]` 表示优先沿维度 1 分配相邻元素。
- `SliceLayout(dim, parent)` 移除父布局的一个轴。二维 tile 的行、列坐标可分别用 `SliceLayout(1, parent)` 和 `SliceLayout(0, parent)` 创建，再扩维广播到同一个二维布局。`gl.convert_layout(value, layout, assert_trivial=False)` 保留逻辑值，改变其在执行单元上的分布。
- 对 `gl.load(pointer, mask=..., other=...)` 和 `gl.store(pointer, value, mask=...)` 分别设置有效地址条件。`gl.where(condition, x, y)` 选择已经求值的结果。读写都接受 `cache_modifier` 和 `eviction_policy`，`load` 还接受 `volatile`。缓存提示的有效取值和效果取决于具体目标。分组映射最后一个不完整的 CTA 组时，可用 `gl.minimum(num_pid_m - first_pid_m, GROUP_M)` 限制组内行块数。
- 对 `gl.exp(x)`、`gl.max(input, axis=...)`、`gl.sum(input, axis=..., keep_dims=False, dtype=None)`，先确定归约轴和精度。对于 Softmax，先减去同一行的最大值，再算指数和分母。尾部填充值必须与数学运算相容。

## 共享内存描述符

`gl.allocate_shared_memory(element_ty, shape, layout, value=None)` 分配有布局的共享存储。使用 `[NUM_STAGES, BM, BK]` 组织流水线时，给单个 `[BM, BK]` tile 指定二维共享布局。通用 swizzle 可用 `SwizzledSharedLayout(vec, per_phase, max_phase, order, cga_layout=[])` 表达；需要线性偏移或带 padding 消冲突时可用 `PaddedSharedLayout` 或 `SharedLinearLayout`。

`tile = smem.index(stage)` 沿首维选取 stage，支持运行时标量 `int32`。所得描述符形状为 `smem.shape[1:]`，沿用原布局。`smem.slice(start, length, dim=0)` 在指定维取子区间，保留维数，三个参数均为编译期整数；还支持 `.permute(...)` 与 `.reshape(...)` 视图。`tile.store(value)` 把分布式张量写入该 tile，`tile.load(layout)` 以指定寄存器布局读出；按索引读写可用 `.gather(indices, axis)`、`.scatter(values, indices, axis)` 与 `.atomic_scatter_*`。

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

这里每行只有一个 program 写结果。线程间归约由 `gl.sum` 的 lowering 处理，源码里无需另加屏障。本例已在 sm86 上通过 `gk check`，与 `x.float().sum(dim=1)` 对照，覆盖 5 行、列数分别为 1、127、128、129、256 的 FP16 输入。

需要矩阵单元或显式异步搬运时，可按架构查找对应入口。[NVIDIA 专用扩展](nvidia/nvidia.md) · [返回 Gluon API 知识库](../../AGENTS.md#gluon-api-知识库)
