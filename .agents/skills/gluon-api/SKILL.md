---
name: gluon-api
description: Triton 3.8.0 Gluon (triton.experimental.gluon) API cheat sheet, layout rules, Ampere (sm86) patterns, and compile error fixes. Use when writing, modifying, or debugging Gluon GPU kernels.
---

# Gluon API 速查（Triton 3.8.0 / NVIDIA Ampere sm86）

Gluon（`triton.experimental.gluon`）要求**显式声明张量/共享内存 layout**，不要混用普通 `triton.language`（`tl.load`/`tl.dot`）的隐式布局习惯。

## 标准导入

```python
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout,
    DotOperandLayout,
    NVMMADistributedLayout,
    NVMMASharedLayout,
    SliceLayout,
    SwizzledSharedLayout,
)
from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2
```

## Layout 速查与约束

- `BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order, cga_layout=[])`
  - `prod(threads_per_warp) == 32`；`prod(warps_per_cta) == num_warps`；`order` 为维度排列（如 `[1, 0]` 表示第 1 维连续）。
  - 离线查映射：`gk layout "BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])" --shape 64,64 [--hardware]`
- `SliceLayout(dim, parent)`
  - 沿 `dim` 对父 layout 降维，用于 1D 坐标（`gl.arange(..., layout=SliceLayout(1, load_layout))`）。
- `NVMMADistributedLayout(version, warps_per_cta, instr_shape, cga_layout=[])`
  - Ampere FP16 TensorCore 累加器常用：`version=[2, 0]`, `instr_shape=[16, 8]`。
- `DotOperandLayout(operand_index, parent, k_width)`
  - `mma_v2` 输入 layout：`operand_index=0` 为左矩阵 A，`1` 为右矩阵 B；FP16 常用 `k_width=2`。
- `SwizzledSharedLayout(vec, per_phase, max_phase, order, cga_layout=[])` / `NVMMASharedLayout(swizzle_byte_width, element_bitwidth, rank=2, transposed=False, fp4_padded=False, cga_layout=[])`
  - `NVMMASharedLayout.get_default_for(block_shape, dtype, transposed=False, fp4_padded=False, cga_layout=None)` 按连续维字节数选择兼容的 swizzle。例如 `NVMMASharedLayout.get_default_for([BM, BK], gl.float16)` 为单个二维 tile 生成布局。分析具体读写路径的 bank conflict 见 [GEMM A tile 示例](../kernel-tuning/SKILL.md#gemm-a-tile-的-bank-conflict)。

## 核心原语（已核实 Triton 3.8.0 签名）

- **张量创建**：
  - `gl.arange(start, end, layout=None)`
  - `gl.full(shape, value, dtype, layout=None)` / `gl.zeros(...)`
  - `gl.cdiv(...)`、`gl.program_id(axis)`、`gl.static_range(arg1, arg2=None, step=None)`
- **共享内存与描述符**：
  - `smem = gl.allocate_shared_memory(element_ty, shape, layout, value=None)`
  - 分配 `[NUM_STAGES, BM, BK]` 环形 buffer 时，layout 描述单个 `[BM, BK]` tile，使用二维 `SwizzledSharedLayout(..., order=[1, 0])` 或 `NVMMASharedLayout(..., rank=2)`。`tile = smem.index(stage)` 接受运行时 `int32` 标量，返回的描述符保留原 layout，形状为 `smem.shape[1:]`。
  - `smem.slice(start, length, dim=0)` 在指定维度取子区间并保留维数，三个参数均须是编译期整数。用 `tile.load(layout)` 读取，用 `tile.store(value)` 写入。
- **边界裁剪与条件选择**：CTA 分组时用 `group_size_m = gl.minimum(num_pid_m - first_pid_m, GROUP_M)` 计算尾组的实际行块数。`gl.where(cond, x, y)` 按元素选值，越界访存仍需在 `gl.load` 或 `gl.store` 上设置 `mask`。
- **Ampere 异步拷贝 (`ampere.async_copy`)**：
  - `async_copy.async_load(smem, pointer, mask=None, cache_modifier='', eviction_policy='', volatile=False)`
  - 前端接受的 `cache_modifier` 为 `""`、`".ca"`、`".cg"`、`".cs"`、`".cv"`，`eviction_policy` 为 `""`、`"evict_last"`、`"evict_first"`。Ampere PTX 的 `cp.async.ca` 支持 4、8、16 字节拷贝，`cp.async.cg` 只支持 16 字节。用 `gk dump-info <eval.py>` 查看后端实际生成的指令。[PTX 指令语法](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)
  - `async_copy.commit_group()`
  - `async_copy.wait_group(num_outstanding=0)`（`0` 等全部；`N` 允许最多 `N` 个未完成 group，用于流水线重叠）
- **Ampere TensorCore 与布局工具**：
  - `acc = mma_v2(a, b, acc, input_precision=None)`
  - `gl.convert_layout(value, layout, assert_trivial=False)`
  - `gl.bank_conflicts(distr_ty, shared_ty)`（返回 bank conflict 数）
  - `gl.load(pointer, mask=None, other=None, ...)` / `gl.store(pointer, value, mask=None, ...)`

## 最小工作骨架（参见 `examples/gemm/gemm.py`）

1. 声明 `load_layout`、`mma_layout`，用 `SliceLayout` 生成 `rows`/`cols`/`ak`/`bk`。
2. `gl.allocate_shared_memory` 分配 `a_smem` / `b_smem`，`gl.full` 初始化 `acc`。
3. K 循环：`async_copy.async_load(...)` -> `commit_group()` -> `wait_group(...)` -> `smem.load(DotOperandLayout(...))` -> `mma_v2(a, b, acc)`。
4. 收尾：`gl.convert_layout(acc.to(gl.float16), load_layout)` -> `gl.store(..., mask=...)`。

## 核实未知 API（严禁凭记忆编造）

遇到不确定签名或其它子模块（如 `mbarrier`、`PaddedSharedLayout`），直接运行：
```bash
.venv/bin/python -c "import inspect, triton.experimental.gluon.language as gl; print(inspect.signature(gl.<名称>))"
```
源码目录：`.venv/lib/python3.12/site-packages/triton/experimental/gluon/`

## 编译错误速查（遇到新报错随时追加）

| 报错特征 | 常见原因 | 处理方式 |
|---|---|---|
| `threads_per_warp` / warp 线程数不合法 | `prod(threads_per_warp) != 32` | 确保 NVIDIA 后端每个 warp 线程乘积为 32 |
| `num_warps` 与 `warps_per_cta` 不匹配 | kernel 启动 `num_warps` 与 layout 不一致 | 保持 `prod(warps_per_cta) == num_warps` |
| `arange` / `load` / `store` 缺 layout 或广播失败 | 1D 偏移缺少显式 `SliceLayout` | 按广播轴补 `SliceLayout(0/1, load_layout)` |
| `mma_v2` 操作数类型/布局错误 | 直接把 `BlockedLayout` 喂给 `mma_v2` | 从 `smem.load(DotOperandLayout(0/1, mma_layout, k_width))` 加载 |
| `OutOfResources: shared memory` | 单 block 共享内存超过 `smem_per_block_optin` | 用 `gk probe` 查上限，减小 tile 或 stage 数 |
| `cp.async does not support transfers smaller than 4 bytes; calculated this as 2 bytes`，可能随后出现 `failed to legalize operation 'ttg.async_copy_global_to_local'` | 编译器只能生成 2B 拷贝。行主序 FP16 GEMM 中，A 的 K 或 B 的 N 为奇数时，相邻行起点相差奇数个 FP16 元素 | 检查源指针、行步长和 mask 允许的连续宽度。无法保证至少 4B 时，可让该路径用 `gl.load(ptr, mask=mask, other=0)` 加载后调用 `tile.store(value)`，保留 stage 同步并用奇数尺寸校验 |
| `mp is already defined. constexpr cannot be reassigned.` | 在 `gl.static_range` 中用 `mp: gl.constexpr = ...` 重复绑定同名常量 | 把循环不变量移到循环外。随迭代变化的中间值可直接写入表达式，或使用普通局部赋值 |
| `'In' object has no attribute '__name__'` | 在 `@g.jit` 内写 `x in (1, 2, 4)` | 改用显式比较，例如 `(x == 1) \| (x == 2) \| (x == 4)` |
| `'Tuple' object has no attribute 'func'` | 在 `@g.jit` 内写 `for x in (1, 2, 4):` | 用 `for i in gl.static_range(3):` 展开，本例可用 `x = 1 << i` 取值 |
| `ValueError: @jit functions should be defined in a Python file` | 在 `python -c` 中定义临时 kernel，JIT 无法取得函数源码 | 将带 `@g.jit` 的函数写入临时 `.py` 文件，再从该文件运行或导入 |
| `Read-only file system: '.../.triton'` | 直接跑 Python 调试时 Triton 尝试写入只读 `~/.triton` | 优先通过 `gk check` / `gk bench` 运行（固定使用 `.gluonkit/triton_cache`），或加 `TRITON_CACHE_DIR=.gluonkit/triton_cache` |
