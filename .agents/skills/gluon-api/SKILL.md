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
  - 共享内存布局，用于避免 bank conflict。

## 核心原语（已核实 Triton 3.8.0 签名）

- **张量创建**：
  - `gl.arange(start, end, layout=None)`
  - `gl.full(shape, value, dtype, layout=None)` / `gl.zeros(...)`
  - `gl.cdiv(...)`、`gl.program_id(axis)`、`gl.static_range(arg1, arg2=None, step=None)`
- **共享内存与描述符**：
  - `smem = gl.allocate_shared_memory(element_ty, shape, layout, value=None)`
  - 描述符方法：`smem.index(idx)`（沿第 0 维切片，适合多 stage 环形 buffer）、`smem.slice(start, length, dim=0)`、`smem.load(layout)`、`smem.store(value)`
- **Ampere 异步拷贝 (`ampere.async_copy`)**：
  - `async_copy.async_load(smem, pointer, mask=None, cache_modifier='', eviction_policy='', volatile=False)`
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
| `Read-only file system: '.../.triton'` | 直接跑 Python 调试时 Triton 尝试写入只读 `~/.triton` | 优先通过 `gk check` / `gk bench` 运行（固定使用 `.gluonkit/triton_cache`），或加 `TRITON_CACHE_DIR=.gluonkit/triton_cache` |
