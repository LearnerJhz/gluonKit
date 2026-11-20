# Gluon 编译与运行排错

以下条目对应本仓库 Triton 3.8.0 环境。先在 traceback 中找最早指向算子源码或目标指令的错误，结合当次输入、布局和编译后端检查相应原因。

## 通用排错（布局与目标资源）

| 报错特征 | 先查什么 | 可验证的改法 |
|---|---|---|
| `threads_per_warp` / warp 线程数不合法 | 检查各维线程数乘积以及目标执行组宽度 | 对齐当前设备的 `warp_size`（例如 NVIDIA 为 `prod(threads_per_warp) == 32`） |
| `num_warps` 与 `warps_per_cta` 不匹配 | 对照启动参数与张量布局 | 令 `prod(warps_per_cta) == num_warps`，重新编译该启动配置 |
| `arange`、`load`、`store` 缺布局或广播失败 | 检查一维坐标扩展后的轴、形状和父布局 | 二维 tile 可从同一父布局为行、列坐标分别派生 `SliceLayout(1, parent)`、`SliceLayout(0, parent)`。[广播与示例](gluon-api-common.md) |
| `OutOfResources: shared memory` | 检查每个 block 的 tile 和 stage 总占用 | 用 `.venv/bin/gk probe` 查当前设备共享内存上限（如 NVIDIA 的 `smem_per_block_optin`），减小 tile 或 stage 数并重新编译 |

## 通用排错（JIT 语法与环境）

| 报错特征 | 先查什么 | 可验证的改法 |
|---|---|---|
| `mp is already defined. constexpr cannot be reassigned.` | 是否在 `gl.static_range` 中多次以 `mp: gl.constexpr = ...` 绑定同名变量 | 循环不变量移到循环外。随迭代变化的值写入表达式，或使用普通局部赋值 |
| `'In' object has no attribute '__name__'` | JIT 函数是否使用了 `x in (1, 2, 4)` | 将该判断展开，例如 `(x == 1) \| (x == 2) \| (x == 4)` |
| `'Tuple' object has no attribute 'func'` | JIT 函数是否直接用 `for x in (1, 2, 4)` 遍历元组 | 对这组三个值用 `for i in gl.static_range(3)`，循环体内令 `x = 1 << i` |
| `ValueError: @jit functions should be defined in a Python file` | 是否通过 `python -c` 定义临时 kernel，JIT 无法取得函数源码 | 将带 `@gluon.jit` 的函数写进 `.py` 文件，运行或导入该文件 |
| `Read-only file system: '.../.triton'` | 直接运行 Python 时，确认 Triton 缓存目标是否可写 | 从仓库根目录使用 `.venv/bin/gk check <eval.py>`，校验通过后再 bench。单独调试 Python 时可设 `TRITON_CACHE_DIR=.gluonkit/triton_cache` |

## NVIDIA 专用排错

| 报错特征 | 先查什么 | 可验证的改法 |
|---|---|---|
| `mma_v2` 的输入类型或布局不合法 | 检查 A、B 是否仍在普通加载布局，以及各自的 operand index 和 parent | 从共享 tile 用对应的 `DotOperandLayout` 加载 A、B，再调用 MMA。[sm86 乘数和累加器布局](nvidia/nvidia-sm86.md) |
| `cp.async does not support transfers smaller than 4 bytes; calculated this as 2 bytes`，有时随后出现 `failed to legalize operation 'ttg.async_copy_global_to_local'` | 检查指针、行步长和 mask。行主序 FP16 GEMM 中 A 的 K 或 B 的 N 为奇数时，相邻行起点相差奇数个 FP16 元素 | 若该路径无法保证至少 4 字节连续拷贝，用 `gl.load(ptr, mask=mask, other=0)` 后接 `tile.store(value)`，保留 stage 间所需的同步。对承诺支持奇数尺寸的接口，用触发问题的尺寸校验 |
