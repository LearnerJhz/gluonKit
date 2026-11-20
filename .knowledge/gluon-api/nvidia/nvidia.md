# B. NVIDIA 专用扩展

设备端扩展位于 `triton.experimental.gluon.language.nvidia.<架构>`（`ampere`、`hopper`、`blackwell`、`rubin`），主机端 TMA 描述符位于 `triton.experimental.gluon.nvidia.hopper`（及 `blackwell` 复导出）；另外 `NVMMADistributedLayout` 与 `NVMMASharedLayout` 从 `gl` 顶层导出但属于 NVIDIA 专用。下列全部接口已对照本机 Triton 3.8.0 核实，实际指令支持仍取决于目标 compute capability、dtype 与 layout 组合。

## 专用类型、布局与主机描述符

| 类别 | 模块与符号 | 说明与约束 |
|---|---|---|
| 顶层 NVIDIA 布局 | `gl.NVMMADistributedLayout`、`gl.NVMMASharedLayout`（含 `.get_default_for`） | 用于 NVIDIA MMA 累加器及共享内存矩阵/TMA tile；`NVMMADistributedLayout` 的 `version=[2, 0]` 对应 Ampere `mma_v2` |
| Barrier 布局 | `ampere.mbarrier.MBarrierLayout`（含 `.multicta`） | 配合 `allocate_shared_memory` 或 `allocate_mbarrier` 为共享内存中的 mbarrier 指定布局 |
| TMEM 布局 | `blackwell.TensorMemoryLayout`、`blackwell.TensorMemoryScalesLayout`、`rubin.TensorMemoryScalesLayout` | 描述 Blackwell/Rubin Tensor Memory 及微缩放（scale）排布；Rubin 版增加 `block_rep_order`（`"mnThenK"` / `"kThenMn"`） |
| 设备端描述符与结果类型 | `hopper.tma.tensor_descriptor` / `tensor_descriptor_type`、`tensor_descriptor_im2col` / `tensor_descriptor_im2col_type`；`hopper.warpgroup_mma_accumulator` / `warpgroup_mma_accumulator_type`；`blackwell.tensor_memory_descriptor` / `tensor_memory_descriptor_type`；`blackwell.clc.clc_result` / `clc_result_type` | 分别对应 TMA 描述符、WGMMA 累加器句柄、TMEM 描述符与 CLC 取消请求返回值 |
| 主机端 TMA 描述符 | `gluon.nvidia.hopper.TensorDescriptor`、`TensorDescriptorIm2Col`（及 `.from_tensor`；`gluon.nvidia.blackwell.TensorDescriptor` 复导出） | 在主机侧基于 `NVMMASharedLayout` 构造并传入 kernel；要求基地址与外维步长满足 16B（`fp4_padded` 需 32B）对齐 |

## 专用操作与辅助接口

| 功能域 | 架构模块与全部入口 | 使用时需要理解的约束 |
|---|---|---|
| 矩阵乘加与等待 | `ampere.mma_v2`；`hopper.warpgroup_mma`、`hopper.warpgroup_mma_wait`、`hopper.warpgroup_mma_init`；`blackwell.tcgen05_mma`、`blackwell.tcgen05_mma_scaled`、`blackwell.tcgen05_mma_barrier_count` | `mma_v2` 要求 `acc` 为 `NVMMADistributedLayout(version=[2, 0])`、A/B 为匹配 `DotOperandLayout`，经 `tt.dot` 发射 PTX `mma.sync`；`warpgroup_mma` 异步执行，消费累加结果前须 `warpgroup_mma_wait`；`tcgen05` 路径将累加器直接存于 TMEM |
| 异步拷贝（cp.async） | `ampere.async_copy.async_load`、`commit_group`、`wait_group`、`mbarrier_arrive` | 从全局内存向共享内存发起拷贝，按组提交和等待或通知 mbarrier；连续宽度、源地址与步长对齐限制实际拷贝字节数 |
| mbarrier 同步 | `ampere.mbarrier`: `allocate_mbarrier`、`init`、`invalidate`、`arrive`、`wait`；`hopper.mbarrier`: `expect`、`arrive(count=...)`、`fence_init_release_cluster`；`rubin.mbarrier`: `arrive(..., cta_mask=...)` | Ampere `arrive` 固定计数 1，Hopper 支持指定 `count` 与 TMA 预期字节 `expect`，Rubin 增加多播掩码 `cta_mask` |
| Cluster 与异步 Shared 顺序 | `hopper.cluster`: `arrive`、`wait`、`barrier`；`hopper.fence_async_shared`、`hopper.async_store` | 用于多 CTA cluster 内同步；`hopper.async_store` 是至少 2 CTA cluster 下向共享内存异步写并通知 mbarrier，不同于 `tma.async_store` |
| TMA 描述符与搬运 | `hopper.tma`: `make_tensor_descriptor`、`async_load`、`async_load_im2col`、`async_store`、`store_wait`、`async_atomic_{add,min,max,and,or,xor}`；`blackwell.tma`: `async_gather`、`async_scatter` | 设备端可用 `make_tensor_descriptor` 或直接传入主机端 `TensorDescriptor`；加载以 mbarrier 通知到达，存储用 `store_wait` 确认完成 |
| TMEM 与 tcgen05 通路 | `blackwell.allocate_tensor_memory`、`blackwell.tcgen05_copy`、`blackwell.tcgen05_commit`；`tensor_memory_descriptor.{load,store,load_min,load_max,get_reg_layout,slice,index}` | TMEM 是独立于共享内存/寄存器的片上矩阵存储；`tcgen05_copy` 从共享内存拷入 TMEM，`tcgen05_commit` 在此前异步 tcgen05 指令完成后触发 mbarrier |
| 原生打包运算 | `blackwell.{add2,sub2,mul2,fma2,min2,max2}`；`rubin.{add4,sub4,mul4,fma4}` | 分别用于二路或四路（FP8/FP4）原生打包算术，普通逐元素加法直接写 `x + y` 即可 |
| CLC 动态调度 | `blackwell.clc`: `try_cancel`、`load_result`、`clc_result.is_canceled`、`clc_result.program_id` | 持久化 kernel 尝试接管尚未启动的 cluster；请求完成后经 `load_result` 读入寄存器，检查 `is_canceled()` 并用 `program_id(dim)` 取新坐标 |

## 全局访存缓存提示与 sm86 Softmax 实测

在 NVIDIA 上，`gl.load(..., cache_modifier=".cg")` 请求经 L2 及以下缓存读取，绕过 L1。`gl.store(..., cache_modifier=".cs")` 发出流式写回提示。可用 `gk api show load store` 查签名，参数语义可在本机 Triton 的 `triton/language/core.py` 中查 `load` 和 `store` 的 docstring，再用 `gk dump-info` 核对生成的 PTX。

以下是 Triton 3.8.0 在 RTX 3050 6GB Laptop GPU（sm86）上对连续 `4096 × 4096` FP16 Softmax 的独立实验。默认版每个 CTA 用 4 个 warp 处理一行，均值 423.034 µs。

| 改动 | 编译证据 | 均值 |
|---|---|---:|
| 读取使用 `.cg` | `ld.global.cg.b16` | 422.941 µs |
| 写回使用 `.cs` | `st.global.cs.b16` | 423.134 µs |
| 每线程连续元素从 1 增至 8 | 评测的基地址与行步长满足对齐，静态 PTX 读写各由 32 条 16-bit 指令变为 4 条 128-bit 指令（`ld.global.v4.b32`、`st.global.v4.b32`） | 427.046 µs |
| 每个 CTA 处理 4 行，每个 warp 独立归约一行 | PTX 中的 `bar.sync` 消失，共享内存编译用量从 16 B 降至 0 B，每线程寄存器从 51 增至 167 | 426.516 µs |

缓存提示已在 PTX 生效，两项延迟变化都在测量波动内。更宽的访存和去掉 CTA 内同步的版本在这个用例上均变慢。换架构或输入布局时，重新验证生成的指令和性能。[完整 Softmax 实验记录](../../../kernels/softMax/REPORT.md) · [换目标时的通用验证步骤](../gluon-api-common.md#换目标时验证访存与归约)

## 复导出、兼容别名与设备数学库

- **继承复导出**：`hopper` 复导出 `ampere` 的 `async_copy` 与 `mma_v2`；`blackwell` 复导出 `ampere`/`hopper` 的常用扩展；`rubin` 复导出 `blackwell` 扩展并补充四路打包运算与新 `mbarrier`/`TensorMemoryScalesLayout`。
- **兼容旧名**：`ampere.async_copy.async_copy_global_to_shared`、 `hopper.tma.async_copy_global_to_shared(_im2col)`、`hopper.tma.async_copy_shared_to_global` 为历史别名，实际语义与 `async_load(_im2col)` / `async_store` 相同，新代码优先用短名。
- **CUDA 设备数学库**：`gl.extra.libdevice` 在 NVIDIA 后端绑定到 CUDA libdevice（共 198 个函数，含带舍入修饰的 `add_rn`/`fma_rn`、快速近似 `fast_*`、位运算 `clz`/`popc`/`brev` 与特殊数学函数）；若需核对符号列表或参数，运行：
  ```bash
  .venv/bin/gk api list language.nvidia.<架构>
  .venv/bin/gk api list triton.language.extra.libdevice
  ```

sm86 上的 FP16 GEMM 用 `NVMMADistributedLayout` 定义累加器，用 `async_copy` 填充共享 tile，按 `DotOperandLayout` 将数据读给 `mma_v2`。网格索引和尾块 mask 的做法也适用于其他后端。[sm86 的布局约束与 GEMM 骨架](nvidia-sm86.md) · [通用核心功能](../gluon-api-common.md) · [排错表](../troubleshooting.md)

使用在线参考时，留意上游页面与本机 Triton 3.8.0 版本的差异。[Triton NVIDIA 扩展文档](https://triton-lang.org/main/gluon/api/nvidia.html) · [CLC 接口说明](https://triton-lang.org/main/gluon/api/generated/triton.experimental.gluon.language.nvidia.blackwell.clc.html)
