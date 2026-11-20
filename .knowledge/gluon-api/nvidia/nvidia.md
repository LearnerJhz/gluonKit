# B. NVIDIA 专用扩展

设备端扩展位于 `triton.experimental.gluon.language.nvidia`。表中从架构模块写起，例如 `ampere.mma_v2`；其中 `NVMMADistributedLayout` 与 `NVMMASharedLayout` 从 `gl` 顶层导出但属于 NVIDIA 专用，TMA 的主机端描述符位于 `triton.experimental.gluon.nvidia.hopper`（及 `blackwell` 复导出）。下列入口已按本机 Triton 3.8.0 核对，具体指令的目标、dtype 和 layout 约束仍需分别验证。

| 功能 | 架构模块与代表入口 | 使用时需要理解的约束 |
|---|---|---|
| 专用布局与主机描述符 | `gl.NVMMADistributedLayout`、`gl.NVMMASharedLayout`、`gluon.nvidia.hopper.TensorDescriptor` / `TensorDescriptorIm2Col` | 前两者虽在 `gl` 下导出，但仅用于 NVIDIA MMA / TMA 路径；`TensorDescriptor` 在主机侧构造并传入 TMA kernel |
| 矩阵乘加（MMA、WGMMA、tcgen05） | `ampere.mma_v2`、`hopper.warpgroup_mma` / `hopper.warpgroup_mma_wait`、`blackwell.tcgen05_mma` / `blackwell.tcgen05_mma_scaled` | Ampere 的 A/B 使用匹配的 dot operand layout。Hopper 的异步 WGMMA 在使用累加结果前要等待。Blackwell 的 tcgen05 路径把累加结果写入 TMEM |
| 异步搬运（cp.async） | `ampere.async_copy.async_load`、`commit_group`、`wait_group`（后二者也在 `ampere.async_copy` 内） | 从全局内存向共享内存发起拷贝，按组提交和等待。连续宽度、源地址与步长对齐会限制可生成的拷贝宽度 |
| TMA（Tensor Memory Accelerator） | `hopper.tma.async_load`、`hopper.tma.async_store`、`hopper.tma.store_wait`、`blackwell.tma.async_gather` / `async_scatter` | 用描述符和坐标搬运 tile。加载以 mbarrier 通知到达，存储按 TMA store 的等待机制确认完成 |
| mbarrier | `ampere.mbarrier.init`、`ampere.mbarrier.arrive`、`ampere.mbarrier.wait`、`hopper.mbarrier.expect`、`rubin.mbarrier.arrive` | 用共享内存中的 barrier 对象跟踪到达计数和 phase。Hopper 的 TMA 加载还要登记预期传输字节数；Rubin 支持多播掩码 `cta_mask` |
| TMEM（Tensor Memory） | `blackwell.TensorMemoryLayout`、`blackwell.allocate_tensor_memory`、`blackwell.tcgen05_copy` | TMEM 是 tcgen05 路径使用的独立存储空间。`tcgen05_copy` 从共享内存复制到 TMEM，读取或复用时须满足相应完成条件 |
| 打包运算 | `blackwell.add2`、`blackwell.fma2`、`rubin.add4`、`rubin.fma4` | 接口表达二路或四路打包算术，支持的元素类型和组合依赖目标指令。普通逐元素加法可以直接写 `x + y` |
| CLC（Cluster Launch Control） | `blackwell.clc.try_cancel`、`blackwell.clc.load_result` | 持久化 kernel 尝试接管尚未启动的 cluster。等请求完成后检查 `result.is_canceled()`，成功时通过 `result.program_id(dim)` 取得对应工作 |

sm86 上的 FP16 GEMM 用 `NVMMADistributedLayout` 定义累加器，用 `async_copy` 填充共享 tile，按 `DotOperandLayout` 将数据读给 `mma_v2`。网格索引和尾块 mask 的做法也适用于其他后端。[sm86 的布局约束与 GEMM 骨架](nvidia-sm86.md) · [通用核心功能](../gluon-api-common.md)

本机安装包中各代扩展定义在 `triton/experimental/gluon/language/nvidia/<架构>/`，未列入上表的接口可用 `.agents/skills/gluon-api/scripts/introspect.py --list language.nvidia.<架构>` 现场查询。使用在线参考时，留意上游页面与本机版本的差异。[Triton NVIDIA 扩展文档](https://triton-lang.org/main/gluon/api/nvidia.html) · [CLC 接口说明](https://triton-lang.org/main/gluon/api/generated/triton.experimental.gluon.language.nvidia.blackwell.clc.html)
