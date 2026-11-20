# GluonKit 架构审视与知识体系设计

**决策建议**　第一笔投入应让 Agent 读懂算子和评测，找回已验证的实验，用可信的参照决定何时停止。跨平台边界先用一个只会执行和计时的测试后端验收。第二种真实平台接入时，再依据实际 lowering、计时器和编译产物扩展接口。

本文基于架构审视时的工作区、本地 Triton 3.8.0 和已有实验记录。性能数字来自历史记录。第 4 节列出现行知识入口和后续规划。CLI 的现行契约见 [AGENTS.md](../AGENTS.md) 和 `gk --help`。

## 1. 用一次 Softmax 调优校准投入方向

主评测是连续 FP16 的 4096 × 4096 矩阵。评测规定的逻辑流量为读输入一次、写输出一次，共 67,108,864 B。这个分子在不同实现间固定，双遍路径对输入的重读会增加实现成本。 [评测](../kernels/softMax/eval_softmax.py) · [单遍及双遍实现](../kernels/softMax/softmax.py)

| 历史 ID | 同一主用例的改变 | 均值 | 有效带宽 |
|---|---|---:|---:|
| `3d9870af` | 双遍基线 | 0.582638 ms | 115.18 GB/s |
| `4e564201` | 整行留在片上，单遍读写 | 0.424696 ms | 158.02 GB/s |
| `bac6f848` | 在单遍版本上把除法写为乘倒数 | 0.424828 ms | 157.97 GB/s |
| `9a7380ce` | 恢复单遍版本，同一源码哈希复测 | 0.424740 ms | 158.00 GB/s |

单遍把耗时降低 **27.1%**。第 3 至 10 轮的八个变体均未测得更低均值。这里值得复用的顺序是先数整次读写、再验证可否让一行数据留在片上。接近同条件 copy 的表现时，提高继续微调所需的证据门槛。单遍适用的最大行宽仍取决于 dtype、资源和目标架构。Triton 的融合 Softmax 教程也把行能否驻留片上作为算法前提。 [历史记录](../.gluonkit/history.jsonl) · [Triton Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

本机旧缓存里有 157.56 GB/s，与单遍数值很接近。缓存未说明计时方法、工作集或 copy 实现，当前 `src/` 也没有读取它。用户试用报告中的 copy 对照可以指导下一次标定。这份缓存本身不足以证明物理 DRAM 饱和。 [本地峰值缓存](../.gluonkit/peaks.json) · [当前指标输入](../src/gluonkit/core/bencher.py)

## 2. 跨平台边界

### 当前最容易影响决策的耦合

| 位置 | 当前行为 | 移植时的后果 |
|---|---|---|
| [CLI](../src/gluonkit/cli.py)、[设备类型](../src/gluonkit/interfaces/types.py) | 入口构造 `NvidiaBackend("cuda:0")`。所有 probe 记录都要提供 `sm_count` 字段 | 设备选择散入上层，为没有 SM 概念的平台填虚值 |
| [Backend Protocol](../src/gluonkit/interfaces/backend.py)、[历史比较](../src/gluonkit/core/history.py) | baseline 的数值单独返回，细节为自由字典。历史比较又直接取字典里的 `avg_ms` | 新后端当次能算 speedup，回看历史时却可能丢掉基准耗时 |
| [调度和持久化](../src/gluonkit/executor.py)、[NVIDIA 计时](../src/gluonkit/backends/nvidia/profiler.py) | executor 写死计时来源。当前实现累加每次调用归属的设备 kernel 时长，多流有重叠时也会累加 | 同为 ms 的事件窗口、端到端延迟和 kernel 时长之和进入同一排名 |
| [NVIDIA 资源摘要](../src/gluonkit/backends/nvidia/dump_info.py)、[bank 分析](../src/gluonkit/backends/nvidia/bank_conflicts.py)、[调优 Skill](../.agents/skills/kernel-tuning/SKILL.md) | 通用流程要求读取寄存器和 occupancy。后端用 driver 的最大可驻留 block 数推算 occupancy，bank 冲突来自离线布局模型 | Agent 会把预测当实测，或在缺少这些能力的平台反复索取同一指标 |

自由字典适合保留厂商原始数据。一旦某字段参与排名、停止、回退、自动路由或公共界面渲染，就应有稳定的类型、单位、范围、来源和缺失语义。这条准入规则比一次设计所有厂商共有的指标更易维护。NVIDIA 自己也将这类 occupancy API 描述为驻留预测。 [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-programming-guide/index.html#occupancy-calculator)

### 最小契约与能力协商

| 对象 | 第一版公共契约 | 负责方 |
|---|---|---|
| 目标与能力 | `backend_id`、目标设备/架构、运行时与编译器标识。按能力返回状态、原因和支持范围 | 后端适配器出具。CLI 在一处选择 backend/device，默认保持当前行为。无设备也可查询离线能力 |
| 校验 | 原有 `prepare(device)` 和 `run/ref`。校验通过后可记录 kernel/eval 哈希、环境和已通过的 case | core 负责比较和凭据。后端提供可用执行环境。当前 eval/比较器依赖 PyTorch，目标平台需验证框架张量互操作 |
| 测量与 baseline | 共同的 `mean_ms`、可用统计、`method`、`scope`、采样条件。baseline 可附加专有细节 | 后端产出计时事实，core 调度并比较同口径的 run/ref。多 kernel 调用保留聚合范围 |
| 诊断报告 | 按 kernel/变体给 `artifacts[]`，每项有 stage、format、path。可选观测有 id、值、单位、对象范围、来源/获取方式 | 后端列出实际产物，允许列表为空。驱动预测、编译器静态估计、运行时计数器分别标明。专有字段进 `extensions.<backend_id>` |
| 推导指标与历史 | 独立计算每个可用速率、比值并记录分母来源。保存测量协议与 schema 版本 | core 只读公共字段。通用 Skill 只按能力、用例和证据行动 |

实现时把现有 `Backend` 缩到“描述目标、取得执行环境、发现能力”。计时、布局分析、IR 导出分别由小的可选 provider 提供。普通 benchmark 和 reference 在口径一致时共用同一 timer。只有厂商需要额外基准细节时才扩展 baseline。命令所需能力在 executor 集中检查一次，前端和 Agent 根据同一份结果做决定。

能力至少有两层。**声明**回答“这个 backend/target 实现了哪些操作、接受哪些条件”，例如某类布局的静态 bank 模型可离线使用。**本次结果**回答“这次实际拿到了什么”，例如计数器受权限限制、某 kernel 没有目标阶段的产物。声明可用不能代替采样成功。报告中的未知值不填零。零是有效的观测值。

举例，下面是假设中的精简后端在 `probe` 的 `info` 内增量提供的能力信息，字段名供接口评审。

```json
{
  "schema_version": 2,
  "backend_id": "minimal",
  "capabilities": {
    "execute": {"state": "available"},
    "timing": {"state": "available", "scope": "device_event_elapsed"},
    "ir_export": {"state": "unsupported", "reason": "backend_has_no_ir_export"},
    "hardware_counters": {"state": "unavailable", "reason": "permission_denied"}
  }
}
```

- `unsupported` 表示该实现或目标没有这项能力，Agent 改走另一条诊断路径。`unavailable` 表示当前环境缺依赖、设备或权限，附原因。`unknown` 用于尚未探测。实际执行出错单列 `error`，保留原始诊断。只有本次命令需要的能力才决定命令成败。
- `history`、签名查询和实现了离线模型的布局工具可以在无设备时工作。`check` 依赖执行和 reference。`bench` 依赖可信计时，缺峰值仍成功返回 ms 与可算的速率。零活动或非正时长不能产生有效样本。更换 timer 必须标出新方法、校准对照、另起可比组。
- 显式请求 `--trace` 或 `bank-conflicts` 且能力缺失，返回 `failed`、退出码 1 和可判读原因。能提供部分诊断时，`dump-info` 可成功并列出已取得的产物和缺项。完全不支持导出或诊断时说明原因。运行失败或数值错误仍然失败。
- 保留顶层单行 `{result, info}` 和退出码。先在内部定义小型 dataclass/TypedDict，由 NVIDIA 适配器同时生成现有公共输出。能够增量扩展的对象增加字段。更改 `check` 成功时的 `null` 或失败字符串等不兼容形状，只放进显式选用的新版协议。历史 reader 同时读旧平铺和现行嵌套记录，缺失来源显示未知。新后端先用内建选择器，第二个实现稳定后再看是否值得做外部插件发现。

**验收边界**　构造一个只支持执行、reference 和设备计时的测试后端，完全不导入 NVIDIA 模块。跑通 CLI 的 check、bench、baseline 和离线 history。移除 timer 后 bench 有确定错误。无 IR、无峰值仍可给出下一步。再用真实目标跑一次点算子和一次带尾块的归约，确认前端、lowering、运行时、张量互操作和可重复计时。测试后端验证控制流，实际硬件验证编译与执行能力。若厂商只支持部分 Gluon 原语，应列出已验证子集及编译失败的最小复现。

## 3. 演进优先级

按「复用度 × 能否改善调优决定 ÷ 实现与长期维护成本」定性排序。成本包括每次 Triton 或厂商工具链升级后的复验。下表的新增命令均为提案。

| 顺序 | 取舍 | 投入 | 复用 / 决策收益 / 成本 | 最小交付与理由 |
|---:|---|---|---|---|
| 1 | Do Now | eval 契约、Reduction/Softmax、带宽诊断卡 | 高 / 高 / 低 | 在现有共享知识库中补齐这些内容，两个 Skill 按任务引导阅读。签名和参数语义由本机安装源码核实 |
| 2 | Do Now | `history list/best/diff`、可逆 `restore` | 高 / 高 / 低至中 | 前三项已有实现。让 Agent 查看同组实验矩阵、恢复字节级快照，并能辨认经过校验的版本 |
| 3 | Do Now | 测量口径、单边带宽指标、同协议 copy 对照 | 高 / 高 / 中 | 直接补足 Softmax 的停止信号。先做一个可复测的 streaming copy 和峰值输入，推迟完整标定套件 |
| 4 | Do Now | 最薄的能力契约、计时/基准公共结构、精简后端测试 | 高 / 高 / 中 | 作为第 2、3 项的数据依赖一起落地。阻止自由字典和 CUDA 假设继续渗入 core |
| 5 | Do Now，按接口范围 | 主 shape 单独 bench，泛型算子增加少量 check-only cases | 高 / 中高 / 低至中 | 先覆盖已存在的尾块、分支切换和误差风险，固定 shape 特化保留原校验 |
| 6 | Defer | 峰值提供者扩展、算力模式标定、丰富的 IR/计数器/指令特征接口 | 中高 / 目前不确定 / 中高 | 先留峰值来源、观测与产物清单。待第二平台或复现问题确定消费者和可验证语义后，再加 provider |
| 7 | Defer | API 自省脚本评估、`gk probe --gluon-api`、全量多 shape bench、外部插件发现 | 中 / 目前低 / 中 | 先评估脚本的查询覆盖和维护成本，再决定是否纳入工作流。多用例性能矩阵需要业务权重和可比组 |
| 8 | Drop/Avoid | core 中解析 PTX 正则、跨 ISA 的“昂贵指令分”、按 `sm86` 猜峰值、硬编码 trace 修正、全库塞进 prompt | 低 / 容易误导 / 高 | 平台工程师可对一个具体假设临时 grep 已导出的 IR。看到 `div.full` 或向量指令只能提出假设，去留由正确性和可比实测决定 |

### 可信的带宽停止信号

新建 eval 的最小契约沿用当前入口。评测脚本命名为 `eval_<name>.py` 并导入同目录算子 `<name>.py`，`prepare(device)` 创建输入并返回无参 `run`。要做数值校验就提供同一输入的 `ref`。说明支持的 shape、dtype、stride、种子和容差，预分配与数据准备放在计时区域外。吞吐字段描述一次被计时调用完成的工作，各 case 独立计数。缺失可靠的 FLOP 约定可省略 `flops`，只报告可解释的速率。 [任务加载与源码快照](../src/gluonkit/executor.py)

1. **定义分子。** `flops` 记录每次逻辑调用的计算约定，如 GEMM 以一次 FMA 计两次。Softmax 的 `5MN` 是算法估算，说明 exp 的计数约定。`bytes_moved` 写明逻辑读写的张量和字节数，按输入/输出 dtype 计算。未来采到的 DRAM 物理字节用独立字段。不同实现的逻辑分子保持一致。
2. **定义分母。** 将厂商理论峰值（精确 SKU、运行模式、资料来源）、同设备微基准（时间、协议、访问模式）和算子 `ref` 分开记录。前两者可以并列展示。算子 reference 提供同任务 speedup。同一来源类别内，优先使用评测作者明确指定且与目标匹配的值，其次是协议匹配的本机标定或已核验的规格。报告标明实际选用值。设备、工具链、协议或功耗状态明显变化时重标定。`probe` 可以展示已知值和出处，重型测量显式触发。
3. **按现有信息计算。** 有时间和字节即可给有效 GB/s，有匹配带宽即可给 `memory_eff_pct = effective_GBps / reference_GBps × 100`。计算侧对应处理，峰值匹配需区分 dtype、累加精度和指令路径（例如 Softmax 不套用 Tensor Core 峰值）。两边数据齐全且口径匹配时，完整模型才给 `max(F/P, B/W) / t × 100` 以及模型预测的限制项，其中 F 是工作量，B 是字节数，P 与 W 分别以 FLOP/s 和 B/s 计，t 为秒。单边比率独立命名。保留旧 `metrics.roofline` 的含义。超过 100% 保留原值并检查字节约定、缓存、标定与计时。 [当前 Roofline](../src/gluonkit/core/roofline.py) · [Roofline 原论文](https://escholarship.org/uc/item/5tz795vq)
4. **校准 copy 的执行路径。** 使用预分配、同 dtype、相近工作集/步长与读写比例的 streaming copy kernel，一次读取和一次写入均计入分子。比较 DRAM 流式表现时选超出相关缓存的工作集或控制缓存状态。小规模、启动受限的任务另组。记录设备、计时范围、预热、采样、缓存、日期及重复结果。NVIDIA 当前 profiler 只统计 kernel 事件。若 `copy_` 走 D2D memcpy 引擎，时长可能漏记。选择同一 kernel timer 可测的实现，或让双方都用等价的设备事件计时。 [活动筛选](../src/gluonkit/backends/nvidia/profiler.py)
5. **把停止写成行动。** 先判断能否减少整次读写或中间张量。单遍已达逻辑最小流量、与匹配 copy 的差距落入跨次重测波动、且正确性通过时，记录“此用例接近实测 copy 表现”，停止局部参数搜索。若缺对照，先补一次。做不了标定则和同范围 `ref`、最佳历史交错复测并注明证据等级。若数据很小、包含多次启动或归约/特殊函数占主导，带宽比低也不能单独定位瓶颈。

同一双遍源码的普通/trace 两条记录分别为 0.582638/0.570063 ms，后者快约 2.16%。当前实现两者都用 `torch.profiler`，先算统计再导出 trace。一对数据不足以估计导出带来的系统偏差。历史记录显式标记 trace 模式，默认同模式比较。接近噪声时交错重复 A/B/A 或多组成对运行，报告跨轮差值与波动。单轮 100 次的 std 不能替代跨轮漂移。 [计时与导出](../src/gluonkit/backends/nvidia/profiler.py)

### 历史与泛化的边界

- `history list` 展示 id/parent、假设、case、校验状态、均值/波动、变化量及其可比性。`best --like <id>` 复用已有分组，补入 case、计时方法/范围、trace、缓存等运行条件。确认带宽排名时还要匹配字节约定与分母。`diff` 可展示跨组源码差异，对跨组速度结论标出不可比原因。将“单次最快”与“同组复测并通过 check”分别呈现。
- `check` 成功写一份小凭据，绑定实际 kernel/eval 哈希、环境和 case 集。旧历史及没有匹配凭据的记录保留，标记 `verification=unknown`。先 check 后 bench 的仓库规则照常执行。`bench` 成功本身只证明取得了样本。
- `restore <id>` 根据记录与当前工作区目标匹配，只恢复算子文件（`kernel_path`）的字节。校验快照 SHA256，展示差异，先把未记录的当前内容存成可找回的快照，检查目标未被并发修改，再原子替换并返回恢复前后哈希。项目外的历史路径可导出快照供检查。恢复后按原 eval 重跑 check。现有 eval、容差和其他工作区编辑继续保留。
- 对承诺多尺寸的算子，评测作者可提供带 `case_id` 的可选 case 工厂。旧 `prepare(device)` 自动成为 primary。按 case 独立准备和校验，任一必测 case 失败即报告其 id。性能仍按 primary 或明确选定的单个 case 入历史。Softmax 用小 M 配列数 4095/4096/4097，可覆盖尾块与单遍/双遍切换。另选常量行，用 FP32 累加验证行和接近 1。这个检查能补逐元素比较的盲点，因为 4096 列均匀概率约为 0.000244，原默认绝对容差 0.01 对这种单元素偏差分辨力有限。由评测作者在新用例定义不变量。调优期间沿用原评测条件。 [Softmax 分支](../kernels/softMax/softmax.py) · [校验器](../src/gluonkit/core/checker.py)

## 4. 知识库应当怎样工作

从 [AGENTS.md](../AGENTS.md) 可直接进入 [Gluon API 知识库](../.knowledge/gluon-api/index.md)，查阅[通用功能](../.knowledge/gluon-api/common.md)或 [NVIDIA 扩展](../.knowledge/gluon-api/platforms/nvidia.md)。仓库硬约束和现行 CLI 由 `AGENTS.md` 维护。Agent 工作流在 `.agents/skills/` 中，`gluon-api` 指导编写和排查 kernel，`kernel-tuning` 指导实验、回退及停止。人和 Agent 共用 `.knowledge/` 中的正文。

### 目录与所有权

当前目录和职责如下。

```text
AGENTS.md                                  # 硬规则、现行命令、共享知识库入口
.agents/skills/
  gluon-api/SKILL.md                       # 编写、调试和查阅 Gluon API 的流程
  kernel-tuning/SKILL.md                   # 评测、瓶颈判断、回退和停止流程
.knowledge/gluon-api/
  index.md                                 # 按问题选通用功能或平台扩展
  common.md                                # 计算、布局、访存、同步、编译启动
  platforms/nvidia.md                      # NVIDIA 专用扩展和架构适用范围
  platforms/nvidia-sm86.md                 # sm86 的布局与 Ampere 原语实作
  troubleshooting.md                       # 按报错查触发条件与验证步骤
```

后续可在 `.knowledge/` 中增加评测契约、算子范式、瓶颈诊断和经验卡。优先补 eval 契约、Reduction/Softmax 与 Memory-bound 诊断，其次按实测补 GEMM、Attention 和 Compute-bound。取得实机证据后，可在 `gluon-api/platforms/` 为对应后端增页。经验卡可放在规划中的 `.knowledge/lessons/`，以卡片头生成短索引，原始记录和候选观察保留在本机 `.gluonkit/`。

**通用层**写算法不变量和验证步骤。Reduction 卡写归约轴、加法/最大值的掩码中性元、累加精度、稳定减最大值、整行驻留与分块在线归约，以及每条路径的读写次数。GEMM 卡写复用与尾 K。Attention 卡写在线更新最大值/归一化分母与输出累加的共同缩放，以及掩码语义。访存诊断从逻辑字节和 copy 对照出发。计算诊断先确认 FLOP 约定、精度与目标指令路径，再试复用、并行度或融合。通用页可以引用目标能力，例如“有静态资源估计时，检查单行驻留的代价”。

**平台层**给出这个 backend、架构和软件构建实际接受的 layout、向量宽度/对齐、同步、异步拷贝、可用资源指标与峰值来源。用小代码片段、成功用例或复现支撑约束。例如“sm86 上的 warp 为 32”放 NVIDIA 页。“按目标的执行组宽度构造布局”留通用页。将不同平台的替代路径放进各自页面，运行时由 capabilities 选取。同名指令或 dtype 在新平台须重新验证。

**API 核实。** 当前直接阅读本机安装的 `triton/experimental/gluon/` 源码。遇到重导出或 JIT 包装时，沿导入找到实际定义，区分用户参数与编译器注入的 `_semantic`。已有的 `introspect.py` 是待评估实现。后续可评估它能否在不同构建中定位真实定义、识别包装，并清楚报告无法解析的符号，再决定是否纳入工作流。

例如，本机的 `gl.sum` 从 `language/_standard.py` 包装的实现可查到 `axis`、`keep_dims` 和 `dtype`。`gl.load` 的参数包括 `mask`、`other`、`cache_modifier` 和 `eviction_policy`。给定参数值、dtype 和 layout 能否在目标设备上编译并得到正确结果，仍需用最小用例运行 check。 [安装包的 Gluon 包装实现](../.venv/lib/python3.12/site-packages/triton/experimental/gluon/language/_standard.py)

编译错误另外处理。保留异常链、编译阶段、算子/原语、归一化消息和最小复现。归一化路径和临时符号，保留会改变诊断的数字，如 2B 与 4B。用这些字段查已验证的错误卡。查不到时给出复现和回源路径，待验证后再写根因。

### 每条经验如何被收录、检索、淘汰

1. **捕获。** 一轮实验结束，从 history 自动带出 id、源码/eval 哈希、case、环境、计时模式和数字。Agent 对有复用价值的事件只补“假设、改动、观察、下一次怎么验证”几行。编译失败没有 bench id 时附最小触发代码和原始错误片段。候选先写本地 inbox，不自动进入 Skill。
2. **定范围。** 一张卡一个主张，用 `kind`（语义、编译错误或调优观察）、算子族、阶段、backend/arch、Triton 构建、dtype/shape/layout、依赖能力作为检索头。记录证据、反例、适用边界、复验方法和状态。对比实验区分提速、退化、噪声内未辨别。编译报错要同时保存触发条件和修复后验证。正反两类结果都有价值。
3. **晋升。** 有原始差异和可复核结果时可提交一张限定范围的卡。数值优化要有对应 check 和相同协议的前后数据，边缘差异要交错复测。缺校验的旧 bench 可以作为观察。多平台复现或算法推导支撑的内容，才提升为通用页里的结论。`.gluonkit/` 已被 Git 忽略，正式卡随库保留最小复现、环境与必要数值摘要，避免换机器后只剩失效的本地 id。
4. **检索。** 默认只加载入口。按当前意图选一页通用材料，再按目标能力/版本选一页平台资料。出现具体错误或实验假设时，过滤索引的 `op/phase/fingerprint/backend/version`，只展开最相关的 1–3 张卡。无匹配则查本机源码、构造最小复现或跑基准实验。开始阶段用 front matter、简单索引和 `rg` 即可。
5. **冲突与过期。** 仓库硬约束优先。签名由本机实现裁决，后端支持由该目标编译/运行裁决，性能由同条件实测裁决。相同条件出现矛盾，挂接反例并标记待复验。版本漂移、复验失败时标 `stale`，替代卡用 `supersedes` 关联，默认检索只展开适用的有效卡。索引由卡片生成避免双写。每次升级 Triton，先重跑少量签名检查和实际用到的最小复现。

**首张经验卡可以这样写。**

| 字段 | `softmax-row-resident-sm86` 的内容 |
|---|---|
| 触发与范围 | 归约 / 内存流量，sm86，Triton 3.8.0，连续 FP16，4096 × 4096，原评测种子和计时协议 |
| 假设与动作 | 一行可放入片上时，在线统计后重读输入的路径多一次读。改为单遍的稳定 Softmax |
| 正向观察 | `3d9870af → 4e564201`，0.582638 → 0.424696 ms。同源复测 `9a7380ce` 为 0.424740 ms |
| 未检出收益 | `4e564201 → bac6f848`，乘倒数的均值差约 0.03%。保留本次条件和差值，不推广为编译器规则 |
| 证据状态与边界 | 历史可复核耗时。当前历史无机器可核验的 check 凭据。补对应源码的 check 后可晋升。其他行宽、架构和工具链需复验。单遍的 4096 阈值只属于这份实现 |
| 给下个 Agent 的动作 | 先数读写、预测驻留代价并通过 check，然后同口径对比双遍/单遍/copy。若在 copy 的跨轮波动内，结束这组微调 |

### 最小落地与验收

**知识库的下一步。** 结合真实任务，在共享知识库中补充 eval 契约、Reduction/Softmax 算法和 Memory-bound 诊断。用现有 Softmax 写首张限定范围的观察卡。让一个新会话从仓库入口找到通用功能，从本机源码查到 `gl.sum` 的真实参数，解释为何 64 MiB 是逻辑流量，并给出 `N=4097` 应验证的分支。

**第二份变更交付可信的实验回退。** 暴露离线历史查询，补计时来源与校验状态，增加可逆 restore。用一条旧记录和一条新记录演练恢复。检查旧结果的验证状态、异组排名的处理、恢复前后的字节和撤销后的内容，并确认主 eval 哈希一致。

**第三份变更交付停止信号与迁移验收。** 实现同协议 copy 标定和独立的带宽比率，使用精简后端验证能力降级。对泛型接口加边界 case。回放 Softmax 的单遍结果，要求 Agent 说出“继续调哪个假设、什么证据足以停止”，并能在缺少 IR 和计数器时作答。新平台到位后，优先验证真实点算子和尾块归约。

记录新算子首次通过 check 前的失败次数、达到最佳可复测结果后的无收益实验轮数，以及从指定历史版本恢复并重新通过 check 的成功率。保留最终耗时和用例覆盖作为约束。这样能区分知识库节省的试错成本、工具提高的可靠性，以及 kernel 自身的性能提升。
