> **GluonKit 目的**：让 Coding Agent 能够独立、可靠地使用 Gluon 编写并调优高性能 GPU 算子

## 核心设计思想：

1. 提供gluon相关知识库，因为gluon是实验性质的且相关文献很少，目的是让agent会用gluon写
2. 制作一套cli工具提供调优过程中的性能数据等关键信息，目的是减少他获取信息的负担，同时能避免他直接读几万行的文件，减少幻觉
3. 制定一套sop，让agent根据sop去操作

---

## 总体分层架构

```mermaid
flowchart TD
    Agent["Coding Agent"]

    subgraph Layer1["1. 规则层"]
        Guides["AGENTS.md (核心规则)"]
        Skills["Skills: gluon-api / kernel-tuning (开发与调优流程)"]
    end

    subgraph Layer2["2. Gluon 知识层"]
        KB[".knowledge/gluon-api/ (文档知识库与排错表)"]
        API["gk api (本机 API 签名与源码查询)"]
        Refs["examples/ & bench-Nvidia3050/ (参考算子示例)"]
    end

    subgraph Layer3["3. 验证层"]
        Layout["gk layout (线程与元素映射可视化)"]
        Bank["gk bank-conflicts (共享内存 Bank 冲突估算)"]
    end

    subgraph Layer4["4. 迭代层"]
        Check["gk check (编译与数值正确性校验)"]
        Base["gk baseline (参考基线测量)"]
        Bench["gk bench (性能测速与 Roofline 评估)"]
        Dump["gk dump-info (编译产物与资源占用分析)"]
        Hist["gk history / restore (实验历史与版本回退)"]
    end

    subgraph Layer5["5. SIMT 后端和相关工具"]
        Proto["Backend Protocol (统一后端抽象接口)"]
        NV["NvidiaBackend (NVIDIA 后端适配)"]
        MX["MetaxBackend (Wave64 后端适配)"]
        Store[".gluonkit/ (缓存、基线与实验数据)"]
    end

    Agent --> Layer1
    Layer1 --> Layer2
    Layer1 --> Layer3
    Layer1 --> Layer4
    Layer4 --> Proto
    Layer3 --> Proto
    Proto --> NV
    Proto --> MX
    Layer4 --> Store
```

---

## 优化核心思想：控制变量 + 迭代

[.agents/skills/kernel-tuning/SKILL.md](.agents/skills/kernel-tuning/SKILL.md) 制定了一个SOP，也有点经验模块的感觉，整体流程是这样的：

```mermaid
flowchart LR
    Probe["1. 环境摸底"]
    Base["2. 建立基线"]
    Hypo["3. 单变量改动"]
    Pre["离线预检"]
    Verify["4. 先校验后测速"]
    Judge{"5. 显著提速?"}
    Keep["保留为新起点"]
    Rollback["回退并归因"]
    Stop["6. 达标交付"]

    Probe --> Base --> Hypo
    Hypo -.-> Pre -.-> Verify
    Hypo --> Verify --> Judge
    Judge -->|"是"| Keep
    Judge -->|"否"| Rollback --> Hypo
    Keep -->|"继续"| Hypo
    Keep -->|"达标"| Stop
```

> 两大模块提供迭代的能力：`gk history + gk restore`

---

## Gluon 知识库设计

### 1. 静态知识库（`.knowledge/gluon-api/`）

1. **通用 API 文档（`gluon-api-common.md`）** 做了一层通用原语的分类和索引
2. **平台专用 API 文档（`nvidia.md`、`nvidia-sm86.md`）** 后续接入其他平台时，在同级新建对应平台目录
3. **Gluon 报错速查文档**

### 2. 动态查询（`gk api`）

获取详细的api参数和gluon源码中的注释说明。这样设计是避免知识库太大挤爆上下文，即用即查的理念

---

## TODO 后续演进的方向

主要应该是知识库方面的，看下怎么更完善，以及提供一些case或者经验教训
