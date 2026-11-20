---
name: gluon-api
description: Write, port, or debug Gluon GPU kernels. Use for Gluon API semantics, layouts, memory, synchronization, JIT launch, target-specific instructions, and compile errors.
---

# Gluon API

从 [AGENTS.md 的 Gluon API 知识库](../../../AGENTS.md#gluon-api-知识库) 按任务选页。通用原语查通用页，目标相关的指令和布局查对应的平台目录。

## 核实未知 API（严禁凭记忆编造）

不确定符号签名、可用子模块或参数约束时，先用内置自省脚本查询当前环境：

```bash
# 列出 gl 或指定后端子模块的全部可用原语、Layout 与子模块
.venv/bin/python .agents/skills/gluon-api/scripts/introspect.py --list [module]

# 一次性查询多个符号的真实签名与首段 docstring
.venv/bin/python .agents/skills/gluon-api/scripts/introspect.py max sum exp load
.venv/bin/python .agents/skills/gluon-api/scripts/introspect.py -m language.nvidia.ampere mma_v2 async_copy.async_load
```

也可直接根据导入路径查看当前安装包 `.venv/lib/python3.12/site-packages/triton/experimental/gluon/` 中的定义。修改 kernel 后按仓库约定用 `gk check` 验证对应目标和用例。
