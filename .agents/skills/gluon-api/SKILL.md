---
name: gluon-api
description: Write, port, or debug Gluon GPU kernels. Use for Gluon API semantics, layouts, memory, synchronization, JIT launch, target-specific instructions, and compile errors.
---

# Gluon API

从 [AGENTS.md 的 Gluon API 知识库](../../../AGENTS.md#gluon-api-知识库) 按任务选页。通用原语查通用页，目标相关的指令和布局查对应的平台目录。

## 核实未知 API（严禁凭记忆编造）

不确定符号签名、可用子模块或参数约束时，先用 `gk api` 查询当前环境（纯文本输出，无需 GPU；查不到的符号写到 stderr 并以退出码 1 结束）：

```bash
# 列出 gl 或指定后端子模块的全部可用原语、Layout 与子模块
.venv/bin/gk api list [module]

# 一次性查询多个符号的真实签名与首段 docstring
.venv/bin/gk api show max sum exp load
.venv/bin/gk api show -m language.nvidia.ampere mma_v2 async_copy.async_load
```

也可直接根据导入路径查看当前安装包 `.venv/lib/python3.12/site-packages/triton/experimental/gluon/` 中的定义。修改 kernel 后按仓库约定用 `gk check` 验证对应目标和用例。
