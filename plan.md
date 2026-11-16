# GluonKit 开发计划

## 0. 目标与范围

做一套给 AI agent 调用的命令行工具，用来开发 Gluon kernel。agent 通过它查询硬件、编译、校验正确性、测性能、记录实验。

- **第一阶段**：在 NVIDIA RTX 3050 Laptop（SM86，Triton 3.8.0）上，用一个 FP16 GEMM Gluon kernel 端到端跑通。
- **之后**：接入国产平台。硬件相关代码全部放在 `backends/<平台>/` 下，其余代码不得直接依赖 CUDA。
- **不做**：自动调参、自动生成 kernel、bank conflict 静态分析、死锁静态分析。

## 1. 环境准备（第一步先做）

- 当前系统 Python 里有 `triton 3.8.0`（装在 `~/.local`），但**没有 torch**。先建 venv，装和本机 CUDA 驱动匹配、且兼容 Triton 3.8 的 torch，再装 pytest、pyyaml。
- 验证命令：`python -c "import torch, triton; print(torch.cuda.is_available(), triton.__version__)"`。
- 把可复现的安装步骤写进 `README.md`。

## 2. 目录结构

```
gluonKit/
├── pyproject.toml              # [project.scripts] gk = "gluonkit.cli:main"
├── README.md
├── AGENTS.md                   # 给使用工具的 agent 看的工作流说明（见第 10 节）
├── src/gluonkit/
│   ├── __init__.py
│   ├── cli.py                  # 统一入口 gk
│   ├── registry.py             # backend 注册与自动发现
│   ├── runner.py               # 通用子进程执行：隔离、超时、崩溃捕获
│   ├── inputs.py               # 解析 --input / --dims，生成输入张量
│   ├── interfaces/
│   │   ├── types.py            # 所有结果数据类
│   │   ├── backend.py          # Backend 协议（硬件相关钩子）
│   │   └── history.py          # 平台无关，直接实现
│   ├── core/                   # 平台无关的实现
│   │   ├── checker.py
│   │   ├── bencher.py
│   │   └── roofline.py
│   ├── backends/
│   │   └── nvidia/
│   │       ├── __init__.py     # register("nvidia", ...)，不在导入时 import torch/triton
│   │       ├── backend.py      # 硬件钩子实现
│   │       ├── probe.py
│   │       ├── compiler.py
│   │       └── profiler.py     # torch.profiler 封装
│   └── data/
│       └── peaks.yaml          # 各设备峰值算力/带宽，由用户填写
├── kernels/
│   └── gemm_fp16.py            # 示例 kernel
└── tests/
    ├── conftest.py             # 提供 requires_gpu 标记
    ├── fixtures/               # 正确/语法错误/结果错误/会崩溃的 kernel 样例
    ├── test_probe.py
    ├── test_compile.py
    ├── test_check.py
    ├── test_bench.py
    └── test_history.py
```

## 3. Kernel 怎么交给工具

**不额外定义约定文件。** kernel 就是一个普通的 `.py`，里面是写好的 `@gluon.jit` kernel，再加一个普通的 Python 启动函数（launcher）。生产环境里的算子本来就是这么写的：

```python
# kernels/gemm_fp16.py
@gluon.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ..., BLOCK_M: gl.constexpr, ...):
    ...

def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    gemm_kernel[grid](a, b, c, M, N, K, ..., num_warps=4)
    return c
```

工具只认一条规则：**用 `文件.py:函数名` 指定一个 Python 函数，它接收张量（和可选的标量参数），返回输出张量或张量 tuple。** 工具从外部调用这个函数：

- 输入怎么造，由命令行描述（见下文），不需要 kernel 文件提供。
- 参考实现用同样的形式指定：`--ref torch.matmul`（可 import 的名字），或 `--ref ref.py:func`。参考函数的参数和 kernel 函数一致。
- 计算量和访存量（用来算 TFLOPS、GB/s 和 roofline 百分比）在命令行用表达式给出，可以不给，不给就只报时间：
  `--flops "2*M*N*K" --bytes "(M*K+K*N+M*N)*2"`
- 如果 kernel 是原地写输出、不返回值，就在 kernel 文件里多写一行包一下，让它返回输出。这是唯一的要求。

### 输入描述（`inputs.py`）

```bash
--dims M=1024,N=1024,K=512        # 一组形状变量；可以重复多次，每组是一个 case
--input "f16[M,K]"                # 第 1 个位置参数
--input "f16[K,N]:t"              # 第 2 个位置参数；:t 表示转置得到的非连续张量
--arg eps=1e-5                    # 关键字标量参数（可选）
--input-file a.pt                 # 也可以直接用保存好的张量文件
```

- 格式：`<dtype>[<shape>][:修饰]`。dtype 支持 f16/bf16/f32/i8/i32 等；shape 里可以写数字，也可以写 `--dims` 里的变量名。
- 修饰（先实现这几个，后续按需加）：`t`（转置后的非连续视图）、`ones`、`zeros`、`randint`。默认用固定 seed 的 `randn`，整数类型默认用 `randint`。
- 每个 case 都用同一个 seed 造输入。kernel 和 ref 各拿一份 **clone**，防止原地修改互相污染。

> [!TIP]
> 参数很长时，用 argparse 的 `fromfile_prefix_chars="@"`，把参数写进一个文本文件，执行 `gk run @gemm.args`。不需要再引入新的配置格式。

## 4. 公共数据结构（`interfaces/types.py`）

全部用 `@dataclass(frozen=True)`，列表字段用 tuple，并提供 `to_dict()`，供 `--json` 输出和写 history。

```python
ProbeResult:   device_name, platform, arch, compiler_version, torch_version,
               smem_per_block_optin, sm_count, vram_bytes, peaks (dict|None), extra: dict
CompileError:  stage ("python"|"frontend"|"ttgir"|"llvm"|"assembler"|"timeout"|"crash"|"unknown"),
               message, line (int|None), raw_tail (只保留 stderr 最后 N 行)
KernelInfo:    name, reg_count, spill_count, smem_bytes, num_warps,
               artifacts: dict[str, path]     # ttgir/llir/ptx 写到文件，这里只返回路径
CompileResult: ok, errors: tuple[CompileError], kernels: tuple[KernelInfo],   # 一次调用可能编译多个 kernel
               warnings: tuple[str], raw: dict
CaseCheck:     dims, passed, max_abs_err, max_rel_err, nan_count, inf_count, atol, rtol, error (str|None)
CheckResult:   passed, cases: tuple[CaseCheck], crash (str|None), seed
Stats:         median, mean, min, max, std, p90, n          # 单位 ms
BenchResult:   dims, device_time: Stats       # 每次调用里所有设备 kernel 的耗时之和（profiler）
               e2e_time: Stats                # 每次调用的墙钟时间，含 launch 开销
               kernels: tuple[(name, calls_per_iter, avg_ms)]   # 每次调用实际启动了哪些 kernel
               baseline_time: Stats|None, speedup_vs_baseline,
               tflops, gbps, roofline_pct, bound_by ("compute"|"memory"|None),
               trace_path (str|None), method ("profiler"|"events"), raw: dict
```

## 5. Backend 协议（换平台只需实现这一层）

```python
class Backend(Protocol):
    name: str                      # "nvidia"
    device: str                    # torch 设备名："cuda"、"npu"、"mlu"……
    def available(self) -> tuple[bool, str]: ...   # (是否可用, 不可用原因)
    def probe(self) -> ProbeResult: ...
    def collect_compiled(self, call) -> CompileResult: ...  # 执行一次 call，收集期间编译出的 kernel 信息
    def synchronize(self) -> None: ...
    def profile(self, fn, iters: int, warmup: int) -> ProfileData: ...   # 每次迭代的设备 kernel 列表
    def event_timer(self, fn, iters: int, warmup: int) -> list[float]: ...
    def flush_cache(self) -> None: ...             # 清 L2，可以留空实现
```

- check、bench、history 的通用逻辑放在 `core/`，只通过 Backend 钩子接触硬件。
- `registry.py` 提供 `register()`、`get(name)` 和 `auto_detect()`：
  - `auto_detect` 记录每个 backend 不可用的原因，全部失败时一起报出来。
  - 支持环境变量 `GK_BACKEND` 强制指定 backend。

## 6. 各模块要求

### 6.1 runner（通用子进程执行）
- 每次 compile、check、bench 都在新的子进程里跑，主进程不 import torch 或 triton。
- 子进程用 importlib 按路径加载 `文件.py`，取出指定的函数。
- 子进程把结果写到一个 JSON 文件，主进程读取。不要靠解析 stdout，避免和 kernel 自己的打印混在一起。
- 要能识别四种情况：正常结束、Python 异常（保留 traceback 末尾）、被信号杀死（segfault 或 CUDA 致命错误）、超时（默认 compile 120s、check/bench 300s，可配置）。
- 每次执行生成一个目录 `runs/<时间戳>-<短hash>/`，存放源码快照、产物和结果 JSON。

### 6.2 probe
- NVIDIA 实现：
  - 用 `torch.cuda.get_device_properties` 拿设备名、SM 数、显存、算力版本。
  - smem 上限用 opt-in 值，可以从 `triton.runtime.driver.active.utils.get_device_properties` 取 `max_shared_mem`。
  - 再加上 triton 和 torch 的版本。
- 峰值数据从 `data/peaks.yaml` 按设备名查。**查不到就返回 None，不准自己编数字**。peaks.yaml 预留 RTX 3050 Laptop 的条目，数值留空，由用户填写。
- `gk probe --gluon-api`：自省 `triton.experimental.gluon.language` 及 `nvidia.ampere`、`hopper`、`blackwell` 子模块，列出可用的函数和 layout 类，防止 agent 调用不存在的 API。
- `gk layout "<layout 表达式>" --shape 64,64 [--hardware]`：调用 layout 对象的 `format_tensor_view` 或 `format_hardware_view`，打印线程、warp 到元素的映射。表达式在受限命名空间里 eval，只暴露 gluon layout 类。

### 6.3 compile
- 在子进程里用第一组 `--dims` 造输入，调用一次 kernel 函数。期间编译出的所有 Triton/Gluon kernel 都收集起来，不需要用户告诉工具 kernel 叫什么。
- 收集方式由开发 agent 调研本机 Triton 3.8.0 后选定，可选方案：
  - `triton.knobs` 里的编译 hook 或 listener；
  - 调用后扫描模块里所有 `JITFunction` / `GluonJITFunction` 的 `device_caches`。
- 从每个 `CompiledKernel` 读 `n_regs`、`n_spills`、`metadata.shared`、`metadata.num_warps`，并把 `asm` 里的 ttgir、llir、ptx 写到 run 目录。
- 出错时归类到 `CompileError.stage`，尽量从 `triton.compiler.errors.CompilationError` 里提取行号。
- 只要 `spill_count > 0`，就加一条 warning。

### 6.4 check
- 对每组 `--dims` 依次执行：造输入 → clone 两份 → 调 kernel 函数和 ref 函数 → 比较。
- 先 `synchronize`，再把结果转成 float32 比较。按 `|a-b| <= atol + rtol*|b|` 逐元素判断，记录最大绝对误差和最大相对误差。
- 输出是 tuple 时逐个比较；结构、shape、dtype 不一致直接判失败，并说明原因。
- NaN 和 Inf 单独计数，只要输出里出现、而参考里没有，就判失败。
- 容差：命令行 `--atol/--rtol` 优先；不给就按输出 dtype 取默认值（f16 1e-2，bf16 2e-2，f32 1e-4）。实际用的值写进结果。
- 一个 case 崩溃（CUDA 错误）后，CUDA 上下文就不能再用了。剩下的 case 标记为"未执行"，并在结果里说明。
- AGENTS.md 里要提醒：一定要包含不能被 BLOCK 整除的边界形状。

### 6.5 bench（独立模块，从外部调用 kernel 函数，以 torch.profiler 为主）
- **主路径（method="profiler"）**：
  - 先预热 `warmup` 次，再在 `torch.profiler.profile(activities=[CPU, CUDA])` 里调用 kernel 函数 `iters` 次（每次调用前可选择清 L2；清 L2 用的 kernel 不计入结果）。
  - 用 `torch.profiler.record_function` 给每次迭代打标记，把设备 kernel 事件按迭代归组。每次迭代的设备耗时 = 这次迭代里所有 kernel 的耗时之和，组成样本列表再算 Stats。
  - 同时按 kernel 名汇总进 `kernels`，agent 能看到函数里实际启动了哪些 kernel（比如多出来的清零 kernel）。
  - 不同 torch 版本的字段名不一样（`device_time` 或 `cuda_time` 等），要做兼容。
  - 可选 `--trace`：导出 chrome trace 到 run 目录。
- **辅助路径（method="events"）**：用 `triton.testing.do_bench` 或 CUDA event 测端到端时间，填进 `e2e_time`。profiler 不可用时，降级成只用这一路，并在结果里注明。
- **baseline**：默认把 `--ref` 当作 baseline，用同样方法测，算出 `speedup_vs_baseline`。可以用 `--baseline` 另外指定，或用 `--no-baseline` 关闭。
- 派生指标都在 `core/roofline.py` 里算（`--flops`、`--bytes` 表达式用当前 case 的 dims 求值）：
  - `tflops = flops / device_time.median`
  - `gbps = bytes / device_time.median`
  - `roofline_pct = 理论最短时间 / 实测时间`，其中理论最短时间 = max(flops/峰值算力, bytes/峰值带宽)
  - `bound_by` 取两者中较大的那一项。
  - 任何输入缺失，对应字段就是 None。
- bench 前**不**自动跑 check，由 `gk run` 负责串联。
- 测试里不设固定耗时门槛。

### 6.6 history
- 存储：`<项目>/.gluonkit/history.jsonl`，每行一条记录。源码按 hash 存到 `.gluonkit/sources/<hash>.py`，不直接写进 jsonl。
- Record 字段：`id, timestamp, kernel (文件:函数), source_hash, parent_id, note, platform, arch, env(triton/torch/driver 版本), invocation (inputs/dims/ref 等命令行参数), compile, check, bench`。
- API：`save`、`get(id)`、`list(kernel=None)`、`best(kernel, dims=None)`、`diff(id_a, id_b)`（源码的 unified diff 加关键指标的变化）。
- `best`：只在相同 kernel、平台、输入描述和 dims 之间比较，只考虑 check 通过且有 bench 结果的记录，取 `device_time.median` 最小的。没有符合条件的记录时返回 None。

## 7. CLI

所有命令都支持 `--json`（输出机器可读结果）、`--backend` 和 `@参数文件`。退出码：`0` 表示成功或通过，`1` 表示 kernel 有问题（编译失败、校验失败），`2` 表示工具自身出错。文本输出要简短，长内容写文件，只打印路径。

```bash
gk probe [--gluon-api]
gk layout "<expr>" --shape 64,64 [--hardware]

# 下面几个命令共用输入描述：--dims（可重复） --input（按位置） --arg --input-file
gk compile kernels/gemm_fp16.py:gemm --dims M=1024,N=1024,K=512 --input "f16[M,K]" --input "f16[K,N]"
gk check   kernels/gemm_fp16.py:gemm --ref torch.matmul \
           --dims M=1024,N=1024,K=512 --dims M=1000,N=777,K=300 \
           --input "f16[M,K]" --input "f16[K,N]" [--atol X --rtol Y] [--seed 0]
gk bench   kernels/gemm_fp16.py:gemm --ref torch.matmul --dims ... --input ... \
           [--flops "2*M*N*K"] [--bytes "(M*K+K*N+M*N)*2"] \
           [--iters 100 --warmup 25] [--trace] [--baseline X | --no-baseline]
gk run     <同上全部参数> --note "num_stages 2->3" [--parent <id>]
           # 依次 compile → check → bench（只有前一步通过才继续），自动写 history，输出摘要
gk run @gemm.args --note "..."          # 参数写在文件里，最常用

gk history list [--kernel kernels/gemm_fp16.py:gemm]
gk history show <id>
gk history best <kernel> [--dims ...]
gk history diff <id_a> <id_b>
```

## 8. 示例 kernel：`kernels/gemm_fp16.py`

- 一个 FP16 输入、FP32 累加的 GEMM，用 Gluon 写，跑在 SM86 上：
  - MMA 用 `NVMMADistributedLayout(version=[2,0], instr_shape=[16,8])`。
  - 操作数用 `DotOperandLayout`。
  - 搬数据用 `gluon.language.nvidia.ampere` 的 `async_copy` 做多级流水。
- 文件里只有 `@gluon.jit` kernel 和 `def gemm(a, b) -> c` 启动函数，没有别的约定内容。
- 先写出能跑对的版本，性能不作要求。参考 Triton 仓库里 Gluon 的 tutorials 和 tests，要以本机 3.8.0 的实际 API 为准。
- 同时提供 `examples/gemm.args`，内容是跑这个 kernel 的完整参数：dims 至少包含 `1024×1024×512`、`1000×777×300`（边界形状）、`2048×2048×2048`，加上 `--ref torch.matmul`、`--flops`、`--bytes`。

## 9. 测试（pytest）

- `tests/conftest.py` 提供 `requires_gpu` 标记，没有 GPU 或 torch 时自动 skip 并说明原因。
- 外部调用（nvidia-smi、子进程、profiler、计时）都可以 mock。

| 文件 | 内容 |
|---|---|
| test_probe.py | 必需字段齐全；peaks 查不到时返回 None；未知 layout 表达式给出错误；backend 不可用时 `auto_detect` 报出原因 |
| test_compile.py | 正确 kernel 编译成功，并拿到寄存器数和 smem；语法错误 kernel 返回 `stage` 和行号；子进程超时、被信号杀死都能识别（用 fixture 脚本模拟）；`文件:函数` 写错时给出清晰报错 |
| test_check.py | 正确 kernel 通过；结果错误的 kernel 失败；atol/rtol 边界判断正确；出现 NaN 判失败；tuple 输出逐个比较；崩溃时保留错误信息；比较逻辑和 `inputs.py` 的解析用纯 CPU tensor 单测 |
| test_bench.py | Stats 统计正确（用固定样本）；`--flops`、`--bytes` 表达式求值和 tflops、gbps、roofline 换算正确；缺 flops 或 peaks 时对应字段为 None；profiler 事件按迭代归组用伪造事件单测；真实 GPU 用例只检查字段存在且为正数 |
| test_history.py | 保存、读取、按 kernel 筛选；`best` 排除校验失败和无 bench 的记录；不同 dims 或输入描述不混在一起比较；没有符合条件的记录时返回 None；diff 输出正确 |

运行：`python -m pytest -q tests/`

## 10. AGENTS.md（给使用工具的 agent 看）

内容要点：
1. 开始前先跑 `gk probe --json` 和 `gk probe --gluon-api`。
2. 写 kernel：一个 `@gluon.jit` kernel 加一个接收张量、返回张量的启动函数。可以参考 `kernels/gemm_fp16.py`。
3. 不确定 layout 时，先用 `gk layout` 看映射。
4. 把调用参数写进一个 `.args` 文件，dims 里一定要包含不能被 BLOCK 整除的边界形状。
5. 每次修改后跑 `gk run @xxx.args --note "<改了什么、为什么>"`。
6. 看结果的方法：
   - 编译失败，看 `errors[].stage` 和 `line`。
   - spill 大于 0，说明寄存器压力大。
   - 校验失败，先看边界形状那个 case。
   - 性能看 `speedup_vs_baseline` 和 `roofline_pct`；`kernels` 里出现预期之外的 kernel，要查原因。
7. 用 `gk history best` 和 `gk history diff` 回溯最好的版本。

## 11. 开发顺序与验收

| 步骤 | 内容 | 验收标准 |
|---|---|---|
| M0 | 环境、pyproject、目录骨架、`types.py`、`registry` | `pip install -e .` 后 `gk --help` 可用 |
| M1 | runner + probe + layout | `gk probe --json` 输出本机真实信息；`gk layout` 能打印一个 BlockedLayout |
| M2 | inputs.py + 示例 GEMM kernel + compile | `gk compile kernels/gemm_fp16.py:gemm ...` 拿到寄存器数、smem、ttgir 路径 |
| M3 | check | 示例 kernel 所有 case 通过；故意改错的 kernel 判失败；会崩溃的 kernel 不拖垮主进程 |
| M4 | bench（profiler 为主） | 输出 device_time、e2e_time、kernels 列表、相对 torch.matmul 的加速比；能导出 trace |
| M5 | history + `gk run` | 连续 `gk run @examples/gemm.args` 两次后，`history best` 和 `history diff` 结果正确 |
| M6 | 测试补齐 + AGENTS.md | `pytest -q tests/` 全部通过（无 GPU 的用例 skip），README 和 AGENTS.md 完成 |

每完成一步，同步补上对应的测试再进入下一步。

## 12. 约束与注意事项

- 主进程不 import torch 或 triton，硬件相关代码只放在 `backends/` 下。
- 不要自己编造硬件峰值数字和 Gluon API。以本机 Triton 3.8.0 的源码为准，路径在 `~/.local/lib/python3.12/site-packages/triton/experimental/gluon/`。
- 输出给 agent 的内容要控制长度：IR、trace、长日志一律写文件，只返回路径。
- 3050 Laptop 会动态调频，性能结果要带上 std 和 p90，agent 才能判断差异是不是噪声。
- `--flops`、`--bytes` 表达式只允许数字、dims 变量和四则运算，用 `ast` 白名单求值，不要直接 `eval`。
