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
GluonKit/
├── pyproject.toml              # [project.scripts] gk = "gluonkit.cli:main"
├── README.md
├── AGENTS.md                   # 给使用工具的 agent 看的工作流说明（见第 10 节）
├── src/gluonkit/
│   ├── __init__.py
│   ├── cli.py                  # 统一入口 gk
│   ├── registry.py             # backend 注册与自动发现
│   ├── spec.py                 # 加载 kernel 文件，校验约定
│   ├── runner.py               # 通用子进程执行：隔离、超时、崩溃捕获
│   ├── interfaces/
│   │   ├── types.py            # 所有结果数据类
│   │   ├── backend.py          # Backend 协议（硬件相关钩子）
│   │   ├── probe.py
│   │   ├── compiler.py
│   │   ├── checker.py
│   │   ├── bencher.py
│   │   └── history.py          # 平台无关，直接实现
│   ├── core/                   # 平台无关的默认实现
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
│   └── gemm_fp16.py            # 示例 kernel，遵守第 3 节的约定
└── tests/
    ├── conftest.py             # 提供 requires_gpu 标记
    ├── fixtures/               # 正确/语法错误/结果错误/会崩溃的 kernel 样例
    ├── test_probe.py
    ├── test_compile.py
    ├── test_check.py
    ├── test_bench.py
    └── test_history.py
```

## 3. Kernel 文件约定（所有工具都依赖它）

每个 kernel 是一个 `.py` 文件，导出下面这些名字。由 `spec.py` 负责加载和校验，缺少必需项时给出明确报错。

```python
# 必需
def make_inputs(case: dict, device: str, seed: int) -> dict:
    """造输入，并预分配输出 buffer。case 例如 {"M":1024,"N":1024,"K":512,"dtype":"float16"}"""

def run(inputs: dict):
    """启动 Gluon kernel：grid、num_warps、constexpr 都写在这里。返回输出 tensor（或 tuple）。
    返回值必须只依赖本次的输入，不能读上一次调用残留的状态。"""

def reference(inputs: dict):
    """PyTorch 参考实现，返回结构和 run 一致"""

CASES: list[dict]               # check 默认覆盖的 case，必须包含不能被 BLOCK 整除的形状

# 可选
def flops(case: dict) -> int: ...        # 浮点运算次数，例如 GEMM 为 2*M*N*K
def bytes_moved(case: dict) -> int: ...  # 理论最少访存字节数，例如 (M*K+K*N+M*N)*elem_size
def baseline(inputs: dict): ...          # 对照实现，例如 torch.matmul；bench 会同时测它算加速比
TOLERANCE: dict = {"atol": 1e-2, "rtol": 1e-2}   # 也可以按 dtype 给不同容差
KERNEL_NAME: str                # profiler 里匹配 kernel 用的名字，默认取 Gluon 函数名
```

## 4. 公共数据结构（`interfaces/types.py`）

全部用 `@dataclass(frozen=True)`，列表字段用 tuple，并提供 `to_dict()`，供 `--json` 输出和写 history。

```python
ProbeResult:   device_name, platform, arch, compiler_version, torch_version,
               smem_per_block_optin, sm_count, vram_bytes, peaks (dict|None), extra: dict
CompileError:  stage ("python"|"frontend"|"ttgir"|"llvm"|"assembler"|"timeout"|"crash"|"unknown"),
               message, line (int|None), raw_tail (只保留 stderr 最后 N 行)
CompileResult: ok, errors: tuple[CompileError], reg_count, spill_count, smem_bytes,
               num_warps, artifacts: dict[str, path]   # ttgir/llir/ptx 写到文件，这里只返回路径
               raw: dict
CaseCheck:     case, passed, max_abs_err, max_rel_err, nan_count, inf_count, atol, rtol, error (str|None)
CheckResult:   passed, cases: tuple[CaseCheck], crash (str|None), seed
Stats:         median, mean, min, max, std, p90, n          # 单位 ms
BenchResult:   case, kernel_time: Stats (profiler 测的纯设备时间),
               e2e_time: Stats (含 launch 开销的墙钟时间),
               baseline_time: Stats|None, speedup_vs_baseline,
               tflops, gbps, roofline_pct, bound_by ("compute"|"memory"|None),
               kernels_launched: tuple[(name, count, avg_ms)],   # run() 实际启动的所有 kernel
               trace_path (str|None), method ("profiler"|"events"), raw: dict
```

## 5. Backend 协议（换平台只需实现这一层）

```python
class Backend(Protocol):
    name: str                      # "nvidia"
    device: str                    # torch 设备名："cuda"、"npu"、"mlu"……
    def available(self) -> tuple[bool, str]: ...   # (是否可用, 不可用原因)
    def probe(self) -> ProbeResult: ...
    def compile_info(self, kernel_path: str, case: dict, out_dir: str) -> CompileResult: ...
    def synchronize(self) -> None: ...
    def profile(self, fn, iters: int, warmup: int, kernel_name: str|None) -> ProfileData: ...
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
- 在子进程里加载 kernel 文件，取 `CASES[0]`（或 `--case` 指定的 case）调 `make_inputs`，然后编译并拿到 `CompiledKernel`。具体做法由开发 agent 调研：用 `kernel.warmup(...)`，或者执行一次 `run` 后从 kernel 的 cache 里取。
- 从 `CompiledKernel` 读 `n_regs`、`n_spills`、`metadata.shared`、`metadata.num_warps`，并把 `asm` 里的 ttgir、llir、ptx 写到 run 目录。
- 出错时归类到 `CompileError.stage`，尽量从 `triton.compiler.errors.CompilationError` 里提取行号。
- 只要 `spill_count > 0`，就在结果里加一条 warning。

### 6.4 check
- 逐个 case 执行：`make_inputs` → `run` → `reference` → 比较。
- 先 `synchronize`，再把结果转成 float32 比较。按 `|a-b| <= atol + rtol*|b|` 逐元素判断，记录最大绝对误差和最大相对误差。
- NaN 和 Inf 单独计数，只要输出里出现、而参考里没有，就判失败。
- shape 或 dtype 不一致直接判失败，并说明原因。
- 容差优先级：命令行参数 > kernel 文件里的 `TOLERANCE` > 默认值。实际用的值写进结果。
- 一个 case 崩溃（CUDA 错误）后，CUDA 上下文就不能再用了。剩下的 case 标记为"未执行"，并在结果里说明。

### 6.5 bench（独立模块，以 torch.profiler 为主）
- **主路径（method="profiler"）**：
  - 先预热 `warmup` 次，再在 `torch.profiler.profile(activities=[CPU, CUDA])` 里调用 `run` `iters` 次（每次调用前可选择清 L2）。
  - 遍历 `prof.events()`，找出设备端的 kernel 事件，按 `KERNEL_NAME` 匹配，取每次调用的设备耗时，组成样本列表再算 Stats。
  - 不同 torch 版本的字段名不一样（`device_time` 或 `cuda_time` 等），要做兼容。
  - 所有被启动的 kernel 都汇总进 `kernels_launched`。如果 `run()` 额外启动了清零之类的 kernel，agent 能看到。
  - 可选 `--trace`：导出 chrome trace 到 run 目录。
- **辅助路径（method="events"）**：用 `triton.testing.do_bench` 或 CUDA event 测端到端时间，填进 `e2e_time`。profiler 不可用时，降级成只用这一路，并在结果里注明。
- 如果 kernel 文件提供了 `baseline`，用同样方法测它，算出 `speedup_vs_baseline`。
- 派生指标都在 `core/roofline.py` 里算：
  - `tflops = flops / kernel_time.median`
  - `gbps = bytes / kernel_time.median`
  - `roofline_pct = 理论最短时间 / 实测时间`，其中理论最短时间 = max(flops/峰值算力, bytes/峰值带宽)
  - `bound_by` 取两者中较大的那一项。
  - 任何输入缺失，对应字段就是 None。
- bench 前**不**自动跑 check，由 `gk run` 负责串联。
- 测试里不设固定耗时门槛。

### 6.6 history
- 存储：`<项目>/.gluonkit/history.jsonl`，每行一条记录。源码按 hash 存到 `.gluonkit/sources/<hash>.py`，不直接写进 jsonl。
- Record 字段：`id, timestamp, kernel_name, kernel_path, source_hash, parent_id, note, platform, arch, env(triton/torch/driver 版本), cases, compile, check, bench`。
- API：`save`、`get(id)`、`list(kernel_name=None)`、`best(kernel_name, case=None)`、`diff(id_a, id_b)`（源码的 unified diff 加关键指标的变化）。
- `best`：只在相同 kernel、平台、case 之间比较，只考虑 check 通过且有 bench 结果的记录，取 `kernel_time.median` 最小的。没有符合条件的记录时返回 None。

## 7. CLI

所有命令都支持 `--json`（输出机器可读结果）和 `--backend`。退出码：`0` 表示成功或通过，`1` 表示 kernel 有问题（编译失败、校验失败），`2` 表示工具自身出错。文本输出要简短，长内容写文件，只打印路径。

```bash
gk probe [--gluon-api]
gk layout "<expr>" --shape 64,64 [--hardware]
gk compile kernels/gemm_fp16.py [--case M=1024,N=1024,K=512]
gk check   kernels/gemm_fp16.py [--case ...]... [--atol X --rtol Y] [--seed 0]
gk bench   kernels/gemm_fp16.py [--case ...] [--iters 100 --warmup 25] [--trace] [--no-baseline]
gk run     kernels/gemm_fp16.py --note "num_stages 2->3" [--parent <id>]
           # 依次 compile → check → bench（只有前一步通过才继续），自动写 history，输出摘要
gk history list [--kernel gemm_fp16]
gk history show <id>
gk history best <kernel> [--case ...]
gk history diff <id_a> <id_b>
```

## 8. 示例 kernel：`kernels/gemm_fp16.py`

- 一个 FP16 输入、FP32 累加的 GEMM，用 Gluon 写，跑在 SM86 上：
  - MMA 用 `NVMMADistributedLayout(version=[2,0], instr_shape=[16,8])`。
  - 操作数用 `DotOperandLayout`。
  - 搬数据用 `gluon.language.nvidia.ampere` 的 `async_copy` 做多级流水。
- 先写出能跑对的版本，性能不作要求。参考 Triton 仓库里 Gluon 的 tutorials 和 tests，要以本机 3.8.0 的实际 API 为准。
- `CASES` 至少包含 `1024×1024×512`、`1000×777×300`（边界形状）、`2048×2048×2048`。
- 提供 `flops`、`bytes_moved`，`baseline` 用 `torch.matmul`。

## 9. 测试（pytest）

- `tests/conftest.py` 提供 `requires_gpu` 标记，没有 GPU 或 torch 时自动 skip 并说明原因。
- 外部调用（nvidia-smi、子进程、profiler、计时）都可以 mock。

| 文件 | 内容 |
|---|---|
| test_probe.py | 必需字段齐全；peaks 查不到时返回 None；未知 layout 表达式给出错误；backend 不可用时 `auto_detect` 报出原因 |
| test_compile.py | 正确 kernel 编译成功，并拿到寄存器数和 smem；语法错误 kernel 返回 `stage` 和行号；子进程超时、被信号杀死都能识别（用 fixture 脚本模拟） |
| test_check.py | 正确 kernel 通过；结果错误的 kernel 失败；atol/rtol 边界判断正确；出现 NaN 判失败；崩溃时保留错误信息；比较逻辑用纯 CPU tensor 单测 |
| test_bench.py | Stats 统计正确（用固定样本）；tflops、gbps、roofline 换算正确；缺 flops 或 peaks 时对应字段为 None；profiler 事件解析用伪造事件单测；真实 GPU 用例只检查字段存在且为正数 |
| test_history.py | 保存、读取、按 kernel 筛选；`best` 排除校验失败和无 bench 的记录；不同 case 不混在一起比较；没有符合条件的记录时返回 None；diff 输出正确 |

运行：`python -m pytest -q tests/`

## 10. AGENTS.md（给使用工具的 agent 看）

内容要点：
1. 开始前先跑 `gk probe --json` 和 `gk probe --gluon-api`。
2. 新 kernel 照第 3 节的约定写，可以复制 `kernels/gemm_fp16.py` 当模板。
3. 不确定 layout 时，先用 `gk layout` 看映射。
4. 每次修改后跑 `gk run <kernel> --note "<改了什么、为什么>"`。
5. 看结果的方法：
   - 编译失败，看 `errors[].stage` 和 `line`。
   - spill 大于 0，说明寄存器压力大。
   - 校验失败，先看边界形状那个 case。
   - 性能看 `speedup_vs_baseline` 和 `roofline_pct`。
6. 用 `gk history best` 和 `gk history diff` 回溯最好的版本。

## 11. 开发顺序与验收

| 步骤 | 内容 | 验收标准 |
|---|---|---|
| M0 | 环境、pyproject、目录骨架、`types.py`、`registry` | `pip install -e .` 后 `gk --help` 可用 |
| M1 | runner + probe + layout | `gk probe --json` 输出本机真实信息；`gk layout` 能打印一个 BlockedLayout |
| M2 | spec + 示例 GEMM kernel + compile | `gk compile kernels/gemm_fp16.py` 拿到寄存器数、smem、ttgir 路径 |
| M3 | check | 示例 kernel 所有 case 通过；故意改错的 kernel 判失败；会崩溃的 kernel 不拖垮主进程 |
| M4 | bench（profiler 为主） | 输出 kernel_time、e2e_time、相对 torch.matmul 的加速比；能导出 trace |
| M5 | history + `gk run` | 连续 `gk run` 两次后，`history best` 和 `history diff` 结果正确 |
| M6 | 测试补齐 + AGENTS.md | `pytest -q tests/` 全部通过（无 GPU 的用例 skip），README 和 AGENTS.md 完成 |

每完成一步，同步补上对应的测试再进入下一步。

## 12. 约束与注意事项

- 主进程不 import torch 或 triton，硬件相关代码只放在 `backends/` 下。
- 不要自己编造硬件峰值数字和 Gluon API。以本机 Triton 3.8.0 的源码为准，路径在 `~/.local/lib/python3.12/site-packages/triton/experimental/gluon/`。
- 输出给 agent 的内容要控制长度：IR、trace、长日志一律写文件，只返回路径。
- 3050 Laptop 会动态调频，性能结果要带上 std 和 p90，agent 才能判断差异是不是噪声。

---