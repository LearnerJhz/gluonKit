from dataclasses import asdict, dataclass, field
from typing import Literal

# gk probe 信息
@dataclass
class ProbeResult:
    device_name: str
    platform: str
    arch: str
    compiler_version: str
    torch_version: str
    smem_per_block_optin: int | None
    sm_count: int
    vram_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)

# gk compile 信息
@dataclass
class CompileResult:
    result: Literal["success", "failed"]
    info: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DumpInfoResult(CompileResult):
    artifacts: dict[str, dict[str, str | None]] = field(default_factory=dict)

# gk check 信息
@dataclass
class CheckResult:
    passed: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

# gk bench 信息
@dataclass
class ProfileKernel:
    name: str
    avg_ms: float


@dataclass
class ProfileData:
    device_time_ms: float          # GPU kernel 总耗时 (ms)
    kernels: list[ProfileKernel]   # 内部已聚合好的 kernel 列表
    trace_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchResult:
    device_time_ms: float
    kernels: list[ProfileKernel]
    baseline_time_ms: float | None = None
    speedup_vs_baseline: float | None = None
    tflops: float | None = None
    gbps: float | None = None
    roofline_pct: float | None = None
    bound_by: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
