from dataclasses import asdict, dataclass, field

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
class CompileError:
    message: str


@dataclass
class CompileResult:
    ok: bool
    errors: list[CompileError]
    artifacts: dict[str, dict[str, str | None]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

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
