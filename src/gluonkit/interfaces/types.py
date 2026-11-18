from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal


def _to_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _to_json_value(asdict(value))
    if isinstance(value, dict):
        return {key: _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_json_value(value.to_dict())
    return value


@dataclass
class Output:
    """Unified JSON output envelope: only `result` and `info`."""

    result: Literal["success", "failed"]
    info: Any = None

    @classmethod
    def ok(cls, info: Any = None) -> "Output":
        return cls(result="success", info=info)

    @classmethod
    def fail(cls, info: str) -> "Output":
        return cls(result="failed", info=info)

    def to_dict(self) -> dict:
        return {"result": self.result, "info": _to_json_value(self.info)}

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

# gk bench 信息
@dataclass
class ProfileKernel:
    name: str
    avg_ms: float


@dataclass
class ProfileData:
    device_time_ms: float          # GPU kernel 总耗时 (ms)
    kernels: list[ProfileKernel]   # 内部已聚合好的 kernel 列表
    std_ms: float = 0.0
    min_ms: float = 0.0
    median_ms: float = 0.0
    trace_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchResult:
    device_time_ms: float
    kernels: list[ProfileKernel]
    std_ms: float = 0.0
    min_ms: float = 0.0
    median_ms: float = 0.0
    baseline_time_ms: float | None = None
    speedup_vs_baseline: float | None = None
    tflops: float | None = None
    gbps: float | None = None
    roofline_pct: float | None = None
    bound_by: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
