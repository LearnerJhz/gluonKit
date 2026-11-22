from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal


def _to_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if not isinstance(value, type) and callable(getattr(value, "to_dict", None)):
        return _to_json_value(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _to_json_value(asdict(value))
    if isinstance(value, dict):
        return {key: _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
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
    warp_size: int = 32

    def to_dict(self) -> dict:
        return asdict(self)


# gk bench 信息
@dataclass
class ProfileData:
    kernel_name: str
    avg_ms: float
    std_ms: float = 0.0
    min_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    trace_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_kernel_dict(self) -> dict[str, Any]:
        return {
            "name": self.kernel_name,
            "avg": self.avg_ms,
            "median": self.median_ms,
            "min": self.min_ms,
            "p95": self.p95_ms,
            "std": self.std_ms,
        }

    def to_baseline_dict(self) -> dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "avg_ms": self.avg_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
        }


@dataclass
class BenchResult:
    """The measured kernel profile with baseline comparison and throughput metrics."""

    kernel_profile: ProfileData
    metrics: dict[str, Any]
    baseline: dict[str, Any] | None = None
    # Peaks used for roofline: {"tflops": ..., "gbps": ...}.
    peaks: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return {
            "trace_path": self.kernel_profile.trace_path,
            "kernel": self.kernel_profile.to_kernel_dict(),
            "metrics": self.metrics,
            "baseline": self.baseline,
            "peaks": self.peaks,
        }
