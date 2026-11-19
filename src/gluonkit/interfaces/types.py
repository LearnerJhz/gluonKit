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
    trace_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchResult:
    """The measured kernel profile with baseline comparison and throughput metrics."""

    kernel_profile: ProfileData
    metrics: dict[str, float | None]
    baseline: dict[str, Any] | None = None
    # Peaks used for roofline: {"tflops", "gbps", "source": {field: "task" | "calibrated:<item>"}}.
    peaks: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        profile = self.kernel_profile
        return {
            "trace_path": profile.trace_path,
            "kernel": {
                "name": profile.kernel_name,
                "avg": profile.avg_ms,
                "median": profile.median_ms,
                "min": profile.min_ms,
                "std": profile.std_ms,
            },
            "metrics": self.metrics,
            "baseline": self.baseline,
            "peaks": self.peaks,
        }
