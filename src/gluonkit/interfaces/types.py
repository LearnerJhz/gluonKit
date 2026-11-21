from dataclasses import asdict, dataclass, field, is_dataclass
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


@dataclass
class ProbeResult:
    """SIMT GPU hardware and software environment metadata returned by `gk probe`."""

    device_name: str
    platform: str
    arch: str
    compiler_version: str
    torch_version: str
    smem_per_block_optin: int | None
    sm_count: int
    vram_bytes: int
    compute_unit_count: int | None = None
    warp_size: int = 32
    max_smem_per_block: int | None = None
    default_smem_per_block: int | None = None
    l2_cache_bytes: int | None = None
    supported_compute_modes: list[str] = field(default_factory=lambda: ["fp16", "bf16", "tf32", "fp32"])
    knowledge_docs: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.compute_unit_count is None:
            self.compute_unit_count = self.sm_count
        if self.max_smem_per_block is None:
            self.max_smem_per_block = self.smem_per_block_optin

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data.get("extra"):
            data.pop("extra", None)
        return data


@dataclass
class KernelSummary:
    """Per-kernel SIMT compilation and occupancy summary from `gk dump-info`."""

    n_regs: int | None
    n_spills: int | None
    n_max_threads: int | None
    shared_bytes: int | None
    num_warps: int | None
    allocated_shared_bytes: int | None = None
    blocks_per_sm: int | None = None
    blocks_per_cu: int | None = None
    occupancy_pct: float | None = None
    warp_size: int = 32
    limiting_factor: str | None = None
    occupancy_source: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.blocks_per_cu is None and self.blocks_per_sm is not None:
            self.blocks_per_cu = self.blocks_per_sm
        elif self.blocks_per_sm is None and self.blocks_per_cu is not None:
            self.blocks_per_sm = self.blocks_per_cu

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data.get("extra"):
            data.pop("extra", None)
        return data


@dataclass
class ProfileData:
    """Per-call kernel duration statistics in milliseconds."""

    kernel_name: str
    avg_ms: float
    std_ms: float = 0.0
    min_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    iters: int = 0
    timing_method: str = "kineto_kernel"
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_kernel_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.kernel_name,
            "avg": self.avg_ms,
            "median": self.median_ms,
            "min": self.min_ms,
            "p95": self.p95_ms,
            "std": self.std_ms,
        }
        if self.iters > 0:
            result["iters"] = self.iters
        if self.timing_method:
            result["timing_method"] = self.timing_method
        return result

    def to_baseline_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kernel_name": self.kernel_name,
            "avg_ms": self.avg_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
        }
        if self.iters > 0:
            result["iters"] = self.iters
        if self.timing_method:
            result["timing_method"] = self.timing_method
        return result


@dataclass
class CheckResult:
    """Numerical verification summary and source identity returned by `gk check`."""

    passed: bool
    atol: float
    rtol: float
    max_abs_err: float
    max_rel_err: float
    eval_path: str | None = None
    eval_source_hash: str | None = None
    kernel_path: str | None = None
    kernel_source_hash: str | None = None
    device_key: str | None = None
    checked_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class BenchResult:
    """The measured kernel profile with baseline comparison and throughput metrics."""

    kernel_profile: ProfileData
    metrics: dict[str, Any]
    baseline: dict[str, Any] | None = None
    # Peaks used for roofline: {"tflops", "gbps", "source": {field: "task" | "calibrated:<item>"}}.
    peaks: dict[str, Any] | None = None
    verified: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "trace_path": self.kernel_profile.trace_path,
            "kernel": self.kernel_profile.to_kernel_dict(),
            "metrics": self.metrics,
            "baseline": self.baseline,
            "peaks": self.peaks,
        }
        if self.verified is not None:
            result["verified"] = self.verified
        return result
