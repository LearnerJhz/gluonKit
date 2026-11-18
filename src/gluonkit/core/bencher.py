"""Benchmark a prepared task using the backend's mean per-call kernel time."""

from gluonkit.core.roofline import calculate_roofline
from gluonkit.interfaces.backend import Backend
from gluonkit.interfaces.types import BenchResult


DEFAULT_ITERS = 100
DEFAULT_WARMUP = 10


def bench_task(task: dict, backend: Backend, *, iters: int = DEFAULT_ITERS,
               warmup: int = DEFAULT_WARMUP, no_baseline: bool = False,
               trace_path: str | None = None) -> BenchResult:
    ref = None if no_baseline else task.get("ref")
    if ref is not None and not callable(ref):
        raise ValueError("ref must be callable when baseline is enabled")
    metadata = {name: task.get(name) for name in ("flops", "bytes_moved", "peaks")}
    calculate_roofline(None, **metadata)  # Validate metadata before executing GPU work.
    measured = backend.profile(task["run"], iters, warmup, trace_path=trace_path)
    baseline_ms = None
    if ref is not None:
        baseline_ms = backend.profile(ref, iters, warmup).device_time_ms
    speedup = None
    if measured.device_time_ms > 0 and baseline_ms is not None and baseline_ms > 0:
        speedup = baseline_ms / measured.device_time_ms
    return BenchResult(
        device_time_ms=measured.device_time_ms,
        kernels=measured.kernels,
        baseline_time_ms=baseline_ms,
        speedup_vs_baseline=speedup,
        trace_path=measured.trace_path,
        **calculate_roofline(measured.device_time_ms, **metadata),
    )
