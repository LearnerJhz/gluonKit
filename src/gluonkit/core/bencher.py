"""Benchmark a prepared task using the backend's mean per-call kernel time."""

from gluonkit.core.peaks import resolve_peaks
from gluonkit.core.roofline import calculate_roofline, validate_metadata
from gluonkit.interfaces.backend import Backend
from gluonkit.interfaces.types import BenchResult


DEFAULT_ITERS = 100
DEFAULT_WARMUP = 10


def bench_task(task: dict, backend: Backend, *, iters: int = DEFAULT_ITERS,
               warmup: int = DEFAULT_WARMUP, no_baseline: bool = False,
               trace_path: str | None = None,
               calibrated_peaks: dict[str, float] | None = None) -> BenchResult:
    """`calibrated_peaks` (from `.gluonkit/peaks.json`) fills peak fields the task omits."""
    ref = None if no_baseline else task.get("ref")
    if ref is not None and not callable(ref):
        raise ValueError("ref must be callable when baseline is enabled")
    peaks, peaks_source = resolve_peaks(task.get("peaks"), task.get("compute_mode"), calibrated_peaks)
    metadata = {"flops": task.get("flops"), "bytes_moved": task.get("bytes_moved"), "peaks": peaks}
    validate_metadata(**metadata)  # Fail on bad metadata before executing GPU work.
    measured = backend.profile(task["run"], iters, warmup, trace_path=trace_path)
    baseline_ms = None
    baseline = None
    if ref is not None:
        baseline_ms, baseline = backend.profile_baseline(ref, iters, warmup)
    speedup = None
    if measured.avg_ms > 0 and baseline_ms is not None and baseline_ms > 0:
        speedup = baseline_ms / measured.avg_ms
    rates = calculate_roofline(measured.avg_ms, **metadata)
    return BenchResult(
        kernel_profile=measured,
        metrics={
            "speedup": speedup,
            "tflops": rates["tflops"],
            "gbps": rates["gbps"],
            "roofline": rates["roofline_pct"],
            "bound_by": rates["bound_by"],
        },
        baseline=baseline,
        peaks=None if peaks is None else {**peaks, "source": peaks_source},
    )
