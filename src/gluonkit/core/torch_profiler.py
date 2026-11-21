"""SIMT GPU kernel timing engines: Kineto correlation (Level 1) and Device Event timing (Level 2)."""

from pathlib import Path
import statistics
from typing import Any, Callable
from uuid import uuid4

from gluonkit.interfaces.types import ProfileData

_RUNTIME_LAUNCH_TYPES = frozenset({
    "cuda_runtime",
    "cuda_driver",
    "hip_runtime",
    "maca_runtime",
    "privateuse1_runtime",
    "privateuse1_driver",
})


def validate_profile_args(fn: Callable[[], object], iters: int, warmup: int) -> None:
    if not callable(fn):
        raise TypeError("fn must be callable")
    if type(iters) is not int or iters <= 0:
        raise ValueError("iters must be a positive integer")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")


def compute_p95(samples: list[float]) -> float:
    """Compute the 95th percentile from a non-empty list of durations."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    pos = 0.95 * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + frac * (ordered[hi] - ordered[lo])


def _summarize_samples(
    per_iter_ms: list[float],
    *,
    kernel_name: str,
    timing_method: str,
    trace_path: str | None = None,
) -> ProfileData:
    avg_ms = statistics.fmean(per_iter_ms)
    std_ms = statistics.stdev(per_iter_ms) if len(per_iter_ms) > 1 else 0.0
    min_ms = min(per_iter_ms)
    median_ms = statistics.median(per_iter_ms)
    p95_ms = compute_p95(per_iter_ms)
    return ProfileData(
        kernel_name=kernel_name,
        avg_ms=avg_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
        iters=len(per_iter_ms),
        timing_method=timing_method,
        trace_path=trace_path,
    )


def allocate_l2_flush_buffer(device: str) -> Any:
    import torch

    props = torch.cuda.get_device_properties(device)
    l2_bytes = getattr(props, "L2_cache_size", 0) or 0
    size = 2 * l2_bytes if l2_bytes > 0 else 256 * 1024 * 1024
    return torch.empty(size, dtype=torch.uint8, device=device)


def collect_kineto_kernels(
    events: Any,
    markers: dict[str, int],
    cache_marker: str,
    device_index: int,
    iters: int,
) -> tuple[list[float], str]:
    """Join GPU kernel events to CPU iteration scopes by correlation IDs, never GPU time windows."""
    from torch.autograd import DeviceType

    cpu_events = [event for event in events if event.device_type == DeviceType.CPU]
    if any(
        not hasattr(event, "activity_type") or not hasattr(event, "linked_correlation_id")
        for event in events
    ):
        raise RuntimeError(
            "torch.profiler events lack activity/correlation fields; "
            "use a compatible PyTorch (validated with 2.14.1) or device_event timing."
        )
    frontend = {
        event.id: event
        for event in cpu_events
        if event.activity_type in ("cpu_op", "user_annotation")
    }
    launches = {
        event.id: event
        for event in cpu_events
        if event.activity_type in _RUNTIME_LAUNCH_TYPES
    }
    per_iter_ms = [0.0] * iters
    kernel_name = ""
    for event in events:
        if (
            event.device_type != DeviceType.CUDA
            or event.device_index != device_index
            or event.activity_type != "kernel"
        ):
            continue
        parent = (
            frontend.get(event.linked_correlation_id)
            if event.linked_correlation_id
            else launches.get(event.id)
        )
        while parent is not None:
            if parent.name == cache_marker:
                break
            iter_idx = markers.get(parent.name)
            if iter_idx is not None:
                duration_ms = event.time_range.elapsed_us() / 1000
                per_iter_ms[iter_idx] += duration_ms
                if not kernel_name:
                    kernel_name = event.name
                break
            parent = parent.cpu_parent
        else:
            raise RuntimeError(
                f"Cannot associate GPU kernel {event.name!r} with an iteration; "
                "launch work on the profiling CPU thread."
            )
    return per_iter_ms, kernel_name


def profile_with_kineto(
    fn: Callable[[], object],
    iters: int,
    warmup: int,
    *,
    device: str = "cuda:0",
    flush_l2: bool = False,
    trace_path: str | None = None,
) -> ProfileData:
    """Level 1 Profiler: exact device kernel durations via PyTorch Kineto correlation."""
    validate_profile_args(fn, iters, warmup)
    import torch

    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError("CUDA profiler activity is unavailable.")
    prefix = f"gluonkit.{uuid4().hex}"
    markers = {f"{prefix}.iteration.{i}": i for i in range(iters)}
    cache_marker = f"{prefix}.flush_l2"
    with torch.cuda.device(device):
        device_index = torch.cuda.current_device()
        buffer = allocate_l2_flush_buffer(device) if flush_l2 else None
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as capture:
            for marker in markers:
                if buffer is not None:
                    with torch.profiler.record_function(cache_marker):
                        buffer.zero_()
                        torch.cuda.synchronize(device)
                with torch.profiler.record_function(marker):
                    fn()
                    torch.cuda.synchronize(device)
        per_iter_ms, kernel_name = collect_kineto_kernels(
            capture.events(), markers, cache_marker, device_index, iters,
        )
        exported = None
        if trace_path is not None:
            path = Path(trace_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            capture.export_chrome_trace(str(path))
            exported = str(path)
    return _summarize_samples(
        per_iter_ms,
        kernel_name=kernel_name,
        timing_method="kineto_kernel",
        trace_path=exported,
    )


def profile_with_events(
    fn: Callable[[], object],
    iters: int,
    warmup: int,
    *,
    device: str = "cuda:0",
    flush_l2: bool = False,
    trace_path: str | None = None,
) -> ProfileData:
    """Level 2 Profiler: per-iteration `torch.cuda.Event(enable_timing=True)` timing.

    Used on SIMT backends (such as early MetaX MACA / custom runtimes) where Kineto
    correlation for raw Triton launches may be unavailable.
    """
    validate_profile_args(fn, iters, warmup)
    import torch
    from triton.compiler.compiler import CompiledKernel

    captured_names: list[str] = []
    original = CompiledKernel.launch_metadata

    def capture_name(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        if not captured_names and getattr(kernel, "name", None):
            captured_names.append(str(kernel.name))
        return original(kernel, *args, **kwargs)

    with torch.cuda.device(device):
        buffer = allocate_l2_flush_buffer(device) if flush_l2 else None
        CompiledKernel.launch_metadata = capture_name
        try:
            for _ in range(warmup):
                fn()
        finally:
            CompiledKernel.launch_metadata = original
        torch.cuda.synchronize(device)

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            if buffer is not None:
                buffer.zero_()
                torch.cuda.synchronize(device)
            start_events[i].record()
            fn()
            end_events[i].record()
        torch.cuda.synchronize(device)
        per_iter_ms = [float(s.elapsed_time(e)) for s, e in zip(start_events, end_events)]

    kernel_name = captured_names[0] if captured_names else getattr(fn, "__name__", "kernel")
    return _summarize_samples(
        per_iter_ms,
        kernel_name=kernel_name,
        timing_method="device_event",
        trace_path=None,
    )
