"""NVIDIA timing primitives. No aggregation policy or automatic fallback here."""

from pathlib import Path
import statistics
from typing import Callable
from uuid import uuid4

from gluonkit.interfaces.types import ProfileData, ProfileKernel


def _validate(fn: Callable[[], object], iters: int, warmup: int) -> None:
    if not callable(fn):
        raise TypeError("fn must be callable")
    if type(iters) is not int or iters <= 0:
        raise ValueError("iters must be a positive integer")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")


def _cache_buffer(device: str):
    import torch

    props = torch.cuda.get_device_properties(device)
    l2_bytes = getattr(props, "L2_cache_size", 0)
    # Streaming writes larger than L2 evict ordinary cached data. This is not
    # a guarantee for CUDA persisting-cache policies. Older torch: 256 MiB.
    size = 2 * l2_bytes if l2_bytes > 0 else 256 * 1024 * 1024
    return torch.empty(size, dtype=torch.uint8, device=device)


def flush_cache(device: str = "cuda:0") -> None:
    """Best-effort L2 eviction; return after the eviction writes complete."""
    import torch

    with torch.cuda.device(device):
        buffer = _cache_buffer(device)
        buffer.zero_()
        torch.cuda.synchronize(device)


def _collect_kernels(events, markers: dict[str, int], cache_marker: str,
                     device_index: int, iters: int) -> tuple[list[float], list[ProfileKernel]]:
    """Join CUDA events to CPU scopes by correlation IDs, never GPU time windows.

    Torch ops use linked_correlation_id; direct Gluon launches use the same
    correlation id as their CUDA runtime/driver launch. Count raw kernel
    events once, excluding copies, memsets and GPU annotation ranges, and
    return per-iteration durations plus aggregated kernel stats.
    """
    from torch.autograd import DeviceType

    cpu_events = [event for event in events if event.device_type == DeviceType.CPU]
    if any(not hasattr(event, "activity_type") or
           not hasattr(event, "linked_correlation_id") for event in events):
        raise RuntimeError("torch.profiler events lack activity/correlation fields; "
                           "use a compatible PyTorch (validated with 2.14.1).")
    frontend = {event.id: event for event in cpu_events
                if event.activity_type in ("cpu_op", "user_annotation")}
    launches = {event.id: event for event in cpu_events
                if event.activity_type in ("cuda_runtime", "cuda_driver")}
    per_iter_ms = [0.0] * iters
    grouped: dict[str, list[float]] = {}
    for event in events:
        if (event.device_type != DeviceType.CUDA or
                event.device_index != device_index or event.activity_type != "kernel"):
            continue
        parent = (frontend.get(event.linked_correlation_id)
                  if event.linked_correlation_id else launches.get(event.id))
        while parent is not None:
            if parent.name == cache_marker:
                break
            iter_idx = markers.get(parent.name)
            if iter_idx is not None:
                duration_ms = event.time_range.elapsed_us() / 1000
                per_iter_ms[iter_idx] += duration_ms
                grouped.setdefault(event.name, []).append(duration_ms)
                break
            parent = parent.cpu_parent
        else:
            raise RuntimeError(f"Cannot associate CUDA kernel {event.name!r} with an "
                               "iteration; launch work on the profiling CPU thread.")
    kernels = [
        ProfileKernel(name=name, avg_ms=sum(durations) / len(durations))
        for name, durations in grouped.items()
    ]
    return per_iter_ms, kernels


def profile(
    fn: Callable[[], object], iters: int, warmup: int, *,
    device: str = "cuda:0", flush_l2: bool = False, trace_path: str | None = None,
) -> ProfileData:
    """Collect mean per-call device kernel duration and aggregated kernel stats in ms.

    fn launches work on the profiling CPU thread and selected device; multiple
    CUDA streams are supported. Warmup and optional eviction are excluded.
    Errors propagate; no silent fallback.
    """
    _validate(fn, iters, warmup)
    import torch

    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError("CUDA profiler activity is unavailable.")
    prefix = f"gluonkit.{uuid4().hex}"
    markers = {f"{prefix}.iteration.{i}": i for i in range(iters)}
    cache_marker = f"{prefix}.flush_l2"
    with torch.cuda.device(device):
        device_index = torch.cuda.current_device()
        buffer = _cache_buffer(device) if flush_l2 else None
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as capture:
            for marker in markers:
                if buffer is not None:
                    with torch.profiler.record_function(cache_marker):
                        buffer.zero_()
                        torch.cuda.synchronize(device)
                with torch.profiler.record_function(marker):
                    fn()
                    torch.cuda.synchronize(device)
        per_iter_ms, kernels = _collect_kernels(
            capture.events(), markers, cache_marker, device_index, iters,
        )
        device_time_ms = statistics.fmean(per_iter_ms)
        std_ms = statistics.stdev(per_iter_ms) if len(per_iter_ms) > 1 else 0.0
        min_ms = min(per_iter_ms)
        median_ms = statistics.median(per_iter_ms)
        exported = None
        if trace_path is not None:
            path = Path(trace_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            capture.export_chrome_trace(str(path))
            exported = str(path)
    return ProfileData(
        device_time_ms=device_time_ms,
        kernels=kernels,
        std_ms=std_ms,
        min_ms=min_ms,
        median_ms=median_ms,
        trace_path=exported,
    )
