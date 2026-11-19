"""Derived rates from mean kernel time (ms), FLOPs and estimated bytes moved."""

from math import isfinite


def validate_metadata(*, flops=None, bytes_moved=None, peaks: dict | None = None) -> None:
    """Reject malformed workload/peak values; cheap, so bench can call it before GPU work."""
    # Peaks use TFLOP/s and GB/s, matching the reported rates.
    if peaks is not None and not isinstance(peaks, dict):
        raise ValueError("peaks must be a dict with tflops and gbps")
    peaks = peaks or {}
    for name, value in (("flops", flops), ("bytes_moved", bytes_moved),
                        ("peaks.tflops", peaks.get("tflops")), ("peaks.gbps", peaks.get("gbps"))):
        if value is not None and (isinstance(value, bool) or
                not isinstance(value, (int, float)) or not isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be a positive finite number")


def calculate_roofline(avg_ms: float | None, *, flops=None,
                       bytes_moved=None, peaks: dict | None = None) -> dict:
    validate_metadata(flops=flops, bytes_moved=bytes_moved, peaks=peaks)
    peaks = peaks or {}
    peak_compute, peak_memory = peaks.get("tflops"), peaks.get("gbps")

    result = dict(tflops=None, gbps=None, roofline_pct=None, bound_by=None)
    if avg_ms is None or not isfinite(avg_ms) or avg_ms <= 0:
        return result
    seconds = avg_ms * 1e-3
    if flops is not None:
        result["tflops"] = flops / seconds / 1e12
    if bytes_moved is not None:
        result["gbps"] = bytes_moved / seconds / 1e9
    # A resource is "in play" once the task declares its workload (flops / bytes_moved). Every
    # declared workload needs its peak, otherwise the bound would silently ignore a resource
    # (e.g. a gemm without compute_mode must not be rated by its memory time alone).
    ideal = {}  # resource -> seconds needed at peak rate
    if flops is not None and peak_compute is not None:
        ideal["compute"] = flops / (peak_compute * 1e12)
    if bytes_moved is not None and peak_memory is not None:
        ideal["memory"] = bytes_moved / (peak_memory * 1e9)
    declared = (flops is not None) + (bytes_moved is not None)
    if ideal and len(ideal) == declared:
        result["bound_by"] = max(ideal, key=ideal.get)  # ties resolve to compute (first inserted)
        result["roofline_pct"] = ideal[result["bound_by"]] / seconds * 100
    return result
