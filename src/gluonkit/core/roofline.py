"""Theoretical hardware peak configuration (`.gluonkit/peaks.json`) and Roofline evaluation."""

from math import isfinite
from pathlib import Path
from typing import Any

from gluonkit.core.storage import atomic_write_json, gluonkit_dir, normalize_path, read_json

PEAKS_FILENAME = "peaks.json"
_INIT_HINT = "run `.venv/bin/python scripts/init_peaks.py` or author `.gluonkit/peaks.json` manually"


def peaks_path(root: str | Path | None = None) -> Path:
    return gluonkit_dir(root, PEAKS_FILENAME)


def _is_positive_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value) and value > 0


def load_peaks(root: str | Path | None = None) -> dict[str, float]:
    """Load theoretical peak values from `.gluonkit/peaks.json` (structured or flat hand-written JSON)."""
    path = peaks_path(root)
    display = normalize_path(path, root)
    if not path.is_file():
        raise FileNotFoundError(f"Peaks file not found: {display} ({_INIT_HINT})")
    data = read_json(path, recovery_hint=_INIT_HINT)
    has_envelope = "peaks" in data
    raw = data["peaks"] if has_envelope else data
    if not isinstance(raw, dict):
        raise ValueError(f"{display} 'peaks' must be a JSON object ({_INIT_HINT})")
    peaks: dict[str, float] = {}
    for key, value in raw.items():
        if not has_envelope and isinstance(value, (str, dict, list, type(None))):
            continue
        if not _is_positive_number(value):
            raise ValueError(f"{display} peak {key!r} must be a positive finite number, got {value!r}")
        peaks[key] = float(value)
    return peaks


def save_peaks(
    peaks: dict[str, float],
    root: str | Path | None = None,
    *,
    device_key: str | None = None,
) -> dict[str, Any]:
    """Persist theoretical hardware peaks to `.gluonkit/peaks.json` atomically."""
    ordered: dict[str, float] = {}
    for key, value in sorted(peaks.items()):
        if not _is_positive_number(value):
            raise ValueError(f"Peak {key!r} must be a positive finite number, got {value!r}")
        ordered[key] = float(value)

    payload: dict[str, Any] = (
        {"device_key": device_key, "peaks": ordered} if device_key is not None else ordered
    )
    atomic_write_json(peaks_path(root), payload)
    return payload


def resolve_peaks(task: dict[str, Any], peak_table: dict[str, float]) -> dict[str, float | None] | None:
    """Resolve the task's required `{tflops, gbps}` theoretical peaks from `.gluonkit/peaks.json`."""
    flops = task.get("flops")
    bytes_moved = task.get("bytes_moved")
    mode = task.get("compute_mode")

    tflops_peak: float | None = None
    if flops is not None or mode is not None:
        if not mode:
            raise ValueError("Task declares 'flops' but no 'compute_mode' (e.g. 'fp16', 'bf16', 'tf32', 'fp32')")
        key = f"tflops_{mode}"
        if key not in peak_table:
            raise ValueError(f"Peak {key!r} is missing from .gluonkit/peaks.json ({_INIT_HINT})")
        tflops_peak = float(peak_table[key])

    gbps_peak: float | None = None
    if bytes_moved is not None:
        if "gbps" not in peak_table:
            raise ValueError(f"Peak 'gbps' is missing from .gluonkit/peaks.json ({_INIT_HINT})")
        gbps_peak = float(peak_table["gbps"])

    if tflops_peak is None and gbps_peak is None:
        return None
    return {"tflops": tflops_peak, "gbps": gbps_peak}


def roofline_model_name(*, flops: float | None = None, bytes_moved: float | None = None) -> str | None:
    if flops is not None and bytes_moved is not None:
        return "dual_roofline"
    if bytes_moved is not None:
        return "memory_only"
    if flops is not None:
        return "compute_only"
    return None


def calculate_roofline(
    avg_ms: float | None,
    *,
    flops: float | None = None,
    bytes_moved: float | None = None,
    peaks: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    """Compute achieved throughput (TFLOP/s, GB/s), bottleneck, and theoretical roofline utilization (%)."""
    peaks = peaks or {}
    peak_compute, peak_memory = peaks.get("tflops"), peaks.get("gbps")
    result: dict[str, Any] = {
        "tflops": None,
        "gbps": None,
        "roofline": None,
        "bound_by": None,
        "roofline_model": roofline_model_name(flops=flops, bytes_moved=bytes_moved),
    }
    if avg_ms is None or avg_ms <= 0:
        return result

    seconds = avg_ms * 1e-3
    if flops is not None:
        result["tflops"] = flops / seconds / 1e12
    if bytes_moved is not None:
        result["gbps"] = bytes_moved / seconds / 1e9

    ideal: dict[str, float] = {}
    if flops is not None and peak_compute:
        ideal["compute"] = flops / (peak_compute * 1e12)
    if bytes_moved is not None and peak_memory:
        ideal["memory"] = bytes_moved / (peak_memory * 1e9)
    declared = (flops is not None) + (bytes_moved is not None)
    if ideal and len(ideal) == declared:
        bound_by = max(ideal, key=ideal.get)
        result["bound_by"] = bound_by
        result["roofline"] = ideal[bound_by] / seconds * 100
    return result
