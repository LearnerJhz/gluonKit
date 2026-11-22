"""Peak configuration hub (`.gluonkit/peaks.json`) and Roofline evaluation."""

from pathlib import Path
from typing import Any

from gluonkit.core.storage import atomic_write_json, gluonkit_dir, normalize_path, read_json

PEAKS_FILENAME = "peaks.json"


def peaks_path(root: str | Path | None = None) -> Path:
    return gluonkit_dir(root, PEAKS_FILENAME)


def load_peaks(root: str | Path | None = None) -> dict[str, float]:
    """Load `.gluonkit/peaks.json` (mandatory hub for `gk bench`)."""
    path = peaks_path(root)
    if not path.is_file():
        display = normalize_path(path, root)
        raise FileNotFoundError(
            f"Peaks file not found: {display} (run `.venv/bin/python scripts/calibrate_peaks.py` first)"
        )
    data = read_json(path)
    raw = data["peaks"] if isinstance(data.get("peaks"), dict) else data
    return {k: float(v) for k, v in raw.items() if not isinstance(v, bool) and isinstance(v, (int, float))}


def save_peaks(updates: dict[str, float], root: str | Path | None = None) -> dict[str, float]:
    """Merge calibrated peak values into `.gluonkit/peaks.json` and persist atomically."""
    path = peaks_path(root)
    peaks = load_peaks(root) if path.is_file() else {}
    for k, v in updates.items():
        peaks[k] = float(v)
    ordered = dict(sorted(peaks.items()))
    atomic_write_json(path, ordered)
    return ordered


def resolve_peaks(task: dict[str, Any], calibrated: dict[str, float]) -> dict[str, float | None] | None:
    """Resolve the task's required `{tflops, gbps}` peaks from `.gluonkit/peaks.json`."""
    flops = task.get("flops")
    bytes_moved = task.get("bytes_moved")
    mode = task.get("compute_mode")

    tflops_peak: float | None = None
    if flops is not None or mode is not None:
        if not mode:
            raise ValueError("Task declares 'flops' but no 'compute_mode' (e.g. 'fp16', 'bf16', 'tf32', 'fp32')")
        key = f"tflops_{mode}"
        if key not in calibrated:
            raise ValueError(
                f"Peak {key!r} is missing from .gluonkit/peaks.json "
                f"(run `.venv/bin/python scripts/calibrate_peaks.py --items {key}`)"
            )
        tflops_peak = float(calibrated[key])

    gbps_peak: float | None = None
    if bytes_moved is not None:
        if "gbps" not in calibrated:
            raise ValueError(
                "Peak 'gbps' is missing from .gluonkit/peaks.json "
                "(run `.venv/bin/python scripts/calibrate_peaks.py --items gbps`)"
            )
        gbps_peak = float(calibrated["gbps"])

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
    """Compute achieved throughput (TFLOP/s, GB/s), bottleneck, and roofline utilization (%)."""
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
