"""Single-device peak configuration: `.gluonkit/peaks.json`, written by `scripts/calibrate_peaks.py` (or hand-authored) and read by `gk bench`.

File layout::

    {
      "device_key": "<platform>:<arch>:<device_name>",
      "calibrated_at": "<iso8601>",
      "env": {"torch": "...", "triton": "..."},
      "peaks": {"gbps": 157.14, "tflops_fp16": 25.3, ...},
      "sources": {"gbps": "torch.neg fp32 (512 MiB)", ...}
    }

When hand-authoring `.gluonkit/peaks.json`, only `"peaks"` is required. Items are merged
on re-calibration for the same device, so `calibrate_peaks.py --items gbps` only
refreshes the bandwidth entry.
"""

from math import isfinite
from pathlib import Path
from typing import Any

from gluonkit.core.storage import (
    atomic_write_json,
    gluonkit_dir,
    normalize_path,
    now_iso,
    read_json,
    resolve_root,
)
from gluonkit.interfaces.types import ProbeResult

PEAKS_FILENAME = "peaks.json"
# Calibrated item name = "tflops_" + the task's `compute_mode` (e.g. "fp16" -> "tflops_fp16").
COMPUTE_PEAK_PREFIX = "tflops_"
MEMORY_PEAK = "gbps"
_CORRUPT_HINT = "delete it and rerun scripts/calibrate_peaks.py"


def device_key(info: ProbeResult) -> str:
    """Stable, human-readable key identifying the GPU platform, arch, and model."""
    return f"{info.platform}:{info.arch}:{info.device_name}"


def _is_positive_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value) and value > 0


class PeakStore:
    def __init__(self, root: str | Path | None = None):
        self.root = resolve_root(root)
        self.path = gluonkit_dir(self.root, PEAKS_FILENAME)

    def display_path(self) -> str:
        return normalize_path(self.path, self.root)

    def load(self) -> dict[str, Any] | None:
        """Load and validate `.gluonkit/peaks.json`, returning None if absent."""
        if not self.path.is_file():
            return None
        data = read_json(self.path, recovery_hint=_CORRUPT_HINT)
        peaks = data.get("peaks")
        if not isinstance(peaks, dict):
            raise ValueError(f"{self.path} must contain a 'peaks' object; {_CORRUPT_HINT}")
        for item, value in peaks.items():
            if not _is_positive_number(value):
                raise ValueError(f"{self.path} peak {item!r} must be a positive finite number; {_CORRUPT_HINT}")
        sources = data.get("sources")
        if sources is not None and not isinstance(sources, dict):
            raise ValueError(f"{self.path} 'sources' must be an object; {_CORRUPT_HINT}")
        return data

    def get(self, info: ProbeResult) -> dict[str, Any] | None:
        """Return the peak record for the active device, validating `device_key` if present."""
        data = self.load()
        if data is None:
            return None
        stored_device = data.get("device_key")
        current_device = device_key(info)
        if stored_device and stored_device != current_device:
            raise ValueError(
                f"{self.display_path()} belongs to {stored_device!r}, "
                f"current device is {current_device!r}; update it or rerun scripts/calibrate_peaks.py"
            )
        return data

    def update(self, info: ProbeResult, results: dict[str, dict[str, Any]],
               env: dict[str, Any] | None = None) -> dict[str, Any]:
        """Merge freshly calibrated items for this device and persist atomically."""
        existing = self.load()
        key = device_key(info)
        same_device = existing is not None and existing.get("device_key") in (None, key)
        peaks = dict(existing.get("peaks") or {}) if same_device else {}
        sources = dict(existing.get("sources") or {}) if same_device else {}
        for item, result in results.items():
            value = result.get("value")
            if not _is_positive_number(value):
                raise ValueError(f"calibration item {item!r} produced an invalid value: {value!r}")
            peaks[item] = float(value)
            source = result.get("source") or result.get("method")
            if source:
                sources[item] = str(source)
        entry: dict[str, Any] = {
            "device_key": key,
            "calibrated_at": now_iso(),
            "env": dict(env or {}),
            "peaks": dict(sorted(peaks.items())),
        }
        if sources:
            entry["sources"] = dict(sorted(sources.items()))
        atomic_write_json(self.path, entry)
        return entry


def resolve_peaks(task_peaks: Any, compute_mode: Any,
                  calibrated: dict[str, float] | None) -> tuple[dict | None, dict | None]:
    """Combine task-declared peaks with calibrated ones; explicit task values win per field.

    `compute_mode` (e.g. "fp16", "bf16", "tf32", "fp32") selects the calibrated
    `tflops_<mode>` item for the compute peak. Without it `tflops` stays None (no
    guessing), so roofline is null for tasks that declare `flops`.

    Returns `(peaks, source)` where `peaks` feeds `calculate_roofline` and `source` maps
    each field to `"task"`, `"calibrated:<item>"` or None. Both are None when nothing is known.
    """
    if task_peaks is not None and not isinstance(task_peaks, dict):
        raise ValueError("peaks must be a dict with tflops and gbps")
    if compute_mode is not None and not isinstance(compute_mode, str):
        raise ValueError("compute_mode must be a string such as 'fp16', 'bf16', 'tf32' or 'fp32'")
    calibrated = calibrated or {}
    compute_item = None if compute_mode is None else COMPUTE_PEAK_PREFIX + compute_mode
    if compute_item is not None and calibrated and compute_item not in calibrated:
        modes = sorted(k[len(COMPUTE_PEAK_PREFIX):] for k in calibrated if k.startswith(COMPUTE_PEAK_PREFIX))
        raise ValueError(f"compute_mode {compute_mode!r} is not calibrated for this device; "
                         f"available: {modes} (run scripts/calibrate_peaks.py --items {compute_item})")
    task_peaks = task_peaks or {}
    peaks: dict[str, Any] = {}
    source: dict[str, str | None] = {}
    for field, item in (("tflops", compute_item), ("gbps", MEMORY_PEAK)):
        peaks[field], source[field] = None, None
        if task_peaks.get(field) is not None:
            peaks[field] = task_peaks[field]
            source[field] = "task"
        elif item is not None and item in calibrated:
            peaks[field] = calibrated[item]
            source[field] = f"calibrated:{item}"
    if all(value is None for value in peaks.values()):
        return None, None
    return peaks, source
