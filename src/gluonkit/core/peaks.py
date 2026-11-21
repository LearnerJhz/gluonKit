"""Per-device calibrated peaks: `.gluonkit/peaks.json`, written by `scripts/calibrate_peaks.py`, read by bench.

File layout (version 1)::

    {
      "version": 1,
      "devices": {
        "<platform>:<arch>:<device_name>": {
          "device_name": "...", "platform": "...", "arch": "...",
          "updated_at": "<iso8601>",
          "peaks":   {"gbps": 160.1, "tflops_fp16": 25.3, ...},   # item -> value
          "details": {"gbps": {"unit": "GB/s", "method": "...", "calibrated_at": "...", ...}}
        }
      }
    }

`peaks` uses the same units as bench metrics (GB/s, TFLOP/s). Items are merged on
re-calibration, so `calibrate_peaks.py --items gbps` only refreshes the bandwidth entry.
"""

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

PEAKS_VERSION = 1
PEAKS_FILENAME = "peaks.json"
# Calibrated item name = "tflops_" + the task's `compute_mode` (e.g. "fp16" -> "tflops_fp16").
COMPUTE_PEAK_PREFIX = "tflops_"
MEMORY_PEAK = "gbps"
_CORRUPT_HINT = "delete it and rerun scripts/calibrate_peaks.py"


def device_key(info: ProbeResult) -> str:
    """Stable, human-readable key: identical GPU models on one host share calibration."""
    return f"{info.platform}:{info.arch}:{info.device_name}"


class PeakStore:
    def __init__(self, root: str | Path | None = None):
        self.root = resolve_root(root)
        self.path = gluonkit_dir(self.root, PEAKS_FILENAME)

    def display_path(self) -> str:
        return normalize_path(self.path, self.root)

    def load(self) -> dict:
        if not self.path.is_file():
            return {"version": PEAKS_VERSION, "devices": {}}
        data = read_json(self.path, recovery_hint=_CORRUPT_HINT)
        if not isinstance(data.get("devices"), dict):
            raise ValueError(f"{self.path} has an unknown layout; {_CORRUPT_HINT}")
        if data.get("version") != PEAKS_VERSION:
            raise ValueError(
                f"{self.path} version {data.get('version')!r} is unsupported "
                f"(expected {PEAKS_VERSION}); {_CORRUPT_HINT}"
            )
        return data

    def get(self, info: ProbeResult) -> dict | None:
        return self.load()["devices"].get(device_key(info))

    def update(self, info: ProbeResult, results: dict[str, dict[str, Any]],
               env: dict[str, Any] | None = None) -> dict:
        """Merge freshly calibrated items into this device's entry and persist atomically."""
        data = self.load()
        key = device_key(info)
        now = now_iso()
        entry = data["devices"].get(key) or {}
        peaks = dict(entry.get("peaks") or {})
        details = dict(entry.get("details") or {})
        for item, result in results.items():
            value = result.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                raise ValueError(f"calibration item {item!r} produced an invalid value: {value!r}")
            peaks[item] = float(value)
            details[item] = {**{k: v for k, v in result.items() if k != "value"},
                             "calibrated_at": now, "env": env or {}}
        entry = {
            "device_name": info.device_name,
            "platform": info.platform,
            "arch": info.arch,
            "updated_at": now,
            "peaks": dict(sorted(peaks.items())),
            "details": dict(sorted(details.items())),
        }
        data["devices"][key] = entry
        atomic_write_json(self.path, data)
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
