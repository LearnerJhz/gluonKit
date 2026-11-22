"""Per-eval stored baseline measurements under `.gluonkit/baseline/`."""

import json
from math import isfinite
from pathlib import Path
from typing import Any

from gluonkit.core.storage import (
    atomic_write_json,
    gluonkit_dir,
    normalize_path,
    now_iso,
    resolve_root,
)
from gluonkit.interfaces.types import ProbeResult

BASELINES_DIRNAME = "baseline"


def device_key(info: ProbeResult) -> str:
    """Stable key identifying the GPU platform, arch, and model."""
    return f"{info.platform}:{info.arch}:{info.device_name}"


def is_valid_ms(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value) and value > 0


def baseline_filename(eval_path: str | Path) -> str:
    """Derive `baseline_<name>.json` from an `eval_<name>.py` (or `<name>.py`) path."""
    stem = Path(eval_path).stem
    name = stem.removeprefix("eval_") if stem.startswith("eval_") and len(stem) > 5 else stem
    return f"baseline_{name}.json"


class BaselineStore:
    def __init__(self, root: str | Path | None = None):
        self.root = resolve_root(root)
        self.dir = gluonkit_dir(self.root, BASELINES_DIRNAME)

    def target_path(self, eval_path: str | Path) -> Path:
        return self.dir / baseline_filename(eval_path)

    def display_path(self, path: Path) -> str:
        return normalize_path(path, self.root)

    def _read(self, eval_path: str | Path) -> dict[str, Any] | None:
        target = self.target_path(eval_path)
        if not target.is_file():
            return None
        return json.loads(target.read_text(encoding="utf-8"))

    def load_record(self, eval_path: str | Path) -> dict[str, Any] | None:
        """Return `{"file": ..., **entry}` if a baseline file exists, otherwise None."""
        entry = self._read(eval_path)
        if entry is None:
            return None
        return {"file": self.display_path(self.target_path(eval_path)), **entry}

    def lookup(self, eval_path: str | Path) -> dict[str, Any]:
        """Load stored baseline summary (`kernel_name` and `avg_ms`)."""
        target = self.target_path(eval_path)
        if not target.is_file():
            raise FileNotFoundError(f"Baseline not found: {self.display_path(target)} (run `gk baseline` first)")
        entry = json.loads(target.read_text(encoding="utf-8"))
        baseline = entry["baseline"]
        return {
            "kernel_name": baseline["kernel_name"],
            "avg_ms": baseline["avg_ms"],
        }

    def save(
        self,
        eval_path: str | Path,
        *,
        baseline: dict[str, Any],
        device_key: str | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        """Persist a measured ref baseline entry atomically."""
        if not isinstance(baseline, dict) or not is_valid_ms(baseline.get("avg_ms")):
            raise ValueError("baseline must contain a finite positive 'avg_ms'")
        stored_baseline = {
            k: v for k, v in baseline.items() if k not in ("source", "measured_at")
        }
        stored_baseline["avg_ms"] = float(stored_baseline["avg_ms"])
        entry: dict[str, Any] = {
            "device_key": device_key,
            "measured_at": now_iso(),
            "baseline": stored_baseline,
        }
        target = self.target_path(eval_path)
        atomic_write_json(target, entry)
        return target, entry
