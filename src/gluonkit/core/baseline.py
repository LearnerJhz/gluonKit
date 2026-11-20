"""Per-eval stored baseline measurements under `.gluonkit/baseline/`."""

import hashlib
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

BASELINES_DIRNAME = "baseline"
_CORRUPT_HINT = "delete it and rerun gk bench <eval.py> --baseline"


def is_valid_ms(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value) and value > 0


def baseline_env(info: Any) -> dict[str, str]:
    """Runtime software versions that invalidate a stored baseline when changed."""
    env: dict[str, str] = {}
    for field, key in (("torch_version", "torch"), ("compiler_version", "triton")):
        value = getattr(info, field, None)
        if value:
            env[key] = value
    return env


def _path_tag(normalized_eval_path: str) -> str:
    return hashlib.sha256(normalized_eval_path.encode("utf-8")).hexdigest()[:8]


class BaselineStore:
    def __init__(self, root: str | Path | None = None):
        self.root = resolve_root(root)
        self.dir = gluonkit_dir(self.root, BASELINES_DIRNAME)

    def _prefix(self, eval_path: str | Path) -> str:
        normalized = normalize_path(eval_path, self.root)
        stem = Path(normalized).stem
        return f"{stem}-{_path_tag(normalized)}-"

    def target_path(self, eval_path: str | Path, eval_source_hash: str) -> Path:
        return self.dir / f"{self._prefix(eval_path)}{eval_source_hash[:12]}.json"

    def display_path(self, path: Path) -> str:
        return normalize_path(path, self.root)

    def lookup(
        self,
        *,
        eval_path: str | Path,
        eval_source_hash: str,
        env: dict[str, Any],
        measurement_options: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Look up a matching stored baseline; return `(baseline_dict, None)` or `(None, miss_reason)`."""
        target = self.target_path(eval_path, eval_source_hash)
        if not target.is_file():
            prefix = self._prefix(eval_path)
            if self.dir.is_dir() and any(
                p.is_file() and p.name.startswith(prefix) and p.suffix == ".json"
                for p in self.dir.iterdir()
            ):
                return None, "eval_changed"
            return None, "not_found"

        entry = read_json(target, recovery_hint=_CORRUPT_HINT)
        if entry.get("env") != env:
            return None, "env_changed"
        if entry.get("measurement_options") != measurement_options:
            return None, "options_changed"
        baseline = entry.get("baseline")
        if not isinstance(baseline, dict) or not is_valid_ms(baseline.get("avg_ms")):
            return None, "invalid"
        return {
            **baseline,
            "measured_at": entry.get("measured_at"),
            "source": "stored",
        }, None

    def save(
        self,
        *,
        eval_path: str | Path,
        eval_source_hash: str,
        env: dict[str, Any],
        measurement_options: dict[str, Any],
        baseline: dict[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        """Persist a measured ref baseline entry atomically in the exact 4-key JSON format."""
        if not isinstance(baseline, dict) or not is_valid_ms(baseline.get("avg_ms")):
            raise ValueError("baseline must contain a finite positive 'avg_ms'")
        stored_baseline = {
            k: v for k, v in baseline.items() if k not in ("source", "measured_at")
        }
        stored_baseline["avg_ms"] = float(stored_baseline["avg_ms"])
        entry = {
            "env": dict(env),
            "measurement_options": dict(measurement_options),
            "measured_at": now_iso(),
            "baseline": stored_baseline,
        }
        target = self.target_path(eval_path, eval_source_hash)
        atomic_write_json(target, entry)
        prefix = self._prefix(eval_path)
        for existing in self.dir.iterdir():
            if existing != target and existing.is_file() and existing.name.startswith(prefix) and existing.suffix == ".json":
                existing.unlink()
        return target, entry
