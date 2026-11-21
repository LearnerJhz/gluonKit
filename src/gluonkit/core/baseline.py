"""Per-eval stored baseline measurements under `.gluonkit/baseline/` with identity verification."""

import hashlib
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

BASELINES_DIRNAME = "baseline"


def is_valid_ms(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value) and value > 0


def baseline_env(info: Any) -> dict[str, str]:
    """Runtime software versions recorded with a stored baseline."""
    env: dict[str, str] = {}
    for field, key in (("torch_version", "torch"), ("compiler_version", "triton")):
        value = getattr(info, field, None)
        if value:
            env[key] = value
    return env


def baseline_filename(eval_path: str | Path) -> str:
    """Derive `baseline_<name>.json` from an `eval_<name>.py` (or `<name>.py`) path."""
    stem = Path(eval_path).stem
    name = stem.removeprefix("eval_") if stem.startswith("eval_") and len(stem) > 5 else stem
    return f"baseline_{name}.json"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    def _infer_legacy_identity(self, eval_rel: str, eval_hash: str) -> bool:
        """Return True if a legacy v1 baseline (without eval_path) matches a recorded history entry."""
        history_path = gluonkit_dir(self.root, "history.jsonl")
        if not history_path.is_file():
            return False
        try:
            for line in history_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("eval_path") == eval_rel and record.get("eval_source_hash") == eval_hash:
                    return True
        except Exception:
            return False
        return False

    def _validate_identity(
        self,
        entry: dict[str, Any],
        eval_path: str | Path,
        *,
        eval_source_hash: str | None = None,
        device_key: str | None = None,
    ) -> None:
        resolved = Path(eval_path).resolve()
        eval_rel = normalize_path(resolved, self.root)
        current_hash = eval_source_hash or (
            _file_sha256(resolved) if resolved.is_file() else None
        )
        display = self.display_path(self.target_path(eval_path))

        stored_path = entry.get("eval_path")
        stored_hash = entry.get("eval_source_hash")
        stored_device = entry.get("device_key")

        if stored_path is not None and stored_path != eval_rel:
            raise ValueError(
                f"Baseline collision in {display}: stored baseline belongs to {stored_path!r}, "
                f"not {eval_rel!r} (run `gk baseline {eval_rel} --refresh` to overwrite)"
            )
        if stored_hash is not None and current_hash is not None and stored_hash != current_hash:
            raise ValueError(
                f"Baseline stale in {display}: evaluation script {eval_rel!r} content changed "
                f"(run `gk baseline {eval_rel} --refresh` to re-measure)"
            )
        if stored_device is not None and device_key is not None and stored_device != device_key:
            raise ValueError(
                f"Baseline device mismatch in {display}: stored baseline was measured on "
                f"{stored_device!r}, current device is {device_key!r} "
                f"(run `gk baseline {eval_rel} --refresh` to re-measure)"
            )
        # If legacy file has no stored_path/stored_hash, verify against history if history exists
        if stored_path is None and current_hash is not None:
            history_path = gluonkit_dir(self.root, "history.jsonl")
            if history_path.is_file() and not self._infer_legacy_identity(eval_rel, current_hash):
                raise ValueError(
                    f"Legacy baseline {display} is unbound and does not match {eval_rel!r} "
                    f"(run `gk baseline {eval_rel} --refresh` to bind and re-measure)"
                )

    def load_record(
        self,
        eval_path: str | Path,
        *,
        eval_source_hash: str | None = None,
        device_key: str | None = None,
    ) -> dict[str, Any] | None:
        """Return `{"file": ..., "status": "reused", **entry}` if valid baseline exists, else None."""
        entry = self._read(eval_path)
        if entry is None:
            return None
        self._validate_identity(
            entry,
            eval_path,
            eval_source_hash=eval_source_hash,
            device_key=device_key,
        )
        return {
            "file": self.display_path(self.target_path(eval_path)),
            "status": "reused",
            **entry,
        }

    def lookup(
        self,
        eval_path: str | Path,
        *,
        eval_source_hash: str | None = None,
        device_key: str | None = None,
    ) -> dict[str, Any]:
        """Load stored baseline summary (`kernel_name` and `avg_ms`), verifying identity."""
        target = self.target_path(eval_path)
        if not target.is_file():
            raise FileNotFoundError(
                f"Baseline not found: {self.display_path(target)} (run `gk baseline` first)"
            )
        entry = json.loads(target.read_text(encoding="utf-8"))
        self._validate_identity(
            entry,
            eval_path,
            eval_source_hash=eval_source_hash,
            device_key=device_key,
        )
        baseline = entry["baseline"]
        return {
            "kernel_name": baseline["kernel_name"],
            "avg_ms": baseline["avg_ms"],
        }

    def save(
        self,
        eval_path: str | Path,
        *,
        env: dict[str, Any],
        measurement_options: dict[str, Any],
        baseline: dict[str, Any],
        eval_source_hash: str | None = None,
        device_key: str | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        """Persist a measured ref baseline entry atomically with eval identity binding."""
        if not isinstance(baseline, dict) or not is_valid_ms(baseline.get("avg_ms")):
            raise ValueError("baseline must contain a finite positive 'avg_ms'")
        resolved = Path(eval_path).resolve()
        eval_rel = normalize_path(resolved, self.root)
        computed_hash = eval_source_hash or (
            _file_sha256(resolved) if resolved.is_file() else None
        )
        stored_baseline = {
            k: v for k, v in baseline.items() if k not in ("source", "measured_at")
        }
        stored_baseline["avg_ms"] = float(stored_baseline["avg_ms"])
        entry: dict[str, Any] = {
            "eval_path": eval_rel,
        }
        if computed_hash is not None:
            entry["eval_source_hash"] = computed_hash
        if device_key is not None:
            entry["device_key"] = device_key
        entry.update({
            "env": dict(env),
            "measurement_options": dict(measurement_options),
            "measured_at": now_iso(),
            "baseline": stored_baseline,
        })
        target = self.target_path(eval_path)
        atomic_write_json(target, entry)
        return target, entry
