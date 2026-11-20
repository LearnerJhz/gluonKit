"""Shared helpers for project-local `.gluonkit/` JSON storage and path normalization."""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

GLUONKIT_DIRNAME = ".gluonkit"


def now_iso() -> str:
    """Return local timezone-aware ISO-8601 timestamp with second precision."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve_root(root: str | Path | None = None) -> Path:
    """Resolve the project root directory (defaults to current working directory)."""
    return Path(root or Path.cwd()).resolve()


def gluonkit_dir(root: str | Path | None = None, *parts: str) -> Path:
    """Resolve `.gluonkit` (or a subdirectory inside it) under `root`."""
    return resolve_root(root).joinpath(GLUONKIT_DIRNAME, *parts)


def normalize_path(path: str | Path, root: str | Path | None = None) -> str:
    """Return project-relative POSIX path when inside `root`, otherwise absolute POSIX path."""
    base = resolve_root(root)
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_json(path: Path, *, recovery_hint: str = "") -> dict[str, Any]:
    """Read and validate a JSON object from `path`."""
    suffix = f"; {recovery_hint}" if recovery_hint else ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON ({exc}){suffix}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} has an unknown layout{suffix}")
    return data


def atomic_write_json(path: Path, data: Any) -> None:
    """Atomically write JSON to `path` via a temporary sibling file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
