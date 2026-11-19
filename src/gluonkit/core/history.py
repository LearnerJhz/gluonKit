"""Experiment history: `.gluonkit/history.jsonl` plus hashed source snapshots."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import uuid


def source_hash(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def _bench_metrics(record: dict) -> dict:
    """Read comparison metrics from either nested or historical flat bench records."""
    bench = record.get("bench") or {}
    if "metrics" not in bench:
        return {**bench, "avg_ms": bench.get("avg_ms", bench.get("device_time_ms"))}
    metrics = bench["metrics"]
    return {
        "avg_ms": bench["kernel"]["avg"],
        "baseline_time_ms": (bench.get("baseline") or {}).get("avg_ms"),
        "speedup_vs_baseline": metrics.get("speedup"),
        "tflops": metrics.get("tflops"),
        "gbps": metrics.get("gbps"),
        "roofline_pct": metrics.get("roofline"),
    }


def summarize_record(record: dict) -> dict:
    """Return a compact, agent-friendly summary of a history record."""
    metrics = _bench_metrics(record)
    return {
        "id": record["id"],
        "parent_id": record.get("parent_id"),
        "kernel_path": record.get("kernel_path"),
        "eval_path": record.get("eval_path"),
        "trace_path": (record.get("bench") or {}).get("trace_path"),
        "note": record.get("note"),
        "avg_ms": metrics.get("avg_ms"),
        "baseline_ms": metrics.get("baseline_time_ms"),
        "speedup": metrics.get("speedup_vs_baseline"),
        "tflops": metrics.get("tflops"),
        "gbps": metrics.get("gbps"),
        "roofline": metrics.get("roofline_pct"),
        "summary": record.get("summary"),
    }


class History:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or Path.cwd()).resolve()
        self.directory = self.root / ".gluonkit"
        self.path = self.directory / "history.jsonl"
        self.sources = self.directory / "sources"

    def normalize_path(self, path: str | Path) -> str:
        """Project-relative POSIX path when inside the project, otherwise absolute."""
        candidate = Path(path)
        resolved = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError:
            return resolved.as_posix()

    def store_source(self, source: bytes) -> str:
        digest = source_hash(source)
        self.sources.mkdir(parents=True, exist_ok=True)
        target = self.sources / f"{digest}.py"
        if not target.exists():
            target.write_bytes(source)
        return digest

    def save(self, record: dict) -> dict:
        record = dict(record)
        record.setdefault("id", uuid.uuid4().hex[:8])
        record.setdefault("timestamp", datetime.now().astimezone().isoformat(timespec="seconds"))
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return record

    def _records(self) -> list[dict]:
        if not self.path.is_file():
            return []
        with self.path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def get(self, record_id: str) -> dict:
        """Look up a record by full ID or unique ID prefix."""
        matches = [r for r in self._records() if r["id"].startswith(record_id)]
        exact = [r for r in matches if r["id"] == record_id]
        if exact:
            return exact[0]
        if not matches:
            raise ValueError(f"No history record matches id {record_id!r}")
        if len(matches) > 1:
            ids = ", ".join(r["id"] for r in matches)
            raise ValueError(f"Ambiguous id {record_id!r}: {ids}")
        return matches[0]

    def list(self, kernel: str | Path | None = None, *, target: str | Path | None = None) -> list[dict]:
        records = self._records()
        if kernel is not None:
            kernel_path = self.normalize_path(kernel)
            records = [r for r in records if r.get("kernel_path") == kernel_path]
        if target is not None:
            normalized = self.normalize_path(target)
            records = [
                r for r in records
                if r.get("kernel_path") == normalized or r.get("eval_path") == normalized
            ]
        return records

    def restore(self, record_id: str) -> dict:
        """Restore the kernel source snapshot for record_id back to its kernel_path."""
        record = self.get(record_id)
        kernel_rel = record.get("kernel_path")
        digest = record.get("kernel_source_hash")
        if not kernel_rel or not digest:
            raise ValueError(f"Record {record['id']} has no kernel source snapshot")
        snapshot_path = self.sources / f"{digest}.py"
        if not snapshot_path.is_file():
            raise FileNotFoundError(f"Kernel snapshot not found: {snapshot_path}")
        snapshot = snapshot_path.read_bytes()
        if source_hash(snapshot) != digest:
            raise ValueError(f"Snapshot integrity check failed for {digest}")
        raw_target = Path(kernel_rel)
        target = raw_target.resolve() if raw_target.is_absolute() else (self.root / raw_target).resolve()
        previous_hash = source_hash(target.read_bytes()) if target.is_file() else None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(snapshot)
        return {
            "id": record["id"],
            "kernel_path": self.normalize_path(target),
            "kernel_source_hash": digest,
            "previous_hash": previous_hash,
            "changed": previous_hash != digest,
            "note": record.get("note"),
        }
