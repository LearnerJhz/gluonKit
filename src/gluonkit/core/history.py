"""Experiment history: `.gluonkit/history.jsonl` plus hashed source snapshots."""

import hashlib
import json
from pathlib import Path
import uuid

from gluonkit.core.storage import gluonkit_dir, normalize_path, resolve_root


def source_hash(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def _extract_metrics(bench: dict) -> dict:
    """Extract summary metrics from nested `BenchResult` or legacy flat records."""
    if "metrics" not in bench:
        return {
            "avg_ms": bench.get("avg_ms", bench.get("device_time_ms")),
            "median_ms": bench.get("median_ms"),
            "min_ms": bench.get("min_ms"),
            "p95_ms": bench.get("p95_ms"),
            "std_ms": bench.get("std_ms"),
            "baseline_ms": bench.get("baseline_time_ms", bench.get("baseline_ms")),
            "speedup": bench.get("speedup_vs_baseline", bench.get("speedup")),
            "tflops": bench.get("tflops"),
            "gbps": bench.get("gbps"),
            "roofline": bench.get("roofline_pct", bench.get("roofline")),
        }
    kernel = bench.get("kernel") or {}
    metrics = bench["metrics"]
    extracted = {
        "avg_ms": kernel.get("avg"),
        "median_ms": kernel.get("median"),
        "min_ms": kernel.get("min"),
        "p95_ms": kernel.get("p95"),
        "std_ms": kernel.get("std"),
        "baseline_ms": (bench.get("baseline") or {}).get("avg_ms"),
        "speedup": metrics.get("speedup"),
        "tflops": metrics.get("tflops"),
        "gbps": metrics.get("gbps"),
        "roofline": metrics.get("roofline"),
    }
    if kernel.get("iters") is not None:
        extracted["iters"] = kernel.get("iters")
    if metrics.get("roofline_model") is not None:
        extracted["roofline_model"] = metrics.get("roofline_model")
    return extracted


def summarize_record(record: dict, *, parent_avg_ms: float | None = None) -> dict:
    """Return a compact, agent-friendly summary of a history record."""
    bench = record.get("bench") or {}
    metrics = _extract_metrics(bench)
    avg_ms = metrics.get("avg_ms")
    delta_pct = None
    if (
        isinstance(avg_ms, (int, float))
        and isinstance(parent_avg_ms, (int, float))
        and parent_avg_ms > 0
    ):
        delta_pct = (avg_ms - parent_avg_ms) / parent_avg_ms * 100.0

    summary_out = {
        "id": record["id"],
        "parent_id": record.get("parent_id"),
        "kernel_path": record.get("kernel_path"),
        "eval_path": record.get("eval_path"),
        "trace_path": bench.get("trace_path"),
        "note": record.get("note"),
        **metrics,
    }
    if delta_pct is not None:
        summary_out["delta_vs_parent_pct"] = delta_pct
    verified = record.get("verified", bench.get("verified"))
    if verified is not None:
        summary_out["verified"] = verified
    summary_out["summary"] = record.get("summary")
    return summary_out


class History:
    def __init__(self, root: str | Path | None = None):
        self.root = resolve_root(root)
        self.directory = gluonkit_dir(self.root)
        self.path = self.directory / "history.jsonl"
        self.sources = self.directory / "sources"

    def normalize_path(self, path: str | Path) -> str:
        """Project-relative POSIX path when inside the project, otherwise absolute."""
        return normalize_path(path, self.root)

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
