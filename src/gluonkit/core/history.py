"""Experiment history: `.gluonkit/history.jsonl` plus hashed source snapshots."""

from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path
import uuid

# Record fields that must match before two measurements may be ranked together.
COMPARABLE_FIELDS = ("kernel_path", "platform", "arch", "env",
                     "eval_source_hash", "measurement_options")  # plus device_info.device_name
METRICS = ("avg_ms", "baseline_time_ms", "speedup_vs_baseline",
           "tflops", "gbps", "roofline_pct", "bound_by")


def source_hash(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def comparable_key(record: dict) -> str:
    """Group key: same kernel, device, platform, environment, eval source and options."""
    values = {name: record.get(name) for name in COMPARABLE_FIELDS}
    values["device_name"] = (record.get("device_info") or {}).get("device_name")
    return json.dumps(values, sort_keys=True)


def _bench_metrics(record: dict) -> dict:
    """Read comparison metrics from either nested or historical flat bench records."""
    bench = record.get("bench") or {}
    if "metrics" not in bench:
        return {**bench, "avg_ms": bench.get("avg_ms", bench.get("device_time_ms"))}
    metrics = bench["metrics"]
    return {
        "avg_ms": bench["kernel"]["avg"],
        "baseline_time_ms": (bench.get("baseline") or {}).get("avg_ms"),
        "speedup_vs_baseline": metrics["speedup"],
        "tflops": metrics["tflops"],
        "gbps": metrics["gbps"],
        "roofline_pct": metrics["roofline"],
    }


def is_rankable(record: dict) -> bool:
    # Records are only written by successful `gk bench`; correctness is `gk check`'s job.
    return _bench_metrics(record).get("avg_ms") is not None


class History:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or Path.cwd()).resolve()
        self.directory = self.root / ".gluonkit"
        self.path = self.directory / "history.jsonl"
        self.sources = self.directory / "sources"

    def normalize_path(self, path: str | Path) -> str:
        """Project-relative POSIX path when inside the project, otherwise absolute."""
        resolved = Path(path).resolve()
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

    def read_source(self, digest: str) -> str:
        return (self.sources / f"{digest}.py").read_text()

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

    def list(self, kernel: str | Path | None = None) -> list[dict]:
        records = self._records()
        if kernel is not None:
            kernel_path = self.normalize_path(kernel)
            records = [r for r in records if r.get("kernel_path") == kernel_path]
        return records

    def best(self, kernel: str | Path | None = None, like_id: str | None = None) -> dict | None:
        """Fastest record within one strictly comparable group.

        Raises ValueError when several comparable groups exist and like_id is not given.
        """
        if kernel is None and like_id is None:
            raise ValueError("best requires a kernel path or --like <id>")
        if like_id is not None:
            like = self.get(like_id)
            if kernel is not None and self.normalize_path(kernel) != like.get("kernel_path"):
                raise ValueError(f"Record {like['id']} is for kernel {like.get('kernel_path')}")
            key = comparable_key(like)
            candidates = [r for r in self._records()
                          if is_rankable(r) and comparable_key(r) == key]
        else:
            candidates = [r for r in self.list(kernel) if is_rankable(r)]
            groups: dict[str, list[dict]] = {}
            for record in candidates:
                groups.setdefault(comparable_key(record), []).append(record)
            if len(groups) > 1:
                lines = [f"{len(groups)} incomparable evaluation/measurement groups; "
                         "choose one with --like <id>:"]
                for members in groups.values():
                    latest = members[-1]
                    lines.append(f"  {latest['id']}  eval {latest.get('eval_source_hash', '')[:12]}  "
                                 f"{(latest.get('device_info') or {}).get('device_name')}  "
                                 f"({len(members)} records)")
                raise ValueError("\n".join(lines))
        if not candidates:
            return None
        return min(candidates, key=lambda r: _bench_metrics(r)["avg_ms"])

    def diff(self, id_a: str, id_b: str) -> dict:
        a, b = self.get(id_a), self.get(id_b)
        reasons = []
        for name in COMPARABLE_FIELDS:
            if a.get(name) != b.get(name):
                reasons.append(f"{name} differs")
        device_a = (a.get("device_info") or {}).get("device_name")
        device_b = (b.get("device_info") or {}).get("device_name")
        if device_a != device_b:
            reasons.append("device_name differs")
        for record in (a, b):
            if not is_rankable(record):
                reasons.append(f"{record['id']} has no bench result")

        def source_diff(field: str, path_field: str) -> str:
            if a.get(field) == b.get(field):
                return ""
            lines_a = self.read_source(a[field]).splitlines(keepends=True)
            lines_b = self.read_source(b[field]).splitlines(keepends=True)
            return "".join(difflib.unified_diff(
                lines_a, lines_b,
                fromfile=f"{a['id']}:{a.get(path_field)}", tofile=f"{b['id']}:{b.get(path_field)}",
            ))

        metrics = {}
        bench_a, bench_b = _bench_metrics(a), _bench_metrics(b)
        for name in METRICS:
            value_a, value_b = bench_a.get(name), bench_b.get(name)
            change = None
            if (isinstance(value_a, (int, float)) and isinstance(value_b, (int, float))
                    and value_a != 0):
                change = (value_b - value_a) / value_a * 100
            metrics[name] = {"a": value_a, "b": value_b, "change_pct": change}
        return {
            "a": a["id"],
            "b": b["id"],
            "comparable": not reasons,
            "reasons": reasons,
            "metrics": metrics,
            "kernel_diff": source_diff("kernel_source_hash", "kernel_path"),
            "eval_diff": source_diff("eval_source_hash", "eval_path"),
        }
