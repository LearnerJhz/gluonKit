"""Benchmark and baseline measurement, roofline calculation, and experiment recording."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator

from gluonkit.core.baseline import BaselineStore, device_key, is_valid_ms
from gluonkit.core.history import History
from gluonkit.core.roofline import calculate_roofline, load_peaks, resolve_peaks
from gluonkit.interfaces.backend import Backend
from gluonkit.interfaces.types import BenchResult

DEFAULT_ITERS = 100
DEFAULT_WARMUP = 10


@dataclass(frozen=True)
class BenchOptions:
    """Sampling options shared between `gk bench` and `gk baseline`."""

    warmup: int = DEFAULT_WARMUP
    iters: int = DEFAULT_ITERS
    flush_l2: bool = True

    def with_overrides(
        self, *, iters: int | None = None, warmup: int | None = None, flush_l2: bool | None = None,
    ) -> "BenchOptions":
        return BenchOptions(
            warmup=self.warmup if warmup is None else warmup,
            iters=self.iters if iters is None else iters,
            flush_l2=self.flush_l2 if flush_l2 is None else flush_l2,
        )


def measure_baseline(
    task: dict,
    backend: Backend,
    *,
    options: BenchOptions | None = None,
    iters: int | None = None,
    warmup: int | None = None,
) -> dict[str, Any]:
    """Measure only `task['ref']` under the standard profiling protocol."""
    opts = (options or BenchOptions()).with_overrides(iters=iters, warmup=warmup)
    ref = task.get("ref")
    if not callable(ref):
        raise ValueError("Evaluation script must define callable 'ref' for baseline")
    measured = backend.profile(ref, opts.iters, opts.warmup, flush_l2=opts.flush_l2)
    if not is_valid_ms(measured.avg_ms):
        raise RuntimeError(f"baseline ref() produced non-positive device kernel time: {measured.avg_ms!r}")
    return measured.to_baseline_dict()


def bench_task(
    task: dict,
    backend: Backend,
    *,
    options: BenchOptions | None = None,
    iters: int | None = None,
    warmup: int | None = None,
    baseline: dict[str, Any] | None = None,
    trace_path: str | None = None,
    peak_table: dict[str, float] | None = None,
) -> BenchResult:
    """Measure `task['run']` and compute speedup + theoretical roofline metrics from `.gluonkit/peaks.json`."""
    opts = (options or BenchOptions()).with_overrides(iters=iters, warmup=warmup)
    peaks = resolve_peaks(task, peak_table or {})
    measured = backend.profile(
        task["run"], opts.iters, opts.warmup, flush_l2=opts.flush_l2, trace_path=trace_path,
    )
    speedup = float(baseline["avg_ms"]) / measured.avg_ms if baseline is not None and measured.avg_ms > 0 else None
    rates = calculate_roofline(
        measured.avg_ms,
        flops=task.get("flops"),
        bytes_moved=task.get("bytes_moved"),
        peaks=peaks,
    )
    return BenchResult(
        kernel_profile=measured,
        metrics={"speedup": speedup, **rates},
        baseline=baseline,
        peaks=peaks,
    )


def collect_kernel_summary(task: dict, backend: Backend, kernel_name: str) -> dict | None:
    """Capture optional backend resource summary after benchmark timing finishes."""
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = backend.dump_info(task["run"], tmp_dir)
    except NotImplementedError:
        return None
    summary_map = info.get("summary") if isinstance(info, dict) else None
    if not isinstance(summary_map, dict) or not summary_map:
        return None
    if kernel_name and kernel_name in summary_map:
        return summary_map[kernel_name]
    if len(summary_map) == 1:
        return next(iter(summary_map.values()))
    return summary_map


def _verify_unchanged(snapshots: list[tuple[Path, bytes]], message: str) -> None:
    for file_path, expected in snapshots:
        if file_path.read_bytes() != expected:
            raise RuntimeError(message)


@contextmanager
def _trace_session(trace: bool, runtime_dir: Path) -> Iterator[str | None]:
    if not trace:
        yield None
        return
    traces_root = runtime_dir / "traces"
    traces_root.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(tempfile.mkdtemp(prefix="bench-", dir=traces_root))
    try:
        yield str(trace_dir / "trace.json")
    except BaseException:
        shutil.rmtree(trace_dir, ignore_errors=True)
        raise


def record_baseline(
    eval_path: Path,
    task: dict,
    backend: Backend,
    *,
    root: str | Path | None = None,
    options: BenchOptions | None = None,
) -> dict[str, Any]:
    """Return existing baseline file if present, otherwise measure `task['ref']` once and save."""
    store = BaselineStore(root)
    existing = store.load_record(eval_path)
    if existing is not None:
        return existing

    info = backend.probe()
    dev_key = device_key(info)
    opts = options or BenchOptions()
    measured_baseline = measure_baseline(task, backend, options=opts)
    saved_path, entry = store.save(
        eval_path,
        baseline=measured_baseline,
        device_key=dev_key,
    )
    return {
        "file": store.display_path(saved_path),
        **entry,
    }


def run_benchmark(
    eval_path: Path,
    kernel_path: Path,
    task: dict,
    backend: Backend,
    *,
    eval_source: bytes,
    kernel_source: bytes,
    trace: bool = False,
    note: str | None = None,
    parent_id: str | None = None,
    root: str | Path | None = None,
    options: BenchOptions | None = None,
) -> dict[str, Any]:
    """Run target benchmark against stored baseline and `.gluonkit/peaks.json`, then persist history."""
    opts = options or BenchOptions()
    history = History(root=root)
    parent = history.get(parent_id)["id"] if parent_id is not None else None
    info = backend.probe()
    peak_table = load_peaks(history.root)

    eval_rel = history.normalize_path(eval_path)
    stored_baseline = BaselineStore(history.root).lookup(eval_path)

    with _trace_session(trace, history.directory) as trace_path:
        result = bench_task(
            task,
            backend,
            options=opts,
            baseline=stored_baseline,
            trace_path=trace_path,
            peak_table=peak_table,
        )
        bench = result.to_dict()
        summary = collect_kernel_summary(task, backend, bench["kernel"]["name"])
        _verify_unchanged(
            [(eval_path, eval_source), (kernel_path, kernel_source)],
            "Kernel or evaluation source changed during bench; not recorded, rerun",
        )
        record = history.save({
            "kernel_path": history.normalize_path(kernel_path),
            "kernel_source_hash": history.store_source(kernel_source),
            "eval_path": eval_rel,
            "eval_source_hash": history.store_source(eval_source),
            "parent_id": parent,
            "note": note,
            "device_info": info.to_dict(),
            "bench": bench,
            "summary": summary,
        })

    return {"history_id": record["id"], **bench, "summary": summary}
