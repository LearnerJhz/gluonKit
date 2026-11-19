"""Command execution entry points and in-process evaluation task loading."""

import ast
from pathlib import Path
import platform
import runpy
import shutil
import sys
import tempfile

from gluonkit.core.bencher import DEFAULT_ITERS, DEFAULT_WARMUP, bench_task
from gluonkit.core.checker import check_task
from gluonkit.core.history import History, summarize_record
from gluonkit.core.peaks import PeakStore, device_key
from gluonkit.interfaces.backend import Backend


def _ensure_available(backend: Backend) -> None:
    available, reason = backend.available()
    if not available:
        raise RuntimeError(reason)


def load_task(eval_path: str, device: str) -> dict:
    """Load an evaluation script and call its prepare(device) entry point."""
    path = Path(eval_path).resolve()
    if not path.is_file():
        raise ValueError(f"Evaluation script not found: {path}")
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    namespace = runpy.run_path(str(path))
    prepare = namespace.get("prepare")
    if not callable(prepare):
        raise ValueError("Evaluation script must define prepare(device)")
    task = prepare(device)
    if not isinstance(task, dict) or not callable(task.get("run")):
        raise ValueError("prepare() must return a dict with callable 'run'")
    return task


def _prepare_task(eval_path: str, backend: Backend) -> tuple[Path, dict]:
    path = Path(eval_path).resolve()
    if not path.is_file():
        raise ValueError(f"Evaluation script not found: {path}")
    _ensure_available(backend)
    return path, load_task(str(path), backend.device)


def _runtime_dir(*parts: str) -> Path:
    """Resolve a runtime subdirectory under project-local `.gluonkit/`."""
    return (Path.cwd() / ".gluonkit").joinpath(*parts).resolve()


def run_probe(backend: Backend, *, root: str | Path | None = None) -> dict:
    """Verify backend availability, probe device information and attach calibrated peaks."""
    _ensure_available(backend)
    info = backend.probe()
    store = PeakStore(root)
    entry = store.get(info)
    peaks: dict = {"file": store.display_path(), "device_key": device_key(info)}
    if entry is None:
        peaks.update(values=None, hint="not calibrated; run `.venv/bin/python scripts/calibrate_peaks.py` to measure peaks for this device")
    else:
        details = entry.get("details") or {}
        peaks.update(
            updated_at=entry.get("updated_at"),
            values=entry.get("peaks"),
            items={item: _brief_detail(details.get(item) or {}) for item in entry.get("peaks") or {}},
        )
    return {**info.to_dict(), "peaks": peaks}


_BRIEF_DETAIL_KEYS = ("unit", "method", "problem", "theoretical_gbps", "efficiency_pct", "calibrated_at")


def _brief_detail(detail: dict) -> dict:
    return {key: detail[key] for key in _BRIEF_DETAIL_KEYS if key in detail}


def run_layout(expression: str, shape: list[int], backend: Backend, *, hardware: bool = False) -> str:
    """Render a Gluon layout mapping offline."""
    return backend.layout(expression, shape, hardware)


def run_bank_conflicts(
    register_layout: str, shared_layout: str, shape: list[int], dtype: str, backend: Backend,
) -> dict[str, int]:
    """Count shared-memory bank conflicts offline."""
    return backend.bank_conflicts(register_layout, shared_layout, shape, dtype)


def run_dump_info(eval_path: str, backend: Backend) -> dict:
    """Execute run() once and export backend analysis artifacts and summary."""
    path, task = _prepare_task(eval_path, backend)
    return backend.dump_info(task["run"], _runtime_dir("dump_info", path.stem))


def run_check(eval_path: str, backend: Backend) -> None:
    """Load task, JIT-compile via run(), and verify numerical correctness against ref()."""
    _, task = _prepare_task(eval_path, backend)
    check_task(task)


def read_kernel_path(eval_source: bytes, eval_path: Path) -> Path:
    """Return the kernel file named by the KERNEL_PATH string literal (relative to the eval script)."""
    tree = ast.parse(eval_source, filename=str(eval_path))
    value = None
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "KERNEL_PATH"):
            if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                raise ValueError("KERNEL_PATH must be a string literal")
            value = node.value.value
    if value is None:
        raise ValueError(f"{eval_path} must declare KERNEL_PATH = \"<kernel file>\" for gk bench")
    kernel = (eval_path.parent / value).resolve()
    if not kernel.is_file():
        raise ValueError(f"KERNEL_PATH file not found: {kernel}")
    return kernel


def _module_loaded_from(path: Path) -> bool:
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file and Path(file).resolve() == path:
            return True
    return False


def _collect_kernel_summary(task: dict, backend: Backend, kernel_name: str) -> dict | None:
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


def run_bench(eval_path: str, backend: Backend, *, trace: bool = False, no_baseline: bool = False,
              note: str | None = None, parent_id: str | None = None) -> dict:
    """Measure mean GPU kernel time, then record the result and both sources in history."""
    path = Path(eval_path).resolve()
    if not path.is_file():
        raise ValueError(f"Evaluation script not found: {path}")
    eval_source = path.read_bytes()
    kernel_file = read_kernel_path(eval_source, path)
    kernel_source = kernel_file.read_bytes()
    history = History()
    parent = history.get(parent_id)["id"] if parent_id is not None else None

    _, task = _prepare_task(str(path), backend)
    if not _module_loaded_from(kernel_file):
        raise ValueError(f"KERNEL_PATH {kernel_file} was not imported by the evaluation script")
    info = backend.probe()
    peak_entry = PeakStore(history.root).get(info)
    calibrated_peaks = (peak_entry or {}).get("peaks")
    trace_dir = None
    trace_path = None
    if trace:
        root = _runtime_dir("traces")
        root.mkdir(parents=True, exist_ok=True)
        trace_dir = Path(tempfile.mkdtemp(prefix="bench-", dir=root))
        trace_path = str(trace_dir / "trace.json")
    try:
        result = bench_task(task, backend, trace_path=trace_path, no_baseline=no_baseline,
                            calibrated_peaks=calibrated_peaks)
    except BaseException:
        if trace_dir is not None:
            shutil.rmtree(trace_dir, ignore_errors=True)
        raise

    # The recorded snapshot must be exactly the code that was measured.
    if path.read_bytes() != eval_source or kernel_file.read_bytes() != kernel_source:
        raise RuntimeError("Kernel or evaluation source changed during bench; not recorded, rerun")
    bench = result.to_dict()
    summary = _collect_kernel_summary(task, backend, bench["kernel"]["name"])
    record = history.save({
        "kernel_path": history.normalize_path(kernel_file),
        "kernel_source_hash": history.store_source(kernel_source),
        "eval_path": history.normalize_path(path),
        "eval_source_hash": history.store_source(eval_source),
        "parent_id": parent,
        "note": note,
        "platform": info.platform,
        "arch": info.arch,
        "device_info": info.to_dict(),
        "env": {
            "python": platform.python_version(),
            "torch": info.torch_version,
            "triton": info.compiler_version,
        },
        "measurement_options": {
            "warmup": DEFAULT_WARMUP,
            "iters": DEFAULT_ITERS,
            "flush_l2": False,
            "timing": "torch.profiler device kernel mean per call",
        },
        "bench": bench,
        "summary": summary,
    })
    return {"history_id": record["id"], **bench}


def run_history(
    target: str | None = None, *, limit: int | None = None, root: str | Path | None = None,
) -> dict:
    """List recorded bench runs, optionally filtered by kernel/eval path and tailed by limit."""
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be a positive integer")
    records = History(root=root).list(target=target)
    total = len(records)
    if limit is not None:
        records = records[-limit:]
    return {
        "total": total,
        "records": [summarize_record(record) for record in records],
    }


def run_restore(record_id: str, *, root: str | Path | None = None) -> dict:
    """Restore a kernel file to the exact source snapshot of a history record."""
    return History(root=root).restore(record_id)
