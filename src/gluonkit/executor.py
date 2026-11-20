"""Command execution entry points and in-process evaluation task loading."""

from pathlib import Path
import runpy
import sys

from gluonkit.core.baseline import BaselineStore
from gluonkit.core.bencher import record_baseline, run_benchmark
from gluonkit.core.checker import check_task
from gluonkit.core.history import History, summarize_record
from gluonkit.core.peaks import PeakStore, device_key
from gluonkit.core.storage import gluonkit_dir
from gluonkit.interfaces.backend import Backend

_BRIEF_DETAIL_KEYS = ("unit", "method", "problem", "theoretical_gbps", "efficiency_pct", "calibrated_at")


def _ensure_available(backend: Backend) -> None:
    available, reason = backend.available()
    if not available:
        raise RuntimeError(reason)


def _read_eval_file(eval_path: str) -> tuple[Path, bytes]:
    path = Path(eval_path).resolve()
    if not path.is_file():
        raise ValueError(f"Evaluation script not found: {path}")
    return path, path.read_bytes()


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


def _prepare_task(eval_path: str | Path, backend: Backend) -> tuple[Path, dict]:
    path = Path(eval_path).resolve()
    if not path.is_file():
        raise ValueError(f"Evaluation script not found: {path}")
    _ensure_available(backend)
    return path, load_task(str(path), backend.device)


def resolve_kernel_path(eval_path: Path) -> Path:
    """Resolve the paired `<name>.py` kernel file next to an `eval_<name>.py` evaluation script."""
    if not (eval_path.name.startswith("eval_") and eval_path.suffix == ".py" and len(eval_path.stem) > 5):
        raise ValueError(f"Evaluation script must be named eval_<kernel>.py: {eval_path.name}")
    kernel = (eval_path.parent / eval_path.name.removeprefix("eval_")).resolve()
    if not kernel.is_file():
        raise ValueError(f"Kernel file not found: {kernel}")
    return kernel


def _module_loaded_from(path: Path) -> bool:
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file and Path(file).resolve() == path:
            return True
    return False


def _brief_detail(detail: dict) -> dict:
    return {key: detail[key] for key in _BRIEF_DETAIL_KEYS if key in detail}


def run_probe(backend: Backend, *, root: str | Path | None = None) -> dict:
    """Verify backend availability, probe device information and attach calibrated peaks."""
    _ensure_available(backend)
    info = backend.probe()
    store = PeakStore(root)
    entry = store.get(info)
    peaks: dict = {"file": store.display_path(), "device_key": device_key(info)}
    if entry is None:
        peaks.update(
            values=None,
            hint="not calibrated; run `.venv/bin/python scripts/calibrate_peaks.py` to measure peaks for this device",
        )
    else:
        details = entry.get("details") or {}
        peaks.update(
            updated_at=entry.get("updated_at"),
            values=entry.get("peaks"),
            items={item: _brief_detail(details.get(item) or {}) for item in entry.get("peaks") or {}},
        )
    return {**info.to_dict(), "peaks": peaks}


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
    return backend.dump_info(task["run"], gluonkit_dir(None, "dump_info", path.stem))


def run_check(eval_path: str, backend: Backend) -> None:
    """Load task, JIT-compile via run(), and verify numerical correctness against ref()."""
    _, task = _prepare_task(eval_path, backend)
    check_task(task)


def run_baseline(
    eval_path: str,
    backend: Backend,
    *,
    root: str | Path | None = None,
) -> dict:
    """Return existing baseline JSON if present, otherwise measure `ref()` once and save it."""
    path = Path(eval_path).resolve()
    if not path.is_file():
        raise ValueError(f"Evaluation script not found: {path}")
    existing = BaselineStore(root).load_record(path)
    if existing is not None:
        return existing
    _, task = _prepare_task(path, backend)
    return record_baseline(path, task, backend, root=root)


def run_bench(
    eval_path: str,
    backend: Backend,
    *,
    trace: bool = False,
    note: str | None = None,
    parent_id: str | None = None,
    root: str | Path | None = None,
) -> dict:
    """Measure mean GPU kernel time and compare against stored baseline."""
    path, eval_source = _read_eval_file(eval_path)
    kernel_file = resolve_kernel_path(path)
    kernel_source = kernel_file.read_bytes()
    _, task = _prepare_task(path, backend)
    if not _module_loaded_from(kernel_file):
        raise ValueError(f"Kernel file {kernel_file} was not imported by the evaluation script")

    return run_benchmark(
        path,
        kernel_file,
        task,
        backend,
        eval_source=eval_source,
        kernel_source=kernel_source,
        trace=trace,
        note=note,
        parent_id=parent_id,
        root=root,
    )


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
