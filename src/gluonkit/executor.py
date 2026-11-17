"""Task loading and in-process execution for evaluation scripts."""

from pathlib import Path
import runpy
import sys
import tempfile
import traceback

from gluonkit.backends.nvidia.backend import NvidiaBackend
from gluonkit.backends.nvidia.compiler import export_ir
from gluonkit.core.checker import check_task
from gluonkit.core.bencher import bench_task
from gluonkit.interfaces.types import BenchResult, CheckResult, CompileError, CompileResult


def load_task(eval_path: str, device: str) -> dict:
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


def run_compile(eval_path: str) -> CompileResult:
    path = Path(eval_path).resolve()
    if not path.is_file():
        raise ValueError(f"Evaluation script not found: {path}")
    try:
        backend = NvidiaBackend()
        available, reason = backend.available()
        if not available:
            raise RuntimeError(reason)
        task = load_task(str(path), backend.device)
        root = Path("runs").resolve()
        root.mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="compile-", dir=root))
        artifacts = export_ir(task["run"], directory / "artifacts")
        return CompileResult(ok=True, errors=[], artifacts=artifacts)
    except Exception:
        return CompileResult(ok=False, errors=[CompileError(traceback.format_exc())])


def run_check(eval_path: str) -> CheckResult:
    path = Path(eval_path).resolve()
    if not path.is_file():
        raise ValueError(f"Evaluation script not found: {path}")
    try:
        backend = NvidiaBackend()
        available, reason = backend.available()
        if not available:
            raise RuntimeError(reason)
        task = load_task(str(path), backend.device)
        return check_task(task)
    except Exception:
        return CheckResult(passed=False, error=traceback.format_exc())


def run_bench(eval_path: str, *, trace: bool = False,
              no_baseline: bool = False) -> BenchResult:
    backend = NvidiaBackend()
    available, reason = backend.available()
    if not available:
        raise RuntimeError(reason)
    task = load_task(eval_path, backend.device)
    trace_path = None
    if trace:
        root = Path("runs").resolve()
        root.mkdir(exist_ok=True)
        trace_path = str(Path(tempfile.mkdtemp(prefix="bench-", dir=root)) / "trace.json")
    return bench_task(task, backend, trace_path=trace_path, no_baseline=no_baseline)
